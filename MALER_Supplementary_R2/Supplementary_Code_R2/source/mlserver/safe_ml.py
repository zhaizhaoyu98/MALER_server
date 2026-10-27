"""Leakage-safe machine-learning utilities used by MALER validation workflows.

The legacy web implementation historically performed imputation, scaling and
feature ranking before cross-validation.  The helpers in this module keep every
learned preprocessing operation inside a scikit-learn ``Pipeline`` so it is fit
on the training fold only.  The module is intentionally compatible with the
Python 3.7 / scikit-learn 1.0 environment used by the project.
"""

from __future__ import absolute_import

import hashlib
import hmac
import io
import json
import os
import platform
import tempfile
import zipfile
from functools import partial
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, label_binarize

from .cv_policy import make_splits, split_index_hash, validate_group_feature_method


MODEL_BUNDLE_VERSION = 1
DEFAULT_RANDOM_STATE = 10


class DataValidationError(ValueError):
    """Raised when an uploaded matrix cannot be analysed safely."""


class ModelBundleError(ValueError):
    """Raised when a model bundle is malformed or fails signature validation."""


class FoldImputer(SimpleImputer):
    """Preserve the feature contract by rejecting all-missing training columns."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        empty = np.flatnonzero(np.isnan(array).all(axis=0))
        if len(empty):
            raise DataValidationError(
                "A training fold has all-missing feature columns (indices %s); "
                "use more observed data or revise the development feature set."
                % ", ".join(map(str, empty[:20])))
        return super(FoldImputer, self).fit(X, y)


def _as_frame(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def validate_feature_matrix(X, max_missing_fraction=0.80, reject_zero_variance=True):
    """Validate and return a numeric feature frame.

    The function never learns replacement values; missing-value imputation is
    deliberately left to the fold-local pipeline.
    """
    frame = _as_frame(X)
    if frame.empty:
        raise DataValidationError("The feature matrix is empty.")
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].astype(str).tolist()[:10]
        raise DataValidationError("Duplicate sample identifiers: %s" % ", ".join(duplicates))
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()[:10]
        raise DataValidationError("Duplicate feature identifiers: %s" % ", ".join(duplicates))

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    introduced = numeric.isna() & ~frame.isna()
    if introduced.any().any():
        bad = introduced.columns[introduced.any()].astype(str).tolist()[:10]
        raise DataValidationError("Non-numeric values were found in features: %s" % ", ".join(bad))
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise DataValidationError("Infinite feature values are not allowed.")

    missing_fraction = numeric.isna().mean(axis=0)
    all_missing = missing_fraction[missing_fraction == 1.0].index.astype(str).tolist()
    if all_missing:
        raise DataValidationError("All-missing features: %s" % ", ".join(all_missing[:10]))
    excessive = missing_fraction[missing_fraction > max_missing_fraction].index.astype(str).tolist()
    if excessive:
        raise DataValidationError(
            "Features exceeding the %.0f%% missing-value limit: %s"
            % (max_missing_fraction * 100.0, ", ".join(excessive[:10]))
        )
    if reject_zero_variance:
        variance = numeric.var(axis=0, skipna=True)
        zero_variance = variance[variance == 0.0].index.astype(str).tolist()
        if zero_variance:
            raise DataValidationError("Zero-variance features: %s" % ", ".join(zero_variance[:10]))
    return numeric


def validate_target(y, task, require_class_counts=True):
    series = pd.Series(y).copy()
    if series.isna().any():
        raise DataValidationError("Outcome values must not be missing.")
    if task in ("binary", "multiclass", "classification"):
        counts = series.value_counts()
        if require_class_counts and len(counts) < 2:
            raise DataValidationError("Classification requires at least two outcome classes.")
        if require_class_counts and counts.min() < 2:
            raise DataValidationError("Each outcome class must contain at least two samples.")
    elif task == "regression":
        converted = pd.to_numeric(series, errors="coerce")
        if converted.isna().any() or np.isinf(converted.to_numpy(dtype=float)).any():
            raise DataValidationError("Regression outcomes must be finite numeric values.")
        series = converted
    else:
        raise DataValidationError("Unsupported task: %s" % task)
    return series.to_numpy()


def _make_scaler(name):
    normalized = (name or "standard").lower()
    if normalized in ("none", "n", "off"):
        return "passthrough"
    if normalized in ("standard", "zscore", "z-score"):
        return StandardScaler()
    if normalized in ("minmax", "min-max"):
        return MinMaxScaler()
    if normalized in ("maxabs", "max-abs"):
        return MaxAbsScaler()
    raise ValueError("Unknown scaler: %s" % name)


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Fold-local minimum-redundancy/maximum-relevance feature selector."""

    def __init__(self, task="classification", k=20):
        self.task = task
        self.k = k

    def fit(self, X, y):
        array = np.asarray(X, dtype=float)
        columns = list(range(array.shape[1]))
        frame = pd.DataFrame(array, columns=columns)
        target = pd.Series(np.asarray(y))
        k = min(max(1, int(self.k)), array.shape[1])
        try:
            if self.task == "classification":
                from mrmr import mrmr_classif
                selected = mrmr_classif(X=frame, y=target, K=k, show_progress=False, n_jobs=1)
            else:
                from mrmr import mrmr_regression
                selected = mrmr_regression(X=frame, y=target, K=k, show_progress=False, n_jobs=1)
        except ImportError:
            raise ValueError("MRMR requires the installed 'mrmr-selection' package.")
        self.selected_indices_ = np.sort(np.asarray(selected, dtype=int))
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask


class CandidateSequentialSelector(BaseEstimator, TransformerMixin):
    """Fold-local FSS/BSS with an explicit 50-feature candidate cap."""

    def __init__(self, estimator, task="classification", k=20, direction="forward",
                 max_candidates=50, cv=3, random_state=DEFAULT_RANDOM_STATE):
        self.estimator = estimator
        self.task = task
        self.k = k
        self.direction = direction
        self.max_candidates = max_candidates
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        array = np.asarray(X, dtype=float)
        candidate_count = min(int(self.max_candidates), array.shape[1])
        selected_count = min(max(1, int(self.k)), candidate_count)
        score_func = f_classif if self.task == "classification" else f_regression
        self.prefilter_ = SelectKBest(score_func=score_func, k=candidate_count).fit(array, y)
        candidate_indices = self.prefilter_.get_support(indices=True)
        candidate_x = self.prefilter_.transform(array)
        if selected_count == candidate_count:
            within_candidates = np.arange(candidate_count)
            self.selector_ = None
        else:
            if self.task == "classification":
                cv = StratifiedKFold(
                    n_splits=int(self.cv), shuffle=True, random_state=int(self.random_state))
                scoring = "balanced_accuracy"
            else:
                cv = KFold(
                    n_splits=int(self.cv), shuffle=True, random_state=int(self.random_state))
                scoring = "r2"
            self.selector_ = SequentialFeatureSelector(
                clone(self.estimator), n_features_to_select=selected_count,
                direction=self.direction, scoring=scoring, cv=cv, n_jobs=1)
            self.selector_.fit(candidate_x, y)
            within_candidates = self.selector_.get_support(indices=True)
        self.selected_indices_ = np.sort(candidate_indices[within_candidates])
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask


def _make_feature_reducer(task, method="select", k=20, random_state=DEFAULT_RANDOM_STATE,
                          estimator=None):
    normalized = (method or "select").lower()
    if normalized in ("select", "selection", "selectkbest", "univariate"):
        score_func = f_classif if task == "classification" else f_regression
        return SelectKBest(score_func=score_func, k=k)
    if normalized in ("pca", "principal-components", "principal_components"):
        return PCA(n_components=k, svd_solver="auto", random_state=random_state)
    if normalized in ("mrmr", "minimum-redundancy-maximum-relevance"):
        return MRMRSelector(task=task, k=k)
    if normalized in ("fss", "forward", "forward-selection"):
        return CandidateSequentialSelector(
            estimator=estimator, task=task, k=k, direction="forward",
            random_state=random_state)
    if normalized in ("bss", "backward", "backward-selection"):
        return CandidateSequentialSelector(
            estimator=estimator, task=task, k=k, direction="backward",
            random_state=random_state)
    if normalized in ("none", "off", "passthrough"):
        return "passthrough"
    raise ValueError("Unknown feature reduction method: %s" % method)


def _classification_estimator(estimator):
    # Custom forms still construct the native XGBoost class; adapt that path too.
    if type(estimator).__module__ == 'xgboost.sklearn' and type(estimator).__name__ == 'XGBClassifier':
        from .labelled_xgboost import LabelledXGBClassifier
        return LabelledXGBClassifier(**estimator.get_params())
    return clone(estimator)


def build_classification_pipeline(
        estimator, k=20, scaler="standard", imputer="median",
        feature_method="select", random_state=DEFAULT_RANDOM_STATE):
    estimator = _classification_estimator(estimator)
    return Pipeline(
        [
            ("imputer", FoldImputer(strategy=imputer)),
            ("scaler", _make_scaler(scaler)),
            ("selector", _make_feature_reducer(
                "classification", feature_method, k, random_state, estimator)),
            ("model", clone(estimator)),
        ]
    )


def build_regression_pipeline(
        estimator, k=20, scaler="standard", imputer="median",
        feature_method="select", random_state=DEFAULT_RANDOM_STATE):
    return Pipeline(
        [
            ("imputer", FoldImputer(strategy=imputer)),
            ("scaler", _make_scaler(scaler)),
            ("selector", _make_feature_reducer(
                "regression", feature_method, k, random_state, estimator)),
            ("model", clone(estimator)),
        ]
    )


class SampleRankTransformer(BaseEstimator, TransformerMixin):
    """Convert each sample to within-sample percentile ranks.

    The operation is independent for every sample and therefore avoids learning
    distributional parameters from an external validation cohort.  It is useful
    when the training and validation matrices were measured on different assay
    platforms but contain the same biological features.
    """

    def fit(self, X, y=None):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("Sample rank transformation requires a 2D matrix.")
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("Sample rank transformation received an incompatible matrix.")
        return pd.DataFrame(array).rank(axis=1, method="average", pct=True).to_numpy(dtype=float)


def build_cross_platform_classification_pipeline(estimator, k=20, imputer="median"):
    """Build a fold-local classifier using sample-wise rank normalization."""
    estimator = _classification_estimator(estimator)
    return Pipeline(
        [
            ("imputer", FoldImputer(strategy=imputer)),
            ("sample_rank", SampleRankTransformer()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("model", clone(estimator)),
        ]
    )


def _decision_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def classification_metrics(y_true, y_pred, scores=None, classes=None,
                           positive_class=None, score_type="probability"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    classes = np.asarray(classes)
    binary = len(classes) == 2
    average = "binary" if binary else "macro"
    if binary and positive_class is None:
        raise DataValidationError("Binary metrics require an explicit positive class.")
    if binary and positive_class not in classes:
        raise DataValidationError("The selected positive class is absent from the model classes.")
    pos_label = positive_class if binary else None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        pos_label=pos_label if binary else 1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificity_values = []
    per_class = {}
    for index, label in enumerate(classes):
        tp = float(cm[index, index])
        fn = float(cm[index, :].sum() - tp)
        fp = float(cm[:, index].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity_values.append(specificity)
        per_class[str(label)] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity_values[list(classes).index(pos_label)] if binary else float(np.nanmean(specificity_values)),
        "f1": f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "per_class": per_class,
    }
    if binary:
        result["positive_class"] = str(pos_label)
    if scores is not None:
        try:
            if binary:
                score_array = np.asarray(scores)
                if score_array.ndim == 2:
                    positive_scores = score_array[:, list(classes).index(pos_label)]
                elif score_type == "decision_function" and pos_label == classes[0]:
                    positive_scores = -score_array
                elif score_type == "probability" and pos_label == classes[0]:
                    positive_scores = 1.0 - score_array
                else:
                    positive_scores = score_array
                binary_truth = (y_true == pos_label).astype(int)
                result["roc_auc"] = roc_auc_score(binary_truth, positive_scores)
                result["pr_auc"] = average_precision_score(binary_truth, positive_scores)
            else:
                score_array = np.asarray(scores)
                truth = label_binarize(y_true, classes=classes)
                result["roc_auc_ovr_macro"] = roc_auc_score(
                    truth, score_array, average="macro", multi_class="ovr"
                )
                result["pr_auc_macro"] = average_precision_score(truth, score_array, average="macro")
        except (ValueError, IndexError):
            pass
    return result


def classification_prediction(model, X, positive_class=None,
                              threshold_policy="estimator_native_predict", threshold=0.5):
    classes = np.asarray(model.classes_)
    binary = len(classes) == 2
    if binary and (positive_class is None or positive_class not in classes):
        raise DataValidationError("Select a positive class from the development labels.")
    if threshold_policy not in ("estimator_native_predict", "explicit_probability_threshold"):
        raise DataValidationError("Unsupported classification threshold policy.")
    if hasattr(model, "predict_proba"):
        scores, score_type = model.predict_proba(X), "probability"
    elif hasattr(model, "decision_function"):
        scores, score_type = model.decision_function(X), "decision_function"
    else:
        scores, score_type = None, "none"
    if threshold_policy == "explicit_probability_threshold":
        if not binary or score_type != "probability" or not 0.0 <= float(threshold) <= 1.0:
            raise DataValidationError("An explicit threshold requires binary positive-class probabilities in [0,1].")
        positive_index = list(classes).index(positive_class)
        negative_class = classes[1 - positive_index]
        prediction = np.where(scores[:, positive_index] >= float(threshold),
                              positive_class, negative_class)
    else:
        prediction = model.predict(X)
    return prediction, scores, score_type


def regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _fixed_threshold_balanced_accuracy(model, X, y, positive_class, threshold):
    prediction, _, _ = classification_prediction(
        model, X, positive_class, "explicit_probability_threshold", threshold)
    return balanced_accuracy_score(y, prediction)


def classification_scorer(positive_class=None,
                          threshold_policy="estimator_native_predict", threshold=0.5):
    if threshold_policy == "estimator_native_predict":
        return "balanced_accuracy"
    if threshold_policy != "explicit_probability_threshold" or not np.isfinite(float(threshold)) \
            or not 0.0 <= float(threshold) <= 1.0:
        raise DataValidationError("Choose a valid threshold policy and a finite cutoff in [0,1].")
    return partial(_fixed_threshold_balanced_accuracy,
                   positive_class=positive_class, threshold=float(threshold))


def _serializable_metric_row(row):
    result = {}
    for key, value in row.items():
        if key == "per_class":
            result[key] = value
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def summarize_fold_metrics(rows):
    numeric_keys = sorted(
        set(
            key
            for row in rows
            for key, value in row.items()
            if key != "per_class" and isinstance(value, (int, float, np.integer, np.floating))
        )
    )
    summary = {}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "n": int(len(values)),
        }
    return summary


def nested_cv_classification(
    X,
    y,
    estimator,
    param_grid=None,
    k_values=(10, 20, 50),
    scaler="standard",
    imputer="median",
    outer_splits=5,
    outer_repeats=2,
    inner_splits=3,
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=1,
    feature_method="select",
    groups=None,
    positive_class=None,
    threshold_policy="estimator_native_predict",
    threshold=0.5,
):
    validate_group_feature_method(groups, feature_method)
    if threshold_policy == "explicit_probability_threshold" and str(feature_method).lower() in (
            "fss", "forward", "forward-selection", "bss", "backward", "backward-selection"):
        raise DataValidationError("FSS/BSS require the estimator-native threshold policy in this release.")
    X = validate_feature_matrix(X)
    y = validate_target(y, "classification")
    max_k = min(X.shape[1], max(k_values))
    pipeline = build_classification_pipeline(
        estimator, k=max_k, scaler=scaler, imputer=imputer,
        feature_method=feature_method, random_state=random_state)
    grid = dict(param_grid or {})
    if feature_method.lower() in ("select", "selection", "selectkbest", "univariate", "mrmr",
                                  "fss", "forward", "forward-selection", "bss", "backward",
                                  "backward-selection"):
        grid["selector__k"] = sorted(set(min(int(k), X.shape[1]) for k in k_values))
    elif feature_method.lower() in ("pca", "principal-components", "principal_components"):
        maximum = min(X.shape[1], int(np.floor(
            X.shape[0] * (outer_splits - 1.0) / outer_splits
            * (inner_splits - 1.0) / inner_splits)))
        grid["selector__n_components"] = sorted(set(min(int(k), maximum) for k in k_values))
    outer, actual_repeats = make_splits(
        X, y, "classification", outer_splits, outer_repeats, random_state, groups)
    rows = []
    for fold, (train_index, test_index) in enumerate(outer, start=1):
        inner, _ = make_splits(
            X.iloc[train_index], y[train_index], "classification", inner_splits,
            seed=random_state + fold,
            groups=None if groups is None else np.asarray(groups)[train_index], inner=True)
        search = GridSearchCV(
            pipeline,
            grid,
            cv=inner,
            scoring=classification_scorer(positive_class, threshold_policy, threshold),
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X.iloc[train_index], y[train_index])
        prediction, scores, score_type = classification_prediction(
            search, X.iloc[test_index], positive_class, threshold_policy, threshold)
        metrics = classification_metrics(
            y[test_index], prediction, scores=scores, classes=search.classes_,
            positive_class=positive_class, score_type=score_type)
        metrics["fold"] = fold
        metrics["n_train"] = len(train_index)
        metrics["n_test"] = len(test_index)
        metrics["best_params"] = search.best_params_
        metrics["selected_features"] = selected_feature_names(
            search.best_estimator_, X.columns)
        rows.append(_serializable_metric_row(metrics))
    return {"folds": rows, "summary": summarize_fold_metrics(rows),
            "split_ids_hash": split_index_hash(outer), "actual_repeats": actual_repeats,
            "candidate_grid": grid, "candidate_count": len(ParameterGrid(grid))}


def nested_cv_regression(
    X,
    y,
    estimator,
    param_grid=None,
    k_values=(10, 20, 50),
    scaler="standard",
    imputer="median",
    outer_splits=5,
    outer_repeats=2,
    inner_splits=3,
    random_state=DEFAULT_RANDOM_STATE,
    n_jobs=1,
    feature_method="select",
    groups=None,
):
    validate_group_feature_method(groups, feature_method)
    X = validate_feature_matrix(X)
    y = validate_target(y, "regression")
    max_k = min(X.shape[1], max(k_values))
    pipeline = build_regression_pipeline(
        estimator, k=max_k, scaler=scaler, imputer=imputer,
        feature_method=feature_method, random_state=random_state)
    grid = dict(param_grid or {})
    if feature_method.lower() in ("select", "selection", "selectkbest", "univariate", "mrmr",
                                  "fss", "forward", "forward-selection", "bss", "backward",
                                  "backward-selection"):
        grid["selector__k"] = sorted(set(min(int(k), X.shape[1]) for k in k_values))
    elif feature_method.lower() in ("pca", "principal-components", "principal_components"):
        maximum = min(X.shape[1], int(np.floor(
            X.shape[0] * (outer_splits - 1.0) / outer_splits
            * (inner_splits - 1.0) / inner_splits)))
        grid["selector__n_components"] = sorted(set(min(int(k), maximum) for k in k_values))
    outer, actual_repeats = make_splits(
        X, y, "regression", outer_splits, outer_repeats, random_state, groups)
    rows = []
    for fold, (train_index, test_index) in enumerate(outer, start=1):
        inner, _ = make_splits(
            X.iloc[train_index], y[train_index], "regression", inner_splits,
            seed=random_state + fold,
            groups=None if groups is None else np.asarray(groups)[train_index], inner=True)
        search = GridSearchCV(
            pipeline,
            grid,
            cv=inner,
            scoring="r2",
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X.iloc[train_index], y[train_index])
        prediction = search.predict(X.iloc[test_index])
        metrics = regression_metrics(y[test_index], prediction)
        metrics["fold"] = fold
        metrics["n_train"] = len(train_index)
        metrics["n_test"] = len(test_index)
        metrics["best_params"] = search.best_params_
        metrics["selected_features"] = selected_feature_names(
            search.best_estimator_, X.columns)
        rows.append(_serializable_metric_row(metrics))
    return {"folds": rows, "summary": summarize_fold_metrics(rows),
            "split_ids_hash": split_index_hash(outer), "actual_repeats": actual_repeats,
            "candidate_grid": grid, "candidate_count": len(ParameterGrid(grid))}


def fit_final_model(X, y, pipeline, param_grid, task, inner_splits=5, random_state=10,
                    n_jobs=1, groups=None, positive_class=None,
                    threshold_policy="estimator_native_predict", threshold=0.5):
    if groups is not None and isinstance(pipeline.named_steps.get("selector"), CandidateSequentialSelector):
        validate_group_feature_method(groups, "fss")
    X = validate_feature_matrix(X)
    y = validate_target(y, task)
    if task in ("binary", "multiclass", "classification"):
        cv, _ = make_splits(X, y, "classification", inner_splits, seed=random_state,
                            groups=groups, inner=True)
        scoring = classification_scorer(positive_class, threshold_policy, threshold)
    else:
        cv, _ = make_splits(X, y, "regression", inner_splits, seed=random_state,
                            groups=groups, inner=True)
        scoring = "r2"
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def selected_feature_names(fitted_pipeline, input_features):
    pipeline = fitted_pipeline.best_estimator_ if hasattr(fitted_pipeline, "best_estimator_") else fitted_pipeline
    selector = pipeline.named_steps.get("selector")
    names = np.asarray(list(map(str, input_features)))
    if selector is None or selector == "passthrough":
        return names.tolist()
    if isinstance(selector, PCA):
        return ["PC%d" % (index + 1) for index in range(selector.n_components_)]
    if not hasattr(selector, "get_support"):
        return []
    return names[selector.get_support()].tolist()


def align_prediction_frame(X, expected_features, reject_unexpected=False):
    frame = _as_frame(X)
    if frame.columns.has_duplicates:
        raise DataValidationError("Prediction data contain duplicate feature identifiers.")
    expected = list(map(str, expected_features))
    frame.columns = list(map(str, frame.columns))
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing:
        raise DataValidationError("Missing required prediction features: %s" % ", ".join(missing[:20]))
    if reject_unexpected and unexpected:
        raise DataValidationError("Unexpected prediction features: %s" % ", ".join(unexpected[:20]))
    aligned = frame.loc[:, expected]
    numeric = aligned.apply(pd.to_numeric, errors="coerce")
    introduced = numeric.isna() & ~aligned.isna()
    if introduced.any().any():
        bad = introduced.columns[introduced.any()].astype(str).tolist()
        raise DataValidationError("Non-numeric prediction values in features: %s" % ", ".join(bad[:20]))
    aligned = numeric
    if aligned.isna().all(axis=0).any():
        bad = aligned.columns[aligned.isna().all(axis=0)].tolist()
        raise DataValidationError("All-missing prediction features: %s" % ", ".join(bad[:20]))
    if np.isinf(aligned.to_numpy(dtype=float)).any():
        raise DataValidationError("Infinite prediction values are not allowed.")
    return aligned, unexpected


def alignment_report(frame, expected_features, cohort=None):
    """报告预测矩阵相对于训练特征契约的偏差.

    Args:
        frame (pandas.DataFrame, 必填): 待检查的预测矩阵.
        expected_features (iterable, 必填): 训练时固定的特征名与顺序.
        cohort (str, 可选): 区分盲集与验证集等来源的标签, 缺省 None.

    Returns:
        dict: 含 reordered, unexpected, n_input, n_expected 的报告; 无偏差时返回 None.

    Note:
        只报告偏差, 不改动数据也不拒绝输入; 重复列名仍由 align_prediction_frame 抛出.

    See Also:
        align_prediction_frame

    Example:
        alignment_report(frame, ["A", "B"], cohort="blind")
    """
    checked = _as_frame(frame)
    expected = [str(name) for name in expected_features]
    columns = [str(name) for name in checked.columns]
    if len(set(columns)) != len(columns):
        return None
    expected_set = set(expected)
    # 输入中属于期望特征的那些列, 其相对顺序即为实际生效的顺序
    present = [name for name in columns if name in expected_set]
    reference = [name for name in expected if name in set(columns)]
    reordered = present if present != reference else []
    unexpected = sorted(set(columns) - expected_set)
    if not reordered and not unexpected:
        return None
    report = {
        "reordered": reordered[:20],
        "unexpected": unexpected[:20],
        "n_reordered": len(reordered),
        "n_unexpected": len(unexpected),
        "n_input": len(columns),
        "n_expected": len(expected),
    }
    if cohort:
        report["cohort"] = str(cohort)
    return report


def _manifest_for_model(model, task, feature_names, metadata=None):
    return {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "task": task,
        "feature_names": list(map(str, feature_names)),
        "model_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "metadata": metadata or {},
    }


def save_signed_model_bundle(path, model, task, feature_names, signing_key, metadata=None):
    if not signing_key:
        raise ModelBundleError("A non-empty signing key is required.")
    manifest = _manifest_for_model(model, task, feature_names, metadata=metadata)
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer, compress=3)
    model_bytes = model_buffer.getvalue()
    signature = hmac.new(
        signing_key.encode("utf-8"), manifest_bytes + b"\n" + model_bytes, hashlib.sha256
    ).hexdigest()
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("model.joblib", model_bytes)
        archive.writestr("signature.sha256", signature + "\n")
    return manifest


def load_signed_model_bundle(path, signing_key, max_model_bytes=100 * 1024 * 1024):
    if not signing_key:
        raise ModelBundleError("A non-empty signing key is required.")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            required = {"manifest.json", "model.joblib", "signature.sha256"}
            if names != required:
                raise ModelBundleError("Unexpected model bundle contents.")
            manifest_bytes = archive.read("manifest.json")
            info = archive.getinfo("model.joblib")
            if info.file_size > max_model_bytes:
                raise ModelBundleError("Model bundle exceeds the configured size limit.")
            model_bytes = archive.read("model.joblib")
            supplied = archive.read("signature.sha256").decode("ascii").strip()
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise ModelBundleError("Invalid model bundle: %s" % exc)
    expected = hmac.new(
        signing_key.encode("utf-8"), manifest_bytes + b"\n" + model_bytes, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise ModelBundleError("Model bundle signature verification failed.")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModelBundleError("Invalid model manifest: %s" % exc)
    if manifest.get("bundle_version") != MODEL_BUNDLE_VERSION:
        raise ModelBundleError("Unsupported model bundle version.")
    recorded_versions = manifest.get("versions") or {}
    for name, current in (("scikit_learn", sklearn.__version__), ("numpy", np.__version__)):
        recorded = str(recorded_versions.get(name) or "")
        if recorded.split(".")[:2] != str(current).split(".")[:2]:
            raise ModelBundleError(
                "This model bundle uses %s %s, but this server uses %s; retrain and export with the current release."
                % (name, recorded or "unknown", current))
    model = joblib.load(io.BytesIO(model_bytes))
    return model, manifest


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
