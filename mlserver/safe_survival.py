"""Leakage-safe survival-analysis helpers for MALER.

All learned preprocessing and univariate Cox ranking are contained in a
scikit-learn Pipeline and are therefore refit for each training fold.
"""

from __future__ import absolute_import

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from .safe_ml import DataValidationError, summarize_fold_metrics, validate_feature_matrix


def validate_survival_target(status, time, require_event_mix=True):
    status_series = pd.Series(status)
    time_series = pd.to_numeric(pd.Series(time), errors="coerce")
    if status_series.isna().any() or time_series.isna().any():
        raise DataValidationError("Survival status and follow-up time must not be missing.")
    if np.isinf(time_series.to_numpy(dtype=float)).any() or (time_series <= 0).any():
        raise DataValidationError("Survival follow-up time must be finite and greater than zero.")
    if status_series.dtype == bool:
        event = status_series.to_numpy(dtype=bool)
    elif np.issubdtype(status_series.dtype, np.number):
        values = set(status_series.astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise DataValidationError("Numeric survival status must use 1=event and 0=censored.")
        event = status_series.astype(int).to_numpy() == 1
    else:
        normalized = status_series.astype(str).str.strip().str.lower()
        mapping = {
            "1": True,
            "1.0": True,
            "event": True,
            "dead": True,
            "deceased": True,
            "death": True,
            "0": False,
            "0.0": False,
            "censored": False,
            "alive": False,
        }
        unknown = sorted(set(normalized) - set(mapping))
        if unknown:
            raise DataValidationError("Unsupported survival status values: %s" % ", ".join(unknown))
        event = normalized.map(mapping).to_numpy(dtype=bool)
    if require_event_mix and (event.sum() < 2 or (~event).sum() < 1):
        raise DataValidationError("Survival analysis requires at least two events and one censored case.")
    target = np.empty(len(event), dtype=[("event", "?"), ("time", "<f8")])
    target["event"] = event
    target["time"] = time_series.to_numpy(dtype=float)
    return target


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


class CoxPHSelectKBest(BaseEstimator, TransformerMixin):
    """Rank features by a vectorized univariate Cox score statistic.

    The statistic is evaluated at beta=0 with Breslow risk sets.  This is both
    censoring-aware and substantially faster than fitting thousands of separate
    Cox models inside every nested-CV fold.
    """

    def __init__(self, k=20, alpha=0.1):
        self.k = k
        self.alpha = alpha

    def fit(self, X, y):
        array = np.asarray(X, dtype=float)
        if self.k <= 0:
            raise ValueError("k must be positive")
        if y.dtype.names is None or set(y.dtype.names) != {"event", "time"}:
            raise ValueError("Cox ranking requires a structured survival outcome.")
        order = np.argsort(y["time"], kind="mergesort")[::-1]
        sorted_x = array[order]
        sorted_time = y["time"][order]
        sorted_event = y["event"][order]
        cumulative_sum = np.cumsum(sorted_x, axis=0)
        cumulative_square = np.cumsum(sorted_x * sorted_x, axis=0)
        score_numerator = np.zeros(array.shape[1], dtype=float)
        score_variance = np.zeros(array.shape[1], dtype=float)
        start = 0
        while start < len(sorted_time):
            end = start
            while end + 1 < len(sorted_time) and sorted_time[end + 1] == sorted_time[start]:
                end += 1
            event_rows = sorted_x[start : end + 1][sorted_event[start : end + 1]]
            event_count = event_rows.shape[0]
            if event_count:
                risk_count = float(end + 1)
                risk_mean = cumulative_sum[end] / risk_count
                risk_variance = cumulative_square[end] / risk_count - risk_mean * risk_mean
                score_numerator += event_rows.sum(axis=0) - event_count * risk_mean
                score_variance += event_count * np.maximum(risk_variance, 0.0)
            start = end + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.abs(score_numerator) / np.sqrt(score_variance + float(self.alpha))
        scores[~np.isfinite(scores)] = -np.inf
        if not np.isfinite(scores).any():
            raise ValueError("No survival feature could be ranked by the Cox model.")
        k = min(int(self.k), array.shape[1])
        order = np.argsort(scores)[::-1]
        self.scores_ = scores
        self.selected_indices_ = np.sort(order[:k])
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


class CoxSequentialSelector(BaseEstimator, TransformerMixin):
    """Censoring-aware fold-local FSS/BSS after a capped Cox prescreen."""

    def __init__(self, estimator, k=20, direction="forward", max_candidates=50,
                 cv=3, random_state=10):
        self.estimator = estimator
        self.k = k
        self.direction = direction
        self.max_candidates = max_candidates
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        array = np.asarray(X, dtype=float)
        candidate_count = min(int(self.max_candidates), array.shape[1])
        selected_count = min(max(1, int(self.k)), candidate_count)
        self.prefilter_ = CoxPHSelectKBest(k=candidate_count).fit(array, y)
        candidate_indices = self.prefilter_.get_support(indices=True)
        candidate_x = self.prefilter_.transform(array)
        if selected_count == candidate_count:
            within_candidates = np.arange(candidate_count)
            self.selector_ = None
        else:
            splitter = StratifiedKFold(
                n_splits=int(self.cv), shuffle=True, random_state=int(self.random_state))
            cv_indices = list(splitter.split(candidate_x, y["event"].astype(int)))
            self.selector_ = SequentialFeatureSelector(
                clone(self.estimator), n_features_to_select=selected_count,
                direction=self.direction, scoring=None, cv=cv_indices, n_jobs=1)
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


def build_survival_pipeline(
        estimator, k=20, scaler="standard", imputer="median",
        feature_method="select", random_state=10):
    normalized = (feature_method or "select").lower()
    if normalized in ("select", "selection", "cox", "cox-score"):
        reducer = CoxPHSelectKBest(k=k)
    elif normalized in ("pca", "principal-components", "principal_components"):
        reducer = PCA(n_components=k, svd_solver="auto", random_state=random_state)
    elif normalized in ("fss", "forward", "forward-selection"):
        reducer = CoxSequentialSelector(
            estimator=estimator, k=k, direction="forward", random_state=random_state)
    elif normalized in ("bss", "backward", "backward-selection"):
        reducer = CoxSequentialSelector(
            estimator=estimator, k=k, direction="backward", random_state=random_state)
    elif normalized in ("none", "off", "passthrough"):
        reducer = "passthrough"
    else:
        raise ValueError("Unknown survival feature reduction method: %s" % feature_method)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer)),
            ("scaler", _make_scaler(scaler)),
            ("selector", reducer),
            ("model", clone(estimator)),
        ]
    )


def _transform_for_survival_model(pipeline, X):
    transformed = X
    for name, step in pipeline.steps:
        if name == "model":
            break
        if step != "passthrough":
            transformed = step.transform(transformed)
    return transformed


def survival_metrics(fitted_pipeline, X_train, y_train, X_test, y_test):
    risk = fitted_pipeline.predict(X_test)
    result = {
        "c_index": float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
    }
    lower = max(float(np.min(y_train["time"])), float(np.min(y_test["time"])))
    upper = min(float(np.max(y_train["time"])), float(np.max(y_test["time"])))
    event_times = y_test["time"][(y_test["event"]) & (y_test["time"] > lower) & (y_test["time"] < upper)]
    if len(event_times) >= 3:
        times = np.unique(np.quantile(event_times, [0.25, 0.50, 0.75]))
        if len(times) >= 2:
            try:
                auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
                result["time_auc_mean"] = float(mean_auc)
                result["time_auc_values"] = [float(value) for value in auc]
                result["time_auc_times"] = [float(value) for value in times]
            except ValueError:
                pass

            model = fitted_pipeline.named_steps["model"]
            if hasattr(model, "predict_survival_function"):
                try:
                    transformed = _transform_for_survival_model(fitted_pipeline, X_test)
                    functions = model.predict_survival_function(transformed)
                    probabilities = np.asarray([[fn(time) for time in times] for fn in functions])
                    result["integrated_brier_score"] = float(
                        integrated_brier_score(y_train, y_test, probabilities, times)
                    )
                except (ValueError, TypeError):
                    pass
    return result


def nested_cv_survival(
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
    random_state=10,
    n_jobs=1,
    feature_method="select",
):
    X = validate_feature_matrix(X)
    if y.dtype.names is None or set(y.dtype.names) != {"event", "time"}:
        raise DataValidationError("Survival target must contain event and time fields.")
    max_k = min(X.shape[1], max(k_values))
    pipeline = build_survival_pipeline(
        estimator, k=max_k, scaler=scaler, imputer=imputer,
        feature_method=feature_method, random_state=random_state)
    grid = dict(param_grid or {})
    normalized_method = feature_method.lower()
    if normalized_method in ("select", "selection", "cox", "cox-score", "fss", "forward",
                              "forward-selection", "bss", "backward", "backward-selection"):
        grid["selector__k"] = sorted(set(min(int(k), X.shape[1]) for k in k_values))
    elif normalized_method in ("pca", "principal-components", "principal_components"):
        maximum = min(X.shape[1], int(np.floor(
            X.shape[0] * (outer_splits - 1.0) / outer_splits
            * (inner_splits - 1.0) / inner_splits)))
        grid["selector__n_components"] = sorted(set(min(int(k), maximum) for k in k_values))
    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=random_state
    )
    rows = []
    event = y["event"].astype(int)
    for fold, (train_index, test_index) in enumerate(outer.split(X, event), start=1):
        inner = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state + fold)
        inner_indices = list(inner.split(X.iloc[train_index], event[train_index]))
        search = GridSearchCV(
            pipeline,
            grid,
            cv=inner_indices,
            scoring=None,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X.iloc[train_index], y[train_index])
        metrics = survival_metrics(
            search.best_estimator_,
            X.iloc[train_index],
            y[train_index],
            X.iloc[test_index],
            y[test_index],
        )
        metrics["fold"] = fold
        metrics["n_train"] = len(train_index)
        metrics["n_test"] = len(test_index)
        metrics["best_params"] = search.best_params_
        selector = search.best_estimator_.named_steps["selector"]
        if hasattr(selector, "get_support"):
            selected = selector.get_support(indices=True)
            metrics["selected_features"] = np.asarray(
                list(map(str, X.columns)))[selected].tolist()
        elif isinstance(selector, PCA):
            metrics["selected_features"] = [
                "PC%d" % (index + 1) for index in range(selector.n_components_)
            ]
        else:
            metrics["selected_features"] = list(map(str, X.columns))
        rows.append(metrics)
    return {"folds": rows, "summary": summarize_fold_metrics(rows)}
