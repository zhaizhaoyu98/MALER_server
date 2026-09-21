"""Run the leakage-safe validation requested during MALER peer review.

This script uses only the example matrices shipped with the repository.  The
declared train/test split is honoured: all model selection is performed on the
training subset and the test subset is evaluated once after model selection.
"""

from __future__ import absolute_import, print_function

import hashlib
import json
import os
import platform
import secrets
import sys
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.tree import SurvivalTree


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import (  # noqa: E402
    _decision_scores,
    build_classification_pipeline,
    build_regression_pipeline,
    classification_metrics,
    file_sha256,
    fit_final_model,
    nested_cv_classification,
    nested_cv_regression,
    regression_metrics,
    save_signed_model_bundle,
    selected_feature_names,
    validate_feature_matrix,
    validate_target,
)
from mlserver.safe_survival import (  # noqa: E402
    build_survival_pipeline,
    nested_cv_survival,
    survival_metrics,
    validate_survival_target,
)


EXAMPLE_DIR = os.path.join(REPOSITORY_ROOT, "mlserver", "static", "cache", "example")
RESULTS_DIR = os.path.join(REPOSITORY_ROOT, "validation", "results")
RANDOM_STATE = 10


def _native(value):
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_native(value), handle, indent=2, sort_keys=True, ensure_ascii=False)


def _read_example(filename):
    path = os.path.join(EXAMPLE_DIR, filename)
    return pd.read_csv(path, index_col=0, dtype=str, low_memory=False), path


def load_classification(filename):
    raw, path = _read_example(filename)
    label = raw.loc["label"].astype(str).str.strip().to_numpy()
    split = raw.loc["set"].astype(str).str.strip().str.lower().to_numpy()
    X = raw.drop(index=["label", "set"]).T
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    X = validate_feature_matrix(X)
    y = validate_target(label, "classification")
    return X, y, split, path


def load_regression(filename):
    raw, path = _read_example(filename)
    label = raw.loc["PURITY"].to_numpy()
    split = raw.loc["set"].astype(str).str.strip().str.lower().to_numpy()
    X = raw.drop(index=["PURITY", "set"]).T
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    X = validate_feature_matrix(X)
    y = validate_target(label, "regression")
    return X, y, split, path


def load_survival(filename):
    raw, path = _read_example(filename)
    split = raw.loc["set"].astype(str).str.strip().str.lower().to_numpy()
    y = validate_survival_target(raw.loc["Status"].to_numpy(), raw.loc["time"].to_numpy())
    X = raw.drop(index=["Status", "time", "set"]).T
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    X = validate_feature_matrix(X)
    return X, y, split, path


def split_declared(X, y, split):
    train_mask = np.isin(split, ["train", "training"])
    test_mask = np.isin(split, ["test", "testing"])
    if not train_mask.any() or not test_mask.any() or (train_mask | test_mask).sum() != len(split):
        raise ValueError("Every sample must be assigned to train/training or test/testing.")
    return X.loc[train_mask], y[train_mask], X.loc[test_mask], y[test_mask]


def describe_dataset(name, X_train, y_train, X_test, y_test, source_path, task):
    result = {
        "dataset": name,
        "task": task,
        "source_path": source_path,
        "source_sha256": file_sha256(source_path),
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "missing_train": int(X_train.isna().sum().sum()),
        "missing_test": int(X_test.isna().sum().sum()),
    }
    if task == "classification":
        result["train_class_counts"] = dict(Counter(map(str, y_train)))
        result["test_class_counts"] = dict(Counter(map(str, y_test)))
    elif task == "survival":
        result["train_events"] = int(y_train["event"].sum())
        result["test_events"] = int(y_test["event"].sum())
        result["train_followup_median"] = float(np.median(y_train["time"]))
        result["test_followup_median"] = float(np.median(y_test["time"]))
    else:
        result["train_target_mean"] = float(np.mean(y_train))
        result["test_target_mean"] = float(np.mean(y_test))
    return result


def feature_stability(folds):
    selections = [set(row.get("selected_features", [])) for row in folds]
    pairwise = []
    for first in range(len(selections)):
        for second in range(first + 1, len(selections)):
            union = selections[first] | selections[second]
            pairwise.append(len(selections[first] & selections[second]) / float(len(union)) if union else 1.0)
    frequency = Counter(feature for selected in selections for feature in selected)
    return {
        "mean_pairwise_jaccard": float(np.mean(pairwise)) if pairwise else 1.0,
        "std_pairwise_jaccard": float(np.std(pairwise, ddof=1)) if len(pairwise) > 1 else 0.0,
        "top_selection_frequency": [
            {"feature": feature, "folds": int(count), "fraction": count / float(len(selections))}
            for feature, count in frequency.most_common(50)
        ],
    }


def fold_intervals(result):
    intervals = {}
    for metric in result.get("summary", {}):
        values = [fold.get(metric) for fold in result["folds"] if fold.get(metric) is not None]
        values = np.asarray(values, dtype=float)
        if values.size:
            intervals[metric] = {
                "fold_percentile_2.5": float(np.percentile(values, 2.5)),
                "fold_percentile_97.5": float(np.percentile(values, 97.5)),
            }
    return intervals


def bootstrap_classification(y, prediction, scores, classes, repetitions=1000):
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for _ in range(repetitions):
        index = rng.randint(0, len(y), len(y))
        if len(np.unique(y[index])) != len(classes):
            continue
        rows.append(classification_metrics(
            y[index], prediction[index],
            scores=None if scores is None else np.asarray(scores)[index],
            classes=classes,
        ))
    result = {}
    for metric in set(key for row in rows for key in row if key != "per_class"):
        values = np.asarray([row[metric] for row in rows if metric in row], dtype=float)
        result[metric] = {
            "estimate": float(classification_metrics(y, prediction, scores=scores, classes=classes)[metric]),
            "bootstrap_95_ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "n_bootstrap": int(len(values)),
        }
    return result


def bootstrap_regression(y, prediction, repetitions=1000):
    rng = np.random.RandomState(RANDOM_STATE)
    point = regression_metrics(y, prediction)
    rows = []
    for _ in range(repetitions):
        index = rng.randint(0, len(y), len(y))
        rows.append(regression_metrics(y[index], prediction[index]))
    return {
        metric: {
            "estimate": float(point[metric]),
            "bootstrap_95_ci": [
                float(np.percentile([row[metric] for row in rows], 2.5)),
                float(np.percentile([row[metric] for row in rows], 97.5)),
            ],
            "n_bootstrap": repetitions,
        }
        for metric in point
    }


def bootstrap_survival(y, risk, repetitions=1000):
    rng = np.random.RandomState(RANDOM_STATE)
    values = []
    for _ in range(repetitions):
        index = rng.randint(0, len(y), len(y))
        if y["event"][index].sum() < 2:
            continue
        try:
            values.append(concordance_index_censored(
                y["event"][index], y["time"][index], risk[index])[0])
        except ValueError:
            continue
    point = concordance_index_censored(y["event"], y["time"], risk)[0]
    return {
        "c_index": {
            "estimate": float(point),
            "bootstrap_95_ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "n_bootstrap": int(len(values)),
        }
    }


def validation_key():
    key = os.environ.get("MALER_MODEL_SIGNING_KEY")
    return key if key else secrets.token_urlsafe(32)


def run_classification(name, filename, signing_key):
    X, y, split, source_path = load_classification(filename)
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    estimator = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=500,
        random_state=RANDOM_STATE, multi_class="ovr")
    started = time.perf_counter()
    nested = nested_cv_classification(
        X_train, y_train, estimator,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        k_values=(10, 20, 50), outer_splits=5, outer_repeats=10,
        inner_splits=3, random_state=RANDOM_STATE, n_jobs=1)
    main_seconds = time.perf_counter() - started
    baseline_started = time.perf_counter()
    no_selection = nested_cv_classification(
        X_train, y_train, estimator,
        param_grid={"model__C": [1.0]}, k_values=(X_train.shape[1],),
        outer_splits=5, outer_repeats=2, inner_splits=3,
        random_state=RANDOM_STATE, n_jobs=1)
    baseline_seconds = time.perf_counter() - baseline_started
    pipeline = build_classification_pipeline(estimator, k=50)
    final = fit_final_model(
        X_train, y_train, pipeline,
        {"selector__k": [10, 20, 50], "model__C": [0.1, 1.0, 10.0]},
        "classification", inner_splits=5, random_state=RANDOM_STATE, n_jobs=1)
    prediction = final.predict(X_test)
    scores = _decision_scores(final, X_test)
    test_metrics = classification_metrics(y_test, prediction, scores=scores, classes=final.classes_)
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_prediction = dummy.predict(X_test)
    dummy_scores = dummy.predict_proba(X_test)
    bundle_path = os.path.join(RESULTS_DIR, name + "_pipeline.maler")
    metadata = {
        "method": "model_bclass" if len(final.classes_) == 2 else "model_mclass",
        "model_name": "LeakageSafeLogisticRegression",
        "classes": {str(value): str(value) for value in final.classes_},
        "pipeline_complete": True,
        "training_dataset": filename,
        "best_params": final.best_params_,
    }
    save_signed_model_bundle(
        bundle_path, final.best_estimator_,
        "binary" if len(final.classes_) == 2 else "multiclass",
        X_train.columns, signing_key, metadata=metadata)
    result = {
        "dataset": describe_dataset(name, X_train, y_train, X_test, y_test, source_path, "classification"),
        "protocol": {"outer_cv": "repeated stratified 5-fold x 10", "inner_cv": "stratified 3-fold"},
        "nested_cv": nested,
        "nested_cv_fold_intervals": fold_intervals(nested),
        "feature_stability": feature_stability(nested["folds"]),
        "no_feature_selection_baseline": {
            "protocol": "repeated stratified 5-fold x 2; fixed C=1",
            "summary": no_selection["summary"],
        },
        "test_metrics": test_metrics,
        "test_bootstrap": bootstrap_classification(y_test, prediction, scores, final.classes_),
        "dummy_test_metrics": classification_metrics(
            y_test, dummy_prediction, scores=dummy_scores, classes=final.classes_),
        "final_best_params": final.best_params_,
        "final_selected_features": selected_feature_names(final, X_train.columns),
        "model_bundle": {"path": bundle_path, "sha256": file_sha256(bundle_path)},
        "performance": {"nested_cv_seconds": main_seconds, "baseline_seconds": baseline_seconds},
    }
    return result


def run_regression(signing_key):
    name = "regression"
    filename = "regression_example.csv"
    X, y, split, source_path = load_regression(filename)
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    estimator = Ridge()
    started = time.perf_counter()
    nested = nested_cv_regression(
        X_train, y_train, estimator,
        param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        k_values=(10, 20, 50), outer_splits=5, outer_repeats=10,
        inner_splits=3, random_state=RANDOM_STATE, n_jobs=1)
    main_seconds = time.perf_counter() - started
    baseline_started = time.perf_counter()
    no_selection = nested_cv_regression(
        X_train, y_train, estimator,
        param_grid={"model__alpha": [10.0]}, k_values=(X_train.shape[1],),
        outer_splits=5, outer_repeats=2, inner_splits=3,
        random_state=RANDOM_STATE, n_jobs=1)
    baseline_seconds = time.perf_counter() - baseline_started
    pipeline = build_regression_pipeline(estimator, k=50)
    final = fit_final_model(
        X_train, y_train, pipeline,
        {"selector__k": [10, 20, 50], "model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "regression", inner_splits=5, random_state=RANDOM_STATE, n_jobs=1)
    prediction = final.predict(X_test)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    bundle_path = os.path.join(RESULTS_DIR, name + "_pipeline.maler")
    save_signed_model_bundle(
        bundle_path, final.best_estimator_, "regression", X_train.columns, signing_key,
        metadata={
            "method": "model_reg", "model_name": "LeakageSafeRidge",
            "pipeline_complete": True, "training_dataset": filename,
            "best_params": final.best_params_,
        })
    return {
        "dataset": describe_dataset(name, X_train, y_train, X_test, y_test, source_path, "regression"),
        "protocol": {"outer_cv": "repeated 5-fold x 10", "inner_cv": "3-fold"},
        "nested_cv": nested,
        "nested_cv_fold_intervals": fold_intervals(nested),
        "feature_stability": feature_stability(nested["folds"]),
        "no_feature_selection_baseline": {
            "protocol": "repeated 5-fold x 2; fixed alpha=10",
            "summary": no_selection["summary"],
        },
        "test_metrics": regression_metrics(y_test, prediction),
        "test_bootstrap": bootstrap_regression(y_test, prediction),
        "dummy_test_metrics": regression_metrics(y_test, dummy.predict(X_test)),
        "final_best_params": final.best_params_,
        "final_selected_features": selected_feature_names(final, X_train.columns),
        "model_bundle": {"path": bundle_path, "sha256": file_sha256(bundle_path)},
        "performance": {"nested_cv_seconds": main_seconds, "baseline_seconds": baseline_seconds},
    }


def run_survival(signing_key):
    name = "survival"
    filename = "survival_example.csv"
    X, y, split, source_path = load_survival(filename)
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    estimator = SurvivalTree(random_state=RANDOM_STATE, min_samples_leaf=10)
    started = time.perf_counter()
    nested = nested_cv_survival(
        X_train, y_train, estimator,
        param_grid={"model__max_depth": [2, 3, 5]},
        k_values=(10, 20, 50), outer_splits=5, outer_repeats=10,
        inner_splits=3, random_state=RANDOM_STATE, n_jobs=1)
    main_seconds = time.perf_counter() - started
    baseline_started = time.perf_counter()
    no_selection = nested_cv_survival(
        X_train, y_train, estimator,
        param_grid={"model__max_depth": [3]}, k_values=(X_train.shape[1],),
        outer_splits=5, outer_repeats=2, inner_splits=3,
        random_state=RANDOM_STATE, n_jobs=1)
    baseline_seconds = time.perf_counter() - baseline_started
    pipeline = build_survival_pipeline(estimator, k=50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_indices = list(cv.split(X_train, y_train["event"].astype(int)))
    final = GridSearchCV(
        pipeline,
        {"selector__k": [10, 20, 50], "model__max_depth": [2, 3, 5]},
        cv=cv_indices, scoring=None, refit=True, n_jobs=1, error_score="raise")
    final.fit(X_train, y_train)
    final.best_estimator_.maler_ytrain_ = y_train
    test_metrics = survival_metrics(final.best_estimator_, X_train, y_train, X_test, y_test)
    risk = final.predict(X_test)
    null_risk = np.zeros(len(y_test), dtype=float)
    bundle_path = os.path.join(RESULTS_DIR, name + "_pipeline.maler")
    save_signed_model_bundle(
        bundle_path, final.best_estimator_, "survival", X_train.columns, signing_key,
        metadata={
            "method": "model_sur", "model_name": "LeakageSafeSurvivalTree",
            "pipeline_complete": True, "training_dataset": filename,
            "best_params": final.best_params_,
        })
    return {
        "dataset": describe_dataset(name, X_train, y_train, X_test, y_test, source_path, "survival"),
        "protocol": {"outer_cv": "event-stratified 5-fold x 10", "inner_cv": "event-stratified 3-fold"},
        "nested_cv": nested,
        "nested_cv_fold_intervals": fold_intervals(nested),
        "feature_stability": feature_stability(nested["folds"]),
        "no_feature_selection_baseline": {
            "protocol": "event-stratified 5-fold x 2; fixed max_depth=3",
            "summary": no_selection["summary"],
        },
        "test_metrics": test_metrics,
        "test_bootstrap": bootstrap_survival(y_test, risk),
        "null_test_c_index": float(concordance_index_censored(
            y_test["event"], y_test["time"], null_risk)[0]),
        "final_best_params": final.best_params_,
        "final_selected_features": np.asarray(list(map(str, X_train.columns)))[
            final.best_estimator_.named_steps["selector"].get_support(indices=True)].tolist(),
        "model_bundle": {"path": bundle_path, "sha256": file_sha256(bundle_path)},
        "performance": {"nested_cv_seconds": main_seconds, "baseline_seconds": baseline_seconds},
    }


def write_tables(results):
    summary_rows = []
    test_rows = []
    for task, result in results["tasks"].items():
        for metric, values in result["nested_cv"]["summary"].items():
            summary_rows.append({
                "task": task, "metric": metric,
                "mean": values["mean"], "sd": values["std"],
                "minimum": values["min"], "maximum": values["max"], "folds": values["n"],
            })
        for metric, value in result["test_metrics"].items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                ci = result.get("test_bootstrap", {}).get(metric, {}).get("bootstrap_95_ci", [None, None])
                test_rows.append({
                    "task": task, "metric": metric, "estimate": value,
                    "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
                })
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "internal_cv_summary.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RESULTS_DIR, "heldout_test_summary.csv"), index=False)
    feature_rows = []
    for task, result in results["tasks"].items():
        for rank, feature in enumerate(result["final_selected_features"], start=1):
            feature_rows.append({"task": task, "rank": rank, "feature": feature})
    pd.DataFrame(feature_rows).to_csv(os.path.join(RESULTS_DIR, "final_selected_features.csv"), index=False)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    signing_key = validation_key()
    started = time.perf_counter()
    results = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "software": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "random_state": RANDOM_STATE,
        "tasks": {},
    }
    tasks = [
        ("binary_classification", lambda: run_classification(
            "binary_classification", "binary_classification_example.csv", signing_key)),
        ("multiclass_classification", lambda: run_classification(
            "multiclass_classification", "multiclass_classification_example.csv", signing_key)),
        ("regression", lambda: run_regression(signing_key)),
        ("survival", lambda: run_survival(signing_key)),
    ]
    for name, runner in tasks:
        task_path = os.path.join(RESULTS_DIR, name + ".json")
        if os.path.exists(task_path) and not os.environ.get("MALER_VALIDATION_FORCE"):
            print("[%s] loading completed result" % name, flush=True)
            with open(task_path, "r", encoding="utf-8") as handle:
                results["tasks"][name] = json.load(handle)
            continue
        print("[%s] starting" % name, flush=True)
        result = runner()
        results["tasks"][name] = result
        _write_json(task_path, result)
        print("[%s] completed" % name, flush=True)
    results["total_seconds"] = time.perf_counter() - started
    _write_json(os.path.join(RESULTS_DIR, "validation_results.json"), results)
    write_tables(results)
    print("Results written to %s" % RESULTS_DIR)


if __name__ == "__main__":
    main()
