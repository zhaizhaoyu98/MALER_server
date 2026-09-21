"""Fold-local PCA sensitivity analysis for MALER classification/regression tasks."""

from __future__ import absolute_import, print_function

import json
import hashlib
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import (  # noqa: E402
    _decision_scores,
    build_classification_pipeline,
    build_regression_pipeline,
    classification_metrics,
    fit_final_model,
    nested_cv_classification,
    nested_cv_regression,
    regression_metrics,
)
from validation.run_reviewer_validation import (  # noqa: E402
    load_classification,
    load_regression,
    split_declared,
)


RESULTS_DIR = os.path.join(REPOSITORY_ROOT, "validation", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "pca_sensitivity.json")
RANDOM_STATE = 10
CHECKPOINT_PATH = OUTPUT_PATH + ".partial"


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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_native(value), handle, indent=2, sort_keys=True)


def run_classification(filename, name):
    X, y, split, source_path = load_classification(filename)
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    estimator = LogisticRegression(solver="liblinear", max_iter=4000, random_state=RANDOM_STATE)
    started = time.perf_counter()
    nested = nested_cv_classification(
        X_train, y_train, estimator,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        k_values=(10, 20, 50), outer_splits=5, outer_repeats=5,
        inner_splits=3, random_state=RANDOM_STATE, feature_method="pca",
    )
    pipeline = build_classification_pipeline(
        estimator, k=50, feature_method="pca", random_state=RANDOM_STATE)
    final = fit_final_model(
        X_train, y_train, pipeline,
        {"selector__n_components": [10, 20, 50], "model__C": [0.1, 1.0, 10.0]},
        "classification", inner_splits=5, random_state=RANDOM_STATE,
    )
    prediction = final.predict(X_test)
    scores = _decision_scores(final, X_test)
    return {
        "name": name,
        "source_path": source_path,
        "source_sha256": _sha256(source_path),
        "nested_cv": nested,
        "heldout_metrics": classification_metrics(
            y_test, prediction, scores=scores, classes=final.classes_),
        "final_best_params": final.best_params_,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "PCA components are predictive dimensions, not gene-level candidate signatures.",
    }


def run_regression():
    X, y, split, source_path = load_regression("regression_example.csv")
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    estimator = Ridge(random_state=RANDOM_STATE)
    started = time.perf_counter()
    nested = nested_cv_regression(
        X_train, y_train, estimator,
        param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        k_values=(10, 20, 50), outer_splits=5, outer_repeats=5,
        inner_splits=3, random_state=RANDOM_STATE, feature_method="pca",
    )
    pipeline = build_regression_pipeline(
        estimator, k=50, feature_method="pca", random_state=RANDOM_STATE)
    final = fit_final_model(
        X_train, y_train, pipeline,
        {"selector__n_components": [10, 20, 50], "model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "regression", inner_splits=5, random_state=RANDOM_STATE,
    )
    prediction = final.predict(X_test)
    return {
        "name": "regression",
        "source_path": source_path,
        "source_sha256": _sha256(source_path),
        "nested_cv": nested,
        "heldout_metrics": regression_metrics(y_test, prediction),
        "final_best_params": final.best_params_,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "PCA components are predictive dimensions, not gene-level candidate signatures.",
    }


def main():
    result = {"protocol": "fold-local PCA; repeated 5-fold x 5 outer CV; 3-fold inner selection"}
    if os.path.isfile(CHECKPOINT_PATH) and not os.environ.get("MALER_VALIDATION_FORCE"):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as handle:
            result.update(json.load(handle))
    jobs = (
        ("binary", lambda: run_classification("binary_classification_example.csv", "binary")),
        ("multiclass", lambda: run_classification("multiclass_classification_example.csv", "multiclass")),
        ("regression", run_regression),
    )
    for name, job in jobs:
        if name in result:
            print("Loaded PCA checkpoint:", name, flush=True)
            continue
        print("PCA sensitivity:", name, flush=True)
        result[name] = job()
        _write_json(CHECKPOINT_PATH, result)
        print("Wrote PCA checkpoint:", name, flush=True)
    _write_json(OUTPUT_PATH, result)
    print("Wrote", OUTPUT_PATH)
    for name in ("binary", "multiclass", "regression"):
        print(name, result[name]["final_best_params"], result[name]["heldout_metrics"])


if __name__ == "__main__":
    main()
