"""Matched-resampling no-feature-selection baselines for the reviewer revision.

The outer and inner splits and model hyperparameter grids match the primary
analysis, but the number of feature-size candidates differs. This is not a
compute-matched or causal feature-selection ablation.
"""

from __future__ import absolute_import, print_function

import json
import hashlib
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sksurv.tree import SurvivalTree


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import nested_cv_classification, nested_cv_regression  # noqa: E402
from mlserver.safe_survival import nested_cv_survival  # noqa: E402
from validation.run_reviewer_validation import (  # noqa: E402
    load_classification,
    load_regression,
    load_survival,
    split_declared,
)
from validation.run_survival_expanded_validation import collapse_tcga_aliquots  # noqa: E402


RESULTS_DIR = os.path.join(REPOSITORY_ROOT, "validation", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "equal_budget_no_selection_baselines.json")
RANDOM_STATE = 10
CHECKPOINT_PATH = OUTPUT_PATH + ".partial"


def compress_no_selection_features(result, n_features):
    for fold in result.get("folds", []):
        fold["selected_features"] = ["ALL_%d_FEATURES" % n_features]
    return result


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


def classification(filename, multiclass=False):
    X, y, split, source_path = load_classification(filename)
    X_train, y_train, _, _ = split_declared(X, y, split)
    started = time.perf_counter()
    result = nested_cv_classification(
        X_train, y_train,
        LogisticRegression(solver="liblinear", class_weight="balanced",
                           max_iter=500, random_state=RANDOM_STATE,
                           multi_class="ovr"),
        param_grid={"model__C": [0.1, 1.0, 10.0]}, k_values=(X_train.shape[1],),
        outer_splits=5, outer_repeats=10, inner_splits=3,
        random_state=RANDOM_STATE, feature_method="none",
    )
    result = compress_no_selection_features(result, X_train.shape[1])
    return {"nested_cv": result, "elapsed_seconds": time.perf_counter() - started,
            "source_path": source_path, "source_sha256": _sha256(source_path),
            "multiclass": bool(multiclass)}


def regression():
    X, y, split, source_path = load_regression("regression_example.csv")
    X_train, y_train, _, _ = split_declared(X, y, split)
    started = time.perf_counter()
    result = nested_cv_regression(
        X_train, y_train, Ridge(random_state=RANDOM_STATE),
        param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        k_values=(X_train.shape[1],), outer_splits=5, outer_repeats=10,
        inner_splits=3, random_state=RANDOM_STATE, feature_method="none",
    )
    result = compress_no_selection_features(result, X_train.shape[1])
    return {"nested_cv": result, "elapsed_seconds": time.perf_counter() - started,
            "source_path": source_path, "source_sha256": _sha256(source_path)}


def survival():
    X, y, split, source_path = load_survival("survival_example.csv")
    # Preserve raw aliquot provenance; collapse below applies the same
    # patient-level training set used by the primary analysis.
    X_train, y_train, _, _ = split_declared(X, y, split, patient_level=False)
    X_train, y_train, duplicate_detail = collapse_tcga_aliquots(X_train, y_train)
    started = time.perf_counter()
    result = nested_cv_survival(
        X_train, y_train, SurvivalTree(random_state=RANDOM_STATE, min_samples_leaf=10),
        param_grid={"model__max_depth": [2, 3, 5]}, k_values=(X_train.shape[1],),
        outer_splits=5, outer_repeats=10, inner_splits=3,
        random_state=RANDOM_STATE, feature_method="none",
    )
    result = compress_no_selection_features(result, X_train.shape[1])
    return {"nested_cv": result, "elapsed_seconds": time.perf_counter() - started,
            "source_path": source_path, "source_sha256": _sha256(source_path),
            "patient_level_training_samples": len(X_train),
            "collapsed_duplicate_patients": len(duplicate_detail)}


def main():
    result = {"protocol": "same 5-fold x 10 outer repeats and 3-fold inner tuning as primary analyses; feature-size candidate count and computational budget differ"}
    if os.path.isfile(CHECKPOINT_PATH) and not os.environ.get("MALER_VALIDATION_FORCE"):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as handle:
            result.update(json.load(handle))
    jobs = (
        ("binary", lambda: classification("binary_classification_example.csv")),
        ("multiclass", lambda: classification("multiclass_classification_example.csv", multiclass=True)),
        ("regression", regression),
        ("survival", survival),
    )
    for name, job in jobs:
        if name in result:
            print("Loaded baseline checkpoint:", name, flush=True)
            continue
        print("Matched-resampling baseline:", name, flush=True)
        result[name] = job()
        _write_json(CHECKPOINT_PATH, result)
        print("Wrote baseline checkpoint:", name, flush=True)
    _write_json(OUTPUT_PATH, result)
    print("Wrote", OUTPUT_PATH)
    print("binary", result["binary"]["nested_cv"]["summary"]["balanced_accuracy"])
    print("multiclass", result["multiclass"]["nested_cv"]["summary"]["balanced_accuracy"])
    print("regression", result["regression"]["nested_cv"]["summary"]["r2"])
    print("survival", result["survival"]["nested_cv"]["summary"]["c_index"])


if __name__ == "__main__":
    main()
