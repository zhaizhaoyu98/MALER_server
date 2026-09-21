"""Expanded, leakage-controlled survival validation for the MALER revision.

This script has two deliberately distinct purposes:

1. Re-audit the repository's high-dimensional TCGA -> CGGA example.  Duplicate
   TCGA aliquots are collapsed to one primary-tumour sample per patient before
   cross-validation.  Candidate models are selected only from repeated nested
   cross-validation on TCGA; CGGA is an external audit cohort and never enters
   model selection.
2. Provide a standard positive-control benchmark with the public GBSG2 breast-
   cancer survival dataset bundled by scikit-survival.  A single stratified
   holdout is declared before fitting, and the winning model family is selected
   solely from repeated nested cross-validation on the development partition.

All learned preprocessing, dimensionality reduction, feature selection and
model fitting occur inside sklearn Pipelines.  The output records every tested
candidate; it never chooses a model from held-out/external performance.
"""

from __future__ import absolute_import, print_function

import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import sksurv
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import (
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.svm import FastSurvivalSVM


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import summarize_fold_metrics, validate_feature_matrix  # noqa: E402
from mlserver.safe_survival import CoxPHSelectKBest  # noqa: E402
from validation.run_reviewer_validation import load_survival, split_declared  # noqa: E402


RESULTS_DIR = os.path.join(REPOSITORY_ROOT, "validation", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "survival_expanded_validation.json")
CACHE_DIR = os.path.join(RESULTS_DIR, "survival_candidate_cache")
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_native(value), handle, indent=2, sort_keys=True, ensure_ascii=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tcga_patient_identifier(sample_identifier):
    value = str(sample_identifier)
    return value[:12] if value.startswith("TCGA-") and len(value) >= 12 else value


def _tcga_sample_priority(sample_identifier):
    """Prefer primary tumour 01A, then other 01 aliquots, then recurrent 02."""
    value = str(sample_identifier)
    sample_code = value[13:15] if len(value) >= 15 else "99"
    vial = value[15:16] if len(value) >= 16 else "Z"
    return (0 if sample_code == "01" else 1, sample_code, vial, value)


def collapse_tcga_aliquots(X, y):
    patient_to_rows = {}
    for row, sample_identifier in enumerate(X.index):
        patient_to_rows.setdefault(tcga_patient_identifier(sample_identifier), []).append(row)
    selected = []
    duplicate_detail = []
    for patient, rows in sorted(patient_to_rows.items()):
        ordered = sorted(rows, key=lambda row: _tcga_sample_priority(X.index[row]))
        selected.append(ordered[0])
        if len(rows) > 1:
            duplicate_detail.append({
                "patient": patient,
                "samples": [str(X.index[row]) for row in rows],
                "selected": str(X.index[ordered[0]]),
                "outcomes_consistent": bool(
                    len(set((bool(y[row]["event"]), float(y[row]["time"])) for row in rows)) == 1
                ),
            })
    selected = np.asarray(sorted(selected), dtype=int)
    return X.iloc[selected].copy(), y[selected].copy(), duplicate_detail


def _preprocessing_pipeline(estimator, reducer):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reduce", reducer),
        ("model", clone(estimator)),
    ])


def candidate_specs_high_dimensional(n_features):
    k_values = sorted(set(min(n_features, value) for value in (20, 50, 100)))
    pca_values = (10, 20, 50)
    return [
        {
            "name": "coxph_ranked",
            "pipeline": _preprocessing_pipeline(CoxPHSurvivalAnalysis(), CoxPHSelectKBest(k=max(k_values))),
            "grid": {"reduce__k": k_values, "model__alpha": [0.1, 1.0, 10.0]},
            "representation": "fold-local univariate Cox ranking",
        },
        {
            "name": "coxnet_ranked",
            "pipeline": _preprocessing_pipeline(
                CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, n_alphas=30, fit_baseline_model=True),
                CoxPHSelectKBest(k=max(k_values)),
            ),
            "grid": {"reduce__k": k_values, "model__l1_ratio": [0.1, 0.5, 0.9]},
            "representation": "fold-local univariate Cox ranking",
        },
        {
            "name": "extra_trees_ranked",
            "pipeline": _preprocessing_pipeline(
                ExtraSurvivalTrees(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
                CoxPHSelectKBest(k=max(k_values)),
            ),
            "grid": {
                "reduce__k": k_values,
                "model__min_samples_leaf": [3, 5, 10],
                "model__max_features": ["sqrt", 0.5],
            },
            "representation": "fold-local univariate Cox ranking",
        },
        {
            "name": "random_forest_ranked",
            "pipeline": _preprocessing_pipeline(
                RandomSurvivalForest(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
                CoxPHSelectKBest(k=max(k_values)),
            ),
            "grid": {
                "reduce__k": k_values,
                "model__min_samples_leaf": [5, 10],
                "model__max_features": ["sqrt", 0.5],
            },
            "representation": "fold-local univariate Cox ranking",
        },
        {
            "name": "gradient_boosting_ranked",
            "pipeline": _preprocessing_pipeline(
                GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE),
                CoxPHSelectKBest(k=max(k_values)),
            ),
            "grid": {
                "reduce__k": k_values,
                "model__n_estimators": [100, 250],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
            },
            "representation": "fold-local univariate Cox ranking",
        },
        {
            "name": "coxph_pca",
            "pipeline": _preprocessing_pipeline(CoxPHSurvivalAnalysis(), PCA(random_state=RANDOM_STATE)),
            "grid": {"reduce__n_components": list(pca_values), "model__alpha": [0.1, 1.0, 10.0]},
            "representation": "fold-local PCA",
        },
        {
            "name": "survival_svm_pca",
            "pipeline": _preprocessing_pipeline(
                FastSurvivalSVM(random_state=RANDOM_STATE, max_iter=1000, tol=1e-5),
                PCA(random_state=RANDOM_STATE),
            ),
            "grid": {"reduce__n_components": list(pca_values), "model__alpha": [0.01, 0.1, 1.0, 10.0]},
            "representation": "fold-local PCA",
        },
    ]


def candidate_specs_low_dimensional():
    passthrough = "passthrough"
    return [
        {
            "name": "coxph",
            "pipeline": _preprocessing_pipeline(CoxPHSurvivalAnalysis(), passthrough),
            # A strictly positive penalty avoids singular fold matrices after
            # one-hot encoding in the legacy scipy/sksurv stack.
            "grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
            "representation": "all encoded clinical features",
        },
        {
            "name": "coxnet",
            "pipeline": _preprocessing_pipeline(
                CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, n_alphas=50, fit_baseline_model=True), passthrough
            ),
            "grid": {"model__l1_ratio": [0.1, 0.5, 0.9]},
            "representation": "all encoded clinical features",
        },
        {
            "name": "extra_trees",
            "pipeline": _preprocessing_pipeline(
                ExtraSurvivalTrees(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1), passthrough
            ),
            "grid": {
                "model__min_samples_leaf": [3, 5, 10, 20],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
            "representation": "all encoded clinical features",
        },
        {
            "name": "random_forest",
            "pipeline": _preprocessing_pipeline(
                RandomSurvivalForest(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1), passthrough
            ),
            "grid": {
                "model__min_samples_leaf": [3, 5, 10, 20],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
            "representation": "all encoded clinical features",
        },
        {
            "name": "gradient_boosting",
            "pipeline": _preprocessing_pipeline(
                GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE), passthrough
            ),
            "grid": {
                "model__n_estimators": [100, 250],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [1, 2],
            },
            "representation": "all encoded clinical features",
        },
        {
            "name": "survival_svm",
            "pipeline": _preprocessing_pipeline(
                FastSurvivalSVM(random_state=RANDOM_STATE, max_iter=1000, tol=1e-5), passthrough
            ),
            "grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
            "representation": "all encoded clinical features",
        },
    ]


def transform_before_model(fitted_pipeline, X):
    transformed = X
    for name, step in fitted_pipeline.steps:
        if name == "model":
            break
        if step != "passthrough":
            transformed = step.transform(transformed)
    return transformed


def survival_metrics(fitted_pipeline, X_train, y_train, X_test, y_test):
    risk = np.asarray(fitted_pipeline.predict(X_test), dtype=float)
    if not np.isfinite(risk).all():
        raise ValueError("Survival model produced non-finite risk scores.")
    result = {
        "c_index": float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0]),
        "risk_all_finite": True,
        "risk_min": float(np.min(risk)),
        "risk_max": float(np.max(risk)),
    }
    lower = max(float(np.min(y_train["time"])), float(np.min(y_test["time"])))
    upper = min(float(np.max(y_train["time"])), float(np.max(y_test["time"])))
    event_times = y_test["time"][
        y_test["event"] & (y_test["time"] > lower) & (y_test["time"] < upper)
    ]
    if len(event_times) >= 3:
        times = np.unique(np.quantile(event_times, [0.25, 0.50, 0.75]))
        if len(times) >= 2:
            try:
                auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
                result.update({
                    "time_auc_mean": float(mean_auc),
                    "time_auc_times": [float(value) for value in times],
                    "time_auc_values": [float(value) for value in auc],
                })
            except ValueError:
                pass
            model = fitted_pipeline.named_steps["model"]
            if hasattr(model, "predict_survival_function"):
                try:
                    transformed = transform_before_model(fitted_pipeline, X_test)
                    functions = model.predict_survival_function(transformed)
                    probabilities = np.asarray([[fn(time) for time in times] for fn in functions])
                    result["integrated_brier_score"] = float(
                        integrated_brier_score(y_train, y_test, probabilities, times)
                    )
                except (ValueError, TypeError):
                    pass
    return result, risk


def bootstrap_c_index(y, risk, repetitions=2000, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    estimates = []
    for _ in range(repetitions):
        index = rng.randint(0, len(y), len(y))
        try:
            estimates.append(float(concordance_index_censored(
                y["event"][index], y["time"][index], risk[index]
            )[0]))
        except ValueError:
            continue
    return {
        "estimate": float(concordance_index_censored(y["event"], y["time"], risk)[0]),
        "bootstrap_95_ci": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n_bootstrap": len(estimates),
    }


def selected_features(fitted_pipeline, columns):
    reducer = fitted_pipeline.named_steps["reduce"]
    if hasattr(reducer, "get_support"):
        indices = reducer.get_support(indices=True)
        return [str(columns[index]) for index in indices]
    return []


def nested_evaluate_candidate(X, y, spec, outer_repeats=5, outer_splits=5, inner_splits=3):
    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=RANDOM_STATE
    )
    event = y["event"].astype(int)
    folds = []
    started = time.perf_counter()
    for fold, (train_index, validation_index) in enumerate(outer.split(X, event), start=1):
        inner = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE + fold
        )
        inner_indices = list(inner.split(X.iloc[train_index], event[train_index]))
        search = GridSearchCV(
            clone(spec["pipeline"]), spec["grid"], cv=inner_indices,
            scoring=None, refit=True, error_score="raise", n_jobs=1,
        )
        search.fit(X.iloc[train_index], y[train_index])
        metrics, _ = survival_metrics(
            search.best_estimator_, X.iloc[train_index], y[train_index],
            X.iloc[validation_index], y[validation_index],
        )
        metrics.update({
            "fold": fold,
            "n_train": len(train_index),
            "n_validation": len(validation_index),
            "best_params": search.best_params_,
            "selected_features": selected_features(search.best_estimator_, X.columns),
        })
        folds.append(metrics)
    return {
        "name": spec["name"],
        "representation": spec["representation"],
        "parameter_grid": spec["grid"],
        "folds": folds,
        "summary": summarize_fold_metrics(folds),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def cached_nested_evaluate(scope, X, y, spec, outer_repeats=5, outer_splits=5, inner_splits=3):
    identity = {
        "cache_version": 1,
        "scope": scope,
        "name": spec["name"],
        "grid": spec["grid"],
        "shape": list(X.shape),
        "events": int(y["event"].sum()),
        "outer_repeats": outer_repeats,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "random_state": RANDOM_STATE,
    }
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "%s_%s_%s.json" % (scope, spec["name"], digest))
    if os.path.isfile(path) and not os.environ.get("MALER_VALIDATION_FORCE"):
        with open(path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        print("Loaded candidate checkpoint:", path, flush=True)
        return cached
    result = nested_evaluate_candidate(
        X, y, spec, outer_repeats=outer_repeats,
        outer_splits=outer_splits, inner_splits=inner_splits)
    _write_json(path, result)
    print("Wrote candidate checkpoint:", path, flush=True)
    return result


def choose_winner(candidate_results):
    ranked = sorted(
        candidate_results,
        key=lambda row: (
            -row["summary"]["c_index"]["mean"],
            row["summary"]["c_index"]["std"],
            row["name"],
        ),
    )
    return ranked[0]["name"], [
        {
            "rank": rank,
            "name": row["name"],
            "mean_c_index": row["summary"]["c_index"]["mean"],
            "sd_c_index": row["summary"]["c_index"]["std"],
        }
        for rank, row in enumerate(ranked, start=1)
    ]


def fit_and_evaluate_candidates(X_train, y_train, X_test, y_test, specs):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    indices = list(cv.split(X_train, y_train["event"].astype(int)))
    rows = []
    for spec in specs:
        search = GridSearchCV(
            clone(spec["pipeline"]), spec["grid"], cv=indices,
            scoring=None, refit=True, error_score="raise", n_jobs=1,
        )
        search.fit(X_train, y_train)
        metrics, risk = survival_metrics(search.best_estimator_, X_train, y_train, X_test, y_test)
        rows.append({
            "name": spec["name"],
            "best_params": search.best_params_,
            "metrics": metrics,
            "c_index_bootstrap": bootstrap_c_index(y_test, risk),
            "selected_features": selected_features(search.best_estimator_, X_train.columns),
        })
    return rows


def run_tcga_cgga():
    X, y, split, source_path = load_survival("survival_example.csv")
    X_train_raw, y_train_raw, X_external, y_external = split_declared(X, y, split)
    X_train, y_train, duplicate_detail = collapse_tcga_aliquots(X_train_raw, y_train_raw)
    specs = candidate_specs_high_dimensional(X_train.shape[1])
    candidate_results = []
    for spec in specs:
        print("TCGA nested CV:", spec["name"], flush=True)
        candidate_results.append(cached_nested_evaluate(
            "tcga", X_train, y_train, spec, outer_repeats=5))
    winner, ranking = choose_winner(candidate_results)
    external_results = fit_and_evaluate_candidates(X_train, y_train, X_external, y_external, specs)
    selected_external = next(row for row in external_results if row["name"] == winner)
    return {
        "design": "TCGA patient-level development; CGGA external audit",
        "source_path": source_path,
        "source_sha256": _sha256(source_path),
        "training_raw_samples": len(X_train_raw),
        "training_unique_patients": len(X_train),
        "training_events": int(y_train["event"].sum()),
        "external_samples": len(X_external),
        "external_events": int(y_external["event"].sum()),
        "n_features": X_train.shape[1],
        "duplicate_patient_audit": duplicate_detail,
        "candidate_results": candidate_results,
        "selection_rule": "highest repeated nested-CV mean C-index; lower SD then name break ties",
        "winner": winner,
        "ranking": ranking,
        "all_external_audit_results": external_results,
        "selected_model_external_result": selected_external,
        "external_use_warning": "CGGA outcomes were already reported previously; treat as an external audit, not a newly untouched cohort.",
    }


def normalize_gbsg2_target(y):
    target = np.empty(len(y), dtype=[("event", "?"), ("time", "<f8")])
    target["event"] = y[y.dtype.names[0]].astype(bool)
    target["time"] = y[y.dtype.names[1]].astype(float)
    return target


def run_gbsg2():
    X_raw, y_raw = load_gbsg2()
    X = pd.get_dummies(X_raw, drop_first=False)
    X.index = ["GBSG2-%04d" % (index + 1) for index in range(len(X))]
    X.columns = [str(column) for column in X.columns]
    X = validate_feature_matrix(X)
    y = normalize_gbsg2_target(y_raw)
    fingerprint = hashlib.sha256()
    fingerprint.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    fingerprint.update(y.tobytes())
    indices = np.arange(len(X))
    train_index, test_index = train_test_split(
        indices, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y["event"].astype(int),
    )
    X_train, y_train = X.iloc[train_index], y[train_index]
    X_test, y_test = X.iloc[test_index], y[test_index]
    specs = candidate_specs_low_dimensional()
    candidate_results = []
    for spec in specs:
        print("GBSG2 nested CV:", spec["name"], flush=True)
        candidate_results.append(cached_nested_evaluate(
            "gbsg2", X_train, y_train, spec, outer_repeats=5))
    winner, ranking = choose_winner(candidate_results)
    test_results = fit_and_evaluate_candidates(X_train, y_train, X_test, y_test, specs)
    selected_test = next(row for row in test_results if row["name"] == winner)
    return {
        "design": "GBSG2 public benchmark; declared stratified 75/25 development/holdout split",
        "source": "scikit-survival bundled GBSG2 dataset",
        "scikit_survival_version": sksurv.__version__,
        "data_fingerprint_sha256": fingerprint.hexdigest(),
        "raw_samples": len(X),
        "development_samples": len(X_train),
        "development_events": int(y_train["event"].sum()),
        "heldout_samples": len(X_test),
        "heldout_events": int(y_test["event"].sum()),
        "encoded_features": X.shape[1],
        "raw_columns": [str(column) for column in X_raw.columns],
        "candidate_results": candidate_results,
        "selection_rule": "highest repeated nested-CV mean C-index; lower SD then name break ties",
        "winner": winner,
        "ranking": ranking,
        "all_heldout_audit_results": test_results,
        "selected_model_heldout_result": selected_test,
        "holdout_use_warning": "Candidate family is selected from development nested CV only; non-winning holdout rows are audit information.",
    }


def main():
    started = time.perf_counter()
    tcga_result = run_tcga_cgga()
    _write_json(OUTPUT_PATH + ".partial", {
        "status": "TCGA/CGGA complete; GBSG2 pending",
        "tcga_cgga": tcga_result,
    })
    result = {
        "protocol": {
            "random_state": RANDOM_STATE,
            "outer_cv": "repeated stratified 5-fold x 5",
            "inner_cv": "stratified 3-fold",
            "bootstrap": 2000,
            "model_selection": "development nested CV only",
            "preprocessing": "median imputation and scaling fitted inside every fold",
        },
        "tcga_cgga": tcga_result,
        "gbsg2": run_gbsg2(),
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(OUTPUT_PATH, result)
    print("Wrote", OUTPUT_PATH)
    print("TCGA winner", result["tcga_cgga"]["winner"],
          result["tcga_cgga"]["selected_model_external_result"]["metrics"])
    print("GBSG2 winner", result["gbsg2"]["winner"],
          result["gbsg2"]["selected_model_heldout_result"]["metrics"])


if __name__ == "__main__":
    main()
