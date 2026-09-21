"""Data-quality and cohort-shift audit for the bundled survival example."""

from __future__ import absolute_import, print_function

import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import ks_2samp


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from mlserver.safe_ml import file_sha256  # noqa: E402
from validation.run_reviewer_validation import load_survival, split_declared  # noqa: E402
from validation.run_survival_expanded_validation import (  # noqa: E402
    collapse_tcga_aliquots,
    tcga_patient_identifier,
)


OUTPUT_PATH = os.path.join(REPOSITORY_ROOT, "validation", "results", "survival_data_quality_audit.json")


def _native(value):
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main():
    X, y, split, source_path = load_survival("survival_example.csv")
    X_train_raw, y_train_raw, X_external, y_external = split_declared(X, y, split)
    X_train, y_train, duplicates = collapse_tcga_aliquots(X_train_raw, y_train_raw)
    training_patients = set(tcga_patient_identifier(value) for value in X_train_raw.index)
    external_patients = set(str(value) for value in X_external.index)

    train_values = X_train.to_numpy(dtype=float)
    external_values = X_external.to_numpy(dtype=float)
    shift_rows = []
    for index, feature in enumerate(X_train.columns):
        train_column = train_values[:, index]
        external_column = external_values[:, index]
        statistic, p_value = ks_2samp(train_column, external_column)
        pooled_scale = float(np.sqrt((np.var(train_column) + np.var(external_column)) / 2.0))
        median_shift = float(np.median(external_column) - np.median(train_column))
        shift_rows.append({
            "feature": str(feature),
            "ks_statistic": float(statistic),
            "ks_p_value": float(p_value),
            "median_shift": median_shift,
            "standardized_median_shift": median_shift / pooled_scale if pooled_scale else None,
        })
    shift_rows.sort(key=lambda row: row["ks_statistic"], reverse=True)
    strong_shift = [row for row in shift_rows if row["ks_statistic"] >= 0.5]

    checks = [
        {"name": "sample_identifier_uniqueness", "passed": not X.index.has_duplicates,
         "evidence": {"samples": len(X), "duplicate_identifiers": int(X.index.duplicated().sum())}},
        {"name": "feature_identifier_uniqueness", "passed": not X.columns.has_duplicates,
         "evidence": {"features": X.shape[1], "duplicate_identifiers": int(X.columns.duplicated().sum())}},
        {"name": "finite_feature_values", "passed": bool(np.isfinite(X.to_numpy(dtype=float)).all()),
         "evidence": {"non_finite_cells": int((~np.isfinite(X.to_numpy(dtype=float))).sum())}},
        {"name": "outcome_validity", "passed": bool(
            np.isfinite(y["time"]).all() and (y["time"] > 0).all() and set(np.unique(y["event"])) <= {False, True}),
         "evidence": {"minimum_time": float(np.min(y["time"])), "maximum_time": float(np.max(y["time"]))}},
        {"name": "cross_cohort_patient_overlap", "passed": not bool(training_patients & external_patients),
         "evidence": {"overlap": sorted(training_patients & external_patients)}},
        {"name": "patient_level_training_uniqueness", "passed": len(duplicates) == 0,
         "evidence": {"raw_samples": len(X_train_raw), "unique_patients": len(X_train),
                      "duplicate_patients": len(duplicates), "extra_aliquots": len(X_train_raw) - len(X_train)}},
        {"name": "duplicate_outcome_consistency", "passed": all(row["outcomes_consistent"] for row in duplicates),
         "evidence": {"duplicate_patient_records": duplicates}},
    ]

    result = {
        "source_path": source_path,
        "source_sha256": file_sha256(source_path),
        "grain": "one column per tumour specimen; rows are outcome fields then gene-expression features",
        "cohort_inference_from_identifiers": {
            "declared_training": dict(Counter("TCGA" if str(value).startswith("TCGA-") else "other" for value in X_train_raw.index)),
            "declared_testing": dict(Counter("CGGA" if str(value).startswith("CGGA") else "other" for value in X_external.index)),
        },
        "training": {
            "raw_samples": len(X_train_raw),
            "unique_patients_after_primary_sample_rule": len(X_train),
            "events_after_collapse": int(y_train["event"].sum()),
            "censored_after_collapse": int((~y_train["event"]).sum()),
            "followup_median": float(np.median(y_train["time"])),
            "followup_iqr": [float(np.percentile(y_train["time"], 25)), float(np.percentile(y_train["time"], 75))],
        },
        "external": {
            "samples": len(X_external),
            "events": int(y_external["event"].sum()),
            "censored": int((~y_external["event"]).sum()),
            "followup_median": float(np.median(y_external["time"])),
            "followup_iqr": [float(np.percentile(y_external["time"], 25)), float(np.percentile(y_external["time"], 75))],
        },
        "checks": checks,
        "passed_checks": sum(bool(row["passed"]) for row in checks),
        "total_checks": len(checks),
        "expected_quality_exception": "Patient-level uniqueness fails before deterministic aliquot collapse; duplicate outcomes are consistent.",
        "cohort_shift": {
            "features_with_ks_at_least_0_5": len(strong_shift),
            "fraction_with_ks_at_least_0_5": len(strong_shift) / float(X.shape[1]),
            "median_ks_statistic": float(np.median([row["ks_statistic"] for row in shift_rows])),
            "top_50_shifted_features": shift_rows[:50],
        },
        "analytical_implications": [
            "The original row-level cross-validation can place aliquots from the same TCGA patient in different folds.",
            "Patient-level collapse is required before internal performance estimation.",
            "The declared test partition is a cross-cohort TCGA-to-CGGA audit, not an internal random holdout.",
            "Large expression-distribution shifts can reduce external transport even when internal cross-validation is valid.",
        ],
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(_native(result), handle, indent=2, sort_keys=True, ensure_ascii=False)
    print("Wrote", OUTPUT_PATH)
    print("checks", result["passed_checks"], "/", result["total_checks"])
    print("duplicate patients", len(duplicates))
    print("KS>=0.5", len(strong_shift), "/", X.shape[1])


if __name__ == "__main__":
    main()
