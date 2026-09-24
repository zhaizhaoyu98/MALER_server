"""Independent cross-platform validation on the real GSE37745 NSCLC cohort."""

from __future__ import absolute_import, print_function

import json
import os
import sqlite3
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold


VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(VALIDATION_DIR)
for directory in (REPOSITORY_ROOT, VALIDATION_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from mlserver.safe_ml import (  # noqa: E402
    _decision_scores,
    build_cross_platform_classification_pipeline,
    classification_metrics,
    file_sha256,
    load_signed_model_bundle,
    save_signed_model_bundle,
    selected_feature_names,
)
from run_reviewer_validation import (  # noqa: E402
    RANDOM_STATE,
    RESULTS_DIR,
    _native,
    _write_json,
    bootstrap_classification,
    describe_dataset,
    feature_stability,
    fold_intervals,
    load_classification,
    split_declared,
    validation_key,
)


EXTERNAL_DIR = os.path.join(VALIDATION_DIR, "external_data")
ZIP_PATH = os.path.join(EXTERNAL_DIR, "E-GEOD-37745.processed.1.zip")
SDRF_PATH = os.path.join(EXTERNAL_DIR, "E-GEOD-37745.sdrf.txt")
METADATA_PATH = os.path.join(EXTERNAL_DIR, "E-GEOD-37745_metadata.json")
MYGENE_PATH = os.path.join(EXTERNAL_DIR, "binary_feature_mygene_mapping.json")
SQLITE_PATH = os.path.join(
    EXTERNAL_DIR, "hgu133plus2.db", "inst", "extdata", "hgu133plus2.sqlite")


def read_internal_selected_features():
    path = os.path.join(RESULTS_DIR, "binary_classification.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["final_selected_features"]


def probe_mapping(features):
    with open(MYGENE_PATH, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    by_query = {}
    for record in records:
        by_query.setdefault(record.get("query"), []).append(record)
    connection = sqlite3.connect(SQLITE_PATH)
    try:
        mapping = {}
        evidence_rows = []
        for feature in features:
            probes = set()
            candidates = by_query.get(feature, [])
            exact = [record for record in candidates if record.get("symbol") == feature]
            for record in exact or candidates:
                reporter = record.get("reporter", {}).get("HG-U133_Plus_2", [])
                if isinstance(reporter, str):
                    reporter = [reporter]
                probes.update(reporter)
                gene_id = str(record.get("_id", ""))
                if gene_id.isdigit():
                    probes.update(row[0] for row in connection.execute(
                        "select probe_id from probes where gene_id = ?", (gene_id,)).fetchall())
            mapping[feature] = sorted(probes)
            evidence_rows.append({
                "feature": feature,
                "mapped": bool(probes),
                "probe_count": len(probes),
                "probes": ";".join(sorted(probes)),
                "mygene_ids": ";".join(sorted(set(
                    str(record.get("_id")) for record in (exact or candidates)
                    if record.get("_id") is not None))),
            })
    finally:
        connection.close()
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(os.path.join(RESULTS_DIR, "gse37745_probe_mapping.csv"), index=False)
    return mapping, evidence


def nested_rank_cv(X, y, estimator):
    k_values = sorted(set([10, 20, min(40, X.shape[1])]))
    pipeline = build_cross_platform_classification_pipeline(estimator, k=max(k_values))
    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_index, test_index) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold)
        search = GridSearchCV(
            pipeline,
            {"selector__k": k_values, "model__C": [0.1, 1.0, 10.0]},
            cv=inner, scoring="balanced_accuracy", refit=True,
            n_jobs=1, error_score="raise")
        search.fit(X.iloc[train_index], y[train_index])
        prediction = search.predict(X.iloc[test_index])
        scores = _decision_scores(search, X.iloc[test_index])
        metrics = classification_metrics(
            y[test_index], prediction, scores=scores, classes=search.classes_)
        metrics.update({
            "fold": fold,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "best_params": search.best_params_,
            "selected_features": selected_feature_names(
                search.best_estimator_, X.columns),
        })
        rows.append(_native(metrics))
    from mlserver.safe_ml import summarize_fold_metrics
    return {"folds": rows, "summary": summarize_fold_metrics(rows)}


def fit_rank_model(X, y, estimator):
    k_values = sorted(set([10, 20, min(40, X.shape[1])]))
    pipeline = build_cross_platform_classification_pipeline(estimator, k=max(k_values))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        {"selector__k": k_values, "model__C": [0.1, 1.0, 10.0]},
        cv=cv, scoring="balanced_accuracy", refit=True,
        n_jobs=1, error_score="raise")
    search.fit(X, y)
    return search


def read_external_expression(feature_probes, selected_features):
    sdrf = pd.read_csv(SDRF_PATH, sep="\t", dtype=str, low_memory=False)
    histology_column = "Characteristics [histology]"
    file_column = "Derived Array Data File"
    source_column = "Source Name"
    subset = sdrf[sdrf[histology_column].str.lower().isin(["adeno", "squamous"])].copy()
    subset["label"] = subset[histology_column].str.lower().map(
        {"adeno": "LUAD", "squamous": "LUSC"})
    required_probes = sorted(set(
        probe for feature in selected_features for probe in feature_probes[feature]))
    samples = []
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError("External cohort archive failed CRC at %s" % bad_member)
        available = set(archive.namelist())
        for _, row in subset.iterrows():
            filename = row[file_column]
            if filename not in available:
                raise ValueError("Processed sample file missing from archive: %s" % filename)
            with archive.open(filename) as handle:
                table = pd.read_csv(handle, sep="\t", index_col=0)
            missing = sorted(set(required_probes) - set(table.index.astype(str)))
            if missing:
                raise ValueError("Sample %s lacks required probes: %s" % (
                    row[source_column], ", ".join(missing[:10])))
            values = {}
            for feature in selected_features:
                probe_values = pd.to_numeric(
                    table.loc[feature_probes[feature], "VALUE"], errors="raise")
                values[feature] = float(np.median(np.asarray(probe_values, dtype=float)))
            values.update({
                "sample": row[source_column],
                "histology": row[histology_column].lower(),
                "label": row["label"],
                "processed_file": filename,
            })
            samples.append(values)
    frame = pd.DataFrame(samples).set_index("sample")
    metadata = frame[["histology", "label", "processed_file"]].copy()
    expression = frame[selected_features].astype(float)
    expression.to_csv(os.path.join(RESULTS_DIR, "gse37745_selected_expression.csv"))
    metadata.to_csv(os.path.join(RESULTS_DIR, "gse37745_samples.csv"))
    return expression, metadata


def cohort_zscore(frame):
    standard_deviation = frame.std(axis=0, ddof=0).replace(0.0, np.nan)
    normalized = (frame - frame.mean(axis=0)) / standard_deviation
    if normalized.isna().any().any():
        raise ValueError("Cohort z-score produced missing values.")
    return normalized


def main():
    started = time.perf_counter()
    for path in (ZIP_PATH, SDRF_PATH, METADATA_PATH, MYGENE_PATH, SQLITE_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    all_selected = read_internal_selected_features()
    mapping, mapping_evidence = probe_mapping(all_selected)
    mapped_features = [feature for feature in all_selected if mapping[feature]]
    unmapped_features = [feature for feature in all_selected if not mapping[feature]]
    if len(mapped_features) < 20:
        raise ValueError("Fewer than 20 selected genes map to GPL570.")

    X, y, split, source_path = load_classification("binary_classification_example.csv")
    X_train, y_train, X_test, y_test = split_declared(X, y, split)
    X_train = X_train[mapped_features]
    X_test = X_test[mapped_features]
    estimator = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=500,
        random_state=RANDOM_STATE)

    nested = nested_rank_cv(X_train, y_train, estimator)
    final = fit_rank_model(X_train, y_train, estimator)
    internal_prediction = final.predict(X_test)
    internal_scores = _decision_scores(final, X_test)
    internal_metrics = classification_metrics(
        y_test, internal_prediction, scores=internal_scores, classes=final.classes_)
    selected_features = selected_feature_names(final, X_train.columns)

    # The fitted selector consumes the complete mapped feature universe and
    # reduces it internally; preserve that input schema for prediction.
    external_X, external_metadata = read_external_expression(mapping, mapped_features)
    external_y = external_metadata["label"].to_numpy()
    external_prediction = final.predict(external_X)
    external_scores = _decision_scores(final, external_X)
    external_metrics = classification_metrics(
        external_y, external_prediction, scores=external_scores, classes=final.classes_)

    # Sensitivity analysis: normalize each assay cohort without using labels,
    # then fit the already selected gene set using the TCGA training cohort.
    train_z = cohort_zscore(X_train[selected_features])
    test_z = (X_test[selected_features] - X_train[selected_features].mean(axis=0)) / \
        X_train[selected_features].std(axis=0, ddof=0)
    external_z = cohort_zscore(external_X[selected_features])
    z_model = clone(estimator).set_params(C=final.best_params_["model__C"])
    z_model.fit(train_z, y_train)
    z_external_prediction = z_model.predict(external_z)
    z_external_scores = z_model.predict_proba(external_z)
    z_internal_prediction = z_model.predict(test_z)
    z_internal_scores = z_model.predict_proba(test_z)

    bundle_path = os.path.join(RESULTS_DIR, "gse37745_cross_platform_pipeline.maler")
    signing_key = validation_key()
    save_signed_model_bundle(
        bundle_path, final.best_estimator_, "binary", mapped_features, signing_key,
        metadata={
            "method": "model_bclass",
            "model_name": "CrossPlatformRankLogisticRegression",
            "classes": {str(value): str(value) for value in final.classes_},
            "pipeline_complete": True,
            "normalization": "within-sample percentile ranks",
            "training_dataset": "binary_classification_example.csv",
            "external_validation_dataset": "GSE37745 / E-GEOD-37745",
            "best_params": final.best_params_,
        })
    loaded, manifest = load_signed_model_bundle(bundle_path, signing_key)
    if not np.array_equal(loaded.predict(external_X), external_prediction):
        raise AssertionError("Saved cross-platform bundle did not reproduce predictions.")

    result = {
        "external_dataset": {
            "accession": "GSE37745 / E-GEOD-37745",
            "design": "independent single-institute NSCLC cohort; Affymetrix HG-U133 Plus 2.0",
            "normalization_from_submitter": "log2 RMA signal",
            "source_urls": {
                "ncbi_geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37745",
                "ebi_api": "https://www.ebi.ac.uk/biostudies/api/v1/studies/E-GEOD-37745",
                "ebi_processed_archive": "https://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEOD/E-GEOD-37745/E-GEOD-37745.processed.1.zip",
            },
            "n_included": int(len(external_y)),
            "class_counts": external_metadata["label"].value_counts().to_dict(),
            "excluded_histology": "large-cell carcinoma (not part of LUAD/LUSC target)",
            "archive_sha256": file_sha256(ZIP_PATH),
            "archive_bytes": os.path.getsize(ZIP_PATH),
            "sdrf_sha256": file_sha256(SDRF_PATH),
            "metadata_sha256": file_sha256(METADATA_PATH),
        },
        "feature_mapping": {
            "starting_internal_selected_features": len(all_selected),
            "gpl570_mapped_features": len(mapped_features),
            "mapped_features": mapped_features,
            "unmapped_features": unmapped_features,
            "final_model_selected_features": selected_features,
            "mygene_response_sha256": file_sha256(MYGENE_PATH),
            "bioconductor_sqlite_sha256": file_sha256(SQLITE_PATH),
        },
        "protocol": {
            "feature_availability_filter": "GPL570 mapping only; no external outcome used",
            "primary_normalization": "within-sample percentile ranks; independent per sample",
            "training_selection": "TCGA training partition only, inner 5-fold CV",
            "internal_validation": "repository-declared TCGA internal held-out partition, excluded from current selection and tuning",
            "external_validation": "all adeno/squamous GSE37745 samples, evaluated once",
            "positive_class": str(final.classes_[-1]),
        },
        "training_nested_cv": nested,
        "training_nested_cv_fold_intervals": fold_intervals(nested),
        "feature_stability": feature_stability(nested["folds"]),
        "best_params": final.best_params_,
        "internal_heldout_test_metrics": internal_metrics,
        "internal_heldout_test_bootstrap": bootstrap_classification(
            y_test, internal_prediction, internal_scores, final.classes_, repetitions=2000),
        "external_primary_metrics": external_metrics,
        "external_primary_bootstrap": bootstrap_classification(
            external_y, external_prediction, external_scores, final.classes_, repetitions=2000),
        "external_confusion_matrix": confusion_matrix(
            external_y, external_prediction, labels=final.classes_).tolist(),
        "sensitivity_cohort_zscore": {
            "note": "External feature-wise z-scores use the unlabeled external cohort distribution.",
            "internal_heldout_test_metrics": classification_metrics(
                y_test, z_internal_prediction, scores=z_internal_scores, classes=final.classes_),
            "external_metrics": classification_metrics(
                external_y, z_external_prediction, scores=z_external_scores, classes=final.classes_),
            "external_bootstrap": bootstrap_classification(
                external_y, z_external_prediction, z_external_scores,
                final.classes_, repetitions=2000),
        },
        "model_bundle": {
            "path": bundle_path,
            "sha256": file_sha256(bundle_path),
            "manifest": manifest,
            "round_trip_verified": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path = os.path.join(RESULTS_DIR, "external_gse37745.json")
    _write_json(output_path, result)
    pd.DataFrame([
        {"analysis": "primary_sample_rank", **{
            key: value for key, value in external_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))}},
        {"analysis": "sensitivity_cohort_zscore", **{
            key: value for key, value in result["sensitivity_cohort_zscore"]["external_metrics"].items()
            if isinstance(value, (int, float, np.integer, np.floating))}},
    ]).to_csv(os.path.join(RESULTS_DIR, "external_gse37745_summary.csv"), index=False)
    print(json.dumps({
        "n_external": result["external_dataset"]["n_included"],
        "class_counts": result["external_dataset"]["class_counts"],
        "mapped_features": len(mapped_features),
        "selected_features": selected_features,
        "external_primary_metrics": external_metrics,
        "external_zscore_metrics": result["sensitivity_cohort_zscore"]["external_metrics"],
        "output": output_path,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
