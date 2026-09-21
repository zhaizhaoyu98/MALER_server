"""Second independent LUAD/LUSC audit using the official GSE50081 matrix.

The model family, feature mapping and selected feature count are reconstructed
deterministically from TCGA exactly as in run_external_gse37745.py.  GSE37745
is used only to calibrate a single probability threshold; GSE50081 is then an
untouched final cohort and never enters feature, parameter or threshold choice.
"""

from __future__ import absolute_import, print_function

import csv
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(VALIDATION_DIR)
for directory in (REPOSITORY_ROOT, VALIDATION_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from mlserver.safe_ml import classification_metrics, file_sha256  # noqa: E402
from run_external_gse37745 import (  # noqa: E402
    fit_rank_model,
    probe_mapping,
    read_internal_selected_features,
)
from run_reviewer_validation import (  # noqa: E402
    RANDOM_STATE,
    RESULTS_DIR,
    _write_json,
    bootstrap_classification,
    load_classification,
    split_declared,
)


EXTERNAL_DIR = os.path.join(VALIDATION_DIR, "external_data")
MATRIX_PATH = os.path.join(EXTERNAL_DIR, "GSE50081_series_matrix.txt.gz")
PROTOCOL_PATH = os.path.join(EXTERNAL_DIR, "GSE50081_protocol.json")
GSE37745_EXPRESSION = os.path.join(RESULTS_DIR, "gse37745_selected_expression.csv")
GSE37745_SAMPLES = os.path.join(RESULTS_DIR, "gse37745_samples.csv")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "external_gse50081.json")
SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50081/"
    "matrix/GSE50081_series_matrix.txt.gz"
)


def read_series_metadata(path):
    accessions = None
    titles = None
    characteristic_rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            fields = next(csv.reader([line.rstrip("\r\n")], delimiter="\t", quotechar='"'))
            key, values = fields[0], fields[1:]
            if key == "!Sample_geo_accession":
                accessions = values
            elif key == "!Sample_title":
                titles = values
            elif key == "!Sample_characteristics_ch1":
                characteristic_rows.append(values)
    if not accessions or not titles or len(accessions) != len(titles):
        raise ValueError("GSE50081 sample metadata is incomplete.")
    rows = []
    for index, accession in enumerate(accessions):
        row = {"geo_accession": accession, "title": titles[index]}
        for values in characteristic_rows:
            if len(values) != len(accessions):
                raise ValueError("Inconsistent GSE50081 characteristic row width.")
            value = values[index]
            if ":" in value:
                name, content = value.split(":", 1)
                row[name.strip().lower()] = content.strip()
        rows.append(row)
    return pd.DataFrame(rows).set_index("geo_accession")


def read_expression(path):
    # Pandas skips GEO metadata rows beginning with ! and consumes the quoted
    # ID_REF header plus 54,675 processed probe rows.
    frame = pd.read_csv(
        path, sep="\t", comment="!", index_col=0, compression="gzip",
        low_memory=False,
    )
    frame.index = frame.index.astype(str)
    frame.columns = [str(column) for column in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame.index.has_duplicates or frame.columns.duplicated().any():
        raise ValueError("GSE50081 expression identifiers are not unique.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("GSE50081 expression matrix contains non-finite values.")
    return frame


def aggregate_genes(probe_frame, mapping, features):
    values = {}
    missing = {}
    available = set(probe_frame.index)
    for feature in features:
        probes = [probe for probe in mapping[feature] if probe in available]
        if not probes:
            missing[feature] = mapping[feature]
            continue
        values[feature] = probe_frame.loc[probes].median(axis=0)
    if missing:
        raise ValueError("GSE50081 is missing mapped genes: %s" % ", ".join(sorted(missing)))
    return pd.DataFrame(values, index=probe_frame.columns)[features].astype(float)


def choose_threshold(y_true, probability, negative_label, positive_label):
    unique = np.unique(np.asarray(probability, dtype=float))
    candidates = np.concatenate((
        [0.0],
        (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else unique,
        [1.0],
    ))
    rows = []
    for threshold in np.unique(candidates):
        prediction = np.where(probability >= threshold, positive_label, negative_label)
        rows.append((
            float(balanced_accuracy_score(y_true, prediction)),
            float(threshold),
        ))
    # Prespecified deterministic tie break: closest to the standard 0.5, then
    # the lower threshold.
    score, threshold = sorted(
        rows, key=lambda row: (-row[0], abs(row[1] - 0.5), row[1]))[0]
    return threshold, score


def main():
    started = time.perf_counter()
    for path in (MATRIX_PATH, PROTOCOL_PATH, GSE37745_EXPRESSION, GSE37745_SAMPLES):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    metadata = read_series_metadata(MATRIX_PATH)
    probe_frame = read_expression(MATRIX_PATH)
    if list(probe_frame.columns) != list(metadata.index):
        raise ValueError("GSE50081 expression and metadata sample order differ.")

    histology = metadata["histology"].str.strip().str.lower()
    label_map = {
        "adenocarcinoma": "LUAD",
        "squamous cell carcinoma": "LUSC",
    }
    included = metadata.loc[histology.isin(label_map)].copy()
    included["label"] = histology.loc[included.index].map(label_map)
    excluded_counts = histology.loc[~histology.isin(label_map)].value_counts().to_dict()

    initial_features = read_internal_selected_features()
    mapping, mapping_evidence = probe_mapping(initial_features)
    mapped_features = [feature for feature in initial_features if mapping[feature]]
    external_X = aggregate_genes(probe_frame[included.index], mapping, mapped_features)
    external_y = included["label"].to_numpy()

    X, y, split, training_path = load_classification("binary_classification_example.csv")
    X_train, y_train, _, _ = split_declared(X, y, split)
    final = fit_rank_model(
        X_train[mapped_features], y_train,
        LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=500,
            random_state=RANDOM_STATE,
        ),
    )
    with open(os.path.join(RESULTS_DIR, "external_gse37745.json"),
              "r", encoding="utf-8") as handle:
        prior = json.load(handle)
    selected = [str(mapped_features[index]) for index in
                final.best_estimator_.named_steps["selector"].get_support(indices=True)]
    if final.best_params_ != prior["best_params"] or selected != prior["feature_mapping"]["final_model_selected_features"]:
        raise AssertionError("Reconstructed TCGA model differs from the locked GSE37745 model.")

    gse37745_X = pd.read_csv(GSE37745_EXPRESSION, index_col=0)[mapped_features]
    gse37745_meta = pd.read_csv(GSE37745_SAMPLES, index_col=0)
    gse37745_y = gse37745_meta.loc[gse37745_X.index, "label"].to_numpy()
    positive_index = list(final.classes_).index(final.classes_[-1])
    calibration_probability = final.predict_proba(gse37745_X)[:, positive_index]
    threshold, calibration_balanced_accuracy = choose_threshold(
        gse37745_y, calibration_probability, final.classes_[0], final.classes_[-1])

    default_prediction = final.predict(external_X)
    probability = final.predict_proba(external_X)
    calibrated_prediction = np.where(
        probability[:, positive_index] >= threshold,
        final.classes_[-1], final.classes_[0],
    )
    default_metrics = classification_metrics(
        external_y, default_prediction, scores=probability, classes=final.classes_)
    calibrated_metrics = classification_metrics(
        external_y, calibrated_prediction, scores=probability, classes=final.classes_)

    external_X.to_csv(os.path.join(RESULTS_DIR, "gse50081_selected_expression.csv"))
    included.to_csv(os.path.join(RESULTS_DIR, "gse50081_samples.csv"))
    mapping_evidence.to_csv(os.path.join(RESULTS_DIR, "gse50081_probe_mapping.csv"), index=False)

    result = {
        "external_dataset": {
            "accession": "GSE50081",
            "design": "independent UHN181 Stage I/II NSCLC cohort; GPL570",
            "source_url": SOURCE_URL,
            "source_page": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50081",
            "normalization_from_submitter": "log2 RMA",
            "matrix_sha256": file_sha256(MATRIX_PATH),
            "matrix_bytes": os.path.getsize(MATRIX_PATH),
            "matrix_probe_rows": int(probe_frame.shape[0]),
            "matrix_samples": int(probe_frame.shape[1]),
            "included_samples": int(len(included)),
            "class_counts": included["label"].value_counts().to_dict(),
            "excluded_histology_counts": excluded_counts,
        },
        "protocol": {
            "locked_protocol_path": PROTOCOL_PATH,
            "locked_protocol_sha256": file_sha256(PROTOCOL_PATH),
            "cohort_rule_locked_before_download_analysis": "include only exact adenocarcinoma and squamous cell carcinoma histology",
            "feature_mapping": "same outcome-blind GPL570 mapping used for GSE37745",
            "model_selection": "TCGA training partition only",
            "threshold_calibration": "single balanced-accuracy threshold selected on GSE37745 only",
            "final_validation": "GSE50081 used once; no feature, model, or threshold tuning",
            "positive_class": str(final.classes_[-1]),
        },
        "quality_checks": {
            "gzip_and_matrix_parse_complete": True,
            "unique_probe_identifiers": bool(not probe_frame.index.has_duplicates),
            "unique_sample_identifiers": bool(not probe_frame.columns.duplicated().any()),
            "metadata_expression_order_identical": True,
            "expression_all_finite": True,
            "all_included_labels_complete": bool(included["label"].notna().all()),
            "all_43_locked_input_genes_mapped": bool(external_X.shape[1] == len(mapped_features) == 43),
            "locked_model_reconstruction_identical": True,
            "gse50081_not_used_for_threshold_selection": True,
        },
        "model_reconstruction": {
            "training_path": training_path,
            "mapped_input_features": mapped_features,
            "selected_features": selected,
            "best_params": final.best_params_,
            "matches_gse37745_locked_model": True,
        },
        "gse37745_calibration": {
            "samples": int(len(gse37745_y)),
            "selected_threshold": float(threshold),
            "balanced_accuracy_at_selected_threshold": float(calibration_balanced_accuracy),
        },
        "gse50081_default_threshold": {
            "metrics": default_metrics,
            "bootstrap": bootstrap_classification(
                external_y, default_prediction, probability, final.classes_, repetitions=2000),
            "confusion_matrix": confusion_matrix(
                external_y, default_prediction, labels=final.classes_).tolist(),
        },
        "gse50081_gse37745_calibrated_threshold": {
            "metrics": calibrated_metrics,
            "bootstrap": bootstrap_classification(
                external_y, calibrated_prediction, probability, final.classes_, repetitions=2000),
            "confusion_matrix": confusion_matrix(
                external_y, calibrated_prediction, labels=final.classes_).tolist(),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["quality_checks"]["all_passed"] = bool(all(result["quality_checks"].values()))
    _write_json(OUTPUT_PATH, result)
    print(json.dumps({
        "included": result["external_dataset"]["included_samples"],
        "class_counts": result["external_dataset"]["class_counts"],
        "threshold": threshold,
        "default": default_metrics,
        "calibrated": calibrated_metrics,
        "output": OUTPUT_PATH,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
