"""Reproducible quality audit for the real GSE37745 validation cohort."""

from __future__ import absolute_import

import hashlib
import json
import os
import zipfile

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "external_data")
RESULTS = os.path.join(HERE, "results")
ARCHIVE = os.path.join(DATA, "E-GEOD-37745.processed.1.zip")
SDRF = os.path.join(DATA, "E-GEOD-37745.sdrf.txt")
EXPRESSION = os.path.join(RESULTS, "gse37745_selected_expression.csv")
SAMPLES = os.path.join(RESULTS, "gse37745_samples.csv")
EXTERNAL_RESULT = os.path.join(RESULTS, "external_gse37745.json")
OUTPUT = os.path.join(RESULTS, "external_data_quality_audit.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    sdrf = pd.read_csv(SDRF, sep="\t", dtype=str, low_memory=False)
    expression = pd.read_csv(EXPRESSION, index_col=0)
    samples = pd.read_csv(SAMPLES, index_col=0, dtype=str)
    with open(EXTERNAL_RESULT, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    derived_column = "Derived Array Data File"
    source_column = "Source Name"
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.namelist()
        bad_member = archive.testzip()
        archive_basenames = set(os.path.basename(item) for item in members)
        derived = set(sdrf[derived_column].dropna())
        missing_derived = sorted(derived - archive_basenames)
        uncompressed_bytes = sum(item.file_size for item in archive.infolist())

    required_labels = {"LUAD", "LUSC"}
    label_counts = samples["label"].value_counts().to_dict()
    matrix_index = set(expression.index.astype(str))
    sample_index = set(samples.index.astype(str))
    finite = bool(np.isfinite(expression.to_numpy(dtype=float)).all())
    final_features = result["feature_mapping"]["final_model_selected_features"]
    mapped_features = result["feature_mapping"]["mapped_features"]
    starting_features = result["feature_mapping"]["starting_internal_selected_features"]

    checks = [
        {
            "check": "archive_integrity",
            "passed": bad_member is None,
            "evidence": "ZIP CRC %s; %d members; %d uncompressed bytes" % (
                "passed" if bad_member is None else "failed at " + bad_member,
                len(members), uncompressed_bytes),
        },
        {
            "check": "archive_sha256",
            "passed": sha256(ARCHIVE) == "3B331E2A531B314B503D0E65D6D2F2259AD32FDE1F40C0F3271998DC38753B27",
            "evidence": sha256(ARCHIVE),
        },
        {
            "check": "sdrf_grain",
            "passed": len(sdrf) == 196 and sdrf[source_column].is_unique,
            "evidence": "%d SDRF rows; %d unique source names" % (
                len(sdrf), sdrf[source_column].nunique()),
        },
        {
            "check": "sdrf_archive_coverage",
            "passed": not missing_derived and len(derived) == 196,
            "evidence": "%d unique derived files; %d missing from archive" % (
                len(derived), len(missing_derived)),
        },
        {
            "check": "analysis_matrix_grain",
            "passed": expression.index.is_unique and expression.columns.is_unique,
            "evidence": "%d samples x %d unique features" % expression.shape,
        },
        {
            "check": "analysis_matrix_completeness_validity",
            "passed": finite and int(expression.isna().sum().sum()) == 0,
            "evidence": "%d missing cells; all finite=%s" % (
                int(expression.isna().sum().sum()), finite),
        },
        {
            "check": "sample_metadata_alignment",
            "passed": matrix_index == sample_index and samples.index.is_unique,
            "evidence": "%d expression IDs; %d metadata IDs; exact match=%s" % (
                len(matrix_index), len(sample_index), matrix_index == sample_index),
        },
        {
            "check": "target_domain",
            "passed": set(label_counts) == required_labels and label_counts == {"LUAD": 106, "LUSC": 66},
            "evidence": json.dumps(label_counts, sort_keys=True),
        },
        {
            "check": "platform_mapping_coverage",
            "passed": len(mapped_features) >= 20 and set(mapped_features) == set(expression.columns),
            "evidence": "%d of %d internal candidates mapped to GPL570 (%.1f%%)" % (
                len(mapped_features), starting_features,
                100.0 * len(mapped_features) / starting_features),
        },
        {
            "check": "locked_feature_availability",
            "passed": len(final_features) == 20 and set(final_features).issubset(expression.columns),
            "evidence": "%d locked features; %d unavailable" % (
                len(final_features), len(set(final_features) - set(expression.columns))),
        },
    ]
    audit = {
        "dataset": "GSE37745 / E-GEOD-37745",
        "intended_grain": "one tumor array per included patient/sample",
        "intended_use": "retrospective locked LUAD-versus-LUSC external validation",
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "limitations": [
            "Submitter-processed log2 RMA expression may retain study/platform effects.",
            "The audit verifies file/sample/feature integrity, not clinical representativeness.",
            "Threshold transport remains weak despite strong ROC-AUC.",
        ],
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
