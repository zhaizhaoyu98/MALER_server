"""Outcome-blind patient-level filtering for TCGA-style sample barcodes."""

import re

import numpy as np


def patient_id(sample_id):
    match = re.match(r"^TCGA[-.]([^-\.]+)[-.]([^-\.]+)(?:[-.]|$)", str(sample_id), re.I)
    return "TCGA-%s-%s" % (match.group(1).upper(), match.group(2).upper()) if match else str(sample_id)


def _sample_priority(sample_id):
    pieces = re.split(r"[-.]", str(sample_id))
    tissue = pieces[3][:2] if len(pieces) > 3 else "99"
    vial = pieces[3][2:3] if len(pieces) > 3 else "Z"
    return (0 if tissue == "01" else 1, tissue, vial, str(sample_id))


def patient_partition_audit(sample_ids, split):
    """Exclude cross-partition patients and collapse within-partition aliquots.

    The caller validates labels first. All cross-partition patients are removed
    from both sets, avoiding a favourable choice between development and test.
    Within a partition, a deterministic sample-type/vial rule selects one row
    without using labels or outcomes. The source matrix is not modified.
    """
    groups = {}
    for row, sample_id in enumerate(sample_ids):
        partition = "train" if split[row] in ("train", "training") else "test"
        groups.setdefault(patient_id(sample_id), {}).setdefault(partition, []).append(row)
    kept = []
    cross = []
    collapsed = []
    for patient, partitions in sorted(groups.items()):
        if len(partitions) > 1:
            cross.append({
                "patient": patient,
                "train_samples": [str(sample_ids[row]) for row in partitions["train"]],
                "test_samples": [str(sample_ids[row]) for row in partitions["test"]],
            })
            continue
        rows = next(iter(partitions.values()))
        selected = min(rows, key=lambda row: _sample_priority(sample_ids[row]))
        kept.append(selected)
        if len(rows) > 1:
            collapsed.append({
                "patient": patient,
                "partition": next(iter(partitions)),
                "samples": [str(sample_ids[row]) for row in rows],
                "selected": str(sample_ids[selected]),
            })
    kept.sort()
    return np.asarray(kept, dtype=int), {
        "rule": "Exclude cross-partition patients from both sets; otherwise keep one aliquot per patient, preferring sample type 01 and vial A without using outcomes.",
        "source_samples": int(len(sample_ids)),
        "retained_patients": int(len(kept)),
        "excluded_cross_partition_patients": cross,
        "collapsed_within_partition_patients": collapsed,
    }
