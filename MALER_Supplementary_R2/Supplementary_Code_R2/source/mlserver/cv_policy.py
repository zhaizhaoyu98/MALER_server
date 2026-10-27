"""Auditable sample- and subject-level CV split policy."""

from __future__ import absolute_import

import hashlib
import json

import numpy as np
from sklearn.model_selection import (
    GroupKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold,
    StratifiedGroupKFold, StratifiedKFold,
)


def split_index_hash(splits):
    payload = [
        [np.asarray(train, dtype=int).tolist(), np.asarray(test, dtype=int).tolist()]
        for train, test in splits
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def make_splits(X, y, task, n_splits, repeats=1, seed=10, groups=None, inner=False):
    """Return verified disjoint splits and the number of distinct repeat blocks.

    GroupKFold has no randomized repeat policy, so grouped regression has one
    repeat. StratifiedGroupKFold attempts independently seeded repeats; exact
    duplicate blocks are omitted instead of being counted as new evidence.
    """
    n_splits, repeats = int(n_splits), int(repeats)
    if n_splits < 2 or repeats < 1:
        raise ValueError("CV requires at least two folds and one repeat.")
    stratify = y["event"].astype(int) if task == "survival" else y
    if groups is None:
        if task in ("classification", "survival"):
            cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                  if inner else RepeatedStratifiedKFold(
                      n_splits=n_splits, n_repeats=repeats, random_state=seed))
        else:
            cv = (KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                  if inner else RepeatedKFold(
                      n_splits=n_splits, n_repeats=repeats, random_state=seed))
        splits = list(cv.split(X, stratify))
        return splits, 1 if inner else repeats

    groups = np.asarray(groups)
    if len(groups) != len(X) or len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough distinct subject IDs for the requested CV folds.")
    if task == "regression":
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        blocks = 1
    else:
        splits, seen = [], set()
        for repeat in range(1 if inner else repeats):
            cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + repeat)
            block = list(cv.split(X, stratify, groups))
            fingerprint = split_index_hash(block)
            if fingerprint not in seen:
                seen.add(fingerprint)
                splits.extend(block)
        blocks = len(seen)
    for train, test in splits:
        if set(groups[train]).intersection(groups[test]):
            raise ValueError("A subject ID crossed a CV fold boundary.")
        if task in ("classification", "survival"):
            train_classes = np.unique(np.asarray(stratify)[train])
            test_classes = np.unique(np.asarray(stratify)[test])
            if len(train_classes) < 2 or len(test_classes) < 2:
                raise ValueError(
                    "A grouped fold lacks a class or event/censor category; use fewer folds or more groups.")
    return splits, blocks
def validate_group_feature_method(groups, feature_method):
    """Reject selectors whose additional internal CV does not accept groups."""
    if groups is not None and str(feature_method).lower() in (
            "fss", "forward", "forward-selection", "bss", "backward", "backward-selection"):
        raise ValueError(
            "Grouped analyses do not support FSS/BSS in this release because their "
            "internal selection folds are not group-aware. Use TopK, MRMR, PCA or no reduction.")

