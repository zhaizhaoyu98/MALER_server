"""Regression tests for the outcome-blind TCGA patient-level filter."""

import unittest

import numpy as np
import pandas as pd

from validation.run_reviewer_validation import _patient_id, split_declared


class PatientPartitionTests(unittest.TestCase):
    def test_hyphen_and_dot_barcodes_share_patient(self):
        self.assertEqual(_patient_id("TCGA-DV-A4W0-01A"), "TCGA-DV-A4W0")
        self.assertEqual(_patient_id("TCGA.DV.A4W0.05A"), "TCGA-DV-A4W0")

    def test_cross_partition_patient_is_excluded_from_both_sets(self):
        frame = pd.DataFrame(
            {"g": [1, 2, 3, 4]},
            index=["TCGA-AA-0001-01A", "TCGA-AA-0001-01B",
                   "TCGA-BB-0002-01A", "TCGA-CC-0003-01A"],
        )
        y = np.asarray([0, 0, 1, 0])
        split = np.asarray(["train", "test", "train", "test"])
        train, train_y, test, test_y, audit = split_declared(
            frame, y, split, return_audit=True)
        self.assertEqual(list(train.index), ["TCGA-BB-0002-01A"])
        self.assertEqual(list(test.index), ["TCGA-CC-0003-01A"])
        self.assertEqual(train_y.tolist(), [1])
        self.assertEqual(test_y.tolist(), [0])
        self.assertEqual(len(audit["excluded_cross_partition_patients"]), 1)

    def test_within_partition_priority_ignores_outcomes(self):
        frame = pd.DataFrame(
            {"g": [1, 2, 3, 4]},
            index=["TCGA-AA-0001-05A", "TCGA-AA-0001-01B",
                   "TCGA-AA-0001-01A", "TCGA-BB-0002-01A"],
        )
        y = np.asarray([1, 0, 1, 0])
        train, train_y, test, _, audit = split_declared(
            frame, y, np.asarray(["train", "train", "train", "test"]),
            return_audit=True)
        self.assertEqual(list(train.index), ["TCGA-AA-0001-01A"])
        self.assertEqual(train_y.tolist(), [1])
        self.assertEqual(len(test), 1)
        self.assertEqual(len(audit["collapsed_within_partition_patients"]), 1)


if __name__ == "__main__":
    unittest.main()
