"""Regression tests for the leakage-free ADNI patient split."""

from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_DIR / "artifacts" / "slice_manifest.csv"
sys.path.insert(0, str(PROJECT_DIR))

from dataset import load_manifest  # noqa: E402


@unittest.skipUnless(MANIFEST_PATH.is_file(), "ADNI manifest is not available")
class PatientSplitTests(unittest.TestCase):
    """Verify the frozen subject-level split and scan completeness."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_manifest(MANIFEST_PATH)

    def test_subjects_are_disjoint_across_splits(self) -> None:
        subjects_by_split: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            subjects_by_split[record.split].add(record.subject_id)

        self.assertTrue(
            subjects_by_split["train"].isdisjoint(
                subjects_by_split["validation"]
            )
        )
        self.assertTrue(
            subjects_by_split["train"].isdisjoint(subjects_by_split["test"])
        )
        self.assertTrue(
            subjects_by_split["validation"].isdisjoint(
                subjects_by_split["test"]
            )
        )
        self.assertEqual(
            {split: len(subjects) for split, subjects in subjects_by_split.items()},
            {"train": 476, "validation": 102, "test": 102},
        )

    def test_every_scan_has_twenty_slices(self) -> None:
        slices_per_scan = Counter(record.scan_id for record in self.records)
        self.assertEqual(len(slices_per_scan), 1526)
        self.assertEqual(set(slices_per_scan.values()), {20})
        self.assertEqual(sum(slices_per_scan.values()), 30520)

    def test_each_subject_has_one_label(self) -> None:
        labels_by_subject: dict[str, set[int]] = defaultdict(set)
        for record in self.records:
            labels_by_subject[record.subject_id].add(record.target)
        conflicts = {
            subject: labels
            for subject, labels in labels_by_subject.items()
            if len(labels) != 1
        }
        self.assertEqual(conflicts, {})
        self.assertEqual(len(labels_by_subject), 680)


if __name__ == "__main__":
    unittest.main()
