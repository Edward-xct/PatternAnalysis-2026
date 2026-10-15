"""Regression tests for the leakage-free ADNI patient split."""

from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_DIR / "artifacts" / "slice_manifest.csv"
sys.path.insert(0, str(PROJECT_DIR))

from dataset import load_manifest, stratified_subject_split  # noqa: E402


class StratifiedSubjectSplitTests(unittest.TestCase):
    """Check the split function itself on synthetic subjects (no data needed).

    The synthetic cohort has the same class sizes as the audited ADNI data:
    221 AD subjects (label 2) and 459 NC subjects (label 0).
    """

    @staticmethod
    def make_labels() -> dict[str, int]:
        return {
            f"{index:03d}_S_{index:04d}": (2 if index < 221 else 0)
            for index in range(680)
        }

    def test_every_subject_assigned_once_with_expected_sizes(self) -> None:
        labels = self.make_labels()
        assignment = stratified_subject_split(labels, 3710, 0.70, 0.15)
        self.assertEqual(set(assignment), set(labels))
        counts = Counter((split, labels[subject]) for subject, split in assignment.items())
        self.assertEqual(
            dict(counts),
            {
                ("train", 2): 155, ("train", 0): 321,
                ("validation", 2): 33, ("validation", 0): 69,
                ("test", 2): 33, ("test", 0): 69,
            },
        )

    def test_split_is_deterministic_for_a_seed(self) -> None:
        labels = self.make_labels()
        first = stratified_subject_split(labels, 3710, 0.70, 0.15)
        second = stratified_subject_split(dict(reversed(labels.items())), 3710, 0.70, 0.15)
        other_seed = stratified_subject_split(labels, 2026, 0.70, 0.15)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)

    def test_invalid_ratios_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stratified_subject_split(self.make_labels(), 3710, 0.90, 0.15)


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
