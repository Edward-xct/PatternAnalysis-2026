"""Shape and metadata checks for the real ADNI validation loader."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_DIR / "artifacts" / "slice_manifest.csv"
sys.path.insert(0, str(PROJECT_DIR))

from dataset import ADNISliceDataset, create_dataloader  # noqa: E402


@unittest.skipUnless(MANIFEST_PATH.is_file(), "ADNI manifest is not available")
class DatasetShapeTests(unittest.TestCase):
    """Check deterministic evaluation transforms and batch metadata."""

    def test_validation_sample_shape_and_type(self) -> None:
        dataset = ADNISliceDataset(
            manifest_path=MANIFEST_PATH,
            split="validation",
            image_size=224,
            verify_paths=True,
        )
        sample = dataset[0]
        self.assertEqual(len(dataset), 4560)
        self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample["image"]).all().item())
        self.assertIn(sample["target"], (0, 1))
        self.assertTrue(sample["subject_id"])
        self.assertTrue(sample["scan_id"])
        self.assertTrue(Path(sample["image_path"]).is_file())

    def test_validation_batch_shape(self) -> None:
        _, loader = create_dataloader(
            manifest_path=MANIFEST_PATH,
            split="validation",
            batch_size=4,
            image_size=224,
            num_workers=0,
            seed=3710,
            balanced_training=False,
        )
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["image"].shape), (4, 3, 224, 224))
        self.assertEqual(tuple(batch["target"].shape), (4,))
        self.assertEqual(len(batch["subject_id"]), 4)
        self.assertEqual(len(batch["scan_id"]), 4)


if __name__ == "__main__":
    unittest.main()
