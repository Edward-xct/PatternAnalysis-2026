"""Synthetic forward tests for both project model families."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules import build_model, count_trainable_parameters  # noqa: E402


class ModelForwardTests(unittest.TestCase):
    """Protect public model names, output shapes, and parameter counts."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(3710)
        torch.set_num_threads(1)

    def test_small_cnn_forward(self) -> None:
        model = build_model("small_cnn").eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            features = model.forward_features(images)
            logits = model(images)
        self.assertEqual(tuple(features.shape), (2, 256))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(count_trainable_parameters(model), 389410)

    def test_convnext_tiny_forward(self) -> None:
        model = build_model("convnext_tiny").eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            features = model.forward_features(images)
            logits = model(images)
        self.assertEqual(tuple(features.shape), (1, 768))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(count_trainable_parameters(model), 27821666)

    def test_models_reject_non_rgb_input(self) -> None:
        images = torch.randn(1, 1, 64, 64)
        for model_name in ("small_cnn", "convnext_tiny"):
            with self.subTest(model=model_name):
                with self.assertRaises(ValueError):
                    build_model(model_name)(images)


if __name__ == "__main__":
    unittest.main()
