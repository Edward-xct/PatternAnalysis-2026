"""Synthetic forward tests for both project model families."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules import (  # noqa: E402
    ConvNeXtTiny,
    build_model,
    convert_torchvision_convnext_state_dict,
    count_trainable_parameters,
    load_model_state_dict,
)


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

    def test_imagenet_variant_uses_own_convnext(self) -> None:
        model = build_model(
            "convnext_tiny_imagenet", initialize_pretrained=False
        ).eval()
        self.assertIsInstance(model, ConvNeXtTiny)
        self.assertEqual(count_trainable_parameters(model), 27821666)
        images = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            logits = model(images)
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_torchvision_weights_reproduce_identical_logits(self) -> None:
        """Converted weights must give the same outputs as torchvision.

        This is the layout of the earlier torchvision-based checkpoint
        (two-class head wrapped in Sequential(Dropout, Linear)). Every tensor
        is randomised first: the default layer scale of 1e-6 would make each
        block almost an identity and hide a wrong mapping.
        """
        from torchvision.models import convnext_tiny

        reference = convnext_tiny(weights=None)
        reference.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 2))
        with torch.no_grad():
            for tensor in reference.state_dict().values():
                if tensor.is_floating_point():
                    tensor.copy_(torch.randn_like(tensor) * 0.1)
        reference.eval()

        own = build_model("convnext_tiny_imagenet", initialize_pretrained=False)
        load_model_state_dict(own, reference.state_dict())
        own.eval()

        images = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            expected = reference(images)
            actual = own(images)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_imagenet_layout_maps_every_backbone_tensor(self) -> None:
        from torchvision.models import convnext_tiny

        source = convnext_tiny(weights=None).state_dict()
        converted, skipped = convert_torchvision_convnext_state_dict(source)
        self.assertEqual(sorted(skipped), ["classifier.2.bias", "classifier.2.weight"])
        own_keys = set(ConvNeXtTiny().state_dict())
        self.assertEqual(
            own_keys - set(converted), {"classifier.1.weight", "classifier.1.bias"}
        )
        self.assertFalse(set(converted) - own_keys)

    def test_models_reject_non_rgb_input(self) -> None:
        images = torch.randn(1, 1, 64, 64)
        for model_name in ("small_cnn", "convnext_tiny"):
            with self.subTest(model=model_name):
                with self.assertRaises(ValueError):
                    build_model(model_name)(images)


if __name__ == "__main__":
    unittest.main()
