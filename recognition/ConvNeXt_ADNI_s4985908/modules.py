"""Neural-network modules for ADNI AD-versus-CN classification.

This file initially provides the simple convolutional baseline required for
comparison with ConvNeXt. The baseline intentionally uses conventional
convolution, batch normalization, ReLU, and max pooling so that any benefit
from the advanced architecture can be evaluated under the same data split.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn


class ConvNormActivation(nn.Sequential):
    """A conventional convolution, batch-normalization, and ReLU block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SmallCNN(nn.Module):
    """Small four-stage CNN used as the project baseline.

    The network receives three-channel 2D MRI slices and returns two logits:
    target 0 is cognitively normal (CN), and target 1 is Alzheimer's disease
    (AD). Global average pooling keeps the classifier compact and permits
    different square input sizes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        channels = (32, 64, 128, 256)
        stages: list[nn.Module] = []
        current_channels = in_channels
        for output_channels in channels:
            stages.extend(
                [
                    ConvNormActivation(current_channels, output_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_channels = output_channels

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Apply explicit, reproducible initialization to trainable layers."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        """Return pooled slice embeddings before the classification head."""
        self._validate_input(images)
        features = self.features(images)
        return self.pool(features).flatten(start_dim=1)

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized class logits with shape ``(batch, 2)``."""
        self._validate_input(images)
        features = self.features(images)
        features = self.pool(features)
        return self.classifier(features)

    @staticmethod
    def _validate_input(images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"Expected input shape (N, C, H, W), received {tuple(images.shape)}"
            )
        if images.shape[1] != 3:
            raise ValueError(
                f"Expected three input channels, received {images.shape[1]}"
            )


def build_model(
    model_name: str,
    num_classes: int = 2,
    dropout: float = 0.30,
) -> nn.Module:
    """Construct a model by stable command-line name."""
    normalized_name = model_name.lower().replace("-", "_")
    if normalized_name == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(
        f"Unknown model {model_name!r}. Available models: ['small_cnn']"
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters for resource reporting."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke_test(batch_size: int, image_size: int, seed: int) -> None:
    """Run one complete optimization step on synthetic data."""
    if batch_size <= 0 or image_size <= 0:
        raise ValueError("batch_size and image_size must be positive")

    torch.manual_seed(seed)
    model = build_model("small_cnn")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    images = torch.randn(batch_size, 3, image_size, image_size)
    targets = torch.arange(batch_size, dtype=torch.long) % 2

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()

    gradients_are_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not gradients_are_finite:
        raise FloatingPointError("Non-finite gradients detected in SmallCNN")
    optimizer.step()

    print("model=small_cnn")
    print(f"trainable_parameters={count_trainable_parameters(model):,}")
    print(f"input_shape={tuple(images.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"finite_gradients={gradients_are_finite}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test model modules")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3710)
    args = parser.parse_args()
    smoke_test(
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
