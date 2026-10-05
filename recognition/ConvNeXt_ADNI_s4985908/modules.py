"""Neural-network modules for ADNI AD-versus-CN classification.

The module contains both models used by the project: a conventional SmallCNN
baseline and a ConvNeXt-Tiny classifier implemented from the core operations
described by Liu et al. (CVPR 2022). Keeping both architectures behind the
same :func:`build_model` interface makes their data and evaluation pipelines
directly comparable.
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


class LayerNorm2d(nn.LayerNorm):
    """Layer normalization over channels for an ``NCHW`` feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__(channels, eps=eps)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(
                "LayerNorm2d expects an NCHW tensor, "
                f"received shape {tuple(features.shape)}"
            )
        features = features.permute(0, 2, 3, 1)
        features = super().forward(features)
        return features.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Drop complete residual paths independently for each training sample."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep_probability = 1.0 - self.probability
        mask_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_mask = torch.empty(
            mask_shape,
            dtype=inputs.dtype,
            device=inputs.device,
        ).bernoulli_(keep_probability)
        return inputs * random_mask / keep_probability


class ConvNeXtBlock(nn.Module):
    """One ConvNeXt residual block.

    Spatial mixing is performed by a 7x7 depthwise convolution. Channel
    mixing then operates in channels-last format using two linear layers with
    a fourfold hidden expansion. Layer scale and stochastic depth stabilize
    optimization of the deep residual stack.
    """

    def __init__(
        self,
        channels: int,
        drop_path_probability: float = 0.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative")

        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pointwise_expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.pointwise_project = nn.Linear(4 * channels, channels)
        self.layer_scale = nn.Parameter(
            torch.full((channels,), float(layer_scale_init))
        )
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        features = self.depthwise_conv(inputs)
        features = features.permute(0, 2, 3, 1)
        features = self.norm(features)
        features = self.pointwise_expand(features)
        features = self.activation(features)
        features = self.pointwise_project(features)
        features = features * self.layer_scale
        features = features.permute(0, 3, 1, 2)
        return residual + self.drop_path(features)


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny classifier for three-channel 2D MRI slices.

    The canonical Tiny configuration uses stage depths ``(3, 3, 9, 3)`` and
    channel dimensions ``(96, 192, 384, 768)``. The model returns slice-level
    logits; the evaluation pipeline averages those logits within each patient
    before computing the project's clinical metrics.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.30,
        stochastic_depth_probability: float = 0.10,
        layer_scale_init: float = 1e-6,
        stage_depths: tuple[int, ...] = (3, 3, 9, 3),
        stage_channels: tuple[int, ...] = (96, 192, 384, 768),
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= stochastic_depth_probability < 1.0:
            raise ValueError("stochastic_depth_probability must be in [0, 1)")
        if len(stage_depths) != len(stage_channels) or not stage_depths:
            raise ValueError(
                "stage_depths and stage_channels must be non-empty and equal length"
            )
        if any(depth <= 0 for depth in stage_depths):
            raise ValueError("Every stage depth must be positive")
        if any(channels <= 0 for channels in stage_channels):
            raise ValueError("Every stage channel count must be positive")

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stage_channels[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(stage_channels[0]),
        )

        total_blocks = sum(stage_depths)
        drop_path_rates = torch.linspace(
            0.0,
            stochastic_depth_probability,
            total_blocks,
        ).tolist()
        current_block = 0
        stages: list[nn.Module] = []
        downsample_layers: list[nn.Module] = []

        for stage_index, (depth, channels) in enumerate(
            zip(stage_depths, stage_channels)
        ):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        channels=channels,
                        drop_path_probability=drop_path_rates[current_block],
                        layer_scale_init=layer_scale_init,
                    )
                )
                current_block += 1
            stages.append(nn.Sequential(*blocks))

            if stage_index < len(stage_channels) - 1:
                downsample_layers.append(
                    nn.Sequential(
                        LayerNorm2d(channels),
                        nn.Conv2d(
                            channels,
                            stage_channels[stage_index + 1],
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.final_norm = nn.LayerNorm(stage_channels[-1], eps=1e-6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(stage_channels[-1], num_classes),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        if min(images.shape[-2:]) < 32:
            raise ValueError("ConvNeXt-Tiny requires spatial dimensions of at least 32")

    def forward_features(self, images: Tensor) -> Tensor:
        """Return the normalized 768-dimensional slice representation."""
        self._validate_input(images)
        features = self.stem(images)
        for stage_index, stage in enumerate(self.stages):
            features = stage(features)
            if stage_index < len(self.downsample_layers):
                features = self.downsample_layers[stage_index](features)
        features = features.mean(dim=(-2, -1))
        return self.final_norm(features)

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized class logits with shape ``(batch, 2)``."""
        return self.classifier(self.forward_features(images))


def build_model(
    model_name: str,
    num_classes: int = 2,
    dropout: float = 0.30,
) -> nn.Module:
    """Construct a model by stable command-line name."""
    normalized_name = model_name.lower().replace("-", "_")
    if normalized_name == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    if normalized_name == "convnext_tiny":
        return ConvNeXtTiny(num_classes=num_classes, dropout=dropout)
    raise ValueError(
        f"Unknown model {model_name!r}. "
        "Available models: ['small_cnn', 'convnext_tiny']"
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters for resource reporting."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke_test(
    model_name: str,
    batch_size: int,
    image_size: int,
    seed: int,
) -> None:
    """Run one complete optimization step on synthetic data."""
    if batch_size <= 0 or image_size <= 0:
        raise ValueError("batch_size and image_size must be positive")

    torch.manual_seed(seed)
    model = build_model(model_name)
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
        raise FloatingPointError(f"Non-finite gradients detected in {model_name}")
    optimizer.step()

    print(f"model={model_name}")
    print(f"trainable_parameters={count_trainable_parameters(model):,}")
    print(f"input_shape={tuple(images.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"finite_gradients={gradients_are_finite}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test model modules")
    parser.add_argument(
        "--model",
        default="small_cnn",
        choices=("small_cnn", "convnext_tiny"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3710)
    args = parser.parse_args()
    smoke_test(
        model_name=args.model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
