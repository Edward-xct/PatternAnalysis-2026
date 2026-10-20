"""Draw the README overview figure: patient pipeline and ConvNeXt-Tiny layout.

The figure is purely schematic. It uses no MRI pixels and no patient data, so
it can be committed. The calibrated thresholds shown are the frozen values
reported in the README (temperature 3.23, NC threshold 0.137, AD threshold 0.5).

Example:
    python scripts/make_pipeline_figure.py --output-dir figures/report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1F2933"
NAVY = "#17365D"
TEAL = "#167D8D"
SAND = "#F4F1EA"
MINT = "#E3F2EF"
SKY = "#E6EEF7"
ROSE = "#F9E6E1"


def box(axis, x, y, w, h, title, body, face, edge=NAVY, title_size=9.5):
    axis.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01",
            facecolor=face, edgecolor=edge, linewidth=1.1,
        )
    )
    axis.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
              fontsize=title_size, fontweight="bold", color=edge)
    axis.text(x + w / 2, y + h - 0.105, body, ha="center", va="top",
              fontsize=7.6, color=INK, linespacing=1.35)


def arrow(axis, start, end, colour=NAVY):
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                        color=colour, linewidth=1.2, shrinkA=2, shrinkB=2)
    )


def draw(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6.4))
    axis.set_xlim(-0.005, 1.005)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Row 1: patient-level decision pipeline.
    axis.text(0.0, 0.985, "a  Patient-level decision pipeline", fontsize=11,
              fontweight="bold", color=INK, va="top")
    row_y, row_h, w, gap = 0.63, 0.29, 0.14, 0.032
    stages = [
        ("Patient", "1–5 MRI visits\n× 20 sagittal\nJPEG slices\n(near midline)", SAND),
        ("Pre-processing", "grayscale → 3 channels\nresize 224 × 224\nImageNet mean/std\n(train: small affine)", SAND),
        ("ConvNeXtTiny", "own implementation\n(modules.py)\nImageNet-initialised\n2 logits per slice", SKY),
        ("Patient aggregation", "mean of all slice\nlogits over every\nscan of the patient\n→ softmax", SKY),
        ("Temperature scaling", "logits / T, T = 3.23\nfitted on validation\nonly; ranking and\naccuracy unchanged", MINT),
        ("Triage rule", "P(AD) ≤ 0.137 → NC\nP(AD) ≥ 0.5 → AD\notherwise → refer\nto a clinician", ROSE),
    ]
    for index, (title, body, face) in enumerate(stages):
        x = index * (w + gap)
        box(axis, x, row_y, w, row_h, title, body, face)
        if index:
            arrow(axis, (x - gap, row_y + row_h / 2), (x, row_y + row_h / 2))

    # Row 2: ConvNeXt-Tiny macro layout.
    axis.text(0.0, 0.53, "b  ConvNeXt-Tiny (27.8M parameters)", fontsize=11,
              fontweight="bold", color=INK, va="top")
    y2, h2 = 0.29, 0.19
    layout = [
        ("Stem", "4×4 conv s4\n+ LayerNorm", 0.085),
        ("Stage 1", "3 blocks\n96 ch, 56×56", 0.085),
        ("Down", "LN + 2×2\nconv, s2", 0.06),
        ("Stage 2", "3 blocks\n192 ch, 28×28", 0.085),
        ("Down", "LN + 2×2\nconv, s2", 0.06),
        ("Stage 3", "9 blocks\n384 ch, 14×14", 0.085),
        ("Down", "LN + 2×2\nconv, s2", 0.06),
        ("Stage 4", "3 blocks\n768 ch, 7×7", 0.085),
        ("Head", "avg pool + LN\nlinear → 2", 0.09),
    ]
    x = 0.0
    previous = None
    for title, body, width in layout:
        face = SKY if title.startswith("Stage") else SAND
        box(axis, x, y2, width, h2, title, body, face, title_size=8.8)
        if previous is not None:
            arrow(axis, (previous, y2 + h2 / 2), (x, y2 + h2 / 2))
        previous = x + width
        x += width + 0.022

    # Row 3: one ConvNeXt block.
    axis.text(0.0, 0.215, "c  One ConvNeXt block (x has C channels)", fontsize=11,
              fontweight="bold", color=INK, va="top")
    y3, h3 = 0.0, 0.15
    block = [
        ("7×7 DW conv", "spatial mixing"),
        ("LayerNorm", "channels-last"),
        ("Linear C → 4C", "expand"),
        ("GELU", ""),
        ("Linear 4C → C", "project"),
        ("Layer scale γ", "init 1e-6"),
        ("DropPath", "p ≤ 0.1"),
    ]
    width, gap3 = 0.092, 0.026
    for index, (title, body) in enumerate(block):
        bx = 0.02 + index * (width + gap3)
        box(axis, bx, y3, width, h3, title, body, MINT, edge=TEAL, title_size=8.2)
        if index:
            arrow(axis, (bx - gap3, y3 + h3 / 2), (bx, y3 + h3 / 2), colour=TEAL)
    end_x = 0.02 + len(block) * (width + gap3) - gap3
    axis.text(end_x + 0.025, y3 + h3 / 2, "+ x", fontsize=12, fontweight="bold",
              color=TEAL, va="center")
    axis.text(end_x + 0.075, y3 + h3 / 2,
              "residual:\noutput = x + DropPath(γ · MLP(LN(DWConv(x))))",
              fontsize=8, color=INK, va="center")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"pipeline_overview.{suffix}", dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(args.output_dir)
    print(f"Wrote: {args.output_dir / 'pipeline_overview.png'}")


if __name__ == "__main__":
    main()
