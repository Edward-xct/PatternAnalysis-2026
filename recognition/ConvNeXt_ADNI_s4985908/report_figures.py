"""Create public, aggregate figures for the final ADNI report.

The script consumes training histories, aggregate evaluation JSON files, and
aggregate calibration tables. It never loads MRI pixels or patient identifiers,
so the generated figures are suitable for the public repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_ORDER = ("smallcnn", "scratch", "pretrained")
MODEL_LABELS = {
    "smallcnn": "SmallCNN",
    "scratch": "ConvNeXt (scratch)",
    "pretrained": "ConvNeXt (ImageNet)",
}
MODEL_COLOURS = {
    "smallcnn": "#0072B2",
    "scratch": "#D55E00",
    "pretrained": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregate figures for the final ADNI report"
    )
    for key in MODEL_ORDER:
        parser.add_argument(f"--{key}-history", type=Path, required=True)
        parser.add_argument(f"--{key}-evaluation", type=Path, required=True)
        parser.add_argument(f"--{key}-result", type=Path, required=True)
        parser.add_argument(f"--{key}-inference", type=Path, required=True)
    parser.add_argument("--pretrained-reliability", type=Path, required=True)
    parser.add_argument("--pretrained-risk-coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_history(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Training history must be a non-empty list: {path}")
    for index, row in enumerate(payload):
        required = {"epoch", "train", "validation"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Malformed history row {index} in {path}")
    return payload


def validate_evaluation(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or not {"validation", "test"} <= payload.keys():
        raise ValueError(f"Evaluation must contain validation and test results: {path}")
    for split in ("validation", "test"):
        for section in ("raw", "calibrated", "selective"):
            if section not in payload[split]:
                raise ValueError(f"Missing {split}.{section} in {path}")
    return payload


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def history_series(
    history: list[dict[str, Any]], section: str, metric: str
) -> list[float]:
    return [
        float("nan") if row[section].get(metric) is None else float(row[section][metric])
        for row in history
    ]


def plot_training_curves(
    histories: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    panels = (
        ("train", "loss", "Training loss", "Cross-entropy loss"),
        ("validation", "slice_loss", "Validation slice loss", "Cross-entropy loss"),
        ("validation", "macro_f1", "Patient-level validation macro-F1", "Macro-F1"),
        ("validation", "auroc", "Patient-level validation AUROC", "AUROC"),
    )
    for axis, (section, metric, title, ylabel) in zip(axes.flat, panels):
        for key in MODEL_ORDER:
            history = histories[key]
            axis.plot(
                [int(row["epoch"]) for row in history],
                history_series(history, section, metric),
                label=MODEL_LABELS[key],
                color=MODEL_COLOURS[key],
                linewidth=1.8,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Training and validation learning curves", y=1.02)
    save_figure(figure, output_dir, "training_curves")


def plot_test_comparison(
    evaluations: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    metrics = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("auroc", "AUROC"),
        ("recall_AD", "AD recall"),
        ("recall_NC", "NC recall"),
    )
    positions = list(range(len(metrics)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5.1))
    for model_index, key in enumerate(MODEL_ORDER):
        result = evaluations[key]["test"]["calibrated"]
        values = [float(result[name]) for name, _ in metrics]
        offsets = [position + (model_index - 1) * width for position in positions]
        bars = axis.bar(
            offsets,
            values,
            width,
            label=MODEL_LABELS[key],
            color=MODEL_COLOURS[key],
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=3,
            fontsize=8,
        )
    axis.axhline(0.80, color="#555555", linestyle="--", linewidth=1, label="0.80 target")
    axis.set_xticks(positions, [label for _, label in metrics])
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel("Patient-level score")
    axis.set_title("Held-out test performance", pad=30)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
    )
    save_figure(figure, output_dir, "test_metric_comparison")


def plot_confusion_matrices(
    evaluations: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    maximum = max(
        int(evaluations[key]["test"]["calibrated"][metric])
        for key in MODEL_ORDER
        for metric in ("tn", "fp", "fn", "tp")
    )
    for axis, key in zip(axes, MODEL_ORDER):
        result = evaluations[key]["test"]["calibrated"]
        matrix = [
            [int(result["tn"]), int(result["fp"])],
            [int(result["fn"]), int(result["tp"])],
        ]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(2):
            row_total = sum(matrix[row])
            for column in range(2):
                value = matrix[row][column]
                percentage = value / row_total if row_total else 0.0
                colour = "white" if value > maximum * 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{value}\n({percentage:.1%})",
                    ha="center",
                    va="center",
                    color=colour,
                    fontweight="bold",
                )
        axis.set_xticks((0, 1), ("NC", "AD"))
        axis.set_yticks((0, 1), ("NC", "AD"))
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(MODEL_LABELS[key])
    figure.suptitle("Held-out test confusion matrices", y=1.03)
    save_figure(figure, output_dir, "test_confusion_matrices")


def plot_generalisation_gap(
    evaluations: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    """Paired validation/test points so the size of the drop is visible.

    Bars that start at zero would hide a 0.05-0.10 drop, while truncated bars
    would exaggerate it; paired points on a shared, labelled axis avoid both.
    """
    metrics = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("auroc", "AUROC"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
    for axis, key in zip(axes, MODEL_ORDER):
        validation = evaluations[key]["validation"]["calibrated"]
        test = evaluations[key]["test"]["calibrated"]
        for position, (name, _) in enumerate(metrics):
            start_value, end_value = float(validation[name]), float(test[name])
            axis.plot(
                (position, position), (start_value, end_value),
                color="#9AA5B1", linewidth=2, zorder=1,
            )
            axis.scatter(position, start_value, color="#56B4E9", s=55, zorder=3,
                         label="Validation" if position == 0 else None)
            axis.scatter(position, end_value, color="#E69F00", s=55, zorder=3,
                         label="Test" if position == 0 else None)
            axis.annotate(f"{start_value:.3f}", (position, start_value), xytext=(9, 0),
                          textcoords="offset points", va="center", fontsize=8)
            axis.annotate(f"{end_value:.3f}", (position, end_value), xytext=(9, 0),
                          textcoords="offset points", va="center", fontsize=8)
        axis.set_xticks(range(len(metrics)), [label for _, label in metrics])
        axis.set_xlim(-0.4, len(metrics) - 0.4)
        axis.set_ylim(0.6, 1.0)
        axis.axhline(0.80, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title(MODEL_LABELS[key])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Patient-level score (axis starts at 0.6)")
    axes[0].legend(frameon=False, loc="lower left")
    figure.suptitle("Validation-to-test generalisation gap", y=1.03)
    save_figure(figure, output_dir, "validation_test_gap")


def plot_resource_profile(
    results: dict[str, dict[str, Any]],
    inference: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    panels = (
        (
            [inference[key]["parameters"] / 1e6 for key in MODEL_ORDER],
            "Parameters (millions)",
        ),
        (
            [results[key]["peak_gpu_vram_gb"] for key in MODEL_ORDER],
            "Peak training VRAM (GB)",
        ),
        (
            [inference[key]["milliseconds_per_slice"] for key in MODEL_ORDER],
            "Inference latency (ms/slice)",
        ),
    )
    labels = [MODEL_LABELS[key].replace("ConvNeXt ", "") for key in MODEL_ORDER]
    colours = [MODEL_COLOURS[key] for key in MODEL_ORDER]
    for axis, (values, title) in zip(axes, panels):
        bars = axis.bar(labels, values, color=colours)
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            padding=3,
            fontsize=8,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, max(values) * 1.18)
    figure.suptitle("Measured computational profile on Rangpur", y=1.03)
    save_figure(figure, output_dir, "resource_profile")


def plot_reliability(rows: list[dict[str, str]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    for variant, colour in (("raw", "#D55E00"), ("calibrated", "#0072B2")):
        selected = [
            row
            for row in rows
            if row["variant"] == variant and row["count"] != "0"
        ]
        axis.plot(
            [float(row["mean_confidence"]) for row in selected],
            [float(row["accuracy"]) for row in selected],
            marker="o",
            linewidth=1.8,
            color=colour,
            label=variant.capitalize(),
        )
    axis.plot(
        (0, 1),
        (0, 1),
        linestyle="--",
        color="#555555",
        label="Perfect calibration",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title("ImageNet ConvNeXt test reliability")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "imagenet_test_reliability")


def plot_risk_coverage(rows: list[dict[str, str]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    for variant, colour in (("raw", "#D55E00"), ("calibrated", "#0072B2")):
        selected = [row for row in rows if row["variant"] == variant]
        axis.plot(
            [float(row["coverage"]) for row in selected],
            [float(row["selective_risk"]) for row in selected],
            linewidth=1.8,
            color=colour,
            label=variant.capitalize(),
        )
    axis.scatter(
        [85 / 102],
        [1 - 73 / 85],
        color="#009E73",
        zorder=4,
        label="Frozen reject rule",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk (1 - accuracy)")
    axis.set_title("ImageNet ConvNeXt test risk-coverage")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "imagenet_test_risk_coverage")


def metric_row(
    key: str,
    evaluation: dict[str, Any],
    result: dict[str, Any],
    inference: dict[str, Any],
) -> dict[str, Any]:
    validation = evaluation["validation"]["calibrated"]
    test_raw = evaluation["test"]["raw"]
    test = evaluation["test"]["calibrated"]
    selective = evaluation["test"]["selective"]
    return {
        "model": MODEL_LABELS[key],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_auroc": validation["auroc"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_auroc": test["auroc"],
        "test_precision_AD": test["precision_AD"],
        "test_recall_AD": test["recall_AD"],
        "test_precision_NC": test["precision_NC"],
        "test_recall_NC": test["recall_NC"],
        "test_ece_raw": test_raw["ece"],
        "test_ece_calibrated": test["ece"],
        "test_brier_calibrated": test["brier"],
        "test_nll_calibrated": test["nll"],
        "test_selective_accuracy": selective["selective_accuracy"],
        "test_coverage": selective["coverage"],
        "test_referral_rate": selective["referral_rate"],
        "temperature": evaluation["calibration"]["temperature"],
        "threshold_NC": evaluation["threshold_selection"]["threshold_CN"],
        "threshold_AD": evaluation["threshold_selection"]["threshold_AD"],
        "parameters": inference["parameters"],
        "peak_training_vram_gb": result["peak_gpu_vram_gb"],
        "inference_ms_per_slice": inference["milliseconds_per_slice"],
        "best_epoch": inference["checkpoint_epoch"],
    }


def write_summary_tables(
    evaluations: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    inference: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    rows = [
        metric_row(key, evaluations[key], results[key], inference[key])
        for key in MODEL_ORDER
    ]
    with (output_dir / "final_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    headers = (
        "Model",
        "Accuracy",
        "Macro-F1",
        "AUROC",
        "AD recall",
        "NC recall",
        "ECE",
        "Latency",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["model"],
                    f"{row['test_accuracy']:.4f}",
                    f"{row['test_macro_f1']:.4f}",
                    f"{row['test_auroc']:.4f}",
                    f"{row['test_recall_AD']:.4f}",
                    f"{row['test_recall_NC']:.4f}",
                    f"{row['test_ece_calibrated']:.4f}",
                    f"{row['inference_ms_per_slice']:.4f} ms",
                )
            )
            + " |"
        )
    (output_dir / "final_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (output_dir / "report_results_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    histories = {
        key: validate_history(
            load_json(getattr(args, f"{key}_history")),
            getattr(args, f"{key}_history"),
        )
        for key in MODEL_ORDER
    }
    evaluations = {
        key: validate_evaluation(
            load_json(getattr(args, f"{key}_evaluation")),
            getattr(args, f"{key}_evaluation"),
        )
        for key in MODEL_ORDER
    }
    results = {
        key: load_json(getattr(args, f"{key}_result")) for key in MODEL_ORDER
    }
    inference = {
        key: load_json(getattr(args, f"{key}_inference")) for key in MODEL_ORDER
    }
    plot_training_curves(histories, args.output_dir)
    plot_test_comparison(evaluations, args.output_dir)
    plot_confusion_matrices(evaluations, args.output_dir)
    plot_generalisation_gap(evaluations, args.output_dir)
    plot_resource_profile(results, inference, args.output_dir)
    plot_reliability(load_csv(args.pretrained_reliability), args.output_dir)
    plot_risk_coverage(load_csv(args.pretrained_risk_coverage), args.output_dir)
    write_summary_tables(evaluations, results, inference, args.output_dir)
    generated = sorted(
        path.name for path in args.output_dir.iterdir() if path.is_file()
    )
    print(json.dumps({"output_dir": str(args.output_dir), "files": generated}, indent=2))


if __name__ == "__main__":
    main()
