"""Create publication-ready aggregate figures for the ADNI experiment.

This script reads training histories and final evaluation JSON files. It never
loads MRI images or patient-level prediction rows, so every generated artifact
is safe to include in the public repository and the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_LABELS = {
    "smallcnn": "SmallCNN",
    "convnext": "ConvNeXt-Tiny",
}
MODEL_COLOURS = {
    "smallcnn": "#0072B2",
    "convnext": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregate report figures from frozen ADNI results"
    )
    parser.add_argument("--smallcnn-history", type=Path, required=True)
    parser.add_argument("--convnext-history", type=Path, required=True)
    parser.add_argument("--smallcnn-evaluation", type=Path, required=True)
    parser.add_argument("--convnext-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_history(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Training history must be a non-empty list: {path}")
    required = {"epoch", "train", "validation"}
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Malformed history row {index} in {path}")
    return payload


def validate_evaluation(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation must be a JSON object: {path}")
    if "validation" not in payload or "test" not in payload:
        raise ValueError(f"Evaluation must contain validation and test: {path}")
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
    values = []
    for row in history:
        value = row[section].get(metric)
        values.append(float("nan") if value is None else float(value))
    return values


def plot_training_curves(
    histories: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panels = (
        ("train", "loss", "Training loss", "Cross-entropy loss"),
        ("validation", "slice_loss", "Validation slice loss", "Cross-entropy loss"),
        ("validation", "macro_f1", "Patient-level validation macro-F1", "Macro-F1"),
        ("validation", "auroc", "Patient-level validation AUROC", "AUROC"),
    )

    for axis, (section, metric, title, ylabel) in zip(axes.flat, panels):
        for model_key, history in histories.items():
            epochs = [int(row["epoch"]) for row in history]
            axis.plot(
                epochs,
                history_series(history, section, metric),
                label=MODEL_LABELS[model_key],
                color=MODEL_COLOURS[model_key],
                linewidth=1.8,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
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
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.8))

    for model_index, model_key in enumerate(("smallcnn", "convnext")):
        result = evaluations[model_key]["test"]["calibrated"]
        offsets = [position + (model_index - 0.5) * width for position in positions]
        values = [float(result[key]) for key, _ in metrics]
        bars = axis.bar(
            offsets,
            values,
            width,
            label=MODEL_LABELS[model_key],
            color=MODEL_COLOURS[model_key],
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)

    axis.axhline(0.80, color="#555555", linestyle="--", linewidth=1, label="0.80 target")
    axis.set_xticks(positions, [label for _, label in metrics])
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Patient-level score")
    axis.set_title("Held-out test performance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="lower center")
    save_figure(figure, output_dir, "test_metric_comparison")


def plot_confusion_matrices(
    evaluations: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    maximum = max(
        int(evaluations[model]["test"]["calibrated"][key])
        for model in evaluations
        for key in ("tn", "fp", "fn", "tp")
    )

    for axis, model_key in zip(axes, ("smallcnn", "convnext")):
        result = evaluations[model_key]["test"]["calibrated"]
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
        axis.set_title(MODEL_LABELS[model_key])

    figure.suptitle("Held-out test confusion matrices", y=1.03)
    save_figure(figure, output_dir, "test_confusion_matrices")


def plot_generalisation_gap(
    evaluations: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    metrics = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("auroc", "AUROC"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)

    for axis, model_key in zip(axes, ("smallcnn", "convnext")):
        positions = list(range(len(metrics)))
        width = 0.36
        validation = evaluations[model_key]["validation"]["calibrated"]
        test = evaluations[model_key]["test"]["calibrated"]
        validation_values = [float(validation[key]) for key, _ in metrics]
        test_values = [float(test[key]) for key, _ in metrics]
        bars_validation = axis.bar(
            [position - width / 2 for position in positions],
            validation_values,
            width,
            label="Validation",
            color="#56B4E9",
        )
        bars_test = axis.bar(
            [position + width / 2 for position in positions],
            test_values,
            width,
            label="Test",
            color="#E69F00",
        )
        axis.bar_label(
            bars_validation,
            labels=[f"{value:.3f}" for value in validation_values],
            padding=2,
            fontsize=8,
        )
        axis.bar_label(
            bars_test,
            labels=[f"{value:.3f}" for value in test_values],
            padding=2,
            fontsize=8,
        )
        axis.set_xticks(positions, [label for _, label in metrics])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(MODEL_LABELS[model_key])
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Patient-level score")
    axes[0].legend(frameon=False, loc="lower center")
    figure.suptitle("Validation-to-test generalisation gap", y=1.03)
    save_figure(figure, output_dir, "validation_test_gap")


def metric_row(model_key: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    test_raw = evaluation["test"]["raw"]
    test_calibrated = evaluation["test"]["calibrated"]
    test_selective = evaluation["test"]["selective"]
    validation = evaluation["validation"]["calibrated"]
    return {
        "model": MODEL_LABELS[model_key],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_auroc": validation["auroc"],
        "test_accuracy": test_calibrated["accuracy"],
        "test_macro_f1": test_calibrated["macro_f1"],
        "test_auroc": test_calibrated["auroc"],
        "test_precision_AD": test_calibrated["precision_AD"],
        "test_recall_AD": test_calibrated["recall_AD"],
        "test_precision_NC": test_calibrated["precision_NC"],
        "test_recall_NC": test_calibrated["recall_NC"],
        "test_ece_raw": test_raw["ece"],
        "test_ece_calibrated": test_calibrated["ece"],
        "test_brier_calibrated": test_calibrated["brier"],
        "test_nll_calibrated": test_calibrated["nll"],
        "test_selective_accuracy": test_selective["selective_accuracy"],
        "test_coverage": test_selective["coverage"],
        "test_referral_rate": test_selective["referral_rate"],
        "temperature": evaluation["calibration"]["temperature"],
        "threshold_NC": evaluation["threshold_selection"]["threshold_CN"],
        "threshold_AD": evaluation["threshold_selection"]["threshold_AD"],
    }


def write_summary_tables(
    evaluations: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    rows = [metric_row(key, evaluations[key]) for key in ("smallcnn", "convnext")]
    with (output_dir / "final_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    headers = (
        "Model",
        "Accuracy",
        "Macro-F1",
        "AUROC",
        "AD precision",
        "AD recall",
        "NC precision",
        "NC recall",
        "ECE",
        "Selective accuracy",
        "Coverage",
    )
    markdown_rows = []
    for row in rows:
        markdown_rows.append(
            (
                row["model"],
                row["test_accuracy"],
                row["test_macro_f1"],
                row["test_auroc"],
                row["test_precision_AD"],
                row["test_recall_AD"],
                row["test_precision_NC"],
                row["test_recall_NC"],
                row["test_ece_calibrated"],
                row["test_selective_accuracy"],
                row["test_coverage"],
            )
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in markdown_rows:
        lines.append(
            "| "
            + " | ".join([str(row[0])] + [f"{float(value):.4f}" for value in row[1:]])
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
        "smallcnn": validate_history(
            load_json(args.smallcnn_history), args.smallcnn_history
        ),
        "convnext": validate_history(
            load_json(args.convnext_history), args.convnext_history
        ),
    }
    evaluations = {
        "smallcnn": validate_evaluation(
            load_json(args.smallcnn_evaluation), args.smallcnn_evaluation
        ),
        "convnext": validate_evaluation(
            load_json(args.convnext_evaluation), args.convnext_evaluation
        ),
    }

    plot_training_curves(histories, args.output_dir)
    plot_test_comparison(evaluations, args.output_dir)
    plot_confusion_matrices(evaluations, args.output_dir)
    plot_generalisation_gap(evaluations, args.output_dir)
    write_summary_tables(evaluations, args.output_dir)

    generated = sorted(path.name for path in args.output_dir.iterdir() if path.is_file())
    print(json.dumps({"output_dir": str(args.output_dir), "files": generated}, indent=2))


if __name__ == "__main__":
    main()
