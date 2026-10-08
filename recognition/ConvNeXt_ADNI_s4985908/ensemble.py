"""Fit and evaluate a leakage-safe patient-level model ensemble.

The fit command may only consume validation predictions.  It standardizes each
model's patient-level logit margin, selects a convex ensemble weight by AUROC,
and then selects the classification threshold by validation accuracy (with
macro-F1 as the first tie-breaker).  The evaluate command applies that frozen
configuration to a different split without any further search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


REQUIRED_COLUMNS = {
    "subject_id",
    "target",
    "mean_logit_NC",
    "mean_logit_AD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit or evaluate a patient-level SmallCNN/ConvNeXt ensemble"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="fit using validation predictions only")
    fit.add_argument("--smallcnn-csv", type=Path, required=True)
    fit.add_argument("--convnext-csv", type=Path, required=True)
    fit.add_argument("--config-out", type=Path, required=True)
    fit.add_argument("--weight-step", type=float, default=0.01)

    evaluate = subparsers.add_parser(
        "evaluate", help="apply a frozen configuration to another split"
    )
    evaluate.add_argument("--smallcnn-csv", type=Path, required=True)
    evaluate.add_argument("--convnext-csv", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument(
        "--allow-test",
        action="store_true",
        help="required acknowledgement before evaluating the held-out test split",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            subject_id = row["subject_id"]
            if subject_id in rows:
                raise ValueError(f"Duplicate subject {subject_id!r} in {path}")
            logit_nc = float(row["mean_logit_NC"])
            logit_ad = float(row["mean_logit_AD"])
            rows[subject_id] = {
                "subject_id": subject_id,
                "target": int(row["target"]),
                "margin": logit_ad - logit_nc,
            }
    if not rows:
        raise ValueError(f"No patient predictions found in {path}")
    return rows


def align_predictions(
    smallcnn_path: Path, convnext_path: Path
) -> list[dict[str, Any]]:
    smallcnn = load_predictions(smallcnn_path)
    convnext = load_predictions(convnext_path)
    if set(smallcnn) != set(convnext):
        only_small = sorted(set(smallcnn) - set(convnext))[:5]
        only_conv = sorted(set(convnext) - set(smallcnn))[:5]
        raise ValueError(
            "Patient sets differ between models: "
            f"only SmallCNN={only_small}, only ConvNeXt={only_conv}"
        )

    aligned = []
    for subject_id in sorted(smallcnn):
        small_row = smallcnn[subject_id]
        conv_row = convnext[subject_id]
        if small_row["target"] != conv_row["target"]:
            raise ValueError(f"Target mismatch for subject {subject_id}")
        aligned.append(
            {
                "subject_id": subject_id,
                "target": small_row["target"],
                "smallcnn_margin": small_row["margin"],
                "convnext_margin": conv_row["margin"],
            }
        )
    return aligned


def binary_auroc(targets: list[int], scores: list[float]) -> float:
    positives = sum(targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    ordered = sorted(zip(scores, targets), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(t for _, t in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def classification_metrics(
    targets: list[int], predictions: list[int], scores: list[float]
) -> dict[str, Any]:
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision_nc = divide(tn, tn + fn)
    recall_nc = divide(tn, tn + fp)
    precision_ad = divide(tp, tp + fp)
    recall_ad = divide(tp, tp + fn)
    f1_nc = divide(2 * precision_nc * recall_nc, precision_nc + recall_nc)
    f1_ad = divide(2 * precision_ad * recall_ad, precision_ad + recall_ad)
    return {
        "patients": len(targets),
        "correct": tn + tp,
        "accuracy": divide(tn + tp, len(targets)),
        "macro_f1": (f1_nc + f1_ad) / 2,
        "auroc": binary_auroc(targets, scores),
        "precision_NC": precision_nc,
        "recall_NC": recall_nc,
        "f1_NC": f1_nc,
        "precision_AD": precision_ad,
        "recall_AD": recall_ad,
        "f1_AD": f1_ad,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def standardization(values: list[float]) -> tuple[float, float]:
    mean = fmean(values)
    std = pstdev(values)
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Model margins have zero or invalid variance")
    return mean, std


def candidate_thresholds(scores: list[float]) -> list[float]:
    unique_scores = sorted(set(scores))
    if len(unique_scores) == 1:
        return [unique_scores[0]]
    thresholds = [unique_scores[0] - 1e-9]
    thresholds.extend(
        (left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:])
    )
    thresholds.append(unique_scores[-1] + 1e-9)
    return thresholds


def fit_ensemble(args: argparse.Namespace) -> None:
    if not 0 < args.weight_step <= 1:
        raise ValueError("--weight-step must be in (0, 1]")
    rows = align_predictions(args.smallcnn_csv, args.convnext_csv)
    targets = [row["target"] for row in rows]
    small_values = [row["smallcnn_margin"] for row in rows]
    conv_values = [row["convnext_margin"] for row in rows]
    small_mean, small_std = standardization(small_values)
    conv_mean, conv_std = standardization(conv_values)
    small_z = [(value - small_mean) / small_std for value in small_values]
    conv_z = [(value - conv_mean) / conv_std for value in conv_values]

    weight_count = round(1.0 / args.weight_step)
    weights = [index / weight_count for index in range(weight_count + 1)]
    best: tuple[tuple[float, ...], dict[str, Any]] | None = None
    for conv_weight in weights:
        scores = [
            conv_weight * conv_score + (1.0 - conv_weight) * small_score
            for conv_score, small_score in zip(conv_z, small_z)
        ]
        auc = binary_auroc(targets, scores)
        for threshold in candidate_thresholds(scores):
            predictions = [int(score >= threshold) for score in scores]
            metrics = classification_metrics(targets, predictions, scores)
            rank = (
                metrics["accuracy"],
                metrics["macro_f1"],
                auc,
                -abs(conv_weight - 0.5),
                -abs(threshold),
            )
            candidate = {
                "convnext_weight": conv_weight,
                "smallcnn_weight": 1.0 - conv_weight,
                "threshold": threshold,
                "metrics": metrics,
            }
            if best is None or rank > best[0]:
                best = (rank, candidate)

    assert best is not None
    config = {
        "schema_version": 1,
        "fit_split": "validation",
        "patient_count": len(rows),
        "smallcnn_margin_mean": small_mean,
        "smallcnn_margin_std": small_std,
        "convnext_margin_mean": conv_mean,
        "convnext_margin_std": conv_std,
        **best[1],
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2, sort_keys=True))


def evaluate_ensemble(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.allow_test:
        raise ValueError("Pass --allow-test to acknowledge held-out test evaluation")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("fit_split") != "validation":
        raise ValueError("Ensemble configuration was not fitted on validation data")
    rows = align_predictions(args.smallcnn_csv, args.convnext_csv)
    targets = [row["target"] for row in rows]
    scores = []
    for row in rows:
        small_z = (
            row["smallcnn_margin"] - config["smallcnn_margin_mean"]
        ) / config["smallcnn_margin_std"]
        conv_z = (
            row["convnext_margin"] - config["convnext_margin_mean"]
        ) / config["convnext_margin_std"]
        score = (
            config["smallcnn_weight"] * small_z
            + config["convnext_weight"] * conv_z
        )
        row["ensemble_score"] = score
        row["predicted_target"] = int(score >= config["threshold"])
        scores.append(score)

    predictions = [row["predicted_target"] for row in rows]
    metrics = classification_metrics(targets, predictions, scores)
    summary = {
        "split": args.split,
        "smallcnn_csv": str(args.smallcnn_csv),
        "convnext_csv": str(args.convnext_csv),
        "config": str(args.config),
        "frozen_convnext_weight": config["convnext_weight"],
        "frozen_smallcnn_weight": config["smallcnn_weight"],
        "frozen_threshold": config["threshold"],
        **metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.split}_ensemble_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = (
            "subject_id",
            "target",
            "smallcnn_margin",
            "convnext_margin",
            "ensemble_score",
            "predicted_target",
            "correct",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fieldnames if field != "correct"},
                    "correct": int(row["target"] == row["predicted_target"]),
                }
            )
    (args.output_dir / f"{args.split}_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        fit_ensemble(args)
    else:
        evaluate_ensemble(args)


if __name__ == "__main__":
    main()
