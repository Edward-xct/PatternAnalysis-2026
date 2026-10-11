"""Validation-fitted equal-weight ensemble for multiple ConvNeXt seeds.

The fit command compares three conventional patient-level scores using
validation predictions only: AD probability, logit margin, and standardized
logit margin. It searches a coarse convex grid of non-negative seed weights
and a decision threshold. The evaluate command applies the frozen
configuration to another split without searching.
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
    "probability_AD",
}
METHOD_ORDER = ("mean_probability", "mean_margin", "standardized_mean_margin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit or evaluate an equal-weight multi-seed ConvNeXt ensemble"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit")
    fit.add_argument(
        "--member",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="validation patient predictions; repeat once per seed",
    )
    fit.add_argument("--config-out", type=Path, required=True)
    fit.add_argument("--weight-step", type=float, default=0.05)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--member",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="patient predictions matching every fitted seed",
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def parse_member(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Member must use NAME=CSV syntax: {specification!r}")
    name, path = specification.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("Member name cannot be empty")
    return name, Path(path)


def load_csv(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = {}
        for raw in reader:
            subject_id = raw["subject_id"]
            if subject_id in rows:
                raise ValueError(f"Duplicate subject {subject_id!r} in {path}")
            logit_nc = float(raw["mean_logit_NC"])
            logit_ad = float(raw["mean_logit_AD"])
            rows[subject_id] = {
                "target": int(raw["target"]),
                "probability": float(raw["probability_AD"]),
                "margin": logit_ad - logit_nc,
            }
    if not rows:
        raise ValueError(f"No patient rows found in {path}")
    return rows


def load_members(specifications: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    members: list[tuple[str, Path, dict[str, dict[str, Any]]]] = []
    names = set()
    for specification in specifications:
        name, path = parse_member(specification)
        if name in names:
            raise ValueError(f"Duplicate member name: {name}")
        names.add(name)
        members.append((name, path, load_csv(path)))
    if len(members) < 2:
        raise ValueError("At least two ensemble members are required")

    reference_subjects = set(members[0][2])
    for name, _, rows in members[1:]:
        if set(rows) != reference_subjects:
            raise ValueError(f"Patient set for {name} differs from the first member")

    aligned = []
    for subject_id in sorted(reference_subjects):
        targets = {rows[subject_id]["target"] for _, _, rows in members}
        if len(targets) != 1:
            raise ValueError(f"Target disagreement for subject {subject_id}")
        aligned.append(
            {
                "subject_id": subject_id,
                "target": targets.pop(),
                "probabilities": {
                    name: rows[subject_id]["probability"] for name, _, rows in members
                },
                "margins": {
                    name: rows[subject_id]["margin"] for name, _, rows in members
                },
            }
        )
    return [name for name, _, _ in members], aligned


def auroc(targets: list[int], scores: list[float]) -> float:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    ordered = sorted(zip(scores, targets), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(target for _, target in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(
    targets: list[int], predictions: list[int], scores: list[float]
) -> dict[str, Any]:
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    precision_nc = safe_divide(tn, tn + fn)
    recall_nc = safe_divide(tn, tn + fp)
    precision_ad = safe_divide(tp, tp + fp)
    recall_ad = safe_divide(tp, tp + fn)
    f1_nc = safe_divide(2 * precision_nc * recall_nc, precision_nc + recall_nc)
    f1_ad = safe_divide(2 * precision_ad * recall_ad, precision_ad + recall_ad)
    return {
        "patients": len(targets),
        "correct": tn + tp,
        "accuracy": safe_divide(tn + tp, len(targets)),
        "macro_f1": (f1_nc + f1_ad) / 2,
        "auroc": auroc(targets, scores),
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


def candidate_thresholds(scores: list[float]) -> list[float]:
    ordered = sorted(set(scores))
    if len(ordered) == 1:
        return ordered
    return [ordered[0] - 1e-9] + [
        (left + right) / 2 for left, right in zip(ordered, ordered[1:])
    ] + [ordered[-1] + 1e-9]


def build_statistics(names: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    statistics = {}
    for name in names:
        values = [row["margins"][name] for row in rows]
        mean = fmean(values)
        std = pstdev(values)
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid margin variance for {name}")
        statistics[name] = {"margin_mean": mean, "margin_std": std}
    return statistics


def scores_for_method(
    method: str,
    names: list[str],
    rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    weights: dict[str, float],
) -> list[float]:
    scores = []
    for row in rows:
        if method == "mean_probability":
            values = [row["probabilities"][name] for name in names]
        elif method == "mean_margin":
            values = [row["margins"][name] for name in names]
        elif method == "standardized_mean_margin":
            values = [
                (row["margins"][name] - statistics[name]["margin_mean"])
                / statistics[name]["margin_std"]
                for name in names
            ]
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        scores.append(sum(weights[name] * value for name, value in zip(names, values)))
    return scores


def convex_weight_grid(names: list[str], step: float) -> list[dict[str, float]]:
    if not 0 < step <= 1:
        raise ValueError("--weight-step must be in (0, 1]")
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError("--weight-step must divide 1.0 exactly")

    vectors: list[dict[str, float]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            counts = prefix + [remaining]
            vectors.append(
                {name: count / units for name, count in zip(names, counts)}
            )
            return
        for count in range(remaining + 1):
            visit(prefix + [count], remaining - count, slots - 1)

    visit([], units, len(names))
    return vectors


def fit(args: argparse.Namespace) -> None:
    names, rows = load_members(args.member)
    targets = [row["target"] for row in rows]
    statistics = build_statistics(names, rows)
    weight_vectors = convex_weight_grid(names, args.weight_step)
    best: tuple[tuple[float, ...], dict[str, Any]] | None = None
    for method_index, method in enumerate(METHOD_ORDER):
        for weights in weight_vectors:
            scores = scores_for_method(method, names, rows, statistics, weights)
            neutral_threshold = 0.5 if method == "mean_probability" else 0.0
            for threshold in candidate_thresholds(scores):
                predictions = [int(score >= threshold) for score in scores]
                result = metrics(targets, predictions, scores)
                equal_weight = 1.0 / len(names)
                weight_penalty = sum(
                    (weight - equal_weight) ** 2 for weight in weights.values()
                )
                rank = (
                    result["accuracy"],
                    result["macro_f1"],
                    result["auroc"],
                    -weight_penalty,
                    -method_index,
                    -abs(threshold - neutral_threshold),
                )
                candidate = {
                    "method": method,
                    "threshold": threshold,
                    "weights": weights,
                    "validation_metrics": result,
                }
                if best is None or rank > best[0]:
                    best = rank, candidate
    assert best is not None
    config = {
        "schema_version": 1,
        "fit_split": "validation",
        "aggregation": "validation_grid_convex_weights",
        "weight_step": args.weight_step,
        "members": names,
        "member_statistics": statistics,
        **best[1],
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.allow_test:
        raise ValueError("Pass --allow-test before evaluating the test split")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("fit_split") != "validation":
        raise ValueError("Configuration was not fitted on validation")
    names, rows = load_members(args.member)
    if names != config["members"]:
        raise ValueError(
            f"Member order/names differ: expected {config['members']}, received {names}"
        )
    scores = scores_for_method(
        config["method"],
        names,
        rows,
        config["member_statistics"],
        config["weights"],
    )
    targets = [row["target"] for row in rows]
    predictions = [int(score >= config["threshold"]) for score in scores]
    result = metrics(targets, predictions, scores)
    summary = {
        "split": args.split,
        "config": str(args.config),
        "members": names,
        "aggregation": config["aggregation"],
        "weights": config["weights"],
        "method": config["method"],
        "frozen_threshold": config["threshold"],
        **result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.split}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "subject_id",
                "target",
                "ensemble_score",
                "predicted_target",
                "correct",
            ),
        )
        writer.writeheader()
        for row, score, prediction in zip(rows, scores, predictions):
            writer.writerow(
                {
                    "subject_id": row["subject_id"],
                    "target": row["target"],
                    "ensemble_score": score,
                    "predicted_target": prediction,
                    "correct": int(row["target"] == prediction),
                }
            )
    (args.output_dir / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        fit(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
