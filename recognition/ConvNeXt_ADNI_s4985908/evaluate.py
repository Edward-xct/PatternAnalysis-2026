"""Calibrate and evaluate patient-level ADNI predictions.

Temperature and reject-option thresholds are fitted exclusively on validation
patients. An optional held-out test CSV is evaluated only after an explicit
``--allow-test`` acknowledgement, using the frozen validation parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


CLASS_NAMES = {0: "NC", 1: "AD"}
REQUIRED_COLUMNS = {
    "subject_id",
    "target",
    "mean_logit_NC",
    "mean_logit_AD",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calibrate patient predictions and select a reject option"
    )
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "evaluation",
    )
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--target-ad-sensitivity", type=float, default=0.90)
    parser.add_argument("--target-cn-specificity", type=float, default=0.90)
    parser.add_argument("--expected-validation-patients", type=int, default=None)
    parser.add_argument("--expected-test-patients", type=int, default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Apply frozen validation calibration and thresholds to test data.",
    )
    args = parser.parse_args()

    if args.ece_bins <= 1:
        raise ValueError("--ece-bins must be greater than 1")
    for name in (
        "min_coverage",
        "target_ad_sensitivity",
        "target_cn_specificity",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("expected_validation_patients", "expected_test_patients"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.test_csv is not None and not args.allow_test:
        raise ValueError(
            "Test evaluation is locked. Use --allow-test only after the model, "
            "calibration method, and reject thresholds are frozen."
        )
    if args.allow_test and args.test_csv is None:
        raise ValueError("--allow-test requires --test-csv")
    return args


def load_patient_predictions(
    path: Path,
    expected_patients: int | None = None,
) -> list[dict[str, Any]]:
    """Read and validate patient logits exported by ``predict.py``."""
    if not path.is_file():
        raise FileNotFoundError(f"Patient prediction CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        seen_subjects: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            subject_id = row["subject_id"]
            if not subject_id:
                raise ValueError(f"Empty subject ID at {path}:{row_number}")
            if subject_id in seen_subjects:
                raise ValueError(f"Duplicate subject {subject_id!r} in {path}")
            seen_subjects.add(subject_id)
            try:
                target = int(row["target"])
                logit_nc = float(row["mean_logit_NC"])
                logit_ad = float(row["mean_logit_AD"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid values at {path}:{row_number}") from error
            if target not in CLASS_NAMES:
                raise ValueError(f"Unsupported target {target} at {path}:{row_number}")
            if not math.isfinite(logit_nc) or not math.isfinite(logit_ad):
                raise ValueError(f"Non-finite logits at {path}:{row_number}")
            rows.append(
                {
                    "subject_id": subject_id,
                    "target": target,
                    "logit_NC": logit_nc,
                    "logit_AD": logit_ad,
                }
            )

    if not rows:
        raise ValueError(f"No patients found in {path}")
    if expected_patients is not None and len(rows) != expected_patients:
        raise ValueError(
            f"Expected {expected_patients} patients in {path}, found {len(rows)}"
        )
    targets = {row["target"] for row in rows}
    if targets != {0, 1}:
        raise ValueError(
            f"Both NC and AD patients are required in {path}; found {sorted(targets)}"
        )
    return rows


def rows_to_tensors(rows: list[dict[str, Any]]) -> tuple[Tensor, Tensor]:
    logits = torch.tensor(
        [[row["logit_NC"], row["logit_AD"]] for row in rows],
        dtype=torch.float64,
    )
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    return logits, targets


def fit_temperature(logits: Tensor, targets: Tensor) -> float:
    """Fit one positive temperature by validation negative log-likelihood."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected two-class logits with shape (patients, 2)")
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = criterion(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().item())
    if not math.isfinite(temperature):
        raise FloatingPointError("Temperature optimization produced a non-finite value")
    return min(max(temperature, 0.05), 10.0)


def probabilities_ad(logits: Tensor, temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits / temperature, dim=1)[:, 1].tolist()


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_auroc(targets: list[int], scores: list[float]) -> float | None:
    positives = sum(target == 1 for target in targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, targets), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            target == 1 for _, target in ordered[index:end]
        )
        index = end
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def reliability_rows(
    targets: list[int], probabilities: list[float], bins: int
) -> list[dict[str, Any]]:
    predictions = [int(probability >= 0.5) for probability in probabilities]
    confidences = [max(probability, 1.0 - probability) for probability in probabilities]
    rows = []
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if confidence >= lower
            and (confidence < upper or (bin_index == bins - 1 and confidence <= upper))
        ]
        if indices:
            mean_confidence = (
                sum(confidences[index] for index in indices) / len(indices)
            )
            accuracy = sum(
                predictions[index] == targets[index] for index in indices
            ) / len(indices)
            gap = abs(accuracy - mean_confidence)
        else:
            mean_confidence = None
            accuracy = None
            gap = None
        rows.append(
            {
                "bin": bin_index,
                "lower": lower,
                "upper": upper,
                "count": len(indices),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return rows


def classification_and_calibration_metrics(
    targets: list[int],
    probabilities: list[float],
    ece_bins: int,
) -> dict[str, Any]:
    if len(targets) != len(probabilities) or not targets:
        raise ValueError("Targets and probabilities must be non-empty and equal length")
    predictions = [int(probability >= 0.5) for probability in probabilities]
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))

    precision_nc = _safe_divide(tn, tn + fn)
    recall_nc = _safe_divide(tn, tn + fp)
    precision_ad = _safe_divide(tp, tp + fp)
    recall_ad = _safe_divide(tp, tp + fn)
    f1_nc = _safe_divide(2 * precision_nc * recall_nc, precision_nc + recall_nc)
    f1_ad = _safe_divide(2 * precision_ad * recall_ad, precision_ad + recall_ad)

    epsilon = 1e-12
    clipped = [
        min(max(probability, epsilon), 1.0 - epsilon)
        for probability in probabilities
    ]
    nll = -sum(
        target * math.log(probability) + (1 - target) * math.log(1 - probability)
        for target, probability in zip(targets, clipped)
    ) / len(targets)
    brier = sum(
        (probability - target) ** 2
        for target, probability in zip(targets, probabilities)
    ) / len(targets)
    calibration_bins = reliability_rows(targets, probabilities, ece_bins)
    ece = sum(
        row["count"] / len(targets) * float(row["absolute_gap"] or 0.0)
        for row in calibration_bins
    )

    return {
        "patients": len(targets),
        "accuracy": _safe_divide(tp + tn, len(targets)),
        "precision_NC": precision_nc,
        "recall_NC": recall_nc,
        "f1_NC": f1_nc,
        "precision_AD": precision_ad,
        "recall_AD": recall_ad,
        "f1_AD": f1_ad,
        "macro_f1": (f1_nc + f1_ad) / 2.0,
        "auroc": _binary_auroc(targets, probabilities),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def selective_metrics(
    targets: list[int],
    probabilities: list[float],
    threshold_cn: float,
    threshold_ad: float,
) -> dict[str, Any]:
    if not 0.0 <= threshold_cn <= 0.5 <= threshold_ad <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= CN <= 0.5 <= AD <= 1")
    decisions = [
        0 if probability <= threshold_cn else 1 if probability >= threshold_ad else None
        for probability in probabilities
    ]
    accepted = [
        index for index, decision in enumerate(decisions) if decision is not None
    ]
    tn = sum(targets[index] == 0 and decisions[index] == 0 for index in accepted)
    fp = sum(targets[index] == 0 and decisions[index] == 1 for index in accepted)
    fn = sum(targets[index] == 1 and decisions[index] == 0 for index in accepted)
    tp = sum(targets[index] == 1 and decisions[index] == 1 for index in accepted)
    accepted_ad = tp + fn
    accepted_nc = tn + fp
    total_ad = sum(targets)
    total_nc = len(targets) - total_ad
    accepted_count = len(accepted)
    return {
        "threshold_CN": threshold_cn,
        "threshold_AD": threshold_ad,
        "accepted_patients": accepted_count,
        "referred_patients": len(targets) - accepted_count,
        "coverage": _safe_divide(accepted_count, len(targets)),
        "referral_rate": 1.0 - _safe_divide(accepted_count, len(targets)),
        "selective_accuracy": _safe_divide(tp + tn, accepted_count),
        "AD_sensitivity_accepted": _safe_divide(tp, accepted_ad),
        "CN_specificity_accepted": _safe_divide(tn, accepted_nc),
        "AD_automation_coverage": _safe_divide(accepted_ad, total_ad),
        "CN_automation_coverage": _safe_divide(accepted_nc, total_nc),
        "AD_false_negative_rate_total": _safe_divide(fn, total_ad),
        "CN_false_positive_rate_total": _safe_divide(fp, total_nc),
        "accepted_AD_patients": accepted_ad,
        "accepted_NC_patients": accepted_nc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def select_reject_thresholds(
    targets: list[int],
    probabilities: list[float],
    min_coverage: float,
    target_ad_sensitivity: float,
    target_cn_specificity: float,
) -> dict[str, Any]:
    """Choose the highest-coverage validation thresholds meeting constraints."""
    candidates_cn = sorted({0.0, 0.5, *(p for p in probabilities if p <= 0.5)})
    candidates_ad = sorted({0.5, 1.0, *(p for p in probabilities if p >= 0.5)})
    feasible: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []

    for threshold_cn in candidates_cn:
        for threshold_ad in candidates_ad:
            metrics = selective_metrics(
                targets,
                probabilities,
                threshold_cn=threshold_cn,
                threshold_ad=threshold_ad,
            )
            both_classes_accepted = (
                metrics["accepted_AD_patients"] > 0
                and metrics["accepted_NC_patients"] > 0
            )
            constraints_satisfied = bool(
                both_classes_accepted
                and metrics["coverage"] >= min_coverage
                and metrics["AD_sensitivity_accepted"] >= target_ad_sensitivity
                and metrics["CN_specificity_accepted"] >= target_cn_specificity
            )
            metrics["constraints_satisfied"] = constraints_satisfied
            fallback.append(metrics)
            if constraints_satisfied:
                feasible.append(metrics)

    if feasible:
        return max(
            feasible,
            key=lambda item: (
                item["coverage"],
                item["selective_accuracy"],
                -item["referral_rate"],
            ),
        )

    valid_fallback = [
        item
        for item in fallback
        if item["accepted_AD_patients"] > 0 and item["accepted_NC_patients"] > 0
    ]
    if not valid_fallback:
        raise RuntimeError("No threshold pair accepts patients from both classes")
    return max(
        valid_fallback,
        key=lambda item: (
            min(item["AD_sensitivity_accepted"], item["CN_specificity_accepted"]),
            item["coverage"],
            item["selective_accuracy"],
        ),
    )


def risk_coverage_rows(
    targets: list[int], probabilities: list[float], variant: str
) -> list[dict[str, Any]]:
    ranked = sorted(
        zip(targets, probabilities),
        key=lambda pair: max(pair[1], 1.0 - pair[1]),
        reverse=True,
    )
    correct = 0
    rows = []
    for accepted_count, (target, probability) in enumerate(ranked, start=1):
        correct += int((probability >= 0.5) == bool(target))
        coverage = accepted_count / len(ranked)
        rows.append(
            {
                "variant": variant,
                "accepted_patients": accepted_count,
                "coverage": coverage,
                "selective_risk": 1.0 - correct / accepted_count,
            }
        )
    return rows


def annotate_rows(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    threshold_cn: float,
    threshold_ad: float,
) -> list[dict[str, Any]]:
    annotated = []
    for row, probability in zip(rows, probabilities):
        prediction = int(probability >= 0.5)
        if probability <= threshold_cn:
            decision = "NC"
        elif probability >= threshold_ad:
            decision = "AD"
        else:
            decision = "REFER"
        annotated.append(
            {
                **row,
                "calibrated_probability_AD": probability,
                "calibrated_prediction": CLASS_NAMES[prediction],
                "calibrated_confidence": max(probability, 1.0 - probability),
                "calibrated_correct": int(prediction == row["target"]),
                "triage_decision": decision,
            }
        )
    return annotated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def plot_reliability(
    raw_rows: list[dict[str, Any]],
    calibrated_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib or omit --plots") from error

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for axis, rows, title in zip(
        axes,
        (raw_rows, calibrated_rows),
        ("Raw softmax", "Temperature scaled"),
    ):
        populated = [row for row in rows if row["count"] > 0]
        axis.plot([0.5, 1.0], [0.5, 1.0], "--", color="black", linewidth=1)
        axis.plot(
            [row["mean_confidence"] for row in populated],
            [row["accuracy"] for row in populated],
            marker="o",
        )
        axis.set_title(title)
        axis.set_xlabel("Mean confidence")
        axis.set_xlim(0.5, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical accuracy")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_risk_coverage(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib or omit --plots") from error

    figure, axis = plt.subplots(figsize=(5, 4))
    for variant in ("raw", "calibrated"):
        selected = [row for row in rows if row["variant"] == variant]
        axis.plot(
            [row["coverage"] for row in selected],
            [row["selective_risk"] for row in selected],
            label=variant,
        )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk (1 - accuracy)")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_split(
    rows: list[dict[str, Any]],
    temperature: float,
    ece_bins: int,
    threshold_cn: float,
    threshold_ad: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    logits, targets_tensor = rows_to_tensors(rows)
    targets = targets_tensor.tolist()
    raw_probabilities = probabilities_ad(logits)
    calibrated_probabilities = probabilities_ad(logits, temperature)
    raw_reliability = reliability_rows(targets, raw_probabilities, ece_bins)
    calibrated_reliability = reliability_rows(
        targets, calibrated_probabilities, ece_bins
    )
    risk_rows = risk_coverage_rows(targets, raw_probabilities, "raw")
    risk_rows.extend(
        risk_coverage_rows(targets, calibrated_probabilities, "calibrated")
    )
    result = {
        "raw": classification_and_calibration_metrics(
            targets, raw_probabilities, ece_bins
        ),
        "calibrated": classification_and_calibration_metrics(
            targets, calibrated_probabilities, ece_bins
        ),
        "selective": selective_metrics(
            targets,
            calibrated_probabilities,
            threshold_cn=threshold_cn,
            threshold_ad=threshold_ad,
        ),
    }
    tables = {
        "raw_reliability": raw_reliability,
        "calibrated_reliability": calibrated_reliability,
        "risk_coverage": risk_rows,
        "annotated": annotate_rows(
            rows,
            calibrated_probabilities,
            threshold_cn=threshold_cn,
            threshold_ad=threshold_ad,
        ),
    }
    return result, tables


def save_split_tables(
    output_dir: Path,
    split: str,
    tables: dict[str, list[dict[str, Any]]],
    make_plots: bool,
) -> None:
    reliability_rows_combined = [
        {"variant": "raw", **row} for row in tables["raw_reliability"]
    ]
    reliability_rows_combined.extend(
        {"variant": "calibrated", **row}
        for row in tables["calibrated_reliability"]
    )
    write_csv(
        output_dir / f"{split}_reliability.csv",
        reliability_rows_combined,
    )
    write_csv(
        output_dir / f"{split}_risk_coverage.csv",
        tables["risk_coverage"],
    )
    write_csv(
        output_dir / f"{split}_calibrated_predictions.csv",
        tables["annotated"],
    )
    if make_plots:
        plot_reliability(
            tables["raw_reliability"],
            tables["calibrated_reliability"],
            output_dir / f"{split}_reliability.png",
        )
        plot_risk_coverage(
            tables["risk_coverage"],
            output_dir / f"{split}_risk_coverage.png",
        )


def main() -> None:
    args = parse_args()
    validation_rows = load_patient_predictions(
        args.validation_csv,
        expected_patients=args.expected_validation_patients,
    )
    validation_logits, validation_targets_tensor = rows_to_tensors(validation_rows)
    validation_targets = validation_targets_tensor.tolist()
    temperature = fit_temperature(validation_logits, validation_targets_tensor)
    validation_calibrated_probabilities = probabilities_ad(
        validation_logits, temperature
    )
    thresholds = select_reject_thresholds(
        validation_targets,
        validation_calibrated_probabilities,
        min_coverage=args.min_coverage,
        target_ad_sensitivity=args.target_ad_sensitivity,
        target_cn_specificity=args.target_cn_specificity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_result, validation_tables = evaluate_split(
        validation_rows,
        temperature=temperature,
        ece_bins=args.ece_bins,
        threshold_cn=thresholds["threshold_CN"],
        threshold_ad=thresholds["threshold_AD"],
    )
    save_split_tables(
        args.output_dir,
        "validation",
        validation_tables,
        make_plots=args.plots,
    )

    result: dict[str, Any] = {
        "calibration": {
            "fit_split": "validation",
            "method": "temperature_scaling",
            "temperature": temperature,
        },
        "threshold_selection": {
            "fit_split": "validation",
            "minimum_coverage": args.min_coverage,
            "target_AD_sensitivity": args.target_ad_sensitivity,
            "target_CN_specificity": args.target_cn_specificity,
            **thresholds,
        },
        "validation": validation_result,
    }

    if args.test_csv is not None:
        test_rows = load_patient_predictions(
            args.test_csv,
            expected_patients=args.expected_test_patients,
        )
        test_result, test_tables = evaluate_split(
            test_rows,
            temperature=temperature,
            ece_bins=args.ece_bins,
            threshold_cn=thresholds["threshold_CN"],
            threshold_ad=thresholds["threshold_AD"],
        )
        save_split_tables(
            args.output_dir,
            "test",
            test_tables,
            make_plots=args.plots,
        )
        result["test"] = test_result

    write_json(args.output_dir / "evaluation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
