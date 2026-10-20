"""Quantify agreement between two patient-level prediction runs.

Used to check that the frozen ImageNet checkpoint gives the same patient
predictions when it is evaluated with this project's own ConvNeXtTiny class as
it did with torchvision's ConvNeXt class. The script reports, rather than
assumes, how close the two runs are:

* the maximum absolute difference in patient mean logits and P(AD), printed in
  scientific notation so that small non-zero differences are visible;
* how many patients have logits that are exactly equal as stored in the CSVs
  (stored values are float32 logits written with Python's ``repr``);
* how many predicted classes differ.

Optionally it also compares every numeric metric in two ``evaluation.json``
files written by ``evaluate.py``. With ``--output-json`` the summary is saved
as aggregate evidence; it contains no subject identifiers.

The check passes when no predicted class differs and the largest logit
difference is within ``--logit-tolerance`` (default 0.05, which allows for
mixed-precision non-determinism). Passing therefore establishes equivalence
within that tolerance; the reported maxima show how close the runs really are.

Example:
    python scripts/compare_predictions.py \
        --reference outputs/final-test-convnext-imagenet/test_patient_predictions.csv \
        --candidate outputs/final-test-convnext-imagenet-own/test_patient_predictions.csv \
        --reference-evaluation outputs/final-results-convnext-imagenet/evaluation.json \
        --candidate-evaluation outputs/final-results-convnext-imagenet-own/evaluation.json \
        --output-json figures/report/checkpoint_equivalence_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read a patient prediction CSV keyed by subject ID."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = {row["subject_id"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError(f"No patient rows found in {path}")
    return rows


def numeric_leaves(payload: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Yield (dotted key, value) for every numeric leaf of a JSON object."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from numeric_leaves(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(payload, bool):
        return
    elif isinstance(payload, (int, float)):
        yield prefix, float(payload)


def compare_predictions(
    reference: dict[str, dict[str, str]], candidate: dict[str, dict[str, str]]
) -> dict[str, Any]:
    max_logit = 0.0
    max_probability = 0.0
    exactly_equal = 0
    class_disagreements = 0
    for subject_id, ref_row in reference.items():
        new_row = candidate[subject_id]
        logit_pairs = [
            (float(ref_row[column]), float(new_row[column]))
            for column in ("mean_logit_NC", "mean_logit_AD")
        ]
        max_logit = max(max_logit, *(abs(a - b) for a, b in logit_pairs))
        max_probability = max(
            max_probability,
            abs(float(ref_row["probability_AD"]) - float(new_row["probability_AD"])),
        )
        exactly_equal += all(a == b for a, b in logit_pairs)
        class_disagreements += ref_row["predicted_class"] != new_row["predicted_class"]
    return {
        "patients": len(reference),
        "reference_correct": sum(int(row["correct"]) for row in reference.values()),
        "candidate_correct": sum(int(row["correct"]) for row in candidate.values()),
        "max_abs_logit_difference": max_logit,
        "max_abs_probability_difference": max_probability,
        "patients_with_exactly_equal_stored_logits": exactly_equal,
        "predicted_class_disagreements": class_disagreements,
    }


def compare_evaluations(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = dict(numeric_leaves(json.loads(reference_path.read_text(encoding="utf-8"))))
    candidate = dict(numeric_leaves(json.loads(candidate_path.read_text(encoding="utf-8"))))
    if set(reference) != set(candidate):
        raise ValueError("The two evaluation files contain different metrics")
    differences = {key: abs(reference[key] - candidate[key]) for key in reference}
    largest_key = max(differences, key=differences.get)
    return {
        "numeric_metrics_compared": len(differences),
        "metrics_exactly_equal": sum(value == 0.0 for value in differences.values()),
        "max_abs_metric_difference": differences[largest_key],
        "metric_with_largest_difference": largest_key,
        "metrics_differing_at_4_decimals": sum(
            round(reference[k], 4) != round(candidate[k], 4) for k in reference
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, default=None)
    parser.add_argument("--candidate-evaluation", type=Path, default=None)
    parser.add_argument("--logit-tolerance", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if (args.reference_evaluation is None) != (args.candidate_evaluation is None):
        parser.error("--reference-evaluation and --candidate-evaluation go together")

    reference = load_rows(args.reference)
    candidate = load_rows(args.candidate)
    if set(reference) != set(candidate):
        print("FAIL: the two files contain different patients")
        return 1

    summary: dict[str, Any] = {
        "reference": args.reference.name,
        "candidate": args.candidate.name,
        "logit_tolerance": args.logit_tolerance,
        "predictions": compare_predictions(reference, candidate),
    }
    if args.reference_evaluation is not None:
        summary["evaluation"] = compare_evaluations(
            args.reference_evaluation, args.candidate_evaluation
        )

    predictions = summary["predictions"]
    passed = (
        predictions["predicted_class_disagreements"] == 0
        and predictions["max_abs_logit_difference"] <= args.logit_tolerance
    )
    summary["passed"] = passed

    for key, value in predictions.items():
        print(f"{key}={value:.3e}" if isinstance(value, float) else f"{key}={value}")
    for key, value in summary.get("evaluation", {}).items():
        print(f"evaluation.{key}={value:.3e}" if isinstance(value, float) else f"evaluation.{key}={value}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote: {args.output_json}")

    if not passed:
        print("FAIL: predictions differ beyond the allowed tolerance")
        return 1
    print("PASS: same predicted class for every patient; logits within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
