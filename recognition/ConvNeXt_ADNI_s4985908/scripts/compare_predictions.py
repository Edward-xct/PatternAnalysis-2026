"""Check that two patient-level prediction files agree.

Used to confirm that the frozen ImageNet checkpoint gives the same patient
predictions when it is evaluated with this project's own ConvNeXtTiny class as
it did with torchvision's ConvNeXt class. Mixed-precision inference can change
logits in the third decimal place, so a small logit tolerance is allowed, but
every patient's predicted class must be identical.

Example:
    python scripts/compare_predictions.py \
        --reference outputs/final-test-convnext-imagenet/test_patient_predictions.csv \
        --candidate outputs/final-test-convnext-imagenet-own/test_patient_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read a patient prediction CSV keyed by subject ID."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = {row["subject_id"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError(f"No patient rows found in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--logit-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    reference = load_rows(args.reference)
    candidate = load_rows(args.candidate)
    if set(reference) != set(candidate):
        print("FAIL: the two files contain different patients")
        return 1

    max_logit_difference = 0.0
    max_probability_difference = 0.0
    class_disagreements = []
    for subject_id, ref_row in reference.items():
        new_row = candidate[subject_id]
        for column in ("mean_logit_NC", "mean_logit_AD"):
            difference = abs(float(ref_row[column]) - float(new_row[column]))
            max_logit_difference = max(max_logit_difference, difference)
        max_probability_difference = max(
            max_probability_difference,
            abs(float(ref_row["probability_AD"]) - float(new_row["probability_AD"])),
        )
        if ref_row["predicted_class"] != new_row["predicted_class"]:
            class_disagreements.append(subject_id)

    reference_correct = sum(int(row["correct"]) for row in reference.values())
    candidate_correct = sum(int(row["correct"]) for row in candidate.values())
    print(f"patients={len(reference)}")
    print(f"reference_correct={reference_correct}")
    print(f"candidate_correct={candidate_correct}")
    print(f"max_abs_logit_difference={max_logit_difference:.6f}")
    print(f"max_abs_probability_difference={max_probability_difference:.6f}")
    print(f"predicted_class_disagreements={len(class_disagreements)}")

    if class_disagreements or max_logit_difference > args.logit_tolerance:
        print("FAIL: predictions differ beyond the allowed tolerance")
        return 1
    print("PASS: predictions are equivalent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
