"""Select and visualise representative patient-level classification errors.

This script consumes the calibrated patient predictions written by
``evaluate.py`` and the leakage-free slice manifest written by
``scripts/audit_data.py``. It ranks false negatives and false positives by
calibrated confidence, then creates one deterministic MRI montage per selected
subject. Generated montages belong in an ignored output directory; raw course
images must not be committed to the public repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from dataset import SliceRecord, load_manifest


CLASS_NAMES = {0: "NC", 1: "AD"}
REQUIRED_COLUMNS = {
    "subject_id",
    "target",
    "calibrated_probability_AD",
    "calibrated_prediction",
    "calibrated_confidence",
    "calibrated_correct",
    "triage_decision",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export representative patient-level ADNI failure cases"
    )
    parser.add_argument("--calibrated-csv", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "artifacts" / "slice_manifest.csv",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "failure_cases",
    )
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--slices-per-case", type=int, default=9)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock held-out test failure analysis after evaluation.",
    )
    args = parser.parse_args()

    if args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if args.slices_per_case <= 0:
        raise ValueError("--slices-per-case must be positive")
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "Test failure analysis is locked. Use --allow-test only after final "
            "test evaluation has been completed."
        )
    return args


def load_calibrated_predictions(path: Path) -> list[dict[str, Any]]:
    """Read and strictly validate calibrated patient predictions."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibrated prediction CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    seen_subjects: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row_number, raw_row in enumerate(reader, start=2):
            subject_id = raw_row["subject_id"].strip()
            if not subject_id:
                raise ValueError(f"Empty subject ID at {path}:{row_number}")
            if subject_id in seen_subjects:
                raise ValueError(f"Duplicate subject {subject_id!r} in {path}")
            seen_subjects.add(subject_id)

            try:
                target = int(raw_row["target"])
                probability_ad = float(raw_row["calibrated_probability_AD"])
                confidence = float(raw_row["calibrated_confidence"])
                correct = int(raw_row["calibrated_correct"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid calibrated values at {path}:{row_number}"
                ) from error

            prediction_name = raw_row["calibrated_prediction"].strip()
            if target not in CLASS_NAMES:
                raise ValueError(f"Unsupported target {target} at {path}:{row_number}")
            if prediction_name not in {"NC", "AD"}:
                raise ValueError(
                    f"Unsupported prediction {prediction_name!r} at "
                    f"{path}:{row_number}"
                )
            if not 0.0 <= probability_ad <= 1.0:
                raise ValueError(
                    f"Probability outside [0, 1] at {path}:{row_number}"
                )
            if not 0.5 <= confidence <= 1.0:
                raise ValueError(f"Invalid confidence at {path}:{row_number}")

            prediction = 1 if prediction_name == "AD" else 0
            expected_correct = int(prediction == target)
            if correct != expected_correct:
                raise ValueError(
                    f"Prediction/correct mismatch at {path}:{row_number}"
                )

            rows.append(
                {
                    "subject_id": subject_id,
                    "target": target,
                    "target_class": CLASS_NAMES[target],
                    "prediction": prediction,
                    "predicted_class": prediction_name,
                    "probability_AD": probability_ad,
                    "confidence": confidence,
                    "correct": correct,
                    "triage_decision": raw_row["triage_decision"].strip(),
                }
            )

    if not rows:
        raise ValueError(f"No patient rows found in {path}")
    return rows


def select_failures(
    rows: Sequence[dict[str, Any]], max_cases: int
) -> list[dict[str, Any]]:
    """Select both error directions first, then fill by confidence."""
    false_negatives = sorted(
        (row for row in rows if row["target"] == 1 and row["prediction"] == 0),
        key=lambda row: (-row["confidence"], row["subject_id"]),
    )
    false_positives = sorted(
        (row for row in rows if row["target"] == 0 and row["prediction"] == 1),
        key=lambda row: (-row["confidence"], row["subject_id"]),
    )

    selected: list[dict[str, Any]] = []
    if false_negatives:
        selected.append({**false_negatives[0], "error_type": "false_negative"})
    if false_positives and len(selected) < max_cases:
        selected.append({**false_positives[0], "error_type": "false_positive"})

    already_selected = {row["subject_id"] for row in selected}
    remaining = [
        {
            **row,
            "error_type": (
                "false_negative" if row["target"] == 1 else "false_positive"
            ),
        }
        for row in (*false_negatives, *false_positives)
        if row["subject_id"] not in already_selected
    ]
    remaining.sort(key=lambda row: (-row["confidence"], row["subject_id"]))
    selected.extend(remaining[: max_cases - len(selected)])
    return selected


def records_by_subject_and_scan(
    manifest_path: Path, split: str
) -> dict[str, dict[str, list[SliceRecord]]]:
    """Index one manifest split by subject and scan."""
    indexed: dict[str, dict[str, list[SliceRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in load_manifest(manifest_path):
        if record.split == split:
            indexed[record.subject_id][record.scan_id].append(record)

    for scans in indexed.values():
        for scan_records in scans.values():
            scan_records.sort(key=lambda record: record.slice_index)
    return indexed


def evenly_spaced(items: Sequence[SliceRecord], count: int) -> list[SliceRecord]:
    """Choose deterministic, evenly spaced items including both endpoints."""
    if not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def choose_scan(scans: dict[str, list[SliceRecord]]) -> tuple[str, list[SliceRecord]]:
    """Choose the fullest scan, breaking ties deterministically by scan ID."""
    if not scans:
        raise ValueError("Cannot choose a scan from an empty subject")
    scan_id = sorted(scans, key=lambda key: (-len(scans[key]), key))[0]
    return scan_id, scans[scan_id]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def create_montage(
    row: dict[str, Any],
    scan_id: str,
    records: Sequence[SliceRecord],
    output_path: Path,
) -> None:
    """Create a compact grayscale montage for one selected failure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to create montages") from error

    if not records:
        raise ValueError(f"No slice records for subject {row['subject_id']}")
    missing = [record.image_path for record in records if not record.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MRI slice: {missing[0]}")

    columns = min(3, len(records))
    rows_count = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(3.0 * columns, 3.0 * rows_count),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")

    for axis, record in zip(axes.flat, records):
        with Image.open(record.image_path) as image:
            axis.imshow(image.convert("L"), cmap="gray")
        axis.set_title(f"slice {record.slice_index}", fontsize=9)
        axis.axis("off")

    figure.suptitle(
        f"{row['error_type'].replace('_', ' ').title()} | "
        f"true={row['target_class']} predicted={row['predicted_class']} | "
        f"P(AD)={row['probability_AD']:.3f} confidence={row['confidence']:.3f}\n"
        f"subject={row['subject_id']} scan={scan_id} "
        f"triage={row['triage_decision']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


def main() -> None:
    args = parse_args()
    predictions = load_calibrated_predictions(args.calibrated_csv)
    selected = select_failures(predictions, args.max_cases)
    manifest_index = records_by_subject_and_scan(args.manifest, args.split)
    output_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(selected, start=1):
        subject_id = row["subject_id"]
        if subject_id not in manifest_index:
            raise ValueError(
                f"Selected subject {subject_id!r} is absent from manifest split "
                f"{args.split!r}"
            )
        scan_id, scan_records = choose_scan(manifest_index[subject_id])
        montage_records = evenly_spaced(scan_records, args.slices_per_case)
        montage_name = f"{rank:02d}_{row['error_type']}_{safe_filename(subject_id)}.png"
        montage_path = args.output_dir / montage_name
        create_montage(row, scan_id, montage_records, montage_path)
        output_rows.append(
            {
                "rank": rank,
                "subject_id": subject_id,
                "error_type": row["error_type"],
                "target_class": row["target_class"],
                "predicted_class": row["predicted_class"],
                "calibrated_probability_AD": row["probability_AD"],
                "calibrated_confidence": row["confidence"],
                "triage_decision": row["triage_decision"],
                "referred": int(row["triage_decision"] == "REFER"),
                "scan_id": scan_id,
                "montage": str(montage_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "failure_cases.csv", output_rows)
    summary = {
        "calibrated_csv": str(args.calibrated_csv),
        "errors_available": sum(row["correct"] == 0 for row in predictions),
        "false_negatives_available": sum(
            row["target"] == 1 and row["prediction"] == 0 for row in predictions
        ),
        "false_positives_available": sum(
            row["target"] == 0 and row["prediction"] == 1 for row in predictions
        ),
        "manifest": str(args.manifest),
        "selected_cases": len(output_rows),
        "split": args.split,
        "warning": (
            "Generated montages contain course MRI images and must not be "
            "committed to the public repository."
        ),
    }
    write_json(args.output_dir / "failure_case_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
