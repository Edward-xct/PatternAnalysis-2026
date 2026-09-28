"""Audit the COMP3710 ADNI dataset and create leakage-free split manifests.

The supplied JPEG folders use a scan-level train/test split. This script
combines both folders, recovers the ADNI subject ID from the metadata JSON,
and creates deterministic, label-stratified patient-level splits.

Only file names and metadata are inspected. Image pixels are not loaded, so
the script is safe to run on the Rangpur login node.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JSON_LABEL_TO_CLASS = {0: "NC", 2: "AD"}
JSON_LABEL_TO_TARGET = {0: 0, 2: 1}
FOLDER_TO_JSON_LABEL = {"NC": 0, "AD": 2}
SUBJECT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})_")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Audit ADNI JPEG slices and create deterministic patient-level "
            "train/validation/test manifests."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/AD_NC"),
        help="Directory containing the supplied train/ and test/ folders.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/meta_data_with_label.json"),
        help="ADNI metadata JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts",
        help="Directory for generated CSV and JSON files.",
    )
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    """Validate split ratios before touching the dataset."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load and minimally validate the course metadata JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError("Metadata JSON must contain an object keyed by scan ID")
    return metadata


def extract_subject_id(entry: dict[str, Any], scan_id: str) -> str:
    """Extract an ADNI subject ID such as 068_S_0473 from a metadata entry."""
    source_path = str(entry.get("raw") or entry.get("masked") or "")
    match = SUBJECT_PATTERN.search(source_path)
    if match is None:
        raise ValueError(
            f"Could not parse subject ID for scan {scan_id} from {source_path!r}"
        )
    return match.group(1)


def collect_slice_records(
    data_root: Path, metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collect and validate every JPEG slice in the supplied folders."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    for original_split in ("train", "test"):
        for class_name in ("AD", "NC"):
            class_dir = data_root / original_split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected directory not found: {class_dir}")

            for image_path in sorted(class_dir.glob("*.jpeg")):
                if image_path in seen_paths:
                    raise ValueError(f"Duplicate image path: {image_path}")
                seen_paths.add(image_path)

                name_parts = image_path.stem.split("_", maxsplit=1)
                if len(name_parts) != 2 or not name_parts[1].isdigit():
                    raise ValueError(f"Unexpected JPEG file name: {image_path.name}")

                scan_id, slice_text = name_parts
                if scan_id not in metadata:
                    raise KeyError(f"Scan {scan_id} is missing from the metadata JSON")

                entry = metadata[scan_id]
                json_label = int(entry["label"])
                expected_label = FOLDER_TO_JSON_LABEL[class_name]
                if json_label != expected_label:
                    raise ValueError(
                        f"Label mismatch for {image_path}: folder={class_name}, "
                        f"JSON label={json_label}"
                    )
                if json_label not in JSON_LABEL_TO_TARGET:
                    raise ValueError(f"Unsupported JSON label {json_label}")

                records.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "subject_id": extract_subject_id(entry, scan_id),
                        "scan_id": scan_id,
                        "slice_index": int(slice_text),
                        "json_label": json_label,
                        "target": JSON_LABEL_TO_TARGET[json_label],
                        "class_name": JSON_LABEL_TO_CLASS[json_label],
                        "original_split": original_split,
                    }
                )

    if not records:
        raise ValueError(f"No JPEG slices found under {data_root}")
    return records


def validate_subject_labels(records: list[dict[str, Any]]) -> dict[str, int]:
    """Require every scan and slice from a subject to have one diagnosis label."""
    labels_by_subject: dict[str, set[int]] = defaultdict(set)
    for record in records:
        labels_by_subject[record["subject_id"]].add(record["json_label"])

    conflicts = {
        subject_id: labels
        for subject_id, labels in labels_by_subject.items()
        if len(labels) != 1
    }
    if conflicts:
        examples = list(conflicts.items())[:10]
        raise ValueError(f"Subjects with inconsistent labels: {examples}")

    return {
        subject_id: next(iter(labels))
        for subject_id, labels in labels_by_subject.items()
    }


def stratified_subject_split(
    subject_labels: dict[str, int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    """Assign whole subjects to deterministic, label-stratified splits."""
    subjects_by_label: dict[int, list[str]] = defaultdict(list)
    for subject_id, json_label in subject_labels.items():
        subjects_by_label[json_label].append(subject_id)

    assignment: dict[str, str] = {}
    for json_label in sorted(subjects_by_label):
        subjects = sorted(subjects_by_label[json_label])
        random.Random(seed + json_label).shuffle(subjects)

        count = len(subjects)
        train_end = round(count * train_ratio)
        val_end = train_end + round(count * val_ratio)

        if train_end == 0 or val_end <= train_end or val_end >= count:
            raise ValueError(
                f"Class {json_label} has too few subjects for the requested ratios"
            )

        for subject_id in subjects[:train_end]:
            assignment[subject_id] = "train"
        for subject_id in subjects[train_end:val_end]:
            assignment[subject_id] = "validation"
        for subject_id in subjects[val_end:]:
            assignment[subject_id] = "test"

    if set(assignment) != set(subject_labels):
        raise AssertionError("Not every subject received exactly one split")
    return assignment


def write_manifests(
    records: list[dict[str, Any]],
    assignment: dict[str, str],
    output_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Write subject-level, slice-level, and summary manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_ids_by_subject: dict[str, set[str]] = defaultdict(set)
    slices_by_subject: Counter[str] = Counter()
    label_by_subject: dict[str, int] = {}
    for record in records:
        subject_id = record["subject_id"]
        scan_ids_by_subject[subject_id].add(record["scan_id"])
        slices_by_subject[subject_id] += 1
        label_by_subject[subject_id] = record["json_label"]

    subject_path = output_dir / "subject_split.csv"
    with subject_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "subject_id",
            "json_label",
            "target",
            "class_name",
            "split",
            "num_scans",
            "num_slices",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for subject_id in sorted(assignment):
            json_label = label_by_subject[subject_id]
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "json_label": json_label,
                    "target": JSON_LABEL_TO_TARGET[json_label],
                    "class_name": JSON_LABEL_TO_CLASS[json_label],
                    "split": assignment[subject_id],
                    "num_scans": len(scan_ids_by_subject[subject_id]),
                    "num_slices": slices_by_subject[subject_id],
                }
            )

    slice_path = output_dir / "slice_manifest.csv"
    slice_fields = [
        "image_path",
        "subject_id",
        "scan_id",
        "slice_index",
        "json_label",
        "target",
        "class_name",
        "split",
        "original_split",
    ]
    sorted_records = sorted(
        records,
        key=lambda item: (
            assignment[item["subject_id"]],
            item["subject_id"],
            item["scan_id"],
            item["slice_index"],
        ),
    )
    with slice_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=slice_fields)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {
                    **record,
                    "split": assignment[record["subject_id"]],
                }
            )

    split_subjects = {
        split_name: {
            subject_id
            for subject_id, assigned_split in assignment.items()
            if assigned_split == split_name
        }
        for split_name in ("train", "validation", "test")
    }
    assert split_subjects["train"].isdisjoint(split_subjects["validation"])
    assert split_subjects["train"].isdisjoint(split_subjects["test"])
    assert split_subjects["validation"].isdisjoint(split_subjects["test"])

    original_subjects = {
        split_name: {
            record["subject_id"]
            for record in records
            if record["original_split"] == split_name
        }
        for split_name in ("train", "test")
    }
    original_overlap = original_subjects["train"] & original_subjects["test"]

    summary: dict[str, Any] = {
        "seed": seed,
        "requested_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "totals": {
            "subjects": len(assignment),
            "scans": len({record["scan_id"] for record in records}),
            "slices": len(records),
        },
        "original_subject_overlap": len(original_overlap),
        "new_subject_overlap": 0,
        "splits": {},
    }

    for split_name in ("train", "validation", "test"):
        split_records = [
            record
            for record in records
            if assignment[record["subject_id"]] == split_name
        ]
        split_ids = split_subjects[split_name]
        summary["splits"][split_name] = {
            "subjects": len(split_ids),
            "subjects_AD": sum(
                label_by_subject[subject_id] == 2 for subject_id in split_ids
            ),
            "subjects_NC": sum(
                label_by_subject[subject_id] == 0 for subject_id in split_ids
            ),
            "scans": len({record["scan_id"] for record in split_records}),
            "slices": len(split_records),
        }

    scan_slice_counts = Counter(record["scan_id"] for record in records)
    irregular_scans = {
        scan_id: count
        for scan_id, count in scan_slice_counts.items()
        if count != 20
    }
    summary["scans_not_having_20_slices"] = irregular_scans

    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote: {subject_path}")
    print(f"Wrote: {slice_path}")
    print(f"Wrote: {summary_path}")


def main() -> None:
    """Run the audit and manifest generation workflow."""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio)
    metadata = load_metadata(args.metadata)
    records = collect_slice_records(args.data_root, metadata)
    subject_labels = validate_subject_labels(records)
    assignment = stratified_subject_split(
        subject_labels=subject_labels,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_manifests(
        records=records,
        assignment=assignment,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
