"""Export slice- and patient-level predictions from an ADNI checkpoint.

Training operates on 2D MRI slices, but the project decision unit is a
patient. This script therefore saves both individual slice predictions and
patient predictions obtained by averaging every available slice logit for a
subject before applying softmax. The held-out test set requires an explicit
``--allow-test`` flag to reduce accidental test-set peeking during development.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from dataset import create_dataloader
from modules import build_model, count_trainable_parameters


CLASS_NAMES = {0: "NC", 1: "AD"}
INFERENCE_SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export patient-level ADNI predictions from a checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "artifacts" / "slice_manifest.csv",
    )
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="validation")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Development smoke-test limit; never allowed for the test split.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock held-out test inference after model selection.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "The held-out test split is locked. Use --allow-test only after "
            "the model, calibration method, and thresholds are frozen."
        )
    if args.split == "test" and args.max_batches is not None:
        raise ValueError("Partial test inference is forbidden; remove --max-batches")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    required = {"model_name", "model_state_dict"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    return payload


def build_checkpoint_model(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("Checkpoint config must be a dictionary")
    model_name = str(checkpoint["model_name"])
    dropout = float(config.get("dropout", 0.30))
    model = build_model(model_name, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _binary_entropy(probability_ad: float) -> float:
    epsilon = 1e-12
    probability = min(max(probability_ad, epsilon), 1.0 - epsilon)
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Run inference and aggregate mean slice logits within each patient."""
    slice_rows: list[dict[str, Any]] = []
    logits_by_subject: dict[str, list[Tensor]] = defaultdict(list)
    scans_by_subject: dict[str, set[str]] = defaultdict(set)
    targets_by_subject: dict[str, int] = {}
    forward_seconds = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        _synchronize(device)
        start_time = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        _synchronize(device)
        forward_seconds += time.perf_counter() - start_time

        logits_cpu = logits.float().cpu()
        probabilities_ad = torch.softmax(logits_cpu, dim=1)[:, 1]
        targets = batch["target"].cpu()

        for index in range(logits_cpu.shape[0]):
            target = int(targets[index].item())
            subject_id = str(batch["subject_id"][index])
            scan_id = str(batch["scan_id"][index])
            slice_index = int(batch["slice_index"][index])
            image_path = str(batch["image_path"][index])
            probability_ad = float(probabilities_ad[index].item())
            prediction = int(probability_ad >= 0.5)

            previous_target = targets_by_subject.get(subject_id)
            if previous_target is not None and previous_target != target:
                raise ValueError(f"Inconsistent labels for subject {subject_id}")
            targets_by_subject[subject_id] = target
            logits_by_subject[subject_id].append(logits_cpu[index])
            scans_by_subject[subject_id].add(scan_id)

            slice_rows.append(
                {
                    "subject_id": subject_id,
                    "scan_id": scan_id,
                    "slice_index": slice_index,
                    "image_path": image_path,
                    "target": target,
                    "target_class": CLASS_NAMES[target],
                    "logit_NC": float(logits_cpu[index, 0].item()),
                    "logit_AD": float(logits_cpu[index, 1].item()),
                    "probability_AD": probability_ad,
                    "predicted_target": prediction,
                    "predicted_class": CLASS_NAMES[prediction],
                    "confidence": max(probability_ad, 1.0 - probability_ad),
                    "correct": int(prediction == target),
                }
            )

    if not slice_rows:
        raise RuntimeError("Inference loader yielded zero slices")

    patient_rows: list[dict[str, Any]] = []
    for subject_id in sorted(logits_by_subject):
        mean_logits = torch.stack(logits_by_subject[subject_id]).mean(dim=0)
        probability_ad = float(torch.softmax(mean_logits, dim=0)[1].item())
        prediction = int(probability_ad >= 0.5)
        target = targets_by_subject[subject_id]
        patient_rows.append(
            {
                "subject_id": subject_id,
                "target": target,
                "target_class": CLASS_NAMES[target],
                "mean_logit_NC": float(mean_logits[0].item()),
                "mean_logit_AD": float(mean_logits[1].item()),
                "probability_AD": probability_ad,
                "predicted_target": prediction,
                "predicted_class": CLASS_NAMES[prediction],
                "confidence": max(probability_ad, 1.0 - probability_ad),
                "entropy_nats": _binary_entropy(probability_ad),
                "correct": int(prediction == target),
                "slice_count": len(logits_by_subject[subject_id]),
                "scan_count": len(scans_by_subject[subject_id]),
            }
        )

    slice_rows.sort(
        key=lambda row: (row["subject_id"], row["scan_id"], row["slice_index"])
    )
    return slice_rows, patient_rows, forward_seconds


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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model, checkpoint_config = build_checkpoint_model(checkpoint, device)
    image_size = args.image_size or int(checkpoint_config.get("image_size", 224))
    amp_enabled = device.type == "cuda" and not args.disable_amp

    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "outputs"
            / "predictions"
            / f"{args.checkpoint.parent.name}-{args.split}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, loader = create_dataloader(
        manifest_path=args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced_training=False,
        verify_paths=args.verify_paths,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    slice_rows, patient_rows, forward_seconds = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        amp_enabled=amp_enabled,
        max_batches=args.max_batches,
    )
    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    slice_path = args.output_dir / f"{args.split}_slice_predictions.csv"
    patient_path = args.output_dir / f"{args.split}_patient_predictions.csv"
    summary_path = args.output_dir / f"{args.split}_prediction_summary.json"
    write_csv(slice_path, slice_rows)
    write_csv(patient_path, patient_rows)

    summary = {
        "amp_enabled": amp_enabled,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "complete_split": args.max_batches is None,
        "device": str(device),
        "forward_seconds": forward_seconds,
        "image_size": image_size,
        "milliseconds_per_slice": 1000.0 * forward_seconds / len(slice_rows),
        "model": str(checkpoint["model_name"]),
        "parameters": count_trainable_parameters(model),
        "patient_output": str(patient_path),
        "patients_processed": len(patient_rows),
        "peak_gpu_vram_gb": peak_vram_gb,
        "slice_output": str(slice_path),
        "slices_available": len(dataset),
        "slices_processed": len(slice_rows),
        "split": args.split,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
