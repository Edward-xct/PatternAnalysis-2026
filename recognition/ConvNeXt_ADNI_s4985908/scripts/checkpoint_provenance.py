"""Record where a reported checkpoint came from.

The final results in the README come from evaluating one frozen checkpoint,
not from re-running training with the current code. This script writes a small
JSON record that lets a reader trace that checkpoint: its SHA-256 hash and size,
the epoch stored inside it, the training hyperparameters from the run's
``config.json``, and the job ID, host, timestamps and Git commit printed by the
Slurm script into the job log. Machine-specific absolute paths are not copied.

Example:
    python scripts/checkpoint_provenance.py \
        --run-dir outputs/convnext-imagenet-seed3710-job650741 \
        --slurm-log logs/convnext-imagenet-650741.out \
        --output figures/report/final_checkpoint_provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_KEYS = (
    "model", "epochs", "batch_size", "learning_rate", "weight_decay", "dropout",
    "seed", "image_size", "amp_enabled", "num_workers", "disable_balanced_sampling",
    "train_slices", "validation_slices", "trainable_parameters",
)
LOG_KEYS = ("job_id", "host", "started", "finished", "git_commit")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_epoch(path: Path) -> Any:
    """Read the epoch stored in the checkpoint, if PyTorch is available."""
    try:
        import torch
    except ImportError:
        return "unavailable (PyTorch not installed)"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload.get("epoch") if isinstance(payload, dict) else None


def log_fields(path: Path | None) -> dict[str, Any]:
    """Collect ``key=value`` lines written by the Slurm scripts."""
    if path is None or not path.is_file():
        return {"available": False}
    fields: dict[str, Any] = {"available": True, "log_file": path.name}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() in LOG_KEYS and key.strip() not in fields:
            fields[key.strip()] = value.strip()
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    record = {
        "run": args.run_dir.name,
        "checkpoint_file": "best.pt",
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_of(checkpoint),
        "checkpoint_epoch": checkpoint_epoch(checkpoint),
        "training_config": {key: config.get(key) for key in CONFIG_KEYS},
        "training_job_log": log_fields(args.slurm_log),
        "note": (
            "Trained with torchvision's ConvNeXt class; the reported results were "
            "obtained by converting this frozen checkpoint into modules.ConvNeXtTiny "
            "and re-evaluating it (see checkpoint_equivalence_*.json). Retraining "
            "with the current code is not expected to reproduce it bit for bit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
