"""Train and validate ADNI classifiers with patient-level evaluation.

The optimization unit is a 2D MRI slice, while all primary validation metrics
are computed after averaging slice logits for each subject. The held-out test
split is untouched by default and is evaluated only with an explicit flag.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from dataset import create_dataloader
from modules import build_model, count_trainable_parameters


CLASS_NAMES = {0: "NC", 1: "AD"}


def parse_args() -> argparse.Namespace:
    """Parse training configuration."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train an ADNI classifier with patient-level validation"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "artifacts" / "slice_manifest.csv",
    )
    parser.add_argument(
        "--model",
        default="small_cnn",
        choices=("small_cnn", "convnext_tiny"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "small_cnn",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--disable-balanced-sampling", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test split once after training finishes.",
    )
    args = parser.parse_args()

    positive_integer_fields = (
        "epochs",
        "batch_size",
        "image_size",
        "log_interval",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    for field in ("max_train_batches", "max_val_batches"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    return args


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested accelerator and fail clearly when unavailable."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch.device(requested)


def _binary_auroc(targets: list[int], scores: list[float]) -> float | None:
    """Compute binary AUROC using average ranks, including tied scores."""
    positives = sum(target == 1 for target in targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, targets), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        group_end = index + 1
        while group_end < len(ordered) and ordered[group_end][0] == ordered[index][0]:
            group_end += 1
        average_rank = ((index + 1) + group_end) / 2.0
        positive_rank_sum += average_rank * sum(
            target == 1 for _, target in ordered[index:group_end]
        )
        index = group_end

    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return float(auc)


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_metrics(
    targets: list[int], probabilities_ad: list[float]
) -> dict[str, float | int | None]:
    """Compute patient-level classification metrics at threshold 0.5."""
    if len(targets) != len(probabilities_ad) or not targets:
        raise ValueError("Targets and probabilities must be non-empty and equally sized")

    predictions = [int(probability >= 0.5) for probability in probabilities_ad]
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))

    precision_nc = _safe_divide(tn, tn + fn)
    recall_nc = _safe_divide(tn, tn + fp)
    f1_nc = _safe_divide(
        2 * precision_nc * recall_nc,
        precision_nc + recall_nc,
    )
    precision_ad = _safe_divide(tp, tp + fp)
    recall_ad = _safe_divide(tp, tp + fn)
    f1_ad = _safe_divide(
        2 * precision_ad * recall_ad,
        precision_ad + recall_ad,
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
        "auroc": _binary_auroc(targets, probabilities_ad),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def aggregate_patient_logits(
    logits: Tensor,
    targets: Tensor,
    subject_ids: Iterable[str],
) -> tuple[list[int], list[float]]:
    """Average slice logits for each patient and return AD probabilities."""
    logits_by_subject: dict[str, list[Tensor]] = defaultdict(list)
    target_by_subject: dict[str, int] = {}

    for slice_logits, slice_target, subject_id in zip(
        logits.cpu(), targets.cpu(), subject_ids
    ):
        target = int(slice_target.item())
        if subject_id in target_by_subject and target_by_subject[subject_id] != target:
            raise ValueError(f"Inconsistent labels for subject {subject_id}")
        target_by_subject[subject_id] = target
        logits_by_subject[subject_id].append(slice_logits)

    patient_targets: list[int] = []
    patient_probabilities_ad: list[float] = []
    for subject_id in sorted(logits_by_subject):
        mean_logits = torch.stack(logits_by_subject[subject_id]).mean(dim=0)
        probability_ad = torch.softmax(mean_logits, dim=0)[1].item()
        patient_targets.append(target_by_subject[subject_id])
        patient_probabilities_ad.append(float(probability_ad))
    return patient_targets, patient_probabilities_ad


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
    log_interval: int,
) -> dict[str, float | int]:
    """Train for one epoch and return slice-level loss statistics."""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        if (batch_index + 1) % log_interval == 0:
            print(
                f"train batch={batch_index + 1} "
                f"mean_loss={loss_sum / sample_count:.6f}"
            )

    if sample_count == 0:
        raise RuntimeError("Training loader yielded zero samples")
    elapsed = time.perf_counter() - start_time
    return {
        "loss": loss_sum / sample_count,
        "slices": sample_count,
        "seconds": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate slice loss and primary patient-level metrics."""
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_subject_ids: list[str] = []
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_subject_ids.extend(str(value) for value in batch["subject_id"])

    if sample_count == 0:
        raise RuntimeError("Evaluation loader yielded zero samples")

    patient_targets, patient_probabilities_ad = aggregate_patient_logits(
        logits=torch.cat(all_logits, dim=0),
        targets=torch.cat(all_targets, dim=0),
        subject_ids=all_subject_ids,
    )
    metrics = compute_binary_metrics(patient_targets, patient_probabilities_ad)
    elapsed = time.perf_counter() - start_time
    return {
        "slice_loss": loss_sum / sample_count,
        "slices": sample_count,
        "seconds": elapsed,
        **metrics,
    }


def _json_ready(value: Any) -> Any:
    """Convert Path and non-finite values to JSON-safe representations."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_dataset, train_loader = create_dataloader(
        manifest_path=args.manifest,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced_training=not args.disable_balanced_sampling,
        verify_paths=args.verify_paths,
    )
    validation_dataset, validation_loader = create_dataloader(
        manifest_path=args.manifest,
        split="validation",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced_training=False,
        verify_paths=args.verify_paths,
    )

    model = build_model(args.model, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_config = {
        **vars(args),
        "resolved_device": str(device),
        "amp_enabled": amp_enabled,
        "train_slices": len(train_dataset),
        "validation_slices": len(validation_dataset),
        "trainable_parameters": count_trainable_parameters(model),
        "class_names": CLASS_NAMES,
    }
    save_json(args.output_dir / "config.json", run_config)
    print(json.dumps(_json_ready(run_config), indent=2, sort_keys=True))

    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_checkpoint = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
            log_interval=args.log_interval,
        )
        validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=args.max_val_batches,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        save_json(args.output_dir / "history.json", history)
        print(json.dumps(_json_ready(epoch_record), indent=2, sort_keys=True))

        macro_f1 = float(validation_metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": _json_ready(run_config),
                },
                best_checkpoint,
            )

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    result: dict[str, Any] = {
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint": str(best_checkpoint),
        "peak_gpu_vram_gb": peak_vram_gb,
        "epochs_completed": args.epochs,
    }

    if args.evaluate_test:
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_loader = create_dataloader(
            manifest_path=args.manifest,
            split="test",
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            seed=args.seed,
            balanced_training=False,
            verify_paths=args.verify_paths,
        )
        result["test"] = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

    save_json(args.output_dir / "result.json", result)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
