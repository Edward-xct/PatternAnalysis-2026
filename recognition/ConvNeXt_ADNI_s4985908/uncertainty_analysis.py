"""Compare uncertainty, triage behaviour and significance across the three models.

This script answers the project's engineering dilemma directly: does the
ConvNeXt quantify uncertainty better than the SmallCNN baseline, and are its
gains statistically meaningful? It reads the output folder that ``evaluate.py``
writes for each model (``evaluation.json``, ``test_calibrated_predictions.csv``,
``test_reliability.csv`` and ``test_risk_coverage.csv``) and produces:

* ``model_uncertainty.json`` and ``model_uncertainty.md``: calibration,
  confidence-versus-correctness, reject-option, significance, error-overlap and
  failure-case tables;
* ``model_reliability.png/.pdf``: raw and calibrated reliability per model,
  with the number of patients in every bin;
* ``model_risk_coverage.png/.pdf``: calibrated risk-coverage curves of all
  three models with their frozen validation-selected operating points.

Subject identifiers are used only to pair the same patient across models.
They are never written to the outputs, so the outputs can be committed.

Example:
    python uncertainty_analysis.py \
        --smallcnn-results outputs/final-results-smallcnn \
        --scratch-results outputs/final-results-convnext \
        --pretrained-results outputs/final-results-convnext-imagenet-own \
        --output-dir figures/report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

MODEL_ORDER = ("smallcnn", "scratch", "pretrained")
MODEL_LABELS = {
    "smallcnn": "SmallCNN",
    "scratch": "ConvNeXt (scratch)",
    "pretrained": "ConvNeXt (ImageNet)",
}
MODEL_COLOURS = {"smallcnn": "#0072B2", "scratch": "#D55E00", "pretrained": "#009E73"}
HIGH_CONFIDENCE = 0.90


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Read calibrated test predictions keyed by subject."""
    patients: dict[str, dict[str, Any]] = {}
    for row in load_csv(path):
        logit_nc, logit_ad = float(row["logit_NC"]), float(row["logit_AD"])
        raw_probability_ad = 1.0 / (1.0 + math.exp(logit_nc - logit_ad))
        patients[row["subject_id"]] = {
            "target": int(row["target"]),
            "probability_ad": float(row["calibrated_probability_AD"]),
            "confidence": float(row["calibrated_confidence"]),
            "raw_confidence": max(raw_probability_ad, 1.0 - raw_probability_ad),
            "correct": int(row["calibrated_correct"]),
            "triage": row["triage_decision"],
        }
    return patients


# --------------------------------------------------------------------------
# Statistics (pure Python so the script runs on a login node)
# --------------------------------------------------------------------------

def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Mann-Whitney estimate of P(score_positive > score_negative)."""
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positive_scores) * len(negative_scores))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return centre - half_width, centre + half_width


def exact_mcnemar(only_first_correct: int, only_second_correct: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    discordant = only_first_correct + only_second_correct
    if discordant == 0:
        return 1.0
    smaller = min(only_first_correct, only_second_correct)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def percentile_interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def model_metrics(
    predictions: dict[str, dict[str, Any]], subjects: Sequence[str]
) -> dict[str, float]:
    """Accuracy, AUROC and AD recall on a (possibly resampled) subject list."""
    positives = [predictions[s]["probability_ad"] for s in subjects if predictions[s]["target"] == 1]
    negatives = [predictions[s]["probability_ad"] for s in subjects if predictions[s]["target"] == 0]
    ad_correct = sum(predictions[s]["correct"] for s in subjects if predictions[s]["target"] == 1)
    return {
        "accuracy": sum(predictions[s]["correct"] for s in subjects) / len(subjects),
        "auroc": auroc(positives, negatives),
        "recall_AD": ad_correct / len(positives),
    }


def bootstrap(
    predictions: dict[str, dict[str, dict[str, Any]]],
    subjects: list[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Class-stratified patient bootstrap of metrics and paired differences."""
    rng = random.Random(seed)
    reference = predictions["smallcnn"]
    ad_subjects = [s for s in subjects if reference[s]["target"] == 1]
    nc_subjects = [s for s in subjects if reference[s]["target"] == 0]
    samples: dict[str, dict[str, list[float]]] = {
        key: {"accuracy": [], "auroc": [], "recall_AD": []} for key in MODEL_ORDER
    }
    differences: dict[str, dict[str, list[float]]] = {
        f"pretrained_minus_{key}": {"accuracy": [], "auroc": []}
        for key in ("smallcnn", "scratch")
    }
    for _ in range(resamples):
        resample = [rng.choice(ad_subjects) for _ in ad_subjects] + [
            rng.choice(nc_subjects) for _ in nc_subjects
        ]
        values = {key: model_metrics(predictions[key], resample) for key in MODEL_ORDER}
        for key in MODEL_ORDER:
            for metric, value in values[key].items():
                samples[key][metric].append(value)
        for baseline in ("smallcnn", "scratch"):
            for metric in ("accuracy", "auroc"):
                differences[f"pretrained_minus_{baseline}"][metric].append(
                    values["pretrained"][metric] - values[baseline][metric]
                )
    return {
        "resamples": resamples,
        "seed": seed,
        "metric_ci95": {
            key: {metric: percentile_interval(v) for metric, v in metrics.items()}
            for key, metrics in samples.items()
        },
        "paired_difference": {
            name: {
                metric: {
                    "ci95": percentile_interval(v),
                    "fraction_not_positive": sum(x <= 0 for x in v) / len(v),
                }
                for metric, v in metrics.items()
            }
            for name, metrics in differences.items()
        },
    }


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def calibration_summary(
    evaluation: dict[str, Any],
    reliability: list[dict[str, str]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Probability quality and whether confidence separates right from wrong."""
    raw_high = [r for r in reliability if r["variant"] == "raw" and float(r["lower"]) >= HIGH_CONFIDENCE]
    raw_high_count = sum(int(r["count"]) for r in raw_high)
    raw_high_correct = sum(
        int(r["count"]) * float(r["accuracy"]) for r in raw_high if int(r["count"])
    )
    correct = [p["confidence"] for p in predictions.values() if p["correct"]]
    wrong = [p["confidence"] for p in predictions.values() if not p["correct"]]
    return {
        "temperature": evaluation["calibration"]["temperature"],
        "raw": {k: evaluation["test"]["raw"][k] for k in ("ece", "nll", "brier")},
        "calibrated": {k: evaluation["test"]["calibrated"][k] for k in ("ece", "nll", "brier")},
        "raw_patients_confidence_at_least_0.9": raw_high_count,
        "raw_accuracy_when_confidence_at_least_0.9": (
            raw_high_correct / raw_high_count if raw_high_count else None
        ),
        # For two classes, temperature scaling does not change the ranking of
        # confidences, so this AUROC is identical before and after calibration.
        "confidence_auroc_for_correctness": auroc(correct, wrong),
        "mean_confidence_correct": sum(correct) / len(correct),
        "mean_confidence_wrong": sum(wrong) / len(wrong),
        "errors": len(wrong),
        "errors_raw_confidence_at_least_0.9": sum(
            p["raw_confidence"] >= HIGH_CONFIDENCE for p in predictions.values() if not p["correct"]
        ),
        "errors_calibrated_confidence_at_least_0.9": sum(c >= HIGH_CONFIDENCE for c in wrong),
    }


def selective_summary(
    evaluation: dict[str, Any], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Frozen reject-option behaviour, including clinically costly outcomes."""
    selective = evaluation["test"]["selective"]
    total_ad = sum(p["target"] == 1 for p in predictions.values())
    total_nc = sum(p["target"] == 0 for p in predictions.values())
    total_errors = sum(not p["correct"] for p in predictions.values())
    accepted_errors = int(selective["fn"]) + int(selective["fp"])
    return {
        "threshold_NC": selective["threshold_CN"],
        "threshold_AD": selective["threshold_AD"],
        "one_sided": float(selective["threshold_AD"]) <= 0.5,
        "coverage": selective["coverage"],
        "referred": int(selective["referred_patients"]),
        "selective_accuracy": selective["selective_accuracy"],
        "AD_auto_cleared": int(selective["fn"]),
        "AD_auto_cleared_rate": int(selective["fn"]) / total_ad,
        "NC_auto_flagged": int(selective["fp"]),
        "NC_auto_flagged_rate": int(selective["fp"]) / total_nc,
        "errors_total": total_errors,
        "errors_referred": total_errors - accepted_errors,
    }


def significance_summary(
    predictions: dict[str, dict[str, dict[str, Any]]], subjects: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {"wilson_ci95": {}, "mcnemar": {}}
    for key in MODEL_ORDER:
        model = predictions[key]
        correct = sum(model[s]["correct"] for s in subjects)
        ad = [s for s in subjects if model[s]["target"] == 1]
        result["wilson_ci95"][key] = {
            "accuracy": wilson_interval(correct, len(subjects)),
            "recall_AD": wilson_interval(sum(model[s]["correct"] for s in ad), len(ad)),
        }
    for baseline in ("smallcnn", "scratch"):
        only_pretrained = sum(
            predictions["pretrained"][s]["correct"] and not predictions[baseline][s]["correct"]
            for s in subjects
        )
        only_baseline = sum(
            predictions[baseline][s]["correct"] and not predictions["pretrained"][s]["correct"]
            for s in subjects
        )
        result["mcnemar"][f"pretrained_vs_{baseline}"] = {
            "only_pretrained_correct": only_pretrained,
            "only_baseline_correct": only_baseline,
            "p_value": exact_mcnemar(only_pretrained, only_baseline),
        }
    return result


def overlap_summary(
    predictions: dict[str, dict[str, dict[str, Any]]], subjects: list[str]
) -> dict[str, Any]:
    errors = {key: {s for s in subjects if not predictions[key][s]["correct"]} for key in MODEL_ORDER}
    pretrained = errors["pretrained"]
    shared = pretrained & errors["smallcnn"] & errors["scratch"]
    ad = {s for s in subjects if predictions["pretrained"][s]["target"] == 1}
    return {
        "errors": {key: len(v) for key, v in errors.items()},
        "pretrained_errors_shared_by_both_baselines": len(shared),
        "pretrained_false_negatives": len(pretrained & ad),
        "pretrained_false_negatives_missed_by_all_three": len(shared & ad),
        "pretrained_errors_unique": len(pretrained - errors["smallcnn"] - errors["scratch"]),
    }


def failure_case_summary(
    predictions: dict[str, dict[str, dict[str, Any]]], subjects: list[str]
) -> list[dict[str, Any]]:
    """Cases A-E: the rule used by failure_cases.py (four most confident false
    negatives plus the most confident false positive of the final model)."""
    final = predictions["pretrained"]
    wrong = [s for s in subjects if not final[s]["correct"]]
    by_confidence = lambda s: (-final[s]["confidence"], s)  # noqa: E731
    false_negatives = sorted((s for s in wrong if final[s]["target"] == 1), key=by_confidence)
    false_positives = sorted((s for s in wrong if final[s]["target"] == 0), key=by_confidence)
    ordered = false_negatives[:1] + false_positives[:1] + false_negatives[1:4]
    cases = []
    for letter, subject in zip("ABCDE", ordered):
        cases.append(
            {
                "case": letter,
                "error": "AD -> NC" if final[subject]["target"] == 1 else "NC -> AD",
                **{
                    f"{key}_probability_AD": predictions[key][subject]["probability_ad"]
                    for key in MODEL_ORDER
                },
                **{f"{key}_triage": predictions[key][subject]["triage"] for key in MODEL_ORDER},
            }
        )
    referred_fn = sorted(
        final[s]["probability_ad"]
        for s in false_negatives
        if final[s]["triage"] == "REFER"
    )
    return cases, referred_fn


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_reliability(
    reliability: dict[str, list[dict[str, str]]],
    calibration: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, grid = plt.subplots(
        2, 3, figsize=(13, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    width = 0.022
    for column, key in enumerate(MODEL_ORDER):
        axis, histogram = grid[0, column], grid[1, column]
        for variant, colour, shift in (("raw", "#D55E00", -width / 2), ("calibrated", "#0072B2", width / 2)):
            all_rows = [r for r in reliability[key] if r["variant"] == variant]
            rows = [r for r in all_rows if int(r["count"])]
            axis.plot(
                [float(r["mean_confidence"]) for r in rows],
                [float(r["accuracy"]) for r in rows],
                color=colour, marker="o", markersize=5, linewidth=1.5,
                label=variant.capitalize(),
            )
            centres = [(float(r["lower"]) + float(r["upper"])) / 2 for r in all_rows]
            counts = [int(r["count"]) for r in all_rows]
            bars = histogram.bar([c + shift for c in centres], counts, width, color=colour, alpha=0.85)
            histogram.bar_label(
                bars, labels=[str(c) if c else "" for c in counts], fontsize=7, padding=1,
            )
        axis.plot((0.5, 1.0), (0.5, 1.0), linestyle="--", color="#555555", linewidth=1)
        stats = calibration[key]
        axis.set_title(
            f"{MODEL_LABELS[key]}\nT={stats['temperature']:.2f}, "
            f"ECE {stats['raw']['ece']:.3f} -> {stats['calibrated']['ece']:.3f}",
            fontsize=10,
        )
        axis.set_ylim(-0.05, 1.08)
        axis.grid(alpha=0.25)
        histogram.set_xlim(0.48, 1.02)
        histogram.set_ylim(0, 105)
        histogram.set_xlabel("Mean predicted confidence")
        histogram.grid(axis="y", alpha=0.25)
        if column:
            axis.tick_params(labelleft=False)
            histogram.tick_params(labelleft=False)
    grid[0, 0].set_ylabel("Observed patient accuracy")
    grid[1, 0].set_ylabel("Patients")
    grid[0, 0].legend(frameon=False, loc="lower right")
    figure.suptitle("Test reliability before and after temperature scaling", y=1.0)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"model_reliability.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_risk_coverage(
    curves: dict[str, list[dict[str, str]]],
    selective: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for key in MODEL_ORDER:
        rows = [r for r in curves[key] if r["variant"] == "calibrated"]
        axis.plot(
            [float(r["coverage"]) for r in rows],
            [float(r["selective_risk"]) for r in rows],
            color=MODEL_COLOURS[key], linewidth=1.6, label=MODEL_LABELS[key],
        )
        point = selective[key]
        axis.scatter(
            [point["coverage"]], [1 - point["selective_accuracy"]],
            color=MODEL_COLOURS[key], edgecolor="black", s=60, zorder=4,
        )
    axis.scatter([], [], color="white", edgecolor="black", s=60, label="Frozen reject rule")
    axis.set_xlim(0.2, 1.0)
    axis.set_ylim(0, 0.3)
    axis.set_xlabel("Coverage (fraction of patients decided automatically)")
    axis.set_ylabel("Selective risk (1 - accuracy)")
    axis.set_title("Test risk-coverage after calibration (most confident patients first)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"model_risk_coverage.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------
# Report tables
# --------------------------------------------------------------------------

def markdown_tables(summary: dict[str, Any]) -> str:
    cal, sel, sig = summary["calibration"], summary["selective"], summary["significance"]
    boot = summary["bootstrap"]
    label = MODEL_LABELS
    lines = ["## Calibration and confidence", ""]
    lines += [
        "| Model | T | ECE raw → cal | NLL raw → cal | Raw conf ≥ 0.9: patients (accuracy) | Confidence AUROC for correctness | Errors with conf ≥ 0.9, raw → cal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in MODEL_ORDER:
        c = cal[key]
        lines.append(
            f"| {label[key]} | {c['temperature']:.2f} | {c['raw']['ece']:.3f} → {c['calibrated']['ece']:.3f} "
            f"| {c['raw']['nll']:.3f} → {c['calibrated']['nll']:.3f} "
            f"| {c['raw_patients_confidence_at_least_0.9']} ({c['raw_accuracy_when_confidence_at_least_0.9']:.3f}) "
            f"| {c['confidence_auroc_for_correctness']:.3f} "
            f"| {c['errors_raw_confidence_at_least_0.9']} → {c['errors_calibrated_confidence_at_least_0.9']} of {c['errors']} |"
        )
    lines += ["", "## Frozen reject option", ""]
    lines += [
        "| Model | NC threshold | AD threshold | Coverage | Selective accuracy | AD auto-cleared | NC auto-flagged | Errors referred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in MODEL_ORDER:
        s = sel[key]
        lines.append(
            f"| {label[key]} | {s['threshold_NC']:.3f} | {s['threshold_AD']:.3f} | {s['coverage']:.3f} "
            f"| {s['selective_accuracy']:.3f} | {s['AD_auto_cleared']}/33 ({s['AD_auto_cleared_rate']:.1%}) "
            f"| {s['NC_auto_flagged']}/69 ({s['NC_auto_flagged_rate']:.1%}) "
            f"| {s['errors_referred']}/{s['errors_total']} |"
        )
    lines += ["", "## Uncertainty of the test metrics", ""]
    lines += [
        "| Model | Accuracy (95% Wilson CI) | AUROC (95% bootstrap CI) | AD recall (95% Wilson CI) |",
        "|---|---:|---:|---:|",
    ]
    for key in MODEL_ORDER:
        acc = summary["metrics"][key]
        w = sig["wilson_ci95"][key]
        b = boot["metric_ci95"][key]
        lines.append(
            f"| {label[key]} | {acc['accuracy']:.3f} ({w['accuracy'][0]:.3f}–{w['accuracy'][1]:.3f}) "
            f"| {acc['auroc']:.3f} ({b['auroc'][0]:.3f}–{b['auroc'][1]:.3f}) "
            f"| {acc['recall_AD']:.3f} ({w['recall_AD'][0]:.3f}–{w['recall_AD'][1]:.3f}) |"
        )
    lines += ["", "| ImageNet ConvNeXt minus | Accuracy difference (95% CI) | McNemar p | AUROC difference (95% CI) |", "|---|---:|---:|---:|"]
    for baseline in ("smallcnn", "scratch"):
        d = boot["paired_difference"][f"pretrained_minus_{baseline}"]
        m = sig["mcnemar"][f"pretrained_vs_{baseline}"]
        point_acc = summary["metrics"]["pretrained"]["accuracy"] - summary["metrics"][baseline]["accuracy"]
        point_auc = summary["metrics"]["pretrained"]["auroc"] - summary["metrics"][baseline]["auroc"]
        lines.append(
            f"| {label[baseline]} | {point_acc:+.3f} ({d['accuracy']['ci95'][0]:+.3f} to {d['accuracy']['ci95'][1]:+.3f}) "
            f"| {m['p_value']:.3f} ({m['only_pretrained_correct']} vs {m['only_baseline_correct']} discordant) "
            f"| {point_auc:+.3f} ({d['auroc']['ci95'][0]:+.3f} to {d['auroc']['ci95'][1]:+.3f}) |"
        )
    o = summary["error_overlap"]
    lines += [
        "",
        "## Error overlap",
        "",
        f"Errors: SmallCNN {o['errors']['smallcnn']}, scratch {o['errors']['scratch']}, "
        f"ImageNet {o['errors']['pretrained']}. ImageNet errors also made by both baselines: "
        f"{o['pretrained_errors_shared_by_both_baselines']}; ImageNet false negatives missed by all "
        f"three models: {o['pretrained_false_negatives_missed_by_all_three']}/{o['pretrained_false_negatives']}; "
        f"errors unique to ImageNet: {o['pretrained_errors_unique']}.",
        "",
        "## Failure cases across models (calibrated P(AD), triage)",
        "",
        "| Case | Error | SmallCNN | ConvNeXt scratch | ConvNeXt ImageNet |",
        "|---|---|---:|---:|---:|",
    ]
    for case in summary["failure_cases"]:
        cells = [
            f"{case[f'{key}_probability_AD']:.3f} ({case[f'{key}_triage']})" for key in MODEL_ORDER
        ]
        lines.append(f"| {case['case']} | {case['error']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Referred ImageNet false negatives, calibrated P(AD): "
        + ", ".join(f"{p:.3f}" for p in summary["referred_false_negative_probabilities"]),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-model uncertainty and significance analysis")
    for key in MODEL_ORDER:
        parser.add_argument(f"--{key}-results", type=Path, required=True,
                            help=f"evaluate.py output folder for {MODEL_LABELS[key]}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3710)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folders = {key: getattr(args, f"{key}_results") for key in MODEL_ORDER}
    evaluations = {key: load_json(folders[key] / "evaluation.json") for key in MODEL_ORDER}
    predictions = {
        key: load_predictions(folders[key] / "test_calibrated_predictions.csv") for key in MODEL_ORDER
    }
    reliability = {key: load_csv(folders[key] / "test_reliability.csv") for key in MODEL_ORDER}
    curves = {key: load_csv(folders[key] / "test_risk_coverage.csv") for key in MODEL_ORDER}

    subjects = sorted(predictions["pretrained"])
    for key in MODEL_ORDER:
        if sorted(predictions[key]) != subjects:
            raise ValueError(f"{key} predictions cover a different set of patients")
        if any(predictions[key][s]["target"] != predictions["pretrained"][s]["target"] for s in subjects):
            raise ValueError(f"{key} predictions disagree on patient labels")

    calibration = {
        key: calibration_summary(evaluations[key], reliability[key], predictions[key])
        for key in MODEL_ORDER
    }
    selective = {key: selective_summary(evaluations[key], predictions[key]) for key in MODEL_ORDER}
    cases, referred_fn = failure_case_summary(predictions, subjects)
    summary = {
        "patients": len(subjects),
        "metrics": {key: model_metrics(predictions[key], subjects) for key in MODEL_ORDER},
        "calibration": calibration,
        "selective": selective,
        "significance": significance_summary(predictions, subjects),
        "bootstrap": bootstrap(predictions, subjects, args.bootstrap, args.seed),
        "error_overlap": overlap_summary(predictions, subjects),
        "failure_cases": cases,
        "referred_false_negative_probabilities": referred_fn,
    }

    with (args.output_dir / "model_uncertainty.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    tables = markdown_tables(summary)
    (args.output_dir / "model_uncertainty.md").write_text(tables, encoding="utf-8")
    plot_reliability(reliability, calibration, args.output_dir)
    plot_risk_coverage(curves, selective, args.output_dir)
    print(tables)


if __name__ == "__main__":
    main()
