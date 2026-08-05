#!/usr/bin/env python3
"""Regenerate active paper tables and figures from compact aggregate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PHASE1_TABLE_COLUMNS = (
    "run_id",
    "stage",
    "mask_preset",
    "selected_epoch",
    "validation_loss",
    "effective_rank_ratio",
    "wrong_context_gap",
    "closed_set_identity_accuracy",
    "held_out_retrieval_accuracy",
    "gait_balanced_accuracy",
)
LEGACY_GFC_PROTOCOL = "legacy_donor_excluded_v1"
GFC_NORMALIZATIONS = {
    "raw_retain_all",
    "raw_effective_rank",
    "pca_effective_rank",
}
CONTEXT_LABELS = {
    "cross_subject": "Different participant",
    "same_subject": "Same participant",
    "temporal_shuffle": "Temporal shuffle",
    "blank": "Blank context",
}
RANK_LABELS = {
    "online_full_view_pooled": "Online pooled recording",
    "ema_target_pre_norm_pooled": "EMA target pooled recording",
    "context_tokens_all_mask_groups": "Context tokens, all masks",
    "context_tokens_large_blocks": "Context tokens, large masks",
    "context_tokens_small_blocks": "Context tokens, small masks",
}


def _require_files(results_dir: Path) -> tuple[Path, Path, Path]:
    paths = (
        results_dir / "phase0_summary.json",
        results_dir / "phase1_summary.csv",
        results_dir / "context_diagnosis.json",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing compact result inputs: " + ", ".join(missing))
    return paths


def _write_phase0_table(source: Path, output_dir: Path) -> Path:
    value = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for checkpoint in value["checkpoints"]:
        rows.append(
            {
                "run_id": value["run_id"],
                "checkpoint": checkpoint["label"],
                "epoch": checkpoint["epoch"],
                "validation_loss": checkpoint["validation_loss"],
                "subject_balanced_loss": checkpoint["subject_balanced_loss"],
                "effective_rank": checkpoint["effective_rank"],
                "effective_rank_ratio": checkpoint["effective_rank_ratio"],
                "wrong_context_gap": checkpoint["wrong_context_gap"],
                "closed_set_identity_accuracy": checkpoint[
                    "closed_set_identity_accuracy"
                ],
                "held_out_retrieval_accuracy": checkpoint[
                    "held_out_retrieval_accuracy"
                ],
                "gait_balanced_accuracy": checkpoint["gait_balanced_accuracy"],
            }
        )
    destination = output_dir / "phase0_table.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def _write_phase1_outputs(source: Path, output_dir: Path) -> tuple[Path, Path]:
    table = pd.read_csv(source)
    missing = [column for column in PHASE1_TABLE_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError("phase1 summary is missing columns: " + ", ".join(missing))
    table_path = output_dir / "phase1_table.csv"
    table.loc[:, PHASE1_TABLE_COLUMNS].to_csv(table_path, index=False)

    positions = np.arange(len(table))
    colors = np.where(table["stage"].astype(str) == "A", "#35618d", "#bb5a3a")
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].bar(positions, table["validation_loss"], color=colors)
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Checkpoint diagnostics disagree across Phase 1")
    axes[1].bar(positions, 100.0 * table["effective_rank_ratio"], color=colors)
    axes[1].set_ylabel("Effective-rank ratio (%)")
    axes[1].set_xticks(positions, table["run_id"], rotation=45, ha="right")
    axes[1].set_xlabel("Run")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(
        handles=(
            Patch(facecolor="#35618d", label="Stage A"),
            Patch(facecolor="#bb5a3a", label="Stage B"),
        ),
        frameon=False,
    )
    figure_path = output_dir / "phase1_diagnostics.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return table_path, figure_path


def _write_context_figure(source: Path, output_dir: Path) -> Path:
    value = json.loads(source.read_text(encoding="utf-8"))
    context = pd.DataFrame(value["context_substitution"])
    ranks = pd.DataFrame(value["representation_rank"])
    context = context.loc[context["condition"] != "self"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(
        context["condition"].map(lambda value: CONTEXT_LABELS.get(value, value)),
        context["participant_mean_gap"],
        color="#35618d",
    )
    axes[0].set_ylabel("Participant-mean loss gap")
    axes[0].set_title("Context substitution")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].barh(
        ranks["representation"].map(lambda value: RANK_LABELS.get(value, value)),
        100.0 * ranks["ratio"],
        color="#bb5a3a",
    )
    axes[1].set_xlabel("Effective-rank ratio (%)")
    axes[1].set_title("Token and pooled breadth")
    axes[1].grid(axis="x", alpha=0.25)
    path = output_dir / "context_diagnosis.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _gfc_summaries(results_dir: Path, output_dir: Path) -> list[tuple[str, dict]]:
    summaries = []
    output_resolved = output_dir.resolve()
    for path in sorted(results_dir.rglob("summary.json")):
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        value = json.loads(path.read_text(encoding="utf-8"))
        if not {"learned", "shortcut", "learned_minus_shortcut"}.issubset(value):
            continue
        summaries.append((str(path.parent.relative_to(results_dir)), value))
    return summaries


def _validate_gfc_summaries(summaries: list[tuple[str, dict]]) -> None:
    by_analysis: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    for label, value in summaries:
        protocol = str(value.get("protocol", ""))
        model_label = str(value.get("model_label", ""))
        split = str(value.get("split", ""))
        normalization = str(value.get("normalization", ""))
        if protocol != LEGACY_GFC_PROTOCOL:
            raise ValueError(
                f"legacy GFC summary {label!r} must declare protocol "
                f"{LEGACY_GFC_PROTOCOL!r}; v2 and mixed summaries cannot enter legacy tables"
            )
        if not model_label or not split or normalization not in GFC_NORMALIZATIONS:
            raise ValueError(
                f"invalid GFC model label, split, or normalization in {label!r}"
            )
        by_analysis.setdefault((model_label, split), []).append((label, value))
    for (model_label, split), group in by_analysis.items():
        normalizations = [str(value["normalization"]) for _, value in group]
        if len(normalizations) != len(set(normalizations)):
            raise ValueError(
                f"duplicate GFC normalization for model {model_label!r}, split {split!r}"
            )
        if set(normalizations) != GFC_NORMALIZATIONS:
            missing = sorted(GFC_NORMALIZATIONS - set(normalizations))
            raise ValueError(
                f"GFC model {model_label!r}, split {split!r} must include primary "
                f"and both sensitivities; missing {missing}"
            )
        reference = group[0][1]
        reference_interval = reference["learned_minus_shortcut"]
        compatible = (
            reference["model_label"],
            reference["seed"],
            reference["feature_dimension"],
            reference["method_settings"],
            reference["training"],
            reference["evaluation"],
            reference_interval["confidence_level"],
            reference_interval["resamples"],
        )
        for label, value in group[1:]:
            interval = value["learned_minus_shortcut"]
            candidate = (
                value["model_label"],
                value["seed"],
                value["feature_dimension"],
                value["method_settings"],
                value["training"],
                value["evaluation"],
                interval["confidence_level"],
                interval["resamples"],
            )
            if candidate != compatible:
                raise ValueError(
                    f"GFC summary {label!r} is incompatible with the other analyses "
                    f"for model {model_label!r}, split {split!r}"
                )


def _write_gfc_outputs(
    summaries: list[tuple[str, dict]], output_dir: Path
) -> tuple[Path, Path] | None:
    if not summaries:
        return None
    _validate_gfc_summaries(summaries)
    rows = []
    for label, value in summaries:
        interval = value["learned_minus_shortcut"]
        rows.append(
            {
                "analysis": label,
                "protocol": value["protocol"],
                "model_label": value["model_label"],
                "split": value["split"],
                "normalization": value["normalization"],
                "participants": value["evaluation"]["participant_count"],
                "learned_top1": value["learned"]["top1"],
                "shortcut_top1": value["shortcut"]["top1"],
                "learned_minus_shortcut": interval["point_estimate"],
                "interval_lower": interval["confidence_interval"]["lower"],
                "interval_upper": interval["confidence_interval"]["upper"],
            }
        )
    table = pd.DataFrame(rows)
    normalization_order = {
        "raw_retain_all": 0,
        "raw_effective_rank": 1,
        "pca_effective_rank": 2,
    }
    table = table.sort_values(
        ["model_label", "split", "normalization"],
        key=lambda values: (
            values.map(normalization_order)
            if values.name == "normalization"
            else values
        ),
        kind="stable",
    ).reset_index(drop=True)
    table_path = output_dir / "legacy_gfc_table.csv"
    table.to_csv(table_path, index=False)
    positions = np.arange(len(table))
    estimates = table["learned_minus_shortcut"].to_numpy(dtype=float)
    lower = table["interval_lower"].to_numpy(dtype=float)
    upper = table["interval_upper"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(max(6.5, 1.3 * len(table)), 4.5), constrained_layout=True)
    axis.vlines(positions, lower, upper, color="#35618d", linewidth=1.5)
    axis.hlines(lower, positions - 0.06, positions + 0.06, color="#35618d")
    axis.hlines(upper, positions - 0.06, positions + 0.06, color="#35618d")
    axis.scatter(positions, estimates, color="#35618d", zorder=3)
    axis.axhline(0.0, color="black", linewidth=1)
    normalization_labels = {
        "raw_retain_all": "primary",
        "raw_effective_rank": "raw ER",
        "pca_effective_rank": "PCA ER",
    }
    plot_labels = [
        f"{model}\n{split}\n{normalization_labels[normalization]}"
        for model, split, normalization in zip(
            table["model_label"], table["split"], table["normalization"]
        )
    ]
    axis.set_xticks(positions, plot_labels, rotation=25, ha="right")
    axis.set_ylabel("Legacy learned minus shortcut top-1")
    axis.set_title("Legacy donor-excluded Grounded Factorial Completion")
    axis.grid(axis="y", alpha=0.25)
    figure_path = output_dir / "legacy_gfc_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return table_path, figure_path


def make_paper_results(results_dir: Path, output_dir: Path) -> list[Path]:
    """Build every maintained table and figure from compact result data."""

    phase0, phase1, context = _require_files(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [_write_phase0_table(phase0, output_dir)]
    generated.extend(_write_phase1_outputs(phase1, output_dir))
    generated.append(_write_context_figure(context, output_dir))
    gfc_outputs = _write_gfc_outputs(_gfc_summaries(results_dir, output_dir), output_dir)
    if gfc_outputs is not None:
        generated.extend(gfc_outputs)
        for stale_name in ("gfc_table.csv", "gfc_comparison.png"):
            (output_dir / stale_name).unlink(missing_ok=True)
    else:
        for stale_name in (
            "legacy_gfc_table.csv",
            "legacy_gfc_comparison.png",
            "gfc_table.csv",
            "gfc_comparison.png",
        ):
            (output_dir / stale_name).unlink(missing_ok=True)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = make_paper_results(args.results_dir, args.output_dir)
    print(json.dumps([str(path) for path in paths], indent=2))


if __name__ == "__main__":
    main()
