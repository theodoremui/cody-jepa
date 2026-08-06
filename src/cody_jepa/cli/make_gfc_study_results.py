#!/usr/bin/env python3
"""Render paper-ready GFC-v2 study outputs from aggregate-only result files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_TABLE_COLUMNS = (
    "model_label",
    "ladder",
    "rung",
    "checkpoint_id",
    "pool_seed",
    "optimization_seed",
    "unique_sequences",
    "training_exposure",
    "primary_analysis_id",
    "primary_ridge_alpha",
    "primary_normalization",
    "learned_top1",
    "learned_mrr",
    "shortcut_top1",
    "shortcut_mrr",
    "hard_control_top1",
    "hard_control_mrr",
    "soft_control_top1",
    "soft_control_mrr",
    "soft_control_target_probability",
    "soft_control_target_nll",
    "soft_temperature",
    "alpha_0_1_learned_top1",
    "alpha_10_learned_top1",
    "raw_effective_rank_learned_top1",
    "pca_effective_rank_learned_top1",
)
LADDER_TABLE_COLUMNS = (
    "ladder",
    "participant_count",
    "small_top1",
    "medium_top1",
    "large_top1",
    "full_top1",
    "full_minus_small",
    "participant_bootstrap_95_lower",
    "participant_bootstrap_95_upper",
)
RUNG_ORDER = ("small", "medium", "large", "full")
PRIMARY_ANALYSIS_ID = "raw_retain_all-alpha-1"
PRIMARY_NORMALIZATION = "raw_retain_all"
SCHEMA_VERSION = "gfc-v2-study-aggregate-v1"
PROTOCOL = "gfc_v2"
GALLERY = "retain_all_8"
ANALYSIS_FREEZE_TAG = "gfc-v2-analysis-freeze-v1"
ROLE_MAP_VERSION = "healthgait-gfc-v2-roles-v1"
PRIVATE_KEYS = {
    "subject_id",
    "subject_ids",
    "participant_id",
    "participant_ids",
    "participant",
    "participants",
    "participant_key",
    "participant_keys",
    "participant_rows",
    "source_video_id",
    "source_video_ids",
    "feature_path",
    "feature_paths",
    "checkpoint_path",
    "checkpoint_paths",
    "role_map_path",
}
OUTPUT_FILENAMES = (
    "gfc_study_run_table.csv",
    "gfc_study_ladder_contrasts.csv",
    "gfc_study_scaling.png",
    "gfc_study_scaling.pdf",
)


def _remove_stale_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILENAMES:
        path = output_dir / name
        if path.is_dir():
            raise ValueError(f"expected generated output to be a file, not a directory: {path}")
        path.unlink(missing_ok=True)


def _require_inputs(aggregate_dir: Path) -> tuple[Path, Path, Path]:
    paths = (
        aggregate_dir / "outcome_summary.json",
        aggregate_dir / "run_table.csv",
        aggregate_dir / "ladder_contrasts.csv",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing aggregate GFC study inputs: " + ", ".join(missing))
    return paths


def _check_private_keys(value: Any, location: str = "outcome_summary") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in PRIVATE_KEYS or normalized.endswith(("_path", "_paths")):
                raise ValueError(f"private field {key!r} is forbidden in {location}")
            _check_private_keys(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_private_keys(item, f"{location}[{index}]")


def _require_exact_columns(table: pd.DataFrame, expected: tuple[str, ...], label: str) -> None:
    actual = tuple(str(column) for column in table.columns)
    private = sorted(set(actual) & PRIVATE_KEYS)
    if private:
        raise ValueError(f"{label} contains forbidden private columns: {private}")
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unexpected = [column for column in actual if column not in expected]
        raise ValueError(
            f"{label} columns do not match the frozen aggregate schema; "
            f"missing={missing}, unexpected={unexpected}, order={list(actual)}"
        )


def _require_finite(table: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} contains nonfinite or nonnumeric values")


def _require_exact_integer(value: object, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} must be exactly {expected}")


def _require_summary_shape(summary: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol",
        "gallery",
        "queries_per_participant",
        "analysis_id",
        "ridge_alpha",
        "normalization",
        "revision",
        "cohort_roles",
        "inference",
        "artifacts",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"outcome_summary.json is missing fields: {missing}")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"study aggregate schema must be {SCHEMA_VERSION!r}")
    if summary["protocol"] != PROTOCOL:
        raise ValueError(
            f"study renderer accepts protocol {PROTOCOL!r} only; legacy and mixed inputs "
            "are forbidden"
        )
    if summary["gallery"] != GALLERY:
        raise ValueError(f"GFC-v2 paper results require gallery {GALLERY!r}")
    _require_exact_integer(
        summary["queries_per_participant"],
        16,
        "GFC-v2 queries per participant",
    )
    if summary["analysis_id"] != PRIMARY_ANALYSIS_ID:
        raise ValueError(f"primary analysis must be {PRIMARY_ANALYSIS_ID!r}")
    if not math.isclose(float(summary["ridge_alpha"]), 1.0):
        raise ValueError("primary ridge alpha must be 1.0")
    if summary["normalization"] != PRIMARY_NORMALIZATION:
        raise ValueError(f"primary normalization must be {PRIMARY_NORMALIZATION!r}")
    revision = summary["revision"]
    if (
        not isinstance(revision, dict)
        or not isinstance(revision.get("code_commit"), str)
        or not revision["code_commit"].strip()
        or revision.get("analysis_freeze_tag") != ANALYSIS_FREEZE_TAG
    ):
        raise ValueError("study aggregate has invalid frozen revision metadata")
    roles = summary["cohort_roles"]
    expected_roles = {
        "version": ROLE_MAP_VERSION,
        "fit_role": "development",
        "evaluation_role": "locked_outcome",
        "assigned_counts": {"development": 80, "locked_outcome": 318},
        "complete_counts": {"development": 76, "locked_outcome": 308},
        "excluded_counts": {"development": 4, "locked_outcome": 10},
    }
    if not isinstance(roles, dict) or any(
        roles.get(key) != value for key, value in expected_roles.items()
    ):
        raise ValueError("study aggregate has invalid frozen cohort role metadata")

    inference = summary["inference"]
    inference_required = {
        "ladder_count",
        "participant_count",
        "rung_order",
        "resolution",
        "mean_full_minus_small",
        "ladder_contrasts",
        "t_interval_95",
        "t_interval_90",
        "crossed_bootstrap_95",
        "superiority",
        "equivalence",
        "decision",
    }
    if not isinstance(inference, dict):
        raise ValueError("outcome_summary.inference must be an object")
    missing = sorted(inference_required - set(inference))
    if missing:
        raise ValueError(f"outcome_summary.inference is missing fields: {missing}")
    _require_exact_integer(inference["ladder_count"], 5, "study inference ladder count")
    participant_count = inference["participant_count"]
    if (
        isinstance(participant_count, bool)
        or not isinstance(participant_count, int)
        or participant_count != 308
    ):
        raise ValueError(
            "study inference must contain all 308 complete outcome participants"
        )
    if tuple(inference["rung_order"]) != RUNG_ORDER:
        raise ValueError(f"study rung order must be {list(RUNG_ORDER)}")
    if not math.isclose(float(inference["resolution"]), 1.0 / 16.0):
        raise ValueError("study decision resolution must be 1/16")
    if len(inference["ladder_contrasts"]) != 5:
        raise ValueError("outcome summary must preserve all five ladder contrasts")
    for name in ("t_interval_95", "t_interval_90", "crossed_bootstrap_95"):
        interval = inference[name]
        if not isinstance(interval, dict) or not {"confidence_level", "lower", "upper"} <= set(
            interval
        ):
            raise ValueError(f"outcome_summary.inference.{name} is not an interval")
        bounds = np.asarray([interval["lower"], interval["upper"]], dtype=float)
        if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"outcome_summary.inference.{name} has invalid bounds")


def _validate_tables(
    summary: dict[str, Any], run_table: pd.DataFrame, ladder_table: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    _require_exact_columns(run_table, RUN_TABLE_COLUMNS, "run_table.csv")
    _require_exact_columns(ladder_table, LADDER_TABLE_COLUMNS, "ladder_contrasts.csv")
    if len(run_table) != 20:
        raise ValueError(f"run_table.csv must contain exactly 20 rows, found {len(run_table)}")
    if len(ladder_table) != 5:
        raise ValueError(
            f"ladder_contrasts.csv must contain exactly five rows, found {len(ladder_table)}"
        )
    if run_table["model_label"].astype(str).duplicated().any():
        raise ValueError("run_table.csv model labels must be unique")
    if run_table["checkpoint_id"].astype(str).duplicated().any():
        raise ValueError("run_table.csv checkpoint IDs must be unique")

    ladder_order = [str(item["ladder"]) for item in summary["inference"]["ladder_contrasts"]]
    if len(ladder_order) != 5 or len(set(ladder_order)) != 5:
        raise ValueError("outcome summary ladder labels must be five unique values")
    if set(run_table["ladder"].astype(str)) != set(ladder_order):
        raise ValueError("run_table.csv ladders do not match outcome_summary.json")
    if set(ladder_table["ladder"].astype(str)) != set(ladder_order):
        raise ValueError("ladder_contrasts.csv ladders do not match outcome_summary.json")
    for ladder, group in run_table.groupby("ladder", sort=False):
        rungs = list(group["rung"].astype(str))
        if len(group) != 4 or set(rungs) != set(RUNG_ORDER):
            raise ValueError(f"ladder {ladder!r} must contain exactly one row per rung")

    if set(run_table["primary_analysis_id"].astype(str)) != {summary["analysis_id"]}:
        raise ValueError("run table primary analysis does not match outcome summary")
    if set(run_table["primary_normalization"].astype(str)) != {summary["normalization"]}:
        raise ValueError("run table primary normalization does not match outcome summary")
    alpha = pd.to_numeric(run_table["primary_ridge_alpha"], errors="coerce").to_numpy()
    if not np.isfinite(alpha).all() or not np.allclose(alpha, float(summary["ridge_alpha"])):
        raise ValueError("run table primary ridge alpha does not match outcome summary")

    numeric_run_columns = (
        "pool_seed",
        "optimization_seed",
        "unique_sequences",
        "training_exposure",
        "primary_ridge_alpha",
        "learned_top1",
        "learned_mrr",
        "shortcut_top1",
        "shortcut_mrr",
        "hard_control_top1",
        "hard_control_mrr",
        "soft_control_top1",
        "soft_control_mrr",
        "soft_control_target_probability",
        "soft_control_target_nll",
        "soft_temperature",
        "alpha_0_1_learned_top1",
        "alpha_10_learned_top1",
        "raw_effective_rank_learned_top1",
        "pca_effective_rank_learned_top1",
    )
    _require_finite(run_table, numeric_run_columns, "run_table.csv")
    _require_finite(ladder_table, LADDER_TABLE_COLUMNS[1:], "ladder_contrasts.csv")
    unit_run_columns = tuple(
        column
        for column in numeric_run_columns
        if column.endswith(("_top1", "_mrr", "_probability"))
    )
    for column in unit_run_columns:
        values = pd.to_numeric(run_table[column]).to_numpy(dtype=float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"run_table.csv.{column} must lie in [0, 1]")
    for column in ("small_top1", "medium_top1", "large_top1", "full_top1"):
        values = pd.to_numeric(ladder_table[column]).to_numpy(dtype=float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"ladder_contrasts.csv.{column} must lie in [0, 1]")
    if (pd.to_numeric(run_table["soft_control_target_nll"]) < 0.0).any():
        raise ValueError("soft control target NLL must be nonnegative")
    temperatures = pd.to_numeric(run_table["soft_temperature"]).to_numpy(dtype=float)
    if ((temperatures <= 0.001) | (temperatures >= 1000.0)).any():
        raise ValueError("soft temperatures must be strictly inside the frozen bounds")
    bounded_columns = (
        "learned_top1",
        "learned_mrr",
        "shortcut_top1",
        "shortcut_mrr",
        "hard_control_top1",
        "hard_control_mrr",
        "soft_control_top1",
        "soft_control_mrr",
        "soft_control_target_probability",
        "alpha_0_1_learned_top1",
        "alpha_10_learned_top1",
        "raw_effective_rank_learned_top1",
        "pca_effective_rank_learned_top1",
    )
    for column in bounded_columns:
        values = pd.to_numeric(run_table[column]).to_numpy(dtype=float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"run_table.csv.{column} must lie in [0, 1]")
    for column in ("unique_sequences", "training_exposure"):
        values = pd.to_numeric(run_table[column]).to_numpy(dtype=float)
        if (values <= 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"run_table.csv.{column} must contain positive integers")

    artifact_counts = summary["artifacts"]
    if artifact_counts.get("run_table") != "run_table.csv":
        raise ValueError("outcome summary must name run_table.csv as its aggregate run table")
    if artifact_counts.get("ladder_contrasts") != "ladder_contrasts.csv":
        raise ValueError(
            "outcome summary must name ladder_contrasts.csv as its aggregate ladder table"
        )
    _require_exact_integer(
        artifact_counts.get("run_table_rows"),
        20,
        "outcome summary aggregate run-table row count",
    )
    _require_exact_integer(
        artifact_counts.get("ladder_contrast_rows"),
        5,
        "outcome summary aggregate ladder row count",
    )
    participant_count = int(summary["inference"]["participant_count"])
    if set(pd.to_numeric(ladder_table["participant_count"]).astype(int)) != {
        participant_count
    }:
        raise ValueError("ladder participant counts do not match outcome summary")

    summary_contrasts = {
        str(item["ladder"]): float(item["full_minus_small"])
        for item in summary["inference"]["ladder_contrasts"]
    }
    table_contrasts = dict(
        zip(
            ladder_table["ladder"].astype(str),
            pd.to_numeric(ladder_table["full_minus_small"]).astype(float),
        )
    )
    if any(
        not math.isclose(table_contrasts[ladder], value, abs_tol=1e-12)
        for ladder, value in summary_contrasts.items()
    ):
        raise ValueError("ladder contrasts disagree between aggregate JSON and CSV")
    for row in ladder_table.itertuples(index=False):
        ladder_runs = run_table.loc[run_table["ladder"].astype(str) == str(row.ladder)]
        learned_by_rung = dict(
            zip(
                ladder_runs["rung"].astype(str),
                pd.to_numeric(ladder_runs["learned_top1"]).astype(float),
            )
        )
        for rung in RUNG_ORDER:
            if not math.isclose(
                learned_by_rung[rung], float(getattr(row, f"{rung}_top1")), abs_tol=1e-12
            ):
                raise ValueError(
                    f"learned top-1 for ladder {row.ladder!r}, rung {rung!r} "
                    "disagrees between aggregate tables"
                )
        if not math.isclose(
            float(row.full_minus_small),
            float(row.full_top1) - float(row.small_top1),
            abs_tol=1e-12,
        ):
            raise ValueError(f"ladder {row.ladder!r} endpoint contrast is inconsistent")
    if not math.isclose(
        float(summary["inference"]["mean_full_minus_small"]),
        float(np.mean(list(table_contrasts.values()))),
        abs_tol=1e-12,
    ):
        raise ValueError("mean ladder contrast disagrees with ladder_contrasts.csv")

    ladder_rank = {ladder: index for index, ladder in enumerate(ladder_order)}
    rung_rank = {rung: index for index, rung in enumerate(RUNG_ORDER)}
    run_table = run_table.assign(
        _ladder_order=run_table["ladder"].astype(str).map(ladder_rank),
        _rung_order=run_table["rung"].astype(str).map(rung_rank),
    ).sort_values(["_ladder_order", "_rung_order"], kind="stable")
    run_table = run_table.drop(columns=["_ladder_order", "_rung_order"]).reset_index(drop=True)
    ladder_table = ladder_table.assign(
        _ladder_order=ladder_table["ladder"].astype(str).map(ladder_rank)
    ).sort_values("_ladder_order", kind="stable")
    ladder_table = ladder_table.drop(columns="_ladder_order").reset_index(drop=True)
    return ladder_order, run_table, ladder_table


def _write_scaling_figure(
    ladder_table: pd.DataFrame, ladder_order: list[str], output_dir: Path
) -> tuple[Path, Path]:
    x = np.arange(len(RUNG_ORDER), dtype=float)
    values = ladder_table.loc[:, [f"{rung}_top1" for rung in RUNG_ORDER]].to_numpy(
        dtype=float
    )
    colors = plt.get_cmap("Blues")(np.linspace(0.38, 0.82, len(ladder_order)))
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for index, ladder in enumerate(ladder_order):
        axis.plot(
            x,
            100.0 * values[index],
            marker="o",
            markersize=4,
            linewidth=1.25,
            color=colors[index],
            alpha=0.8,
            label=ladder,
        )
    axis.plot(
        x,
        100.0 * values.mean(axis=0),
        marker="o",
        markersize=5,
        linewidth=2.4,
        color="#252525",
        label="Mean",
        zorder=5,
    )
    axis.set_xticks(x, [rung.capitalize() for rung in RUNG_ORDER])
    axis.set_xlabel("Nested GaitLU training-data rung")
    axis.set_ylabel("Learned GFC-v2 top-1 (%)")
    axis.set_title("GFC-v2 scaling across five prespecified ladders")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    axis.margins(x=0.04)

    png_path = output_dir / "gfc_study_scaling.png"
    pdf_path = output_dir / "gfc_study_scaling.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "GFC-v2 scaling across five prespecified ladders",
            "Subject": "Aggregate-only GFC-v2 study result",
        },
    )
    plt.close(figure)
    return png_path, pdf_path


def make_gfc_study_results(aggregate_dir: Path, output_dir: Path) -> list[Path]:
    """Validate aggregate study inputs and render the frozen paper tables and figure."""

    _remove_stale_outputs(output_dir)
    summary_path, run_path, ladder_path = _require_inputs(aggregate_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("outcome_summary.json must contain one JSON object")
    _check_private_keys(summary)
    _require_summary_shape(summary)
    run_table = pd.read_csv(run_path)
    ladder_table = pd.read_csv(ladder_path)
    ladder_order, run_table, ladder_table = _validate_tables(
        summary, run_table, ladder_table
    )

    paper_run_path = output_dir / "gfc_study_run_table.csv"
    paper_ladder_path = output_dir / "gfc_study_ladder_contrasts.csv"
    run_table.to_csv(paper_run_path, index=False)
    ladder_table.to_csv(paper_ladder_path, index=False)
    png_path, pdf_path = _write_scaling_figure(ladder_table, ladder_order, output_dir)
    return [paper_run_path, paper_ladder_path, png_path, pdf_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        required=True,
        help="Directory containing outcome_summary.json and the two aggregate CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = make_gfc_study_results(args.aggregate_dir, args.output_dir)
    print(json.dumps([str(path) for path in paths], indent=2))


if __name__ == "__main__":
    main()
