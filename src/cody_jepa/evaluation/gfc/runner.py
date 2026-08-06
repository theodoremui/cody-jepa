#!/usr/bin/env python3
"""Run learned and shortcut Grounded Factorial Completion from a feature table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    CANONICAL_CELLS,
    Cell,
    EXPECTED_GALLERY_SIZE,
    EXPECTED_QUERIES,
    FactorBlocks,
    GFC_PROTOCOL,
    Recording,
    aggregate_windows,
    evaluate_cohort,
)
from .controls import (
    evaluate_independent_factor_controls,
    fit_shared_temperature,
)
from .inference import (
    bootstrap_gfc_gain,
    paired_cohort_gfc_gain,
    plan_prospective_power,
)
from .normalization import (
    fit_factor_adapter,
    fit_pca_normalizer,
    fit_raw_normalizer,
)
from .oracle import compile_healthgait_gfc_v2_protocol
from .roles import (
    DEVELOPMENT_ROLE,
    EXPECTED_ASSIGNED_COUNTS,
    EXPECTED_COMPLETE_COUNTS,
    LOCKED_OUTCOME_ROLE,
    ROLE_MAP_VERSION,
    load_role_map,
    role_lookup,
)
from ...config.gfc import load_gfc_config
from ..features import read_feature_table


@dataclass(frozen=True)
class RecordingRow:
    subject_id: str
    recording_id: str
    source_video_id: str
    direction_clip_id: str
    split: str
    cell: Cell
    window_ids: tuple[str, ...]
    learned: np.ndarray
    shortcut: np.ndarray


@dataclass(frozen=True)
class PreparedGFCData:
    """Role-selected rows and arrays shared by a model's five analyses."""

    learned_columns: tuple[str, ...]
    training_rows: tuple[RecordingRow, ...]
    evaluation_rows: tuple[RecordingRow, ...]
    training_exclusions: tuple[str, ...]
    private_roles: pd.DataFrame | None
    complete_counts: dict[str, int] | None
    excluded_counts: dict[str, int] | None
    train_learned: np.ndarray
    train_shortcut: np.ndarray
    train_subjects: tuple[str, ...]
    train_cells: tuple[Cell, ...]
    evaluation_learned: np.ndarray
    evaluation_shortcut: np.ndarray

def _feature_columns(table: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in table.columns if str(column).startswith("feature_")]
    try:
        columns.sort(key=lambda column: int(column.removeprefix("feature_")))
    except ValueError as error:
        raise ValueError("learned feature columns must be feature_0 through feature_D") from error
    if columns != [f"feature_{index}" for index in range(len(columns))]:
        raise ValueError("learned feature columns must be contiguous from feature_0")
    if not columns:
        raise ValueError("feature table has no learned feature columns")
    return columns


def _text(row: pd.Series, column: str) -> str:
    value = str(row[column])
    if not value or value != value.strip() or value.casefold() == "nan":
        raise ValueError(f"{column} values must be nonempty text")
    return value


def _recording_rows(
    table: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[RecordingRow], list[str]]:
    subject_column = str(config["subject_column"])
    recording_column = str(config["recording_column"])
    split_column = str(config["split_column"])
    factor_columns = list(config["factors"]["order"])
    shortcut_columns = list(config["shortcut"]["columns"])
    learned_columns = _feature_columns(table)
    required = {
        subject_column,
        recording_column,
        split_column,
        "window_start",
        "num_frames",
        "fps",
        "source_video_id",
        "direction_clip_id",
        *factor_columns,
        *shortcut_columns,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("feature table is missing columns: " + ", ".join(missing))
    numeric_columns = [
        *learned_columns,
        *shortcut_columns,
        "window_start",
        "num_frames",
        "fps",
    ]
    numeric = table[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("feature and shortcut columns must contain finite numbers")
    num_frames = numeric["num_frames"].to_numpy(dtype=np.float64)
    fps = numeric["fps"].to_numpy(dtype=np.float64)
    if (
        (num_frames <= 0.0).any()
        or not np.equal(num_frames, np.floor(num_frames)).all()
        or (fps <= 0.0).any()
    ):
        raise ValueError("num_frames must be positive integers and fps must be positive")
    signed = numeric["shortcut_horizontal_centroid_drift_signed"].to_numpy(
        dtype=np.float64
    )
    absolute = numeric["shortcut_horizontal_centroid_drift_absolute"].to_numpy(
        dtype=np.float64
    )
    if (
        (np.abs(signed) > 1.0).any()
        or (absolute < 0.0).any()
        or (absolute > 1.0).any()
        or not np.allclose(absolute, np.abs(signed), rtol=1e-7, atol=1e-9)
    ):
        raise ValueError("centroid-drift shortcut values are inconsistent")
    area_names = [
        "shortcut_foreground_area_mean",
        "shortcut_foreground_area_std",
        "shortcut_foreground_area_q25",
        "shortcut_foreground_area_median",
        "shortcut_foreground_area_q75",
    ]
    areas = numeric[area_names].to_numpy(dtype=np.float64)
    if (areas < 0.0).any() or (areas > 1.0).any():
        raise ValueError("foreground-area shortcut values must lie in [0, 1]")
    if not np.all(
        numeric["shortcut_foreground_area_q25"]
        <= numeric["shortcut_foreground_area_median"]
    ) or not np.all(
        numeric["shortcut_foreground_area_median"]
        <= numeric["shortcut_foreground_area_q75"]
    ):
        raise ValueError("foreground-area shortcut quantiles must be ordered")
    if not np.allclose(
        numeric["shortcut_log_frame_count"].to_numpy(dtype=np.float64),
        np.log(num_frames),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("shortcut_log_frame_count is inconsistent with num_frames")
    if not np.allclose(
        numeric["shortcut_duration_seconds"].to_numpy(dtype=np.float64),
        num_frames / fps,
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("shortcut_duration_seconds is inconsistent with num_frames and fps")
    expected_windows = int(config["recording_aggregation"]["windows_per_recording"])
    if expected_windows != 3:
        raise ValueError("the maintained Health&Gait evaluator requires three windows")

    rows: list[RecordingRow] = []
    for recording_id, group in table.groupby(recording_column, sort=True, dropna=False):
        if len(group) != expected_windows:
            raise ValueError(
                f"recording {recording_id!r} has {len(group)} rows; expected {expected_windows}"
            )
        first = group.iloc[0]
        identity_columns = [
            subject_column,
            split_column,
            "source_video_id",
            "direction_clip_id",
            *factor_columns,
        ]
        for column in identity_columns:
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(f"recording {recording_id!r} changes {column!r} across windows")
        if _text(first, "direction_clip_id") != str(recording_id):
            raise ValueError("direction_clip_id must equal the direction-level recording_id")
        starts = pd.to_numeric(group["window_start"], errors="raise").astype(int)
        if starts.nunique() != expected_windows or (starts < 0).any():
            raise ValueError(f"recording {recording_id!r} has invalid window starts")
        shortcut_values = group[shortcut_columns].to_numpy(dtype=np.float64)
        if not np.allclose(shortcut_values, shortcut_values[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"recording-level shortcut values change across {recording_id!r} windows"
            )
        cell = Cell(*(_text(first, column) for column in factor_columns))
        recording_text = str(recording_id)
        rows.append(
            RecordingRow(
                subject_id=_text(first, subject_column),
                recording_id=recording_text,
                source_video_id=_text(first, "source_video_id"),
                direction_clip_id=_text(first, "direction_clip_id"),
                split=_text(first, split_column),
                cell=cell,
                window_ids=tuple(
                    f"{recording_text}::window={start}" for start in sorted(starts.tolist())
                ),
                learned=aggregate_windows(
                    group[learned_columns].to_numpy(dtype=np.float64),
                    expected_count=expected_windows,
                    label="learned features",
                ),
                shortcut=aggregate_windows(
                    shortcut_values,
                    expected_count=expected_windows,
                    label="shortcut cues",
                ),
            )
        )
    source_owners: dict[str, tuple[str, str, str]] = {}
    by_factor_pair: dict[tuple[str, str, str], list[RecordingRow]] = {}
    for row in rows:
        owner = (row.subject_id, row.cell.speed, row.cell.clothing)
        previous_owner = source_owners.setdefault(row.source_video_id, owner)
        if previous_owner != owner:
            raise ValueError(
                f"source_video_id {row.source_video_id!r} spans distinct participant/factor groups"
            )
        by_factor_pair.setdefault(owner, []).append(row)
    for owner, pair_rows in by_factor_pair.items():
        sources = {row.source_video_id for row in pair_rows}
        if len(sources) != 1:
            raise ValueError(
                "direction clips for participant/speed/clothing must share one source video: "
                f"{owner!r} has {sorted(sources)!r}"
            )
        clips_by_direction: dict[str, str] = {}
        for row in pair_rows:
            prior_clip = clips_by_direction.setdefault(
                row.cell.direction, row.direction_clip_id
            )
            if prior_clip != row.direction_clip_id:
                raise ValueError(
                    f"{owner!r} has multiple direction clips for {row.cell.direction!r}"
                )
        if len(clips_by_direction) == 2 and len(set(clips_by_direction.values())) != 2:
            raise ValueError(f"{owner!r} must use distinct clips for the two directions")
    return rows, learned_columns


def _complete_training_rows(rows: list[RecordingRow]) -> tuple[list[RecordingRow], list[str]]:
    complete: list[RecordingRow] = []
    excluded: list[str] = []
    expected = set(CANONICAL_CELLS)
    by_subject: dict[str, list[RecordingRow]] = {}
    for row in rows:
        by_subject.setdefault(row.subject_id, []).append(row)
    for subject_id in sorted(by_subject):
        subject_rows = by_subject[subject_id]
        cells = [row.cell for row in subject_rows]
        if len(cells) != len(set(cells)):
            raise ValueError(f"training subject {subject_id!r} has a duplicate factorial cell")
        if set(cells) != expected:
            excluded.append(subject_id)
        else:
            complete.extend(sorted(subject_rows, key=lambda row: row.cell.canonical_index))
    if not complete:
        raise ValueError("no complete training participants are available for fitted transforms")
    return complete, excluded


def _normalizer_factory(name: str):
    choices = {
        "raw_retain_all": (fit_raw_normalizer, "retain_all"),
        "raw_effective_rank": (fit_raw_normalizer, "effective_rank"),
        "pca_effective_rank": (fit_pca_normalizer, "effective_rank"),
    }
    try:
        return choices[name]
    except KeyError as error:
        raise ValueError(f"unknown normalization analysis {name!r}") from error


def _make_recordings(
    rows: list[RecordingRow], factor_values: dict[str, np.ndarray]
) -> list[Recording]:
    if set(factor_values) != {"speed", "clothing", "direction"}:
        raise ValueError("transformed blocks must cover all three factors")
    if any(values.shape[0] != len(rows) for values in factor_values.values()):
        raise ValueError("transformed block row counts do not match recording rows")
    return [
        Recording(
            subject_id=row.subject_id,
            recording_id=row.recording_id,
            source_video_id=row.source_video_id,
            cell=row.cell,
            factor_blocks=FactorBlocks(
                speed=factor_values["speed"][index],
                clothing=factor_values["clothing"][index],
                direction=factor_values["direction"][index],
            ),
            window_ids=row.window_ids,
        )
        for index, row in enumerate(rows)
    ]


def _public_participant_labels(subject_ids: list[str]) -> dict[str, str]:
    return {
        subject_id: f"participant_{index:04d}"
        for index, subject_id in enumerate(sorted(set(subject_ids)), start=1)
    }


def _public_query(value: dict[str, Any], participant_label: str) -> dict[str, Any]:
    """Remove private source identifiers while retaining the scientific query."""

    result = dict(value)
    result["subject_id"] = participant_label
    for role in ("target", "donor_u", "donor_v"):
        item = dict(result[role])
        cell = item["cell"]
        item["recording_id"] = (
            f"{participant_label}/{cell['speed']}:{cell['clothing']}:{cell['direction']}"
        )
        result[role] = item
    gallery = []
    for source in result["gallery"]:
        item = dict(source)
        cell = item["cell"]
        item["recording_id"] = (
            f"{participant_label}/{cell['speed']}:{cell['clothing']}:{cell['direction']}"
        )
        gallery.append(item)
    result["gallery"] = gallery
    return result


def _write_summary_files(summary: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    interval_values = summary["learned_minus_shortcut"]
    confidence_interval = interval_values["confidence_interval"]
    pd.DataFrame(
        [
            {
                "protocol": summary["protocol"],
                "gallery": summary["gallery"],
                "queries_per_participant": summary["queries_per_participant"],
                "factor_heads": summary["factor_heads"],
                "source_independence_verified": summary[
                    "source_independence_verified"
                ],
                "split": summary["split"],
                "normalization": summary["normalization"],
                "model_label": summary["model_label"],
                "seed": summary["seed"],
                "participants": summary["evaluation"]["participant_count"],
                "excluded_participants": summary["evaluation"][
                    "excluded_participant_count"
                ],
                "learned_top1": summary["learned"]["top1"],
                "shortcut_top1": summary["shortcut"]["top1"],
                "learned_minus_shortcut": interval_values["point_estimate"],
                "interval_lower": confidence_interval["lower"],
                "interval_upper": confidence_interval["upper"],
                "confidence_level": interval_values["confidence_level"],
                "bootstrap_resamples": interval_values["resamples"],
            }
        ]
    ).to_csv(directory / "summary.csv", index=False)


def _aggregate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove participant rows while retaining aggregate exclusion reporting."""

    patterns: dict[tuple[str, str], int] = {}
    for item in summary["exclusions"]:
        cells = json.dumps(item["missing_cells"], sort_keys=True, separators=(",", ":"))
        key = (str(item["reason"]), cells)
        patterns[key] = patterns.get(key, 0) + 1
    aggregate = dict(summary)
    aggregate["exclusions"] = [
        {
            "reason": reason,
            "missing_cells": json.loads(cells),
            "participant_count": count,
        }
        for (reason, cells), count in sorted(patterns.items())
    ]
    return aggregate


def _validate_output_directories(
    output_dir: Path, aggregate_output_dir: Path | None
) -> tuple[Path, Path | None]:
    detailed = output_dir.expanduser().resolve()
    if aggregate_output_dir is None:
        return detailed, None
    aggregate = aggregate_output_dir.expanduser().resolve()
    if (
        aggregate == detailed
        or aggregate in detailed.parents
        or detailed in aggregate.parents
    ):
        raise ValueError("detailed and aggregate output directories must not overlap")
    if aggregate.exists() and not aggregate.is_dir():
        raise ValueError("aggregate output path must be a directory")
    if aggregate.is_dir():
        unexpected = sorted(
            path.name
            for path in aggregate.iterdir()
            if path.name not in {"summary.csv", "summary.json"}
        )
        if unexpected:
            raise ValueError(
                "aggregate output directory contains non-summary files: "
                + ", ".join(unexpected)
            )
    return detailed, aggregate


def _prepare_gfc_data(
    table: pd.DataFrame,
    config: dict[str, Any],
    split: str,
    role_map: Path | None,
) -> PreparedGFCData:
    """Select roles, validate grids, and aggregate recordings exactly once."""

    split_map = dict(config["split_map"])
    evaluable_splits = ("development", "confirmation", LOCKED_OUTCOME_ROLE)
    if split not in evaluable_splits:
        raise ValueError(f"split must be one of {list(evaluable_splits)}")
    locked = split == LOCKED_OUTCOME_ROLE
    if locked and role_map is None:
        raise ValueError("locked_outcome evaluation requires --role-map")
    if not locked and role_map is not None:
        raise ValueError("--role-map is reserved for locked_outcome evaluation")
    adapter_fit_split = str(config["adapter"]["fit_split"])
    normalization_fit_split = str(config["normalization"]["fit_split"])
    if normalization_fit_split != adapter_fit_split:
        raise ValueError("adapter and normalization fit splits must be identical")
    training_label = str(split_map[adapter_fit_split])
    evaluation_label = None if locked else str(split_map[split])
    split_column = str(config["split_column"])
    if split_column not in table.columns:
        raise ValueError(f"feature table is missing columns: {split_column}")
    raw_splits = table[split_column].astype(str)
    private_roles = None
    subject_roles: dict[str, str] | None = None
    if locked:
        subject_column = str(config["subject_column"])
        if subject_column not in table.columns:
            raise ValueError(f"feature table is missing columns: {subject_column}")
        archive_subjects = sorted(set(table[subject_column].astype(str)))
        private_roles = load_role_map(role_map, expected_subject_ids=archive_subjects)
        subject_roles = role_lookup(private_roles)
        # Historical archive splits have no selection authority in locked mode.
        relevant_table = table.copy()
    else:
        unknown_splits = sorted(set(raw_splits) - set(split_map.values()))
        if unknown_splits:
            raise ValueError(
                "feature table contains unsupported split labels: "
                + ", ".join(repr(value) for value in unknown_splits)
            )
        relevant_table = table.loc[
            raw_splits.isin({training_label, evaluation_label})
        ].copy()
    all_rows, learned_columns = _recording_rows(relevant_table, config)
    if locked:
        assert subject_roles is not None
        training_candidates = [
            row for row in all_rows if subject_roles[row.subject_id] == DEVELOPMENT_ROLE
        ]
        evaluation_rows = [
            row for row in all_rows if subject_roles[row.subject_id] == LOCKED_OUTCOME_ROLE
        ]
    else:
        training_candidates = [row for row in all_rows if row.split == training_label]
        evaluation_rows = [row for row in all_rows if row.split == evaluation_label]
    training_rows, training_exclusions = _complete_training_rows(training_candidates)
    if not evaluation_rows:
        raise ValueError(f"feature table has no recordings for {split!r}")

    complete_counts = None
    excluded_counts = None
    if locked:
        complete_outcome_rows, outcome_exclusions = _complete_training_rows(evaluation_rows)
        complete_counts = {
            DEVELOPMENT_ROLE: len({row.subject_id for row in training_rows}),
            LOCKED_OUTCOME_ROLE: len({row.subject_id for row in complete_outcome_rows}),
        }
        excluded_counts = {
            DEVELOPMENT_ROLE: len(training_exclusions),
            LOCKED_OUTCOME_ROLE: len(outcome_exclusions),
        }
        if complete_counts != EXPECTED_COMPLETE_COUNTS:
            raise ValueError(
                f"complete role counts must be {EXPECTED_COMPLETE_COUNTS}, got {complete_counts}"
            )
        expected_excluded = {
            role: EXPECTED_ASSIGNED_COUNTS[role] - EXPECTED_COMPLETE_COUNTS[role]
            for role in EXPECTED_ASSIGNED_COUNTS
        }
        if excluded_counts != expected_excluded:
            raise ValueError(
                f"excluded role counts must be {expected_excluded}, got {excluded_counts}"
            )
    training_subjects = {row.subject_id for row in training_rows}
    evaluation_subjects = {row.subject_id for row in evaluation_rows}
    overlap = sorted(training_subjects & evaluation_subjects)
    if overlap:
        raise ValueError(
            "training and evaluation participants must be disjoint; overlap: "
            + ", ".join(overlap[:10])
        )

    return PreparedGFCData(
        learned_columns=tuple(learned_columns),
        training_rows=tuple(training_rows),
        evaluation_rows=tuple(evaluation_rows),
        training_exclusions=tuple(training_exclusions),
        private_roles=private_roles,
        complete_counts=complete_counts,
        excluded_counts=excluded_counts,
        train_learned=np.stack([row.learned for row in training_rows]),
        train_shortcut=np.stack([row.shortcut for row in training_rows]),
        train_subjects=tuple(row.subject_id for row in training_rows),
        train_cells=tuple(row.cell for row in training_rows),
        evaluation_learned=np.stack([row.learned for row in evaluation_rows]),
        evaluation_shortcut=np.stack([row.shortcut for row in evaluation_rows]),
    )


def run_gfc_table(
    table: pd.DataFrame,
    config_path: Path,
    split: str,
    output_dir: Path,
    *,
    model_label: str,
    normalization: str | None = None,
    ridge_alpha: float | None = None,
    role_map: Path | None = None,
    write_queries: bool = False,
    aggregate_output_dir: Path | None = None,
    model_metadata: dict[str, Any] | None = None,
    revision: dict[str, Any] | None = None,
    _prepared: PreparedGFCData | None = None,
    _adapter_cache: dict[float, tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]]
    | None = None,
) -> dict[str, Any]:
    """Evaluate an already-loaded feature table without re-reading its archive."""

    if not isinstance(model_label, str) or not model_label.strip() or model_label != model_label.strip():
        raise ValueError("model_label must be nonempty text without surrounding whitespace")
    output_dir, aggregate_output_dir = _validate_output_directories(
        output_dir, aggregate_output_dir
    )
    config = load_gfc_config(config_path)
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    locked = split == LOCKED_OUTCOME_ROLE
    prepared = _prepared or _prepare_gfc_data(table, config, split, role_map)
    learned_columns = prepared.learned_columns
    training_rows = list(prepared.training_rows)
    evaluation_rows = list(prepared.evaluation_rows)
    training_exclusions = list(prepared.training_exclusions)
    private_roles = prepared.private_roles
    complete_counts = prepared.complete_counts
    excluded_counts = prepared.excluded_counts
    train_learned = prepared.train_learned
    train_shortcut = prepared.train_shortcut
    train_subjects = list(prepared.train_subjects)
    train_cells = list(prepared.train_cells)
    analysis_name = normalization or str(config["normalization"]["primary"])
    alpha = float(config["adapter"]["alpha"] if ridge_alpha is None else ridge_alpha)
    declared = [
        item
        for item in config["analyses"]
        if item["normalization"] == analysis_name
        and float(item["ridge_alpha"]) == alpha
    ]
    if len(declared) != 1:
        allowed = [
            f"{item['normalization']}@alpha={float(item['ridge_alpha']):g}"
            for item in config["analyses"]
        ]
        raise ValueError(
            "unsupported normalization/ridge-alpha analysis; expected one of "
            + ", ".join(allowed)
        )
    analysis = declared[0]
    factor_names = tuple(compile_healthgait_gfc_v2_protocol().design.factor_names)
    cached_fit = _adapter_cache.get(alpha) if _adapter_cache is not None else None
    if cached_fit is None:
        adapters = {
            f"{representation}_{factor}": fit_factor_adapter(
                train_learned if representation == "learned" else train_shortcut,
                train_subjects,
                train_cells,
                factor_name=factor,
                alpha=alpha,
            )
            for representation in ("learned", "shortcut")
            for factor in factor_names
        }
        training_inputs = {"learned": train_learned, "shortcut": train_shortcut}
        raw_adapter_outputs = {
            representation: {
                factor: adapters[f"{representation}_{factor}"].transform(
                    training_inputs[representation]
                )
                for factor in factor_names
            }
            for representation in ("learned", "shortcut")
        }
        if _adapter_cache is not None:
            _adapter_cache[alpha] = (adapters, raw_adapter_outputs)
    else:
        adapters, raw_adapter_outputs = cached_fit

    fit_normalizer, dimension_policy = _normalizer_factory(analysis_name)
    scale_floor = float(config["normalization"]["scale_floor"])
    block_epsilon = float(config["normalization"]["block_l2_epsilon"])

    normalizers = {}
    for representation in ("learned", "shortcut"):
        for factor in factor_names:
            name = f"{representation}_{factor}"
            normalizers[name] = fit_normalizer(
                raw_adapter_outputs[representation][factor],
                dimension_policy=dimension_policy,
                scale_floor=scale_floor,
                zero_norm_epsilon=block_epsilon,
            )

    evaluation_learned = prepared.evaluation_learned
    evaluation_shortcut = prepared.evaluation_shortcut
    evaluation_inputs = {
        "learned": evaluation_learned,
        "shortcut": evaluation_shortcut,
    }
    transformed = {
        representation: {
            factor: normalizers[f"{representation}_{factor}"].transform(
                adapters[f"{representation}_{factor}"].transform(
                    evaluation_inputs[representation]
                )
            )
            for factor in factor_names
        }
        for representation in ("learned", "shortcut")
    }

    temperature_fit = None
    controls = None
    if locked and bool(analysis["controls"]):
        temperature_fit = fit_shared_temperature(
            raw_adapter_outputs["learned"],
            train_cells,
            bounds=tuple(config["independent_factor_controls"]["temperature"]["bounds"]),
        )
        raw_evaluation_blocks = {
            factor: adapters[f"learned_{factor}"].transform(evaluation_learned)
            for factor in factor_names
        }
        controls = evaluate_independent_factor_controls(
            _make_recordings(evaluation_rows, raw_evaluation_blocks),
            temperature=temperature_fit.temperature,
            tie_tolerance=float(config["ties"]["absolute_tolerance"]),
        )

    seed = int(config["bootstrap"]["seed"])
    distance = config["distance"]
    tie_tolerance = float(config["ties"]["absolute_tolerance"])
    learned = evaluate_cohort(
        _make_recordings(evaluation_rows, transformed["learned"]),
        split=split,
        seed=seed,
        representation="learned",
        tie_tolerance=tie_tolerance,
        zero_norm_epsilon=float(distance["zero_norm_epsilon"]),
    )
    shortcut = evaluate_cohort(
        _make_recordings(evaluation_rows, transformed["shortcut"]),
        split=split,
        seed=seed,
        representation="shortcut",
        tie_tolerance=tie_tolerance,
        zero_norm_epsilon=float(distance["zero_norm_epsilon"]),
    )
    if [item.to_dict() for item in learned.exclusions] != [
        item.to_dict() for item in shortcut.exclusions
    ]:
        raise RuntimeError("learned and shortcut paths produced different exclusions")
    if controls is not None and [item.to_dict() for item in learned.exclusions] != [
        item.to_dict() for item in controls.exclusions
    ]:
        raise RuntimeError("learned and independent-factor controls produced different exclusions")
    paired = paired_cohort_gfc_gain(learned.participants, shortcut.participants)
    interval = bootstrap_gfc_gain(
        paired,
        resamples=int(config["bootstrap"]["resamples"]),
        seed=seed,
        confidence_level=float(config["bootstrap"]["confidence_level"]),
    )
    power: dict[str, Any] | None = None
    if split == "development":
        try:
            power = plan_prospective_power(
                paired,
                minimum_effect=float(config["power"]["effect"]),
                alpha=float(config["power"]["alpha"]),
                target_power=float(config["power"]["minimum_power"]),
            ).to_dict()
            power["estimable"] = True
        except ValueError as error:
            power = {"estimable": False, "reason": str(error)}

    participant_rows = []
    public_labels = _public_participant_labels(
        [item.subject_id for item in learned.participants]
        + [item.subject_id for item in learned.exclusions]
    )
    learned_by_subject = {item.subject_id: item for item in learned.participants}
    shortcut_by_subject = {item.subject_id: item for item in shortcut.participants}
    for contrast in paired.participants:
        participant_row = {
                "participant": public_labels[contrast.subject_id],
                "learned_top1": contrast.learned_top1,
                "shortcut_top1": contrast.shortcut_top1,
                "learned_minus_shortcut": contrast.difference,
                "learned_mrr": learned_by_subject[contrast.subject_id].mrr,
                "shortcut_mrr": shortcut_by_subject[contrast.subject_id].mrr,
                "learned_donor_u_attraction": learned_by_subject[
                    contrast.subject_id
                ].donor_u_attraction,
                "learned_donor_v_attraction": learned_by_subject[
                    contrast.subject_id
                ].donor_v_attraction,
                "shortcut_donor_u_attraction": shortcut_by_subject[
                    contrast.subject_id
                ].donor_u_attraction,
                "shortcut_donor_v_attraction": shortcut_by_subject[
                    contrast.subject_id
                ].donor_v_attraction,
            }
        if controls is not None:
            hard_by_subject = {item.subject_id: item for item in controls.hard_participants}
            soft_by_subject = {item.subject_id: item for item in controls.soft_participants}
            hard = hard_by_subject[contrast.subject_id]
            soft = soft_by_subject[contrast.subject_id]
            participant_row.update(
                {
                    "hard_control_top1": hard.top1,
                    "hard_control_mrr": hard.mrr,
                    "soft_control_top1": soft.top1,
                    "soft_control_mrr": soft.mrr,
                    "soft_control_target_probability": soft.target_probability,
                    "soft_control_target_nll": soft.target_nll,
                    "learned_minus_hard_control_top1": contrast.learned_top1 - hard.top1,
                    "learned_minus_soft_control_top1": contrast.learned_top1 - soft.top1,
                }
            )
        participant_rows.append(participant_row)
    summary: dict[str, Any] = {
        "protocol": GFC_PROTOCOL,
        "gallery": f"retain_all_{EXPECTED_GALLERY_SIZE}",
        "queries_per_participant": EXPECTED_QUERIES,
        "factor_heads": "three_matched_ridge_heads",
        "source_independence_verified": True,
        "split": split,
        "normalization": analysis_name,
        "analysis_id": analysis["analysis_id"],
        "ridge_alpha": alpha,
        "model_label": model_label,
        "seed": seed,
        "feature_dimension": len(learned_columns),
        "method_settings": {
            key: config[key]
            for key in (
                "split_map",
                "factors",
                "recording_aggregation",
                "complete_case",
                "protocol",
                "distance",
                "ties",
                "adapter",
                "normalization",
                "analyses",
                "independent_factor_controls",
                "shortcut",
                "metrics",
                "primary_metric",
                "primary_contrast",
                "bootstrap",
                "power",
            )
        },
        "training": {
            "participant_count": len({row.subject_id for row in training_rows}),
            "recording_count": len(training_rows),
            "incomplete_participant_count": len(training_exclusions),
        },
        "evaluation": {
            "participant_count": len(learned.participants),
            "excluded_participant_count": len(learned.exclusions),
            "recording_count": len(evaluation_rows),
        },
        "learned": {
            "top1": learned.top1,
            "mrr": learned.mrr,
            "donor_u_attraction": learned.donor_u_attraction,
            "donor_v_attraction": learned.donor_v_attraction,
        },
        "shortcut": {
            "top1": shortcut.top1,
            "mrr": shortcut.mrr,
            "donor_u_attraction": shortcut.donor_u_attraction,
            "donor_v_attraction": shortcut.donor_v_attraction,
        },
        "learned_minus_shortcut": interval.to_dict(),
        "adapter_diagnostics": {
            name: {
                "factor_name": fit.factor_name,
                "fit_row_count": fit.fit_row_count,
                "input_dimension": fit.input_dimension,
                "output_dimension": fit.output_dimension,
            }
            for name, fit in adapters.items()
        },
        "normalizer_diagnostics": {
            name: fit.diagnostics() for name, fit in normalizers.items()
        },
        "exclusions": [
            {
                "participant": public_labels[item.subject_id],
                "reason": item.reason,
                "missing_cells": [cell.to_dict() for cell in item.missing_cells],
            }
            for item in learned.exclusions
        ],
        "prospective_power": power,
    }
    if model_metadata is not None:
        if not isinstance(model_metadata, dict):
            raise TypeError("model_metadata must be an object")
        summary["model"] = {**model_metadata, "label": model_label}
    if revision is not None:
        if not isinstance(revision, dict):
            raise TypeError("revision must be an object")
        summary["revision"] = dict(revision)
    if locked:
        assert private_roles is not None
        summary["cohort_roles"] = {
            "version": ROLE_MAP_VERSION,
            "fit_role": DEVELOPMENT_ROLE,
            "evaluation_role": LOCKED_OUTCOME_ROLE,
            "assigned_counts": dict(EXPECTED_ASSIGNED_COUNTS),
            "complete_counts": complete_counts,
            "excluded_counts": excluded_counts,
        }
        summary["independent_factor_controls"] = None
        if controls is not None and temperature_fit is not None:
            summary["independent_factor_controls"] = {
                **controls.aggregate_dict(),
                "temperature": temperature_fit.to_dict(),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(participant_rows).to_csv(output_dir / "participants.csv", index=False)
    _write_summary_files(summary, output_dir)
    if aggregate_output_dir is not None:
        _write_summary_files(_aggregate_summary(summary), aggregate_output_dir)
    query_path = output_dir / "queries.jsonl"
    if write_queries:
        with query_path.open("w", encoding="utf-8") as handle:
            for participant in (*learned.participants, *shortcut.participants):
                for query in participant.queries:
                    handle.write(
                        json.dumps(
                            _public_query(
                                query.to_dict(), public_labels[participant.subject_id]
                            ),
                            sort_keys=True,
                        )
                        + "\n"
                    )
    else:
        query_path.unlink(missing_ok=True)
    return summary


def run_gfc_analysis_suite(
    table: pd.DataFrame,
    config_path: Path,
    output_root: Path,
    *,
    model_label: str,
    role_map: Path,
    model_metadata: dict[str, Any],
    revision: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the frozen five analyses with shared row preparation and ridge fits."""

    config = load_gfc_config(config_path)
    prepared = _prepare_gfc_data(table, config, LOCKED_OUTCOME_ROLE, role_map)
    adapter_cache: dict[
        float, tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]
    ] = {}
    results = []
    for analysis in config["analyses"]:
        results.append(
            run_gfc_table(
                table,
                config_path,
                LOCKED_OUTCOME_ROLE,
                output_root / str(analysis["analysis_id"]),
                model_label=model_label,
                normalization=str(analysis["normalization"]),
                ridge_alpha=float(analysis["ridge_alpha"]),
                role_map=role_map,
                model_metadata=model_metadata,
                revision=revision,
                _prepared=prepared,
                _adapter_cache=adapter_cache,
            )
        )
    return results


def run_gfc(
    features: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    *,
    model_label: str,
    normalization: str | None = None,
    ridge_alpha: float | None = None,
    role_map: Path | None = None,
    write_queries: bool = False,
    aggregate_output_dir: Path | None = None,
    model_metadata: dict[str, Any] | None = None,
    revision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one feature archive and run the single-analysis evaluator."""

    if features.suffix.casefold() == ".npz":
        # Scientific validation is role/split scoped in run_gfc_table.
        table, _ = read_feature_table(features, validate=False)
    else:
        table = pd.read_csv(features)
    return run_gfc_table(
        table,
        config_path,
        split,
        output_dir,
        model_label=model_label,
        normalization=normalization,
        ridge_alpha=ridge_alpha,
        role_map=role_map,
        write_queries=write_queries,
        aggregate_output_dir=aggregate_output_dir,
        model_metadata=model_metadata,
        revision=revision,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--normalization")
    parser.add_argument("--ridge-alpha", type=float)
    parser.add_argument("--role-map", type=Path)
    parser.add_argument("--write-queries", action="store_true")
    parser.add_argument(
        "--aggregate-output-dir",
        type=Path,
        help="Optional directory receiving only summary.json and summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_gfc(
        args.features,
        args.config,
        args.split,
        args.output_dir,
        model_label=args.model_label,
        normalization=args.normalization,
        ridge_alpha=args.ridge_alpha,
        role_map=args.role_map,
        write_queries=args.write_queries,
        aggregate_output_dir=args.aggregate_output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PreparedGFCData",
    "RecordingRow",
    "run_gfc",
    "run_gfc_analysis_suite",
    "run_gfc_table",
]
