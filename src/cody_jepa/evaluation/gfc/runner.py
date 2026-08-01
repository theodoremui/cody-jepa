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
    GFC_PROTOCOL,
    Recording,
    aggregate_windows,
    evaluate_cohort,
)
from .inference import (
    bootstrap_gfc_gain,
    paired_cohort_gfc_gain,
    plan_prospective_power,
)
from .normalization import (
    fit_condition_adapter,
    fit_gait_adapter,
    fit_pca_normalizer,
    fit_raw_normalizer,
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
    shortcut_condition: np.ndarray
    shortcut_gait: np.ndarray

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
    shortcut_condition_columns = list(config["shortcut"]["condition_columns"])
    shortcut_gait_columns = list(config["shortcut"]["gait_columns"])
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
        *shortcut_condition_columns,
        *shortcut_gait_columns,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("feature table is missing columns: " + ", ".join(missing))
    numeric_columns = [
        *learned_columns,
        *shortcut_condition_columns,
        *shortcut_gait_columns,
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
        shortcut_columns = [*shortcut_condition_columns, *shortcut_gait_columns]
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
                shortcut_condition=aggregate_windows(
                    shortcut_values[:, : len(shortcut_condition_columns)],
                    expected_count=expected_windows,
                    label="condition shortcuts",
                ),
                shortcut_gait=aggregate_windows(
                    shortcut_values[:, len(shortcut_condition_columns) :],
                    expected_count=expected_windows,
                    label="gait shortcuts",
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
    rows: list[RecordingRow], condition: np.ndarray, gait: np.ndarray
) -> list[Recording]:
    if condition.shape[0] != len(rows) or gait.shape[0] != len(rows):
        raise ValueError("transformed block row counts do not match recording rows")
    return [
        Recording(
            row.subject_id,
            row.recording_id,
            row.cell,
            condition[index],
            gait[index],
            row.window_ids,
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
    for role in ("target", "condition_donor", "gait_donor"):
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


def run_gfc(
    features: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    *,
    model_label: str,
    normalization: str | None = None,
    write_queries: bool = False,
    aggregate_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the public features-to-summary research path."""

    if not isinstance(model_label, str) or not model_label.strip() or model_label != model_label.strip():
        raise ValueError("model_label must be nonempty text without surrounding whitespace")
    output_dir, aggregate_output_dir = _validate_output_directories(
        output_dir, aggregate_output_dir
    )
    config = load_gfc_config(config_path)
    if features.suffix.casefold() == ".npz":
        table, _ = read_feature_table(features)
    else:
        # The GFC runner intentionally accepts its narrower documented CSV
        # contract in addition to the full feature-export table contract.
        table = pd.read_csv(features)
    all_rows, learned_columns = _recording_rows(table, config)
    split_map = dict(config["split_map"])
    unknown_splits = sorted(
        {row.split for row in all_rows} - set(split_map.values())
    )
    if unknown_splits:
        raise ValueError(
            "feature table contains unsupported split labels: "
            + ", ".join(repr(value) for value in unknown_splits)
        )
    evaluable_splits = ("development", "confirmation")
    if split not in evaluable_splits:
        raise ValueError(f"split must be one of {list(evaluable_splits)}")
    adapter_fit_split = str(config["adapter"]["fit_split"])
    normalization_fit_split = str(config["normalization"]["fit_split"])
    if normalization_fit_split != adapter_fit_split:
        raise ValueError("adapter and normalization fit splits must be identical")
    training_label = str(split_map[adapter_fit_split])
    evaluation_label = str(split_map[split])
    training_rows, training_exclusions = _complete_training_rows(
        [row for row in all_rows if row.split == training_label]
    )
    evaluation_rows = [row for row in all_rows if row.split == evaluation_label]
    if not evaluation_rows:
        raise ValueError(f"feature table has no recordings for {split!r}")
    training_subjects = {row.subject_id for row in training_rows}
    evaluation_subjects = {row.subject_id for row in evaluation_rows}
    overlap = sorted(training_subjects & evaluation_subjects)
    if overlap:
        raise ValueError(
            "training and evaluation participants must be disjoint; overlap: "
            + ", ".join(overlap[:10])
        )

    train_learned = np.stack([row.learned for row in training_rows])
    train_subjects = [row.subject_id for row in training_rows]
    train_cells = [row.cell for row in training_rows]
    alpha = float(config["adapter"]["alpha"])
    condition_adapter = fit_condition_adapter(
        train_learned, train_subjects, train_cells, alpha=alpha
    )
    gait_adapter = fit_gait_adapter(train_learned, train_subjects, train_cells, alpha=alpha)

    analysis_name = normalization or str(config["normalization"]["primary"])
    allowed_analyses = {
        str(config["normalization"]["primary"]),
        *(str(item) for item in config["normalization"].get("sensitivities", [])),
    }
    if analysis_name not in allowed_analyses:
        raise ValueError(f"normalization must be one of {sorted(allowed_analyses)}")
    fit_normalizer, dimension_policy = _normalizer_factory(analysis_name)
    scale_floor = float(config["normalization"]["scale_floor"])
    block_epsilon = float(config["normalization"]["block_l2_epsilon"])

    train_learned_condition = condition_adapter.transform(train_learned)
    train_learned_gait = gait_adapter.transform(train_learned)
    train_shortcut_condition = np.stack(
        [row.shortcut_condition for row in training_rows]
    )
    train_shortcut_gait = np.stack([row.shortcut_gait for row in training_rows])
    normalizers = {
        "learned_condition": fit_normalizer(
            train_learned_condition,
            dimension_policy=dimension_policy,
            scale_floor=scale_floor,
            zero_norm_epsilon=block_epsilon,
        ),
        "learned_gait": fit_normalizer(
            train_learned_gait,
            dimension_policy=dimension_policy,
            scale_floor=scale_floor,
            zero_norm_epsilon=block_epsilon,
        ),
        "shortcut_condition": fit_normalizer(
            train_shortcut_condition,
            dimension_policy=dimension_policy,
            scale_floor=scale_floor,
            zero_norm_epsilon=block_epsilon,
        ),
        "shortcut_gait": fit_normalizer(
            train_shortcut_gait,
            dimension_policy=dimension_policy,
            scale_floor=scale_floor,
            zero_norm_epsilon=block_epsilon,
        ),
    }

    evaluation_learned = np.stack([row.learned for row in evaluation_rows])
    learned_condition = normalizers["learned_condition"].transform(
        condition_adapter.transform(evaluation_learned)
    )
    learned_gait = normalizers["learned_gait"].transform(
        gait_adapter.transform(evaluation_learned)
    )
    shortcut_condition = normalizers["shortcut_condition"].transform(
        np.stack([row.shortcut_condition for row in evaluation_rows])
    )
    shortcut_gait = normalizers["shortcut_gait"].transform(
        np.stack([row.shortcut_gait for row in evaluation_rows])
    )

    seed = int(config["bootstrap"]["seed"])
    distance = config["distance"]
    tie_tolerance = float(config["ties"]["absolute_tolerance"])
    learned = evaluate_cohort(
        _make_recordings(evaluation_rows, learned_condition, learned_gait),
        split=split,
        seed=seed,
        representation="learned",
        condition_weight=float(distance["condition_weight"]),
        gait_weight=float(distance["gait_weight"]),
        tie_tolerance=tie_tolerance,
        zero_norm_epsilon=float(distance["zero_norm_epsilon"]),
    )
    shortcut = evaluate_cohort(
        _make_recordings(evaluation_rows, shortcut_condition, shortcut_gait),
        split=split,
        seed=seed,
        representation="shortcut",
        condition_weight=float(distance["condition_weight"]),
        gait_weight=float(distance["gait_weight"]),
        tie_tolerance=tie_tolerance,
        zero_norm_epsilon=float(distance["zero_norm_epsilon"]),
    )
    if [item.to_dict() for item in learned.exclusions] != [
        item.to_dict() for item in shortcut.exclusions
    ]:
        raise RuntimeError("learned and shortcut paths produced different exclusions")
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
        participant_rows.append(
            {
                "participant": public_labels[contrast.subject_id],
                "learned_top1": contrast.learned_top1,
                "shortcut_top1": contrast.shortcut_top1,
                "learned_minus_shortcut": contrast.difference,
                "learned_mrr": learned_by_subject[contrast.subject_id].mrr,
                "shortcut_mrr": shortcut_by_subject[contrast.subject_id].mrr,
                "learned_donor_attraction": learned_by_subject[
                    contrast.subject_id
                ].donor_attraction,
                "shortcut_donor_attraction": shortcut_by_subject[
                    contrast.subject_id
                ].donor_attraction,
            }
        )
    summary: dict[str, Any] = {
        "protocol": GFC_PROTOCOL,
        "split": split,
        "normalization": analysis_name,
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
                "query",
                "gallery",
                "distance",
                "ties",
                "adapter",
                "normalization",
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
            "donor_attraction": learned.donor_attraction,
        },
        "shortcut": {
            "top1": shortcut.top1,
            "mrr": shortcut.mrr,
            "donor_attraction": shortcut.donor_attraction,
        },
        "learned_minus_shortcut": interval.to_dict(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--normalization")
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
        write_queries=args.write_queries,
        aggregate_output_dir=args.aggregate_output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
