"""Frozen five-ladder GFC-v2 study validation, inference, and aggregation.

This module deliberately keeps private participant rows in memory.  The only files
written by :func:`summarize_study` are aggregate study artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ...config.gfc import load_gfc_config
from ...training.checkpoint import load_checkpoint
from ..features import read_feature_table
from .core import CANONICAL_CELLS
from .roles import (
    EXPECTED_ASSIGNED_COUNTS,
    ROLE_MAP_VERSION,
    build_role_map,
    load_role_map,
)


STUDY_SCHEMA_VERSION = "gfc-v2-study-aggregate-v1"
CHECKPOINT_METADATA_VERSION = "gfc-v2-training-checkpoint-v1"
ANALYSIS_FREEZE_TAG = "gfc-v2-analysis-freeze-v1"
PROTOCOL = "gfc_v2"
GALLERY = "retain_all_8"
QUERIES_PER_PARTICIPANT = 16
PRIMARY_ANALYSIS_ID = "raw_retain_all-alpha-1"
PRIMARY_NORMALIZATION = "raw_retain_all"
PRIMARY_RIDGE_ALPHA = 1.0
RUNG_ORDER = ("small", "medium", "large", "full")
EXPECTED_LADDERS = 5
EXPECTED_MODELS = EXPECTED_LADDERS * len(RUNG_ORDER)
EXPECTED_COMPLETE_COUNTS = {"development": 76, "locked_outcome": 308}
RESOLUTION = 1.0 / QUERIES_PER_PARTICIPANT
DEFAULT_PARTICIPANT_BOOTSTRAPS = 10_000
DEFAULT_CROSSED_BOOTSTRAPS = 10_000
DEFAULT_STUDY_SEED = 20_260_805

REGISTRY_COLUMNS = (
    "model_label",
    "ladder",
    "rung",
    "checkpoint_id",
    "checkpoint_path",
    "feature_path",
    "pool_seed",
    "optimization_seed",
    "unique_sequences",
    "training_exposure",
)

ANALYSIS_IDS = (
    PRIMARY_ANALYSIS_ID,
    "raw_retain_all-alpha-0.1",
    "raw_retain_all-alpha-10",
    "raw_effective_rank-alpha-1",
    "pca_effective_rank-alpha-1",
)

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

LADDER_CONTRAST_COLUMNS = (
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


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_integer(value: object, label: str) -> int:
    numeric = _finite_float(value, label)
    if numeric <= 0 or numeric != math.floor(numeric):
        raise ValueError(f"{label} must be a positive integer")
    return int(numeric)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be nonempty text without outer whitespace")
    return value


def load_study_registry(path: Path) -> pd.DataFrame:
    """Load and structurally validate the private 5x4 model registry."""

    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read study registry: {error}") from error
    return validate_study_registry(table)


def validate_study_registry(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the frozen registry contract without opening outcome values."""

    if list(table.columns) != list(REGISTRY_COLUMNS):
        raise ValueError(
            "study registry must have exactly these columns in order: "
            + ",".join(REGISTRY_COLUMNS)
        )
    if len(table) != EXPECTED_MODELS:
        raise ValueError(f"study registry must contain exactly {EXPECTED_MODELS} rows")
    result = table.copy()
    text_columns = REGISTRY_COLUMNS[:6]
    for column in text_columns:
        result[column] = [_text(value, f"registry {column}") for value in result[column]]
    if any(
        label in {".", ".."} or "/" in label or "\\" in label
        for label in result["model_label"]
    ):
        raise ValueError("registry model_label values must be safe single path components")
    for column in ("pool_seed", "optimization_seed"):
        values: list[int] = []
        for value in result[column]:
            numeric = _finite_float(value, f"registry {column}")
            if numeric < 0 or numeric != math.floor(numeric):
                raise ValueError(f"registry {column} must contain nonnegative integers")
            values.append(int(numeric))
        result[column] = values
    for column in ("unique_sequences", "training_exposure"):
        result[column] = [
            _positive_integer(value, f"registry {column}") for value in result[column]
        ]

    for unique_column in ("model_label", "checkpoint_id", "checkpoint_path", "feature_path"):
        if result[unique_column].duplicated().any():
            raise ValueError(f"registry {unique_column} values must be unique across all models")
    for path_column in ("checkpoint_path", "feature_path"):
        canonical_paths = result[path_column].map(
            lambda value: str(Path(value).expanduser().resolve())
        )
        if canonical_paths.duplicated().any():
            raise ValueError(
                f"registry {path_column} values must resolve to unique files across all models"
            )
    folded = result["model_label"].str.casefold()
    if folded.duplicated().any():
        raise ValueError("registry model labels must not collide by case")
    ladders = sorted(result["ladder"].unique())
    if len(ladders) != EXPECTED_LADDERS:
        raise ValueError(f"study registry must contain exactly {EXPECTED_LADDERS} ladders")
    expected_rungs = set(RUNG_ORDER)
    for ladder, rows in result.groupby("ladder", sort=True):
        rung_counts = rows["rung"].value_counts().to_dict()
        if set(rung_counts) != expected_rungs or any(count != 1 for count in rung_counts.values()):
            raise ValueError(
                f"ladder {ladder!r} must contain exactly one of each rung {RUNG_ORDER}"
            )
        if rows["pool_seed"].nunique() != 1 or rows["optimization_seed"].nunique() != 1:
            raise ValueError(f"ladder {ladder!r} must use one pool and optimization seed")
        ordered = rows.set_index("rung").loc[list(RUNG_ORDER)]
        sizes = ordered["unique_sequences"].to_numpy(dtype=np.int64)
        if not (2_000 <= sizes[0] <= 3_000):
            raise ValueError(f"ladder {ladder!r} small rung must be near 2.5k sequences")
        if not (20_000 <= sizes[1] <= 30_000):
            raise ValueError(f"ladder {ladder!r} medium rung must be near 25k sequences")
        if not (200_000 <= sizes[2] <= 300_000):
            raise ValueError(f"ladder {ladder!r} large rung must be near 250k sequences")
        if not sizes[3] > sizes[2]:
            raise ValueError(f"ladder {ladder!r} full rung must exceed the large rung")
    full_sizes = result.loc[result["rung"] == "full", "unique_sequences"]
    if full_sizes.nunique() != 1:
        raise ValueError("all study ladders must use the same full-data sequence count")
    if result["training_exposure"].nunique() != 1:
        raise ValueError("all study models must have equal training exposure")
    ladder_seeds = result.groupby("ladder", sort=True)[
        ["pool_seed", "optimization_seed"]
    ].first()
    if not np.equal(
        ladder_seeds["pool_seed"].to_numpy(),
        ladder_seeds["optimization_seed"].to_numpy(),
    ).all():
        raise ValueError("pool and optimization seeds must be shared within each ladder")
    if ladder_seeds["pool_seed"].nunique() != EXPECTED_LADDERS:
        raise ValueError("the five study ladders must use five distinct replicate seeds")
    rung_rank = {rung: index for index, rung in enumerate(RUNG_ORDER)}
    result["_rung_rank"] = result["rung"].map(rung_rank)
    return result.sort_values(["ladder", "_rung_rank"], kind="stable").drop(
        columns="_rung_rank"
    ).reset_index(drop=True)


def _git_output(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise ValueError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _validate_frozen_revision(repo_root: Path, freeze_tag: str) -> str:
    status = _git_output(repo_root, "status", "--porcelain")
    if status:
        raise ValueError("study preflight requires a clean Git worktree")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    tag_type = _git_output(repo_root, "cat-file", "-t", f"refs/tags/{freeze_tag}")
    if tag_type != "tag":
        raise ValueError(f"analysis freeze tag {freeze_tag!r} must be annotated")
    tagged = _git_output(repo_root, "rev-list", "-n", "1", freeze_tag)
    if tagged != head:
        raise ValueError(f"analysis freeze tag {freeze_tag!r} must resolve to HEAD")
    return head


def _validate_destinations(output_root: Path, aggregate_output: Path) -> None:
    output = output_root.expanduser().resolve()
    aggregate = aggregate_output.expanduser().resolve()
    if output == aggregate or output in aggregate.parents or aggregate in output.parents:
        raise ValueError("study output and aggregate output directories must not overlap")
    for path, label in ((output, "study output"), (aggregate, "aggregate output")):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"{label} destination must be absent or an empty directory")


def _feature_participants(path: Path) -> tuple[set[str], set[str]]:
    table, _ = read_feature_table(path)
    required = {
        "subject_id",
        "recording_id",
        "speed",
        "clothing",
        "direction",
        "window_start",
    }
    if not required <= set(table):
        raise ValueError(f"feature archive {path} lacks participant/factor columns")
    subjects = set(table["subject_id"].astype(str))
    if not subjects:
        raise ValueError(f"feature archive {path} contains no participants")
    expected_cells = {
        (cell.speed, cell.clothing, cell.direction) for cell in CANONICAL_CELLS
    }
    recording_groups = table.groupby("recording_id", sort=False, dropna=False)
    window_counts = recording_groups.size()
    distinct_starts = recording_groups["window_start"].nunique(dropna=False)
    if (window_counts != 3).any() or (distinct_starts != 3).any():
        raise ValueError(f"feature archive {path} must have three windows per recording")
    recordings = table.drop_duplicates("recording_id")
    complete = set()
    for subject_id, rows in recordings.groupby("subject_id", sort=False):
        cell_rows = list(
            zip(
                rows["speed"].astype(str),
                rows["clothing"].astype(str),
                rows["direction"].astype(str),
            )
        )
        cells = set(cell_rows)
        if len(cell_rows) != len(cells):
            raise ValueError(
                f"feature archive {path} has duplicate factorial cells for a participant"
            )
        if cells == expected_cells:
            complete.add(str(subject_id))
        elif not cells < expected_cells:
            raise ValueError(
                f"feature archive {path} has invalid factorial cells for a participant"
            )
    return subjects, complete


def _validate_checkpoint_metadata(state: Mapping[str, Any], row: Any) -> None:
    """Require the small registry agreement block embedded in every eligible model."""

    metadata = state.get("study_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"checkpoint {row.checkpoint_id!r} lacks GFC study metadata"
        )
    expected = {
        "version": CHECKPOINT_METADATA_VERSION,
        "training_dataset": "GaitLU-1M",
        "checkpoint_kind": "final_step",
        "model_label": row.model_label,
        "checkpoint_id": row.checkpoint_id,
        "pool_seed": row.pool_seed,
        "optimization_seed": row.optimization_seed,
        "unique_sequences": row.unique_sequences,
        "training_exposure": row.training_exposure,
    }
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"checkpoint {row.checkpoint_id!r} study metadata disagrees with the registry: "
            f"{mismatched}"
        )


def preflight_study(
    registry: pd.DataFrame | Path,
    *,
    role_map: Path,
    output_root: Path,
    aggregate_output: Path,
    repo_root: Path = Path("."),
    freeze_tag: str = ANALYSIS_FREEZE_TAG,
    require_frozen_revision: bool = True,
) -> dict[str, object]:
    """Perform the no-outcome preflight gate for a complete private study."""

    records = (
        load_study_registry(registry)
        if isinstance(registry, (str, Path))
        else validate_study_registry(registry)
    )
    _validate_destinations(output_root, aggregate_output)
    code_commit = (
        _validate_frozen_revision(repo_root.expanduser().resolve(), freeze_tag)
        if require_frozen_revision
        else "unverified-test-revision"
    )
    expected_subjects: set[str] | None = None
    expected_complete: set[str] | None = None
    for row in records.itertuples(index=False):
        checkpoint_path = Path(row.checkpoint_path).expanduser().resolve()
        feature_path = Path(row.feature_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint is not readable: {checkpoint_path}")
        state = load_checkpoint(checkpoint_path)
        _validate_checkpoint_metadata(state, row)
        configured_steps = state.get("config", {}).get("steps")
        if configured_steps is None or int(state.get("global_step", -1)) != int(configured_steps):
            raise ValueError(f"checkpoint {row.checkpoint_id!r} is not the final-step checkpoint")
        if not feature_path.is_file():
            raise ValueError(f"feature archive is not readable: {feature_path}")
        subjects, complete = _feature_participants(feature_path)
        if expected_subjects is None:
            expected_subjects = subjects
            expected_complete = complete
        elif subjects != expected_subjects:
            raise ValueError("all feature archives must contain the same participant coverage")
        elif complete != expected_complete:
            raise ValueError("all feature archives must contain the same complete participants")
        sidecar = feature_path.with_suffix(feature_path.suffix + ".metadata.json")
        if sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid feature metadata sidecar for {feature_path}") from error
            sidecar_checkpoint = metadata.get("checkpoint")
            if not isinstance(sidecar_checkpoint, str) or not sidecar_checkpoint.strip():
                raise ValueError(
                    f"feature metadata sidecar for {feature_path} lacks checkpoint provenance"
                )
            if Path(sidecar_checkpoint).expanduser().resolve() != checkpoint_path:
                raise ValueError(
                    f"feature archive {feature_path} does not agree with its registry checkpoint"
                )
    assert expected_subjects is not None and expected_complete is not None
    roles = load_role_map(role_map, expected_subject_ids=expected_subjects)
    counts = roles["role"].value_counts().to_dict()
    if counts != EXPECTED_ASSIGNED_COUNTS:
        raise ValueError("role-map counts differ from the frozen study contract")
    role_lookup = dict(zip(roles["subject_id"], roles["role"]))
    complete_counts = {
        role: sum(role_lookup[subject_id] == role for subject_id in expected_complete)
        for role in EXPECTED_ASSIGNED_COUNTS
    }
    if complete_counts != EXPECTED_COMPLETE_COUNTS:
        raise ValueError(
            f"complete role counts must be {EXPECTED_COMPLETE_COUNTS}, got {complete_counts}"
        )
    return {
        "eligible_models": len(records),
        "ladders": records["ladder"].nunique(),
        "role_map_version": ROLE_MAP_VERSION,
        "assigned_counts": counts,
        "complete_counts": complete_counts,
        "code_commit": code_commit,
        "analysis_freeze_tag": freeze_tag,
    }


@dataclass(frozen=True)
class PrimaryRun:
    """Private in-memory participant outcome for one primary ladder/rung cell."""

    model_label: str
    ladder: str
    rung: str
    participant_top1: Mapping[str, float]
    summary: Mapping[str, Any]


def _interval(values: np.ndarray, confidence_level: float) -> dict[str, float]:
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, (tail, 1.0 - tail), method="linear")
    return {
        "confidence_level": float(confidence_level),
        "lower": float(lower),
        "upper": float(upper),
    }


def _student_t_interval(values: np.ndarray, confidence_level: float) -> dict[str, float]:
    if values.shape != (EXPECTED_LADDERS,) or not np.isfinite(values).all():
        raise ValueError("Student-t inference requires five finite ladder contrasts")
    mean = float(np.mean(values, dtype=np.float64))
    standard_error = float(stats.sem(values, ddof=1))
    critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, df=EXPECTED_LADDERS - 1))
    half_width = critical * standard_error
    return {
        "confidence_level": float(confidence_level),
        "degrees_of_freedom": EXPECTED_LADDERS - 1,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _validate_summary_contract(runs: Sequence[PrimaryRun]) -> dict[str, Any]:
    fields = (
        "protocol",
        "gallery",
        "queries_per_participant",
        "analysis_id",
        "ridge_alpha",
        "normalization",
        "method_settings",
    )
    first = dict(runs[0].summary)
    if not isinstance(first.get("method_settings"), Mapping):
        raise ValueError("primary summaries require frozen method settings")
    for field in fields:
        if any(run.summary.get(field) != first.get(field) for run in runs[1:]):
            raise ValueError(f"primary summaries have mixed {field}")
    revision = first.get("revision")
    roles = first.get("cohort_roles")
    if not isinstance(revision, Mapping) or not isinstance(roles, Mapping):
        raise ValueError("primary summaries require revision and cohort_roles metadata")
    for name, value in (
        ("protocol", PROTOCOL),
        ("gallery", GALLERY),
        ("queries_per_participant", QUERIES_PER_PARTICIPANT),
        ("analysis_id", PRIMARY_ANALYSIS_ID),
        ("ridge_alpha", PRIMARY_RIDGE_ALPHA),
        ("normalization", PRIMARY_NORMALIZATION),
    ):
        if first.get(name) != value:
            raise ValueError(f"primary summary {name} must be {value!r}")
    for run in runs[1:]:
        if run.summary.get("revision") != revision:
            raise ValueError("primary summaries have mixed code commit or freeze tag")
        if run.summary.get("cohort_roles") != roles:
            raise ValueError("primary summaries have mixed cohort role metadata")
    if revision.get("analysis_freeze_tag") != ANALYSIS_FREEZE_TAG:
        raise ValueError("primary summaries use the wrong analysis freeze tag")
    if roles.get("version") != ROLE_MAP_VERSION:
        raise ValueError("primary summaries use the wrong role-map version")
    expected_role_metadata = {
        "fit_role": "development",
        "evaluation_role": "locked_outcome",
        "assigned_counts": EXPECTED_ASSIGNED_COUNTS,
        "complete_counts": EXPECTED_COMPLETE_COUNTS,
        "excluded_counts": {
            role: EXPECTED_ASSIGNED_COUNTS[role] - EXPECTED_COMPLETE_COUNTS[role]
            for role in EXPECTED_ASSIGNED_COUNTS
        },
    }
    if any(roles.get(key) != value for key, value in expected_role_metadata.items()):
        raise ValueError("primary summaries have invalid frozen cohort role metadata")
    return first


def infer_five_ladder_study(
    runs: Sequence[PrimaryRun],
    *,
    participant_resamples: int = DEFAULT_PARTICIPANT_BOOTSTRAPS,
    crossed_resamples: int = DEFAULT_CROSSED_BOOTSTRAPS,
    seed: int = DEFAULT_STUDY_SEED,
) -> dict[str, Any]:
    """Compute paired four-rung curves and five-ladder primary inference."""

    if len(runs) != EXPECTED_MODELS:
        raise ValueError(f"inference requires exactly {EXPECTED_MODELS} primary runs")
    participant_resamples = _positive_integer(participant_resamples, "participant_resamples")
    crossed_resamples = _positive_integer(crossed_resamples, "crossed_resamples")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _validate_summary_contract(runs)
    cells: dict[tuple[str, str], PrimaryRun] = {}
    for run in runs:
        key = (run.ladder, run.rung)
        if key in cells:
            raise ValueError(f"duplicate primary ladder/rung cell: {key!r}")
        if run.rung not in RUNG_ORDER:
            raise ValueError(f"unknown study rung {run.rung!r}")
        cells[key] = run
    ladders = sorted({run.ladder for run in runs})
    if len(ladders) != EXPECTED_LADDERS or set(cells) != {
        (ladder, rung) for ladder in ladders for rung in RUNG_ORDER
    }:
        raise ValueError("inference requires a complete five-ladder by four-rung grid")

    participant_sets = [set(run.participant_top1) for run in runs]
    if not participant_sets[0] or any(items != participant_sets[0] for items in participant_sets[1:]):
        raise ValueError("all primary runs must contain the same complete outcome participants")
    participants = sorted(participant_sets[0])
    expected_participants = EXPECTED_COMPLETE_COUNTS["locked_outcome"]
    if len(participants) != expected_participants:
        raise ValueError(
            f"primary runs require all {expected_participants} complete outcome participants"
        )
    matrices: list[np.ndarray] = []
    ladder_results: list[dict[str, Any]] = []
    generator = np.random.Generator(np.random.PCG64(seed))
    for ladder in ladders:
        matrix = np.asarray(
            [
                [
                    _finite_float(
                        cells[(ladder, rung)].participant_top1[participant],
                        "participant top1",
                    )
                    for rung in RUNG_ORDER
                ]
                for participant in participants
            ],
            dtype=np.float64,
        )
        if ((matrix < 0.0) | (matrix > 1.0)).any():
            raise ValueError("participant top1 values must lie in [0, 1]")
        matrices.append(matrix)
        indices = generator.integers(
            0,
            len(participants),
            size=(participant_resamples, len(participants)),
            endpoint=False,
        )
        sampled_rungs = np.mean(matrix[indices], axis=1, dtype=np.float64)
        rung_means = np.mean(matrix, axis=0, dtype=np.float64)
        endpoint_samples = sampled_rungs[:, -1] - sampled_rungs[:, 0]
        ladder_results.append(
            {
                "ladder": ladder,
                "participant_count": len(participants),
                "rung_means": {
                    rung: float(rung_means[index]) for index, rung in enumerate(RUNG_ORDER)
                },
                "full_minus_small": float(rung_means[-1] - rung_means[0]),
                "participant_bootstrap": {
                    "resamples": participant_resamples,
                    "seed": seed,
                    "rung_intervals_95": {
                        rung: _interval(sampled_rungs[:, index], 0.95)
                        for index, rung in enumerate(RUNG_ORDER)
                    },
                    "endpoint_interval_95": _interval(endpoint_samples, 0.95),
                },
            }
        )

    contrasts = np.asarray(
        [item["full_minus_small"] for item in ladder_results], dtype=np.float64
    )
    t95 = _student_t_interval(contrasts, 0.95)
    t90 = _student_t_interval(contrasts, 0.90)
    mean_contrast = float(np.mean(contrasts, dtype=np.float64))

    crossed = np.empty(crossed_resamples, dtype=np.float64)
    matrix_stack = np.stack(matrices)
    for index in range(crossed_resamples):
        ladder_draw = generator.integers(0, EXPECTED_LADDERS, EXPECTED_LADDERS)
        participant_draw = generator.integers(0, len(participants), len(participants))
        sampled = matrix_stack[ladder_draw][:, participant_draw, :]
        endpoints = np.mean(sampled[:, :, -1] - sampled[:, :, 0], axis=1)
        crossed[index] = np.mean(endpoints, dtype=np.float64)
    crossed_interval = _interval(crossed, 0.95)
    crossed_interval.update({"resamples": crossed_resamples, "seed": seed})

    superiority = t95["lower"] > 0.0
    equivalence = t90["lower"] >= -RESOLUTION and t90["upper"] <= RESOLUTION
    if superiority and mean_contrast >= RESOLUTION:
        decision = "Meaningful positive"
    elif superiority:
        decision = "Positive but small"
    elif equivalence:
        decision = "Equivalent at the 6.25-point resolution"
    else:
        decision = "Inconclusive"
    return {
        "ladder_count": EXPECTED_LADDERS,
        "participant_count": len(participants),
        "rung_order": list(RUNG_ORDER),
        "resolution": RESOLUTION,
        "mean_full_minus_small": mean_contrast,
        "ladder_contrasts": [
            {"ladder": item["ladder"], "full_minus_small": item["full_minus_small"]}
            for item in ladder_results
        ],
        "participant_bootstraps": ladder_results,
        "t_interval_95": t95,
        "t_interval_90": t90,
        "crossed_bootstrap_95": crossed_interval,
        "superiority": bool(superiority),
        "equivalence": bool(equivalence),
        "decision": decision,
    }


def _nested(summary: Mapping[str, Any], *paths: Sequence[str]) -> float:
    for path in paths:
        value: Any = summary
        try:
            for key in path:
                value = value[key]
        except (KeyError, TypeError):
            continue
        return _finite_float(value, ".".join(path))
    raise ValueError("summary is missing aggregate metric: " + " or ".join(".".join(p) for p in paths))


def _read_analysis(directory: Path, *, participants: bool) -> tuple[dict[str, Any], dict[str, float]]:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing analysis summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid analysis summary: {summary_path}") from error
    rows: dict[str, float] = {}
    if participants:
        participant_path = directory / "participants.csv"
        if not participant_path.is_file():
            raise ValueError(f"missing private participant output: {participant_path}")
        table = pd.read_csv(participant_path, dtype={"participant": str})
        if "participant" not in table or "learned_top1" not in table:
            raise ValueError("participant output requires participant and learned_top1 columns")
        if table["participant"].duplicated().any() or table.empty:
            raise ValueError("participant output must contain unique participant keys")
        rows = {
            _text(participant, "private participant key"): _finite_float(value, "learned_top1")
            for participant, value in zip(table["participant"], table["learned_top1"])
        }
    return summary, rows


def _run_table_row(registry_row: Any, analyses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary = analyses[PRIMARY_ANALYSIS_ID]
    controls = primary.get("independent_factor_controls", {})
    hard = controls.get("hard", {})
    soft = controls.get("soft", {})
    temperature = controls.get("temperature", {})
    return {
        "model_label": registry_row.model_label,
        "ladder": registry_row.ladder,
        "rung": registry_row.rung,
        "checkpoint_id": registry_row.checkpoint_id,
        "pool_seed": registry_row.pool_seed,
        "optimization_seed": registry_row.optimization_seed,
        "unique_sequences": registry_row.unique_sequences,
        "training_exposure": registry_row.training_exposure,
        "primary_analysis_id": PRIMARY_ANALYSIS_ID,
        "primary_ridge_alpha": PRIMARY_RIDGE_ALPHA,
        "primary_normalization": PRIMARY_NORMALIZATION,
        "learned_top1": _nested(primary, ("learned", "top1")),
        "learned_mrr": _nested(primary, ("learned", "mrr")),
        "shortcut_top1": _nested(primary, ("shortcut", "top1")),
        "shortcut_mrr": _nested(primary, ("shortcut", "mrr")),
        "hard_control_top1": _nested(hard, ("top1",)),
        "hard_control_mrr": _nested(hard, ("mrr",)),
        "soft_control_top1": _nested(soft, ("top1",)),
        "soft_control_mrr": _nested(soft, ("mrr",)),
        "soft_control_target_probability": _nested(
            soft, ("target_probability",), ("mean_target_probability",)
        ),
        "soft_control_target_nll": _nested(soft, ("target_nll",), ("mean_target_nll",)),
        "soft_temperature": _nested(temperature, ("fitted_temperature",), ("temperature",)),
        "alpha_0_1_learned_top1": _nested(
            analyses["raw_retain_all-alpha-0.1"], ("learned", "top1")
        ),
        "alpha_10_learned_top1": _nested(
            analyses["raw_retain_all-alpha-10"], ("learned", "top1")
        ),
        "raw_effective_rank_learned_top1": _nested(
            analyses["raw_effective_rank-alpha-1"], ("learned", "top1")
        ),
        "pca_effective_rank_learned_top1": _nested(
            analyses["pca_effective_rank-alpha-1"], ("learned", "top1")
        ),
    }


def _validate_model_analyses(
    registry_row: Any, analyses: Mapping[str, Mapping[str, Any]]
) -> None:
    expected_settings = {
        "raw_retain_all-alpha-1": ("raw_retain_all", 1.0),
        "raw_retain_all-alpha-0.1": ("raw_retain_all", 0.1),
        "raw_retain_all-alpha-10": ("raw_retain_all", 10.0),
        "raw_effective_rank-alpha-1": ("raw_effective_rank", 1.0),
        "pca_effective_rank-alpha-1": ("pca_effective_rank", 1.0),
    }
    primary = analyses[PRIMARY_ANALYSIS_ID]
    for analysis_id, summary in analyses.items():
        normalization, alpha = expected_settings[analysis_id]
        if summary.get("analysis_id") != analysis_id:
            raise ValueError(f"analysis directory {analysis_id!r} contains mismatched output")
        if summary.get("normalization") != normalization or summary.get("ridge_alpha") != alpha:
            raise ValueError(f"analysis {analysis_id!r} records unsupported effective settings")
        for field in (
            "protocol",
            "gallery",
            "queries_per_participant",
            "revision",
            "cohort_roles",
            "method_settings",
        ):
            if summary.get(field) != primary.get(field):
                raise ValueError(f"model analyses have mixed {field}")
        model = summary.get("model", {})
        expected_model = {
            "label": registry_row.model_label,
            "ladder": registry_row.ladder,
            "rung": registry_row.rung,
            "checkpoint_id": registry_row.checkpoint_id,
            "pool_seed": registry_row.pool_seed,
            "optimization_seed": registry_row.optimization_seed,
            "unique_sequences": registry_row.unique_sequences,
            "training_exposure": registry_row.training_exposure,
        }
        if not isinstance(model, Mapping) or any(
            model.get(key) != value for key, value in expected_model.items()
        ):
            raise ValueError("analysis model metadata does not match the private registry")


def _assert_no_private_content(value: object) -> None:
    forbidden_keys = {
        "subject_id",
        "participant",
        "participants",
        "participant_rows",
        "feature_path",
        "checkpoint_path",
        "role_map_path",
    }
    if isinstance(value, Mapping):
        overlap = forbidden_keys & set(value)
        if overlap:
            raise ValueError("aggregate artifact contains private fields: " + ",".join(sorted(overlap)))
        for item in value.values():
            _assert_no_private_content(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_private_content(item)


def summarize_study(
    registry: pd.DataFrame | Path,
    output_root: Path,
    aggregate_output: Path,
    *,
    participant_resamples: int = DEFAULT_PARTICIPANT_BOOTSTRAPS,
    crossed_resamples: int = DEFAULT_CROSSED_BOOTSTRAPS,
    seed: int = DEFAULT_STUDY_SEED,
) -> dict[str, Any]:
    """Read complete private outputs and write only the three aggregate artifacts."""

    records = (
        load_study_registry(registry)
        if isinstance(registry, (str, Path))
        else validate_study_registry(registry)
    )
    aggregate_output = aggregate_output.expanduser().resolve()
    if aggregate_output.exists() and (
        not aggregate_output.is_dir() or any(aggregate_output.iterdir())
    ):
        raise ValueError("aggregate output destination must be absent or empty")
    root = output_root.expanduser().resolve()
    primary_runs: list[PrimaryRun] = []
    run_rows: list[dict[str, Any]] = []
    first_summary: dict[str, Any] | None = None
    for registry_row in records.itertuples(index=False):
        analyses: dict[str, Mapping[str, Any]] = {}
        participant_top1: dict[str, float] = {}
        for analysis_id in ANALYSIS_IDS:
            summary, participants = _read_analysis(
                root / registry_row.model_label / analysis_id,
                participants=analysis_id == PRIMARY_ANALYSIS_ID,
            )
            analyses[analysis_id] = summary
            if participants:
                participant_top1 = participants
        _validate_model_analyses(registry_row, analyses)
        primary = dict(analyses[PRIMARY_ANALYSIS_ID])
        if first_summary is None:
            first_summary = primary
        primary_runs.append(
            PrimaryRun(
                model_label=registry_row.model_label,
                ladder=registry_row.ladder,
                rung=registry_row.rung,
                participant_top1=participant_top1,
                summary=primary,
            )
        )
        run_rows.append(_run_table_row(registry_row, analyses))
    assert first_summary is not None
    inference = infer_five_ladder_study(
        primary_runs,
        participant_resamples=participant_resamples,
        crossed_resamples=crossed_resamples,
        seed=seed,
    )
    bootstrap_by_ladder = {
        item["ladder"]: item for item in inference["participant_bootstraps"]
    }
    ladder_rows = []
    for item in inference["ladder_contrasts"]:
        details = bootstrap_by_ladder[item["ladder"]]
        endpoint = details["participant_bootstrap"]["endpoint_interval_95"]
        ladder_rows.append(
            {
                "ladder": item["ladder"],
                "participant_count": details["participant_count"],
                **{
                    f"{rung}_top1": details["rung_means"][rung] for rung in RUNG_ORDER
                },
                "full_minus_small": item["full_minus_small"],
                "participant_bootstrap_95_lower": endpoint["lower"],
                "participant_bootstrap_95_upper": endpoint["upper"],
            }
        )
    public_roles = {
        key: first_summary["cohort_roles"][key]
        for key in (
            "version",
            "fit_role",
            "evaluation_role",
            "assigned_counts",
            "complete_counts",
            "excluded_counts",
        )
    }
    outcome = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "gallery": GALLERY,
        "queries_per_participant": QUERIES_PER_PARTICIPANT,
        "analysis_id": PRIMARY_ANALYSIS_ID,
        "ridge_alpha": PRIMARY_RIDGE_ALPHA,
        "normalization": PRIMARY_NORMALIZATION,
        "revision": dict(first_summary["revision"]),
        "cohort_roles": public_roles,
        "inference": inference,
        "artifacts": {
            "run_table": "run_table.csv",
            "run_table_rows": EXPECTED_MODELS,
            "ladder_contrasts": "ladder_contrasts.csv",
            "ladder_contrast_rows": EXPECTED_LADDERS,
        },
    }
    _assert_no_private_content(outcome)
    _assert_no_private_content(run_rows)
    _assert_no_private_content(ladder_rows)
    aggregate_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows, columns=RUN_TABLE_COLUMNS).to_csv(
        aggregate_output / "run_table.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(ladder_rows, columns=LADDER_CONTRAST_COLUMNS).to_csv(
        aggregate_output / "ladder_contrasts.csv", index=False, lineterminator="\n"
    )
    (aggregate_output / "outcome_summary.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outcome


def run_study(
    registry: pd.DataFrame,
    *,
    load_archive: Callable[[Path], object],
    evaluate_analysis: Callable[[object, pd.Series, str], object],
) -> list[object]:
    """Load each archive once and stop at the first failed declared analysis.

    The injected callbacks keep orchestration independent from the single-model I/O
    layer.  The production CLI supplies those callbacks after preflight.
    """

    records = validate_study_registry(registry)
    outputs: list[object] = []
    for _, row in records.iterrows():
        archive = load_archive(Path(row["feature_path"]))
        for analysis_id in ANALYSIS_IDS:
            outputs.append(evaluate_analysis(archive, row, analysis_id))
    return outputs


def run_registered_study(
    registry: pd.DataFrame | Path,
    *,
    config_path: Path,
    role_map: Path,
    output_root: Path,
    aggregate_output: Path,
    repo_root: Path = Path("."),
    freeze_tag: str = ANALYSIS_FREEZE_TAG,
) -> list[dict[str, Any]]:
    """Preflight and evaluate all five analyses for every registered model."""

    records = (
        load_study_registry(registry)
        if isinstance(registry, (str, Path))
        else validate_study_registry(registry)
    )
    preflight = preflight_study(
        records,
        role_map=role_map,
        output_root=output_root,
        aggregate_output=aggregate_output,
        repo_root=repo_root,
        freeze_tag=freeze_tag,
    )
    config = load_gfc_config(config_path)
    declared = {
        item["analysis_id"]: (item["normalization"], float(item["ridge_alpha"]))
        for item in config["analyses"]
    }
    if tuple(declared) != ANALYSIS_IDS:
        raise ValueError("evaluator config does not declare the frozen five study analyses")
    from .runner import run_gfc_analysis_suite

    root = output_root.expanduser().resolve()
    revision = {
        "code_commit": preflight["code_commit"],
        "analysis_freeze_tag": freeze_tag,
    }
    results: list[dict[str, Any]] = []
    for row in records.itertuples(index=False):
        feature_path = Path(row.feature_path).expanduser().resolve()
        table, _ = read_feature_table(feature_path, validate=False)
        model_metadata = {
            "checkpoint_id": row.checkpoint_id,
            "checkpoint_path": row.checkpoint_path,
            "ladder": row.ladder,
            "rung": row.rung,
            "pool_seed": row.pool_seed,
            "optimization_seed": row.optimization_seed,
            "unique_sequences": row.unique_sequences,
            "training_exposure": row.training_exposure,
        }
        model_results = run_gfc_analysis_suite(
            table,
            config_path,
            root / row.model_label,
            model_label=row.model_label,
            role_map=role_map,
            model_metadata=model_metadata,
            revision=revision,
        )
        for analysis_id, result in zip(ANALYSIS_IDS, model_results):
            if result.get("analysis_id") != analysis_id:
                raise RuntimeError("single-model runner returned the wrong analysis")
            results.append(result)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    role_map = subparsers.add_parser("build-role-map")
    role_map.add_argument("--manifest", type=Path, required=True)
    role_map.add_argument("--output", type=Path, required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--registry", type=Path, required=True)
    preflight.add_argument("--role-map", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--aggregate-output", type=Path, required=True)
    preflight.add_argument("--repo-root", type=Path, default=Path("."))
    preflight.add_argument("--freeze-tag", default=ANALYSIS_FREEZE_TAG)
    run = subparsers.add_parser("run")
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--role-map", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--aggregate-output", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--freeze-tag", default=ANALYSIS_FREEZE_TAG)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--registry", type=Path, required=True)
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.add_argument("--aggregate-output", type=Path, required=True)
    summarize.add_argument("--participant-resamples", type=int, default=DEFAULT_PARTICIPANT_BOOTSTRAPS)
    summarize.add_argument("--crossed-resamples", type=int, default=DEFAULT_CROSSED_BOOTSTRAPS)
    summarize.add_argument("--seed", type=int, default=DEFAULT_STUDY_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "build-role-map":
        roles = build_role_map(args.manifest, args.output)
        result = {
            "role_map_version": ROLE_MAP_VERSION,
            "output": str(args.output),
            "assigned_counts": roles["role"].value_counts().to_dict(),
        }
    elif args.command == "preflight":
        result = preflight_study(
            args.registry,
            role_map=args.role_map,
            output_root=args.output_root,
            aggregate_output=args.aggregate_output,
            repo_root=args.repo_root,
            freeze_tag=args.freeze_tag,
        )
    elif args.command == "run":
        outputs = run_registered_study(
            args.registry,
            config_path=args.config,
            role_map=args.role_map,
            output_root=args.output_root,
            aggregate_output=args.aggregate_output,
            repo_root=args.repo_root,
            freeze_tag=args.freeze_tag,
        )
        result = {"models": EXPECTED_MODELS, "analyses": len(outputs)}
    else:
        result = summarize_study(
            args.registry,
            args.output_root,
            args.aggregate_output,
            participant_resamples=args.participant_resamples,
            crossed_resamples=args.crossed_resamples,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "ANALYSIS_FREEZE_TAG",
    "ANALYSIS_IDS",
    "CHECKPOINT_METADATA_VERSION",
    "LADDER_CONTRAST_COLUMNS",
    "PRIMARY_ANALYSIS_ID",
    "PrimaryRun",
    "REGISTRY_COLUMNS",
    "RUN_TABLE_COLUMNS",
    "STUDY_SCHEMA_VERSION",
    "infer_five_ladder_study",
    "load_study_registry",
    "main",
    "preflight_study",
    "run_study",
    "run_registered_study",
    "summarize_study",
    "validate_study_registry",
]
