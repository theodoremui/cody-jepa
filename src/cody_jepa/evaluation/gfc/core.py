"""Grounded Factorial Completion for the Health&Gait 2 x 2 x 2 design.

The evaluator consumes one pair of recording-level representation blocks per
factorial cell.  Window aggregation, donor exclusion, deterministic tie
handling, and participant weighting are scientific parts of the method.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import itertools
import math

import numpy as np


SPEEDS = ("UGS", "FGS")
CLOTHING = ("WoJ", "WJ")
DIRECTIONS = ("R2L", "L2R")
GFC_PROTOCOL = "legacy_donor_excluded_v1"
EXPECTED_WINDOWS = 3
EXPECTED_CELLS = 8
EXPECTED_QUERIES = 24
EXPECTED_GALLERY_SIZE = 6
TIE_ABS_TOLERANCE = 1e-12
ZERO_NORM_EPSILON = 1e-12
SCORING_DTYPE = np.dtype(np.float64)


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be nonempty text without outer whitespace")
    return value


@dataclass(frozen=True, order=True)
class Cell:
    """One Health&Gait factorial cell."""

    speed: str
    clothing: str
    direction: str

    def __post_init__(self) -> None:
        if self.speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}, got {self.speed!r}")
        if self.clothing not in CLOTHING:
            raise ValueError(f"clothing must be one of {CLOTHING}, got {self.clothing!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")

    @property
    def canonical_index(self) -> tuple[int, int, int]:
        return (
            SPEEDS.index(self.speed),
            CLOTHING.index(self.clothing),
            DIRECTIONS.index(self.direction),
        )

    @property
    def key(self) -> str:
        return f"{self.speed}:{self.clothing}:{self.direction}"

    def to_dict(self) -> dict[str, str]:
        return {
            "speed": self.speed,
            "clothing": self.clothing,
            "direction": self.direction,
        }


CANONICAL_CELLS = tuple(
    Cell(speed, clothing, direction)
    for speed, clothing, direction in itertools.product(SPEEDS, CLOTHING, DIRECTIONS)
)


def _finite_vector(value: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    vector = np.array(value, dtype=SCORING_DTYPE, copy=True)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a nonempty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    vector.setflags(write=False)
    return vector


def aggregate_windows(
    windows: Sequence[Sequence[float] | np.ndarray],
    *,
    expected_count: int = EXPECTED_WINDOWS,
    label: str = "features",
) -> np.ndarray:
    """Average aligned window vectors in float64 before any fitted transform."""

    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    try:
        matrix = np.asarray(windows, dtype=SCORING_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} windows must have one consistent width") from error
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{label} windows must have shape [W, D]")
    if matrix.shape[0] != expected_count:
        raise ValueError(f"{label} requires exactly {expected_count} windows, got {matrix.shape[0]}")
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"{label} windows contain non-finite values")
    result = np.mean(matrix, axis=0, dtype=SCORING_DTYPE)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Recording:
    """One recording-level condition block and gait block."""

    subject_id: str
    recording_id: str
    cell: Cell
    condition_block: np.ndarray = field(repr=False, compare=False)
    gait_block: np.ndarray = field(repr=False, compare=False)
    window_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _nonempty_text(self.subject_id, "subject_id"))
        object.__setattr__(self, "recording_id", _nonempty_text(self.recording_id, "recording_id"))
        if not isinstance(self.cell, Cell):
            raise TypeError("cell must be a Cell")
        object.__setattr__(
            self, "condition_block", _finite_vector(self.condition_block, "condition_block")
        )
        object.__setattr__(self, "gait_block", _finite_vector(self.gait_block, "gait_block"))
        ids = tuple(_nonempty_text(item, "window_id") for item in self.window_ids)
        if len(ids) != EXPECTED_WINDOWS:
            raise ValueError(f"recording requires exactly {EXPECTED_WINDOWS} source windows")
        if len(set(ids)) != len(ids):
            raise ValueError("window_ids must be unique within a recording")
        object.__setattr__(self, "window_ids", ids)

    @classmethod
    def from_windows(
        cls,
        *,
        subject_id: str,
        recording_id: str,
        cell: Cell,
        condition_windows: Sequence[Sequence[float] | np.ndarray],
        gait_windows: Sequence[Sequence[float] | np.ndarray],
        window_ids: Sequence[str],
    ) -> "Recording":
        ids = tuple(window_ids)
        if len(condition_windows) != len(gait_windows) or len(ids) != len(condition_windows):
            raise ValueError("condition, gait, and window identifiers must have equal counts")
        return cls(
            subject_id=subject_id,
            recording_id=recording_id,
            cell=cell,
            condition_block=aggregate_windows(condition_windows, label="condition"),
            gait_block=aggregate_windows(gait_windows, label="gait"),
            window_ids=ids,
        )


@dataclass(frozen=True)
class Query:
    query_index: int
    subject_id: str
    target_id: str
    target_cell: Cell
    condition_donor_id: str
    condition_donor_cell: Cell
    gait_donor_id: str
    gait_donor_cell: Cell
    gallery_ids: tuple[str, ...]
    gallery_cells: tuple[Cell, ...]


def _opposite_speed(speed: str) -> str:
    return SPEEDS[1 - SPEEDS.index(speed)]


def _recording_index(recordings: Sequence[Recording]) -> dict[Cell, Recording]:
    if not recordings:
        raise ValueError("a participant has no recordings")
    if len({item.subject_id for item in recordings}) != 1:
        raise ValueError("participant recordings must have one subject_id")
    identifiers = [item.recording_id for item in recordings]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("recording_id values must be unique within a participant")
    windows = [window for item in recordings for window in item.window_ids]
    if len(set(windows)) != len(windows):
        raise ValueError("window_id values must be unique across participant recordings")
    index: dict[Cell, Recording] = {}
    for recording in recordings:
        if recording.cell in index:
            raise ValueError(f"duplicate factorial cell {recording.cell.key!r}")
        index[recording.cell] = recording
    if len({item.condition_block.size for item in recordings}) != 1:
        raise ValueError("condition block width must be constant within a participant")
    if len({item.gait_block.size for item in recordings}) != 1:
        raise ValueError("gait block width must be constant within a participant")
    return index


def missing_cells(recordings: Sequence[Recording]) -> tuple[Cell, ...]:
    index = _recording_index(recordings)
    return tuple(cell for cell in CANONICAL_CELLS if cell not in index)


def build_queries(recordings: Sequence[Recording]) -> tuple[Query, ...]:
    """Build 24 canonical queries, removing both donors from each gallery."""

    index = _recording_index(recordings)
    absent = tuple(cell for cell in CANONICAL_CELLS if cell not in index)
    if absent or len(index) != EXPECTED_CELLS:
        raise ValueError(
            "fixed GFC requires all eight cells; missing cells: "
            + (", ".join(cell.key for cell in absent) or "none")
        )
    subject_id = next(iter(index.values())).subject_id
    queries: list[Query] = []
    for target_cell in CANONICAL_CELLS:
        target = index[target_cell]
        condition_cell = Cell(
            _opposite_speed(target_cell.speed), target_cell.clothing, target_cell.direction
        )
        condition_donor = index[condition_cell]
        gait_cells = tuple(
            cell
            for cell in CANONICAL_CELLS
            if cell.speed == target_cell.speed
            and (cell.clothing, cell.direction) != (target_cell.clothing, target_cell.direction)
        )
        for gait_cell in gait_cells:
            gait_donor = index[gait_cell]
            donor_ids = {condition_donor.recording_id, gait_donor.recording_id}
            gallery_recordings = tuple(
                index[cell] for cell in CANONICAL_CELLS if index[cell].recording_id not in donor_ids
            )
            if (
                len(gallery_recordings) != EXPECTED_GALLERY_SIZE
                or target.recording_id not in {item.recording_id for item in gallery_recordings}
            ):
                raise RuntimeError("constructed gallery violates donor exclusion")
            queries.append(
                Query(
                    query_index=len(queries),
                    subject_id=subject_id,
                    target_id=target.recording_id,
                    target_cell=target_cell,
                    condition_donor_id=condition_donor.recording_id,
                    condition_donor_cell=condition_cell,
                    gait_donor_id=gait_donor.recording_id,
                    gait_donor_cell=gait_cell,
                    gallery_ids=tuple(item.recording_id for item in gallery_recordings),
                    gallery_cells=tuple(item.cell for item in gallery_recordings),
                )
            )
    if len(queries) != EXPECTED_QUERIES:
        raise RuntimeError("factor design did not produce 24 queries")
    return tuple(queries)


def cosine_distance(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> float:
    """Float64 cosine distance with stable finite-vector scaling."""

    if not math.isfinite(zero_norm_epsilon) or zero_norm_epsilon <= 0:
        raise ValueError("zero_norm_epsilon must be finite and positive")
    a = np.asarray(left, dtype=SCORING_DTYPE)
    b = np.asarray(right, dtype=SCORING_DTYPE)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("cosine inputs must be nonempty vectors with equal shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("cosine inputs contain non-finite values")

    def unit(vector: np.ndarray) -> np.ndarray | None:
        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            return None
        scaled = vector / scale
        norm = float(np.sqrt(np.sum(scaled * scaled, dtype=SCORING_DTYPE)))
        if scale <= zero_norm_epsilon / norm:
            return None
        return scaled / norm

    unit_a = unit(a)
    unit_b = unit(b)
    if unit_a is None or unit_b is None:
        return 1.0
    similarity = float(np.dot(unit_a, unit_b))
    return 1.0 - min(1.0, max(-1.0, similarity))


def _validate_weights(condition_weight: float, gait_weight: float) -> tuple[float, float]:
    condition = float(condition_weight)
    gait = float(gait_weight)
    if not math.isfinite(condition) or not math.isfinite(gait) or condition < 0 or gait < 0:
        raise ValueError("distance weights must be finite and nonnegative")
    if not math.isclose(condition + gait, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("distance weights must sum to one")
    return condition, gait


def mixed_distance(
    query_condition: np.ndarray,
    query_gait: np.ndarray,
    gallery_recording: Recording,
    *,
    condition_weight: float = 0.5,
    gait_weight: float = 0.5,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> tuple[float, float, float]:
    condition_weight, gait_weight = _validate_weights(condition_weight, gait_weight)
    condition = cosine_distance(
        query_condition,
        gallery_recording.condition_block,
        zero_norm_epsilon=zero_norm_epsilon,
    )
    gait = cosine_distance(
        query_gait,
        gallery_recording.gait_block,
        zero_norm_epsilon=zero_norm_epsilon,
    )
    combined = float(condition_weight * condition + gait_weight * gait)
    return condition, gait, combined


@dataclass(frozen=True)
class RankScores:
    strictly_closer_count: int
    tie_size: int
    average_rank: float
    top1: float
    reciprocal_rank: float


def rank_target(
    distances: Mapping[str, float],
    target_id: str,
    *,
    tolerance: float = TIE_ABS_TOLERANCE,
) -> RankScores:
    if not distances or target_id not in distances:
        raise ValueError("target must be present in nonempty distances")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    values = {str(key): float(value) for key, value in distances.items()}
    if len(values) != len(distances):
        raise ValueError("distance identifiers must remain unique as text")
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("rank distances contain non-finite values")
    target = values[target_id]
    closer = sum(value < target - tolerance for value in values.values())
    tie_size = sum(abs(value - target) <= tolerance for value in values.values())
    average_rank = closer + (tie_size + 1) / 2.0
    return RankScores(
        strictly_closer_count=closer,
        tie_size=tie_size,
        average_rank=average_rank,
        top1=(1.0 / tie_size if closer == 0 else 0.0),
        reciprocal_rank=1.0 / average_rank,
    )


@dataclass(frozen=True)
class DistanceEntry:
    recording_id: str
    cell: Cell
    condition_distance: float
    gait_distance: float
    combined_distance: float

    def to_dict(self, target_id: str) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "cell": self.cell.to_dict(),
            "condition_distance": self.condition_distance,
            "gait_distance": self.gait_distance,
            "combined_distance": self.combined_distance,
            "is_target": self.recording_id == target_id,
        }


@dataclass(frozen=True)
class QueryResult:
    """Numerical result for one scientifically identified query."""

    split: str
    seed: int
    representation: str
    query: Query
    distances: tuple[DistanceEntry, ...]
    rank: RankScores
    donor_attraction: float

    @property
    def top1(self) -> float:
        return self.rank.top1

    @property
    def reciprocal_rank(self) -> float:
        return self.rank.reciprocal_rank

    @property
    def scientific_key(self) -> tuple[object, ...]:
        return (
            self.split,
            self.seed,
            self.query.subject_id,
            self.query.target_id,
            self.query.target_cell,
            self.query.condition_donor_id,
            self.query.condition_donor_cell,
            self.query.gait_donor_id,
            self.query.gait_donor_cell,
            tuple(zip(self.query.gallery_ids, self.query.gallery_cells)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "seed": self.seed,
            "representation": self.representation,
            "subject_id": self.query.subject_id,
            "query_index": self.query.query_index,
            "target": {"recording_id": self.query.target_id, "cell": self.query.target_cell.to_dict()},
            "condition_donor": {
                "recording_id": self.query.condition_donor_id,
                "cell": self.query.condition_donor_cell.to_dict(),
            },
            "gait_donor": {
                "recording_id": self.query.gait_donor_id,
                "cell": self.query.gait_donor_cell.to_dict(),
            },
            "gallery": [item.to_dict(self.query.target_id) for item in self.distances],
            "strictly_closer_count": self.rank.strictly_closer_count,
            "tie_size": self.rank.tie_size,
            "average_rank": self.rank.average_rank,
            "top1": self.top1,
            "reciprocal_rank": self.reciprocal_rank,
            "donor_attraction": self.donor_attraction,
        }


def evaluate_query(
    query: Query,
    recordings_by_id: Mapping[str, Recording],
    *,
    split: str,
    seed: int,
    representation: str,
    condition_weight: float = 0.5,
    gait_weight: float = 0.5,
    tie_tolerance: float = TIE_ABS_TOLERANCE,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> QueryResult:
    split = _nonempty_text(split, "split")
    representation = _nonempty_text(representation, "representation")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(recordings_by_id) != {item.recording_id for item in recordings_by_id.values()}:
        raise ValueError("recordings_by_id keys must match recording_id values")
    canonical = build_queries(tuple(recordings_by_id.values()))
    if not 0 <= query.query_index < len(canonical) or query != canonical[query.query_index]:
        raise ValueError("query does not match the canonical factor construction")
    if query.condition_donor_id in query.gallery_ids or query.gait_donor_id in query.gallery_ids:
        raise ValueError("both donors must be excluded from the gallery")
    condition_donor = recordings_by_id[query.condition_donor_id]
    gait_donor = recordings_by_id[query.gait_donor_id]
    entries: list[DistanceEntry] = []
    for identifier, cell in zip(query.gallery_ids, query.gallery_cells):
        recording = recordings_by_id[identifier]
        if recording.cell != cell:
            raise ValueError("gallery factor identity differs from the query")
        condition, gait, combined = mixed_distance(
            condition_donor.condition_block,
            gait_donor.gait_block,
            recording,
            condition_weight=condition_weight,
            gait_weight=gait_weight,
            zero_norm_epsilon=zero_norm_epsilon,
        )
        entries.append(DistanceEntry(identifier, cell, condition, gait, combined))
    rank = rank_target(
        {entry.recording_id: entry.combined_distance for entry in entries},
        query.target_id,
        tolerance=tie_tolerance,
    )
    _, _, donor_distance = mixed_distance(
        condition_donor.condition_block,
        gait_donor.gait_block,
        gait_donor,
        condition_weight=condition_weight,
        gait_weight=gait_weight,
        zero_norm_epsilon=zero_norm_epsilon,
    )
    target_distance = next(
        entry.combined_distance for entry in entries if entry.recording_id == query.target_id
    )
    if abs(target_distance - donor_distance) <= tie_tolerance:
        attraction = 0.5
    else:
        attraction = float(target_distance < donor_distance - tie_tolerance)
    return QueryResult(split, seed, representation, query, tuple(entries), rank, attraction)


@dataclass(frozen=True)
class ParticipantScores:
    subject_id: str
    representation: str
    queries: tuple[QueryResult, ...]
    top1: float
    mrr: float
    donor_attraction: float

    def to_dict(self, *, include_queries: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "subject_id": self.subject_id,
            "representation": self.representation,
            "query_count": len(self.queries),
            "top1": self.top1,
            "mrr": self.mrr,
            "donor_attraction": self.donor_attraction,
        }
        if include_queries:
            result["queries"] = [query.to_dict() for query in self.queries]
        return result


def evaluate_participant(
    recordings: Sequence[Recording],
    *,
    split: str = "development",
    seed: int = 0,
    representation: str = "learned",
    condition_weight: float = 0.5,
    gait_weight: float = 0.5,
    tie_tolerance: float = TIE_ABS_TOLERANCE,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> ParticipantScores:
    queries = build_queries(recordings)
    by_id = {item.recording_id: item for item in recordings}
    results = tuple(
        evaluate_query(
            query,
            by_id,
            split=split,
            seed=seed,
            representation=representation,
            condition_weight=condition_weight,
            gait_weight=gait_weight,
            tie_tolerance=tie_tolerance,
            zero_norm_epsilon=zero_norm_epsilon,
        )
        for query in queries
    )
    return ParticipantScores(
        subject_id=queries[0].subject_id,
        representation=representation,
        queries=results,
        top1=float(np.mean([item.top1 for item in results], dtype=SCORING_DTYPE)),
        mrr=float(np.mean([item.reciprocal_rank for item in results], dtype=SCORING_DTYPE)),
        donor_attraction=float(
            np.mean([item.donor_attraction for item in results], dtype=SCORING_DTYPE)
        ),
    )


@dataclass(frozen=True)
class ParticipantExclusion:
    subject_id: str
    reason: str
    missing_cells: tuple[Cell, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "reason": self.reason,
            "missing_cells": [cell.to_dict() for cell in self.missing_cells],
        }


@dataclass(frozen=True)
class CohortScores:
    participants: tuple[ParticipantScores, ...]
    exclusions: tuple[ParticipantExclusion, ...]
    top1: float | None
    mrr: float | None
    donor_attraction: float | None


def evaluate_cohort(
    recordings: Sequence[Recording],
    *,
    split: str = "development",
    seed: int = 0,
    representation: str = "learned",
    condition_weight: float = 0.5,
    gait_weight: float = 0.5,
    tie_tolerance: float = TIE_ABS_TOLERANCE,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> CohortScores:
    """Evaluate complete participants and weight participants equally."""

    if not recordings:
        raise ValueError("cohort has no recordings")
    grouped: dict[str, list[Recording]] = defaultdict(list)
    recording_ids: set[str] = set()
    window_ids: set[str] = set()
    for recording in recordings:
        if recording.recording_id in recording_ids:
            raise ValueError(f"duplicate global recording_id {recording.recording_id!r}")
        if window_ids.intersection(recording.window_ids):
            raise ValueError("window_id values must be globally unique")
        recording_ids.add(recording.recording_id)
        window_ids.update(recording.window_ids)
        grouped[recording.subject_id].append(recording)

    participants: list[ParticipantScores] = []
    exclusions: list[ParticipantExclusion] = []
    for subject_id in sorted(grouped):
        subject_recordings = grouped[subject_id]
        absent = missing_cells(subject_recordings)  # also makes duplicates a hard error
        if absent:
            exclusions.append(
                ParticipantExclusion(subject_id, "incomplete_factorial_grid", absent)
            )
            continue
        participants.append(
            evaluate_participant(
                subject_recordings,
                split=split,
                seed=seed,
                representation=representation,
                condition_weight=condition_weight,
                gait_weight=gait_weight,
                tie_tolerance=tie_tolerance,
                zero_norm_epsilon=zero_norm_epsilon,
            )
        )
    if not participants:
        return CohortScores((), tuple(exclusions), None, None, None)
    return CohortScores(
        tuple(participants),
        tuple(exclusions),
        float(np.mean([item.top1 for item in participants], dtype=SCORING_DTYPE)),
        float(np.mean([item.mrr for item in participants], dtype=SCORING_DTYPE)),
        float(
            np.mean([item.donor_attraction for item in participants], dtype=SCORING_DTYPE)
        ),
    )


__all__ = [
    "CANONICAL_CELLS",
    "CLOTHING",
    "DIRECTIONS",
    "EXPECTED_CELLS",
    "EXPECTED_GALLERY_SIZE",
    "EXPECTED_QUERIES",
    "EXPECTED_WINDOWS",
    "GFC_PROTOCOL",
    "SPEEDS",
    "TIE_ABS_TOLERANCE",
    "Cell",
    "CohortScores",
    "DistanceEntry",
    "ParticipantExclusion",
    "ParticipantScores",
    "Query",
    "QueryResult",
    "RankScores",
    "Recording",
    "aggregate_windows",
    "build_queries",
    "cosine_distance",
    "evaluate_cohort",
    "evaluate_participant",
    "evaluate_query",
    "missing_cells",
    "mixed_distance",
    "rank_target",
]
