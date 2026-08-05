"""Grounded Factorial Completion v2 for the Health&Gait 2 x 2 x 2 design."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np

from .oracle import compile_healthgait_gfc_v2_protocol


_COMPILED_PROTOCOL = compile_healthgait_gfc_v2_protocol()
_FACTOR_NAMES = _COMPILED_PROTOCOL.design.factor_names
SPEEDS = _COMPILED_PROTOCOL.design.factors[0].values
CLOTHING = _COMPILED_PROTOCOL.design.factors[1].values
DIRECTIONS = _COMPILED_PROTOCOL.design.factors[2].values
GFC_PROTOCOL = _COMPILED_PROTOCOL.name
EXPECTED_WINDOWS = 3
EXPECTED_CELLS = len(_COMPILED_PROTOCOL.design.cells)
EXPECTED_QUERIES = len(_COMPILED_PROTOCOL.queries)
EXPECTED_GALLERY_SIZE = len(_COMPILED_PROTOCOL.queries[0].gallery)
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


CANONICAL_CELLS = tuple(Cell(*values) for values in _COMPILED_PROTOCOL.design.cells)


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
class FactorBlocks:
    """Three immutable recording-level vectors, one for each design factor."""

    speed: np.ndarray = field(repr=False, compare=False)
    clothing: np.ndarray = field(repr=False, compare=False)
    direction: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in _FACTOR_NAMES:
            object.__setattr__(self, name, _finite_vector(getattr(self, name), f"{name}_block"))

    def for_factor(self, factor_name: str) -> np.ndarray:
        if factor_name not in _FACTOR_NAMES:
            raise ValueError(f"unknown factor {factor_name!r}")
        return getattr(self, factor_name)


@dataclass(frozen=True)
class Recording:
    """One recording with source lineage and three matched factor blocks."""

    subject_id: str
    recording_id: str
    source_video_id: str
    cell: Cell
    factor_blocks: FactorBlocks
    window_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _nonempty_text(self.subject_id, "subject_id"))
        object.__setattr__(self, "recording_id", _nonempty_text(self.recording_id, "recording_id"))
        object.__setattr__(
            self, "source_video_id", _nonempty_text(self.source_video_id, "source_video_id")
        )
        if not isinstance(self.cell, Cell):
            raise TypeError("cell must be a Cell")
        if not isinstance(self.factor_blocks, FactorBlocks):
            raise TypeError("factor_blocks must be FactorBlocks")
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
        source_video_id: str,
        cell: Cell,
        speed_windows: Sequence[Sequence[float] | np.ndarray],
        clothing_windows: Sequence[Sequence[float] | np.ndarray],
        direction_windows: Sequence[Sequence[float] | np.ndarray],
        window_ids: Sequence[str],
    ) -> "Recording":
        ids = tuple(window_ids)
        counts = {len(ids), len(speed_windows), len(clothing_windows), len(direction_windows)}
        if len(counts) != 1:
            raise ValueError("factor windows and window identifiers must have equal counts")
        return cls(
            subject_id=subject_id,
            recording_id=recording_id,
            source_video_id=source_video_id,
            cell=cell,
            factor_blocks=FactorBlocks(
                speed=aggregate_windows(speed_windows, label="speed"),
                clothing=aggregate_windows(clothing_windows, label="clothing"),
                direction=aggregate_windows(direction_windows, label="direction"),
            ),
            window_ids=ids,
        )


@dataclass(frozen=True)
class Query:
    query_index: int
    protocol: str
    subject_id: str
    target_id: str
    target_cell: Cell
    focal_factor: str
    donor_u_id: str
    donor_u_cell: Cell
    donor_v_id: str
    donor_v_cell: Cell
    factor_sources: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    gallery_cells: tuple[Cell, ...]
    source_independence_verified: bool


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
    for factor_name in _FACTOR_NAMES:
        widths = {item.factor_blocks.for_factor(factor_name).size for item in recordings}
        if len(widths) != 1:
            raise ValueError(f"{factor_name} block width must be constant within a participant")
    return index


def missing_cells(recordings: Sequence[Recording]) -> tuple[Cell, ...]:
    index = _recording_index(recordings)
    return tuple(cell for cell in CANONICAL_CELLS if cell not in index)


def build_queries(recordings: Sequence[Recording]) -> tuple[Query, ...]:
    """Materialize the oracle-defined GFC-v2 queries for one participant."""

    index = _recording_index(recordings)
    absent = tuple(cell for cell in CANONICAL_CELLS if cell not in index)
    if absent or len(index) != EXPECTED_CELLS:
        raise ValueError(
            "fixed GFC requires all eight cells; missing cells: "
            + (", ".join(cell.key for cell in absent) or "none")
        )
    queries: list[Query] = []
    for compiled in _COMPILED_PROTOCOL.queries:
        target_cell = Cell(*compiled.target)
        donor_u_cell, donor_v_cell = (Cell(*cell) for cell in compiled.donors)
        target = index[target_cell]
        donor_u = index[donor_u_cell]
        donor_v = index[donor_v_cell]
        if donor_u.source_video_id == target.source_video_id:
            raise ValueError("donor_u shares source_video_id with target")
        if donor_v.source_video_id == target.source_video_id:
            raise ValueError("donor_v shares source_video_id with target")
        gallery_cells = tuple(Cell(*cell) for cell in compiled.gallery)
        gallery_recordings = tuple(index[cell] for cell in gallery_cells)
        factor_sources = tuple(
            "donor_u" if source_index == 0 else "donor_v"
            for source_index in compiled.factor_sources
        )
        queries.append(
            Query(
                query_index=compiled.query_index,
                protocol=_COMPILED_PROTOCOL.name,
                subject_id=target.subject_id,
                target_id=target.recording_id,
                target_cell=target_cell,
                focal_factor=compiled.focal_factor,
                donor_u_id=donor_u.recording_id,
                donor_u_cell=donor_u_cell,
                donor_v_id=donor_v.recording_id,
                donor_v_cell=donor_v_cell,
                factor_sources=factor_sources,
                gallery_ids=tuple(item.recording_id for item in gallery_recordings),
                gallery_cells=gallery_cells,
                source_independence_verified=True,
            )
        )
    if len(queries) != EXPECTED_QUERIES:
        raise RuntimeError("compiled factor design produced an unexpected query count")
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


def mixed_distance(
    query_blocks: FactorBlocks,
    gallery_recording: Recording,
    *,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> tuple[float, float, float, float]:
    """Return per-factor cosine distances and their equal arithmetic mean."""

    if not isinstance(query_blocks, FactorBlocks):
        raise TypeError("query_blocks must be FactorBlocks")
    distances = tuple(
        cosine_distance(
            query_blocks.for_factor(factor_name),
            gallery_recording.factor_blocks.for_factor(factor_name),
            zero_norm_epsilon=zero_norm_epsilon,
        )
        for factor_name in _FACTOR_NAMES
    )
    return (*distances, float(np.mean(distances, dtype=SCORING_DTYPE)))


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
    speed_distance: float
    clothing_distance: float
    direction_distance: float
    combined_distance: float

    def to_dict(self, target_id: str) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "cell": self.cell.to_dict(),
            "speed_distance": self.speed_distance,
            "clothing_distance": self.clothing_distance,
            "direction_distance": self.direction_distance,
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
    donor_u_attraction: float
    donor_v_attraction: float

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
            self.query.protocol,
            self.query.subject_id,
            self.query.target_id,
            self.query.target_cell,
            self.query.focal_factor,
            self.query.donor_u_id,
            self.query.donor_u_cell,
            self.query.donor_v_id,
            self.query.donor_v_cell,
            self.query.factor_sources,
            tuple(zip(self.query.gallery_ids, self.query.gallery_cells)),
            self.query.source_independence_verified,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "seed": self.seed,
            "representation": self.representation,
            "protocol": self.query.protocol,
            "subject_id": self.query.subject_id,
            "query_index": self.query.query_index,
            "focal_factor": self.query.focal_factor,
            "target": {"recording_id": self.query.target_id, "cell": self.query.target_cell.to_dict()},
            "donor_u": {
                "recording_id": self.query.donor_u_id,
                "cell": self.query.donor_u_cell.to_dict(),
            },
            "donor_v": {
                "recording_id": self.query.donor_v_id,
                "cell": self.query.donor_v_cell.to_dict(),
            },
            "factor_sources": dict(zip(_FACTOR_NAMES, self.query.factor_sources)),
            "source_independence_verified": self.query.source_independence_verified,
            "gallery": [item.to_dict(self.query.target_id) for item in self.distances],
            "strictly_closer_count": self.rank.strictly_closer_count,
            "tie_size": self.rank.tie_size,
            "average_rank": self.rank.average_rank,
            "top1": self.top1,
            "reciprocal_rank": self.reciprocal_rank,
            "donor_u_attraction": self.donor_u_attraction,
            "donor_v_attraction": self.donor_v_attraction,
        }


def _compose_query_blocks(query: Query, donor_u: Recording, donor_v: Recording) -> FactorBlocks:
    donors = {"donor_u": donor_u, "donor_v": donor_v}
    if len(query.factor_sources) != len(_FACTOR_NAMES):
        raise ValueError("query must declare one donor source per factor")
    if any(role not in donors for role in query.factor_sources):
        raise ValueError("query contains an unknown factor-source role")
    blocks = {
        factor_name: donors[role].factor_blocks.for_factor(factor_name)
        for factor_name, role in zip(_FACTOR_NAMES, query.factor_sources)
    }
    return FactorBlocks(**blocks)


def _donor_attraction(
    target_distance: float,
    donor_distance: float,
    *,
    tolerance: float,
) -> float:
    if abs(target_distance - donor_distance) <= tolerance:
        return 0.5
    return float(donor_distance < target_distance - tolerance)


def evaluate_query(
    query: Query,
    recordings_by_id: Mapping[str, Recording],
    *,
    split: str,
    seed: int,
    representation: str,
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
    if tuple(query.gallery_ids) != tuple(item.recording_id for item in recordings_by_id.values()):
        # Input mapping order is not scientific, so validate set and canonical cell order below.
        if set(query.gallery_ids) != set(recordings_by_id):
            raise ValueError("retain-all gallery must contain every participant recording")
    target = recordings_by_id[query.target_id]
    donor_u = recordings_by_id[query.donor_u_id]
    donor_v = recordings_by_id[query.donor_v_id]
    if donor_u.source_video_id == target.source_video_id:
        raise ValueError("donor_u shares source_video_id with target")
    if donor_v.source_video_id == target.source_video_id:
        raise ValueError("donor_v shares source_video_id with target")
    query_blocks = _compose_query_blocks(query, donor_u, donor_v)
    entries: list[DistanceEntry] = []
    for identifier, cell in zip(query.gallery_ids, query.gallery_cells):
        recording = recordings_by_id[identifier]
        if recording.cell != cell:
            raise ValueError("gallery factor identity differs from the query")
        speed, clothing, direction, combined = mixed_distance(
            query_blocks,
            recording,
            zero_norm_epsilon=zero_norm_epsilon,
        )
        entries.append(DistanceEntry(identifier, cell, speed, clothing, direction, combined))
    rank = rank_target(
        {entry.recording_id: entry.combined_distance for entry in entries},
        query.target_id,
        tolerance=tie_tolerance,
    )
    target_distance = next(
        entry.combined_distance for entry in entries if entry.recording_id == query.target_id
    )
    donor_u_distance = next(
        entry.combined_distance for entry in entries if entry.recording_id == query.donor_u_id
    )
    donor_v_distance = next(
        entry.combined_distance for entry in entries if entry.recording_id == query.donor_v_id
    )
    return QueryResult(
        split,
        seed,
        representation,
        query,
        tuple(entries),
        rank,
        _donor_attraction(target_distance, donor_u_distance, tolerance=tie_tolerance),
        _donor_attraction(target_distance, donor_v_distance, tolerance=tie_tolerance),
    )


@dataclass(frozen=True)
class ParticipantScores:
    subject_id: str
    representation: str
    queries: tuple[QueryResult, ...]
    top1: float
    mrr: float
    donor_u_attraction: float
    donor_v_attraction: float

    def to_dict(self, *, include_queries: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "subject_id": self.subject_id,
            "representation": self.representation,
            "query_count": len(self.queries),
            "top1": self.top1,
            "mrr": self.mrr,
            "donor_u_attraction": self.donor_u_attraction,
            "donor_v_attraction": self.donor_v_attraction,
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
        donor_u_attraction=float(
            np.mean([item.donor_u_attraction for item in results], dtype=SCORING_DTYPE)
        ),
        donor_v_attraction=float(
            np.mean([item.donor_v_attraction for item in results], dtype=SCORING_DTYPE)
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
    donor_u_attraction: float | None
    donor_v_attraction: float | None


def evaluate_cohort(
    recordings: Sequence[Recording],
    *,
    split: str = "development",
    seed: int = 0,
    representation: str = "learned",
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
                tie_tolerance=tie_tolerance,
                zero_norm_epsilon=zero_norm_epsilon,
            )
        )
    if not participants:
        return CohortScores((), tuple(exclusions), None, None, None, None)
    return CohortScores(
        tuple(participants),
        tuple(exclusions),
        float(np.mean([item.top1 for item in participants], dtype=SCORING_DTYPE)),
        float(np.mean([item.mrr for item in participants], dtype=SCORING_DTYPE)),
        float(
            np.mean([item.donor_u_attraction for item in participants], dtype=SCORING_DTYPE)
        ),
        float(
            np.mean([item.donor_v_attraction for item in participants], dtype=SCORING_DTYPE)
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
    "FactorBlocks",
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
