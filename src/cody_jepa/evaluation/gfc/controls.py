"""Independent-factor completion controls for the locked GFC-v2 study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .core import (
    SCORING_DTYPE,
    TIE_ABS_TOLERANCE,
    Cell,
    ParticipantExclusion,
    Query,
    Recording,
    build_queries,
    missing_cells,
    rank_target,
)
from .oracle import compile_healthgait_gfc_v2_protocol


TEMPERATURE_BOUNDS = (0.001, 1000.0)
_PROTOCOL = compile_healthgait_gfc_v2_protocol()
_FACTOR_NAMES = _PROTOCOL.design.factor_names
_FACTOR_LABELS = {
    factor.name: tuple(factor.values) for factor in _PROTOCOL.design.factors
}


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    fit_row_count: int
    factor_observation_count: int
    bounds: tuple[float, float]
    converged: bool
    boundary_solution: bool
    iterations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "fitted_temperature": self.temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "fit_row_count": self.fit_row_count,
            "factor_observation_count": self.factor_observation_count,
            "bounds": list(self.bounds),
            "converged": self.converged,
            "boundary_solution": self.boundary_solution,
            "iterations": self.iterations,
        }


def _factor_targets(cells: Sequence[Cell], factor_name: str) -> np.ndarray:
    labels = _FACTOR_LABELS[factor_name]
    return np.asarray([labels.index(getattr(cell, factor_name)) for cell in cells], dtype=int)


def _temperature_nll(
    factor_scores: Mapping[str, np.ndarray], cells: Sequence[Cell], temperature: float
) -> float:
    if not math.isfinite(temperature) or temperature <= 0.0:
        return math.inf
    losses: list[np.ndarray] = []
    for factor_name in _FACTOR_NAMES:
        scores = factor_scores[factor_name] / temperature
        targets = _factor_targets(cells, factor_name)
        losses.append(logsumexp(scores, axis=1) - scores[np.arange(len(cells)), targets])
    value = float(np.mean(np.concatenate(losses), dtype=SCORING_DTYPE))
    return value if math.isfinite(value) else math.inf


def fit_shared_temperature(
    factor_scores: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    cells: Sequence[Cell],
    *,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> TemperatureFit:
    """Fit one scalar temperature to pooled development-row factor NLL."""

    if set(factor_scores) != set(_FACTOR_NAMES):
        raise ValueError("temperature scores must cover exactly the three canonical factors")
    if not cells:
        raise ValueError("temperature fitting requires development rows")
    matrices: dict[str, np.ndarray] = {}
    for factor_name in _FACTOR_NAMES:
        values = np.asarray(factor_scores[factor_name], dtype=SCORING_DTYPE)
        if values.shape != (len(cells), len(_FACTOR_LABELS[factor_name])):
            raise ValueError(f"{factor_name} scores must have shape ({len(cells)}, 2)")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{factor_name} scores contain non-finite values")
        matrices[factor_name] = values
    lower, upper = (float(value) for value in bounds)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("temperature bounds must be finite, positive, and increasing")
    log_bounds = (math.log(lower), math.log(upper))
    result = minimize_scalar(
        lambda log_temperature: _temperature_nll(
            matrices, cells, math.exp(float(log_temperature))
        ),
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    temperature = math.exp(float(result.x))
    before = _temperature_nll(matrices, cells, 1.0)
    after = _temperature_nll(matrices, cells, temperature)
    lower_nll = _temperature_nll(matrices, cells, lower)
    upper_nll = _temperature_nll(matrices, cells, upper)
    boundary_tolerance = 1e-7 * (log_bounds[1] - log_bounds[0])
    boundary = (
        float(result.x) - log_bounds[0] <= boundary_tolerance
        or log_bounds[1] - float(result.x) <= boundary_tolerance
        # A saturated objective can give the bounded optimizer an arbitrary
        # interior point even though the optimum includes a configured edge.
        or lower_nll <= after + 1e-12
        or upper_nll <= after + 1e-12
    )
    converged = bool(result.success) and all(
        math.isfinite(value) for value in (temperature, before, after)
    )
    fit = TemperatureFit(
        temperature=temperature,
        nll_before=before,
        nll_after=after,
        fit_row_count=len(cells),
        factor_observation_count=len(cells) * len(_FACTOR_NAMES),
        bounds=(lower, upper),
        converged=converged,
        boundary_solution=boundary,
        iterations=int(result.nfev),
    )
    if not converged:
        raise RuntimeError("shared-temperature optimization failed or returned non-finite output")
    if boundary:
        raise RuntimeError("shared-temperature optimization reached a configured boundary")
    return fit


@dataclass(frozen=True)
class ControlParticipantScores:
    subject_id: str
    top1: float
    mrr: float
    target_probability: float | None = None
    target_nll: float | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"top1": self.top1, "mrr": self.mrr}
        if self.target_probability is not None:
            result["target_probability"] = self.target_probability
        if self.target_nll is not None:
            result["target_nll"] = self.target_nll
        return result


@dataclass(frozen=True)
class IndependentFactorControls:
    hard_participants: tuple[ControlParticipantScores, ...]
    soft_participants: tuple[ControlParticipantScores, ...]
    exclusions: tuple[ParticipantExclusion, ...]
    hard_top1: float
    hard_mrr: float
    soft_top1: float
    soft_mrr: float
    soft_target_probability: float
    soft_target_nll: float
    top1_agreement: bool

    def aggregate_dict(self) -> dict[str, object]:
        return {
            "hard": {"top1": self.hard_top1, "mrr": self.hard_mrr},
            "soft": {
                "top1": self.soft_top1,
                "mrr": self.soft_mrr,
                "target_probability": self.soft_target_probability,
                "target_nll": self.soft_target_nll,
            },
            "top1_agreement": self.top1_agreement,
        }


def _composed_scores(
    query: Query, by_id: Mapping[str, Recording]
) -> dict[str, np.ndarray]:
    donors = {"donor_u": by_id[query.donor_u_id], "donor_v": by_id[query.donor_v_id]}
    return {
        factor_name: donors[source].factor_blocks.for_factor(factor_name)
        for factor_name, source in zip(_FACTOR_NAMES, query.factor_sources)
    }


def _hard_log_masses(scores: np.ndarray, *, tolerance: float) -> np.ndarray:
    maximum = float(np.max(scores))
    winners = np.abs(scores - maximum) <= tolerance
    result = np.full(scores.shape, -math.inf, dtype=SCORING_DTYPE)
    result[winners] = -math.log(int(np.sum(winners)))
    return result


def _soft_log_masses(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = scores / temperature
    return scaled - logsumexp(scaled)


def _gallery_log_scores(
    query: Query,
    factor_log_masses: Mapping[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for identifier, cell in zip(query.gallery_ids, query.gallery_cells):
        value = 0.0
        for factor_name in _FACTOR_NAMES:
            label_index = _FACTOR_LABELS[factor_name].index(getattr(cell, factor_name))
            value += float(factor_log_masses[factor_name][label_index])
        values[identifier] = value
    return values


def _rank_log_masses(
    log_masses: Mapping[str, float],
    target_id: str,
    *,
    tolerance: float,
):
    """Rank probabilities in log space without collapsing tiny masses to zero."""

    distances = {
        key: -float(value)
        for key, value in log_masses.items()
        if math.isfinite(float(value))
    }
    if not distances:
        raise FloatingPointError("gallery has no finite probability mass")
    if len(distances) != len(log_masses):
        impossible_distance = max(distances.values()) + max(1.0, 2.0 * tolerance)
        distances.update(
            (key, impossible_distance)
            for key, value in log_masses.items()
            if not math.isfinite(float(value))
        )
    return rank_target(distances, target_id, tolerance=tolerance)


def _evaluate_control_participant(
    recordings: Sequence[Recording],
    *,
    temperature: float,
    tie_tolerance: float,
) -> tuple[ControlParticipantScores, ControlParticipantScores]:
    queries = build_queries(recordings)
    by_id = {recording.recording_id: recording for recording in recordings}
    hard_top1: list[float] = []
    hard_mrr: list[float] = []
    soft_top1: list[float] = []
    soft_mrr: list[float] = []
    target_probabilities: list[float] = []
    target_nlls: list[float] = []
    for query in queries:
        scores = _composed_scores(query, by_id)
        hard_logs = {
            factor_name: _hard_log_masses(values, tolerance=tie_tolerance)
            for factor_name, values in scores.items()
        }
        soft_logs = {
            factor_name: _soft_log_masses(values, temperature=temperature)
            for factor_name, values in scores.items()
        }
        hard_gallery = _gallery_log_scores(query, hard_logs)
        soft_gallery = _gallery_log_scores(query, soft_logs)
        # rank_target minimizes. Negative log mass preserves probability ordering
        # without collapsing very small calibrated probabilities into zero ties.
        hard_rank = _rank_log_masses(
            hard_gallery, query.target_id, tolerance=tie_tolerance
        )
        soft_rank = _rank_log_masses(
            soft_gallery,
            query.target_id,
            # Every soft log-probability difference is divided by T. Scaling
            # the absolute tie tolerance the same way keeps ranking invariant
            # to calibration while probabilities and NLL remain informative.
            tolerance=tie_tolerance / temperature,
        )
        target_log_probability = soft_gallery[query.target_id]
        hard_top1.append(hard_rank.top1)
        hard_mrr.append(hard_rank.reciprocal_rank)
        soft_top1.append(soft_rank.top1)
        soft_mrr.append(soft_rank.reciprocal_rank)
        target_probabilities.append(math.exp(target_log_probability))
        target_nlls.append(-target_log_probability)
    subject_id = queries[0].subject_id
    hard = ControlParticipantScores(
        subject_id,
        float(np.mean(hard_top1, dtype=SCORING_DTYPE)),
        float(np.mean(hard_mrr, dtype=SCORING_DTYPE)),
    )
    soft = ControlParticipantScores(
        subject_id,
        float(np.mean(soft_top1, dtype=SCORING_DTYPE)),
        float(np.mean(soft_mrr, dtype=SCORING_DTYPE)),
        float(np.mean(target_probabilities, dtype=SCORING_DTYPE)),
        float(np.mean(target_nlls, dtype=SCORING_DTYPE)),
    )
    return hard, soft


def evaluate_independent_factor_controls(
    recordings: Sequence[Recording],
    *,
    temperature: float,
    tie_tolerance: float = TIE_ABS_TOLERANCE,
) -> IndependentFactorControls:
    """Score hard and soft completion controls with equal participant weight."""

    if not recordings:
        raise ValueError("control cohort has no recordings")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    grouped: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        grouped[recording.subject_id].append(recording)
    hard: list[ControlParticipantScores] = []
    soft: list[ControlParticipantScores] = []
    exclusions: list[ParticipantExclusion] = []
    for subject_id in sorted(grouped):
        absent = missing_cells(grouped[subject_id])
        if absent:
            exclusions.append(
                ParticipantExclusion(subject_id, "incomplete_factorial_grid", absent)
            )
            continue
        hard_item, soft_item = _evaluate_control_participant(
            grouped[subject_id],
            temperature=temperature,
            tie_tolerance=tie_tolerance,
        )
        hard.append(hard_item)
        soft.append(soft_item)
    if not hard:
        raise ValueError("control cohort has no complete participants")
    hard_top1 = float(np.mean([item.top1 for item in hard], dtype=SCORING_DTYPE))
    soft_top1 = float(np.mean([item.top1 for item in soft], dtype=SCORING_DTYPE))
    return IndependentFactorControls(
        tuple(hard),
        tuple(soft),
        tuple(exclusions),
        hard_top1,
        float(np.mean([item.mrr for item in hard], dtype=SCORING_DTYPE)),
        soft_top1,
        float(np.mean([item.mrr for item in soft], dtype=SCORING_DTYPE)),
        float(
            np.mean([item.target_probability for item in soft], dtype=SCORING_DTYPE)
        ),
        float(np.mean([item.target_nll for item in soft], dtype=SCORING_DTYPE)),
        all(
            math.isclose(hard_item.top1, soft_item.top1, rel_tol=0.0, abs_tol=tie_tolerance)
            for hard_item, soft_item in zip(hard, soft)
        ),
    )


__all__ = [
    "IndependentFactorControls",
    "TEMPERATURE_BOUNDS",
    "TemperatureFit",
    "evaluate_independent_factor_controls",
    "fit_shared_temperature",
]
