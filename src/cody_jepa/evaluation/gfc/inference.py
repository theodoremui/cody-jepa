"""Participant-level paired inference for learned-over-shortcut GFC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from .core import EXPECTED_QUERIES, ParticipantScores, QueryResult, SCORING_DTYPE


DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_728
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class ParticipantContrast:
    subject_id: str
    learned_top1: float
    shortcut_top1: float
    difference: float

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "learned_top1": self.learned_top1,
            "shortcut_top1": self.shortcut_top1,
            "difference": self.difference,
        }


@dataclass(frozen=True)
class PairedCohort:
    participants: tuple[ParticipantContrast, ...]
    mean_difference: float

    def to_dict(self) -> dict[str, object]:
        return {
            "participant_count": len(self.participants),
            "mean_difference": self.mean_difference,
            "participants": [item.to_dict() for item in self.participants],
        }


def _validated_query_map(
    result: ParticipantScores, expected_representation: str
) -> dict[tuple[object, ...], QueryResult]:
    if result.representation != expected_representation:
        raise ValueError(f"participant result is not the {expected_representation!r} path")
    if len(result.queries) != EXPECTED_QUERIES:
        raise ValueError("participant result must contain exactly 24 queries")
    if any(item.representation != expected_representation for item in result.queries):
        raise ValueError("query representation labels are inconsistent")
    if any(item.query.subject_id != result.subject_id for item in result.queries):
        raise ValueError("participant result mixes subjects")
    query_map = {item.scientific_key: item for item in result.queries}
    if len(query_map) != EXPECTED_QUERIES:
        raise ValueError("participant result has duplicate scientific query keys")
    if sorted(item.query.query_index for item in result.queries) != list(range(EXPECTED_QUERIES)):
        raise ValueError("participant result has missing or duplicate query indices")
    recalculated = float(np.mean([item.top1 for item in result.queries], dtype=SCORING_DTYPE))
    if result.top1 != recalculated:
        raise ValueError("participant top1 is not its equal-query mean")
    return query_map


def paired_participant_lsg(
    learned: ParticipantScores,
    shortcut: ParticipantScores,
    *,
    learned_representation: str = "learned",
    shortcut_representation: str = "shortcut",
) -> ParticipantContrast:
    """Pair on scientific query keys, then subtract participant mean top-1."""

    if learned.subject_id != shortcut.subject_id:
        raise ValueError("learned and shortcut participants differ")
    learned_queries = _validated_query_map(learned, learned_representation)
    shortcut_queries = _validated_query_map(shortcut, shortcut_representation)
    if set(learned_queries) != set(shortcut_queries):
        missing = len(set(learned_queries) - set(shortcut_queries))
        extra = len(set(shortcut_queries) - set(learned_queries))
        raise ValueError(
            f"learned and shortcut scientific query keys differ (missing={missing}, extra={extra})"
        )
    difference = float(np.float64(learned.top1) - np.float64(shortcut.top1))
    return ParticipantContrast(
        subject_id=learned.subject_id,
        learned_top1=learned.top1,
        shortcut_top1=shortcut.top1,
        difference=difference,
    )


def paired_cohort_lsg(
    learned: Sequence[ParticipantScores],
    shortcut: Sequence[ParticipantScores],
    *,
    learned_representation: str = "learned",
    shortcut_representation: str = "shortcut",
) -> PairedCohort:
    learned_by_subject = {item.subject_id: item for item in learned}
    shortcut_by_subject = {item.subject_id: item for item in shortcut}
    if len(learned_by_subject) != len(learned) or len(shortcut_by_subject) != len(shortcut):
        raise ValueError("paired cohorts contain duplicate participant results")
    if not learned_by_subject or set(learned_by_subject) != set(shortcut_by_subject):
        raise ValueError("learned and shortcut cohorts must contain the same participants")
    participants = tuple(
        paired_participant_lsg(
            learned_by_subject[subject_id],
            shortcut_by_subject[subject_id],
            learned_representation=learned_representation,
            shortcut_representation=shortcut_representation,
        )
        for subject_id in sorted(learned_by_subject)
    )
    return PairedCohort(
        participants=participants,
        mean_difference=float(
            np.mean([item.difference for item in participants], dtype=np.float64)
        ),
    )


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    participant_count: int
    point_estimate: float
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    resamples: int
    seed: int

    @property
    def positive_supported(self) -> bool:
        return self.confidence_interval_lower > 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "participant_count": self.participant_count,
            "point_estimate": self.point_estimate,
            "confidence_level": self.confidence_level,
            "confidence_interval": {
                "lower": self.confidence_interval_lower,
                "upper": self.confidence_interval_upper,
            },
            "resamples": self.resamples,
            "seed": self.seed,
            "positive_supported": self.positive_supported,
        }


def bootstrap_lsg(
    cohort: PairedCohort,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> BootstrapResult:
    """Percentile bootstrap over paired participant differences."""

    if not isinstance(cohort, PairedCohort):
        raise TypeError("cohort must be a PairedCohort")
    margins = np.asarray([item.difference for item in cohort.participants], dtype=np.float64)
    if margins.ndim != 1 or margins.size < 2 or not np.isfinite(margins).all():
        raise ValueError("bootstrap requires at least two finite participant differences")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    generator = np.random.Generator(np.random.PCG64(seed))
    resample_means = np.empty(resamples, dtype=np.float64)
    batch_size = 512
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = generator.integers(
            0, margins.size, size=(stop - start, margins.size), endpoint=False
        )
        resample_means[start:stop] = np.mean(margins[indices], axis=1, dtype=np.float64)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(resample_means, (tail, 1.0 - tail), method="linear")
    return BootstrapResult(
        metric="learned_top1_minus_shortcut_top1",
        participant_count=margins.size,
        point_estimate=cohort.mean_difference,
        confidence_level=float(confidence_level),
        confidence_interval_lower=float(lower),
        confidence_interval_upper=float(upper),
        resamples=resamples,
        seed=seed,
    )


@dataclass(frozen=True)
class ProspectivePowerResult:
    participant_count: int
    minimum_effect: float
    alpha: float
    target_power: float
    sample_standard_deviation: float
    planned_power: float

    @property
    def meets_target(self) -> bool:
        return self.planned_power >= self.target_power

    def to_dict(self) -> dict[str, object]:
        return {
            "participant_count": self.participant_count,
            "minimum_effect": self.minimum_effect,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "sample_standard_deviation": self.sample_standard_deviation,
            "planned_power": self.planned_power,
            "meets_target": self.meets_target,
        }


def plan_prospective_power(
    cohort: PairedCohort,
    *,
    minimum_effect: float = 1.0 / 24.0,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> ProspectivePowerResult:
    """Two-sided paired-mean noncentral-t planning calculation.

    This is optional development-set planning, not confirmatory inference.
    """

    margins = np.asarray([item.difference for item in cohort.participants], dtype=np.float64)
    if margins.size < 2 or not np.isfinite(margins).all():
        raise ValueError("power planning requires at least two finite differences")
    for value, label in (
        (minimum_effect, "minimum_effect"),
        (alpha, "alpha"),
        (target_power, "target_power"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if minimum_effect <= 0.0 or not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("power parameters are outside their valid ranges")
    standard_deviation = float(np.std(margins, ddof=1, dtype=np.float64))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("power planning requires positive participant variation")
    try:
        from scipy import stats
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("power planning requires scipy") from error
    degrees_of_freedom = margins.size - 1
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    noncentrality = minimum_effect * math.sqrt(float(margins.size)) / standard_deviation
    power = float(
        stats.nct.cdf(-critical, degrees_of_freedom, noncentrality)
        + stats.nct.sf(critical, degrees_of_freedom, noncentrality)
    )
    return ProspectivePowerResult(
        participant_count=margins.size,
        minimum_effect=float(minimum_effect),
        alpha=float(alpha),
        target_power=float(target_power),
        sample_standard_deviation=standard_deviation,
        planned_power=min(1.0, max(0.0, power)),
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "BootstrapResult",
    "PairedCohort",
    "ParticipantContrast",
    "ProspectivePowerResult",
    "bootstrap_lsg",
    "paired_cohort_lsg",
    "paired_participant_lsg",
    "plan_prospective_power",
]
