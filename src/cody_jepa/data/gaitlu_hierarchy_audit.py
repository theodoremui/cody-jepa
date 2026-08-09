"""Outcome-blind support audit for the hierarchical GaitLU intervention."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import random
import statistics


def anchor_count(num_frames: int, *, clip_length: int = 16, spacing: int = 8) -> int:
    """Return the number of regularly spaced valid clip starts."""

    if num_frames < clip_length:
        return 0
    return (num_frames - clip_length) // spacing + 1


def _closest_group_prefix(group_names, groups, target):
    chosen = []
    count = 0
    for name in group_names:
        size = len(groups[name])
        if chosen and abs(count - target) <= abs(count + size - target):
            break
        chosen.append(name)
        count += size
        if count == target:
            break
    return chosen


def _occupied_probability(draws: int, probability: float) -> float:
    if probability >= 1.0:
        return 1.0
    return -math.expm1(draws * math.log1p(-probability))


def _expected_support(anchor_counts, draws):
    sequence_count = len(anchor_counts)
    frozen = sequence_count * _occupied_probability(draws, 1.0 / sequence_count)
    resampled = sum(
        count * _occupied_probability(draws, 1.0 / (sequence_count * count))
        for count in anchor_counts
    )
    return frozen, resampled


def _mean_distinct_window_overlap(count: int, *, clip_length: int, spacing: int) -> float:
    """Mean frame-overlap fraction over distinct anchor pairs in one sequence."""

    if count < 2:
        return 0.0
    overlap_sum = 0.0
    for lag in range(1, min(count, math.ceil(clip_length / spacing))):
        overlap = max(0, clip_length - lag * spacing) / clip_length
        overlap_sum += (count - lag) * overlap
    return overlap_sum / math.comb(count, 2)


def _pool_metrics(rows, *, draws, clip_length, anchor_spacing):
    anchor_counts = [
        anchor_count(int(row["num_frames"]), clip_length=clip_length, spacing=anchor_spacing)
        for row in rows
    ]
    nonoverlap_counts = [
        anchor_count(int(row["num_frames"]), clip_length=clip_length, spacing=clip_length)
        for row in rows
    ]
    frozen, resampled = _expected_support(anchor_counts, draws)
    _, nonoverlap_support = _expected_support(nonoverlap_counts, draws)
    median_anchors = float(statistics.median(anchor_counts))
    support_ratio = resampled / frozen
    return {
        "actual_sequences": len(rows),
        "median_8_frame_anchors": median_anchors,
        "median_nonoverlapping_anchors": float(statistics.median(nonoverlap_counts)),
        "fraction_with_at_least_4_anchors": sum(value >= 4 for value in anchor_counts)
        / len(anchor_counts),
        "expected_frozen_support": frozen,
        "expected_resampled_support": resampled,
        "expected_resampled_to_frozen_ratio": support_ratio,
        "expected_nonoverlap_to_frozen_ratio": nonoverlap_support / frozen,
        "mean_distinct_window_overlap_fraction": statistics.fmean(
            _mean_distinct_window_overlap(
                count, clip_length=clip_length, spacing=anchor_spacing
            )
            for count in anchor_counts
        ),
        "gate_pass": median_anchors >= 4 and support_ratio >= 4,
    }


def audit_hierarchical_support(
    inventory_path,
    *,
    draws=4_096_000,
    holdout_size=10_000,
    holdout_seed=20_260_806,
    pool_seeds=tuple(range(8)),
    low_target=2_500,
    high_target=250_000,
    clip_length=16,
    anchor_spacing=8,
):
    """Audit treatment support without reading labels or outcome data."""

    pool_seeds = tuple(map(int, pool_seeds))
    if not pool_seeds or len(set(pool_seeds)) != len(pool_seeds):
        raise ValueError("pool_seeds must be nonempty and distinct")
    if min(draws, holdout_size, low_target, high_target) <= 0:
        raise ValueError("draws, holdout size, and pool targets must be positive")
    if low_target >= high_target:
        raise ValueError("low_target must be smaller than high_target")
    inventory_path = Path(inventory_path).expanduser().resolve()
    with inventory_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sequence_id", "source_group", "num_frames", "eligible"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"inventory is missing required columns: {sorted(missing)}")
    canonical_rows = [row for row in rows if row["eligible"].casefold() == "true"]
    temporally_eligible = [
        row
        for row in canonical_rows
        if anchor_count(
            int(row["num_frames"]), clip_length=clip_length, spacing=anchor_spacing
        )
        >= 2
    ]
    groups = {}
    for row in temporally_eligible:
        groups.setdefault(row["source_group"], []).append(row)
    group_names = sorted(groups)
    random.Random(int(holdout_seed)).shuffle(group_names)
    holdout_names = set(_closest_group_prefix(group_names, groups, int(holdout_size)))
    training_rows = [
        row for row in temporally_eligible if row["source_group"] not in holdout_names
    ]
    training_groups = {}
    for row in training_rows:
        training_groups.setdefault(row["source_group"], []).append(row)

    report = {
        "schema": "gaitlu-hierarchical-support-audit-v1",
        "inventory": str(inventory_path),
        "clip_length": int(clip_length),
        "anchor_spacing": int(anchor_spacing),
        "maximum_adjacent_overlap_fraction": max(
            0.0, (clip_length - anchor_spacing) / clip_length
        ),
        "exposure_used_for_gate": int(draws),
        "exact_deduplicated_eligible_before_temporal_rule": len(canonical_rows),
        "excluded_by_two_anchor_rule": len(canonical_rows) - len(temporally_eligible),
        "eligible_after_two_anchor_rule": len(temporally_eligible),
        "holdout_sequences": len(temporally_eligible) - len(training_rows),
        "training_sequences": len(training_rows),
        "pool_audits": [],
        "gate_pass": False,
        "failure_reasons": [],
    }
    if len(training_rows) < high_target:
        report["failure_reasons"].append(
            f"only {len(training_rows)} training sequences remain for target {high_target}"
        )
        return report

    for replicate, pool_seed in enumerate(pool_seeds):
        ordered_groups = sorted(training_groups)
        random.Random(int(pool_seed)).shuffle(ordered_groups)
        low_names = set(
            _closest_group_prefix(ordered_groups, training_groups, int(low_target))
        )
        high_names = set(
            _closest_group_prefix(ordered_groups, training_groups, int(high_target))
        )
        if not low_names < high_names:
            report["failure_reasons"].append(
                f"replicate {replicate} cannot form strictly nested pools"
            )
            continue
        for support, names in (("low", low_names), ("high", high_names)):
            selected = [row for row in training_rows if row["source_group"] in names]
            metrics = _pool_metrics(
                selected,
                draws=int(draws),
                clip_length=int(clip_length),
                anchor_spacing=int(anchor_spacing),
            )
            metrics.update(
                {
                    "replicate": replicate,
                    "pool_seed": int(pool_seed),
                    "sequence_support": support,
                }
            )
            report["pool_audits"].append(metrics)
            if not metrics["gate_pass"]:
                report["failure_reasons"].append(
                    f"replicate {replicate} {support} pool fails support gate"
                )
    expected_audits = 2 * len(pool_seeds)
    report["gate_pass"] = (
        len(report["pool_audits"]) == expected_audits
        and all(row["gate_pass"] for row in report["pool_audits"])
        and not report["failure_reasons"]
    )
    return report


__all__ = ["anchor_count", "audit_hierarchical_support"]
