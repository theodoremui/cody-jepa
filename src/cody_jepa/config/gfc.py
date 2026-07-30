"""Schema validation for the maintained GFC evaluation protocol."""

from pathlib import Path
from typing import Any
import json

import numpy as np

from ..evaluation.gfc.core import CLOTHING, DIRECTIONS, SPEEDS


def load_gfc_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GFC config must be a JSON object")
    validate_gfc_config(value)
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"unsupported {label}: expected {expected!r}, got {actual!r}")


def _require_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _require_equal(set(value), expected, f"{label} fields")
    return value


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite_real(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_gfc_config(config: dict[str, Any]) -> None:
    """Reject scientific settings the maintained evaluator would not honor."""

    _require_fields(
        config,
        {
            "subject_column",
            "recording_column",
            "split_column",
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
        },
        "top-level config",
    )
    for key in ("subject_column", "recording_column", "split_column"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    split_map = _require_fields(
        config["split_map"],
        {"development_train", "development", "confirmation"},
        "split_map",
    )
    _require_equal(
        split_map,
        {"development_train": "train", "development": "val", "confirmation": "test"},
        "split map",
    )
    factors = _require_fields(config["factors"], {"order", "values"}, "factors")
    factor_values = _require_fields(
        factors["values"], {"speed", "clothing", "direction"}, "factor values"
    )
    _require_equal(factors["order"], ["speed", "clothing", "direction"], "factor order")
    _require_equal(factor_values["speed"], list(SPEEDS), "speed values")
    _require_equal(factor_values["clothing"], list(CLOTHING), "clothing values")
    _require_equal(factor_values["direction"], list(DIRECTIONS), "direction values")
    aggregation = _require_fields(
        config["recording_aggregation"],
        {"windows_per_recording", "method", "dtype"},
        "recording aggregation",
    )
    _require_equal(_require_integer(aggregation["windows_per_recording"], "window count", minimum=1), 3, "window count")
    _require_equal(aggregation["method"], "mean", "recording aggregation")
    _require_equal(aggregation["dtype"], "float64", "recording aggregation dtype")
    complete_case = _require_fields(
        config["complete_case"],
        {"required_cells", "missing_cell", "duplicate_cell"},
        "complete case",
    )
    _require_equal(_require_integer(complete_case["required_cells"], "complete-cell count", minimum=1), 8, "complete-cell count")
    _require_equal(complete_case["missing_cell"], "exclude_participant", "missing-cell policy")
    _require_equal(complete_case["duplicate_cell"], "error", "duplicate-cell policy")
    query = _require_fields(
        config["query"],
        {
            "targets_per_participant",
            "condition_donor",
            "gait_donors",
            "gait_donors_per_target",
            "queries_per_participant",
        },
        "query",
    )
    _require_equal(_require_integer(query["targets_per_participant"], "target count", minimum=1), 8, "target count")
    _require_equal(
        query["condition_donor"],
        "same_clothing_direction_opposite_speed",
        "condition donor rule",
    )
    _require_equal(
        query["gait_donors"],
        "same_speed_all_other_clothing_direction_pairs",
        "gait donor rule",
    )
    _require_equal(_require_integer(query["gait_donors_per_target"], "gait donor count", minimum=1), 3, "gait donor count")
    _require_equal(_require_integer(query["queries_per_participant"], "query count", minimum=1), 24, "query count")
    gallery = _require_fields(
        config["gallery"],
        {
            "exclude_condition_donor",
            "exclude_gait_donor",
            "size",
            "target_count",
            "distractor_count",
        },
        "gallery",
    )
    _require_equal(gallery["exclude_condition_donor"], True, "condition donor exclusion")
    _require_equal(gallery["exclude_gait_donor"], True, "gait donor exclusion")
    _require_equal(_require_integer(gallery["size"], "gallery size", minimum=1), 6, "gallery size")
    _require_equal(_require_integer(gallery["target_count"], "gallery target count", minimum=1), 1, "gallery target count")
    _require_equal(_require_integer(gallery["distractor_count"], "gallery distractor count", minimum=1), 5, "gallery distractor count")
    distance = _require_fields(
        config["distance"],
        {"metric", "condition_weight", "gait_weight", "dtype", "zero_norm_epsilon"},
        "distance",
    )
    _require_equal(distance["metric"], "cosine", "distance metric")
    _require_equal(distance["dtype"], "float64", "distance dtype")
    condition_weight = _require_finite_real(
        distance["condition_weight"], "condition weight"
    )
    gait_weight = _require_finite_real(distance["gait_weight"], "gait weight")
    _require_equal(
        condition_weight + gait_weight,
        1.0,
        "distance-weight sum",
    )
    if _require_finite_real(
        distance["zero_norm_epsilon"], "zero_norm_epsilon"
    ) <= 0.0:
        raise ValueError("zero_norm_epsilon must be positive")
    ties = _require_fields(
        config["ties"],
        {"absolute_tolerance", "relative_tolerance", "rank", "top1"},
        "ties",
    )
    _require_equal(
        _require_finite_real(ties["relative_tolerance"], "tie relative tolerance"),
        0.0,
        "tie relative tolerance",
    )
    _require_equal(ties["rank"], "average_occupied_rank", "tie rank")
    _require_equal(ties["top1"], "fractional_credit", "tie top-1")
    if _require_finite_real(
        ties["absolute_tolerance"], "tie absolute tolerance"
    ) < 0.0:
        raise ValueError("tie absolute tolerance must be nonnegative")
    adapter = _require_fields(
        config["adapter"],
        {
            "type",
            "alpha",
            "fit_split",
            "input_standardization",
            "condition_labels",
            "gait_labels",
            "condition_output_dim",
            "gait_output_dim",
        },
        "adapter",
    )
    _require_equal(adapter["type"], "ridge", "adapter type")
    _require_equal(adapter["input_standardization"], "population", "adapter standardization")
    _require_equal(adapter["condition_labels"], ["WoJ", "WJ", "R2L", "L2R"], "condition labels")
    _require_equal(adapter["gait_labels"], ["UGS", "FGS"], "gait labels")
    _require_equal(_require_integer(adapter["condition_output_dim"], "condition output width", minimum=1), 4, "condition output width")
    _require_equal(_require_integer(adapter["gait_output_dim"], "gait output width", minimum=1), 2, "gait output width")
    _require_equal(adapter["fit_split"], "development_train", "adapter fit split")
    if _require_finite_real(adapter["alpha"], "adapter alpha") <= 0.0:
        raise ValueError("adapter alpha and fit split must be valid")
    normalization = _require_fields(
        config["normalization"],
        {
            "fit_split",
            "primary",
            "sensitivities",
            "standardization",
            "scale_floor",
            "block_l2_epsilon",
            "effective_rank_rounding",
            "pca_solver",
        },
        "normalization",
    )
    _require_equal(normalization["standardization"], "population", "normalization standardization")
    _require_equal(normalization["effective_rank_rounding"], "nearest_half_up", "rank rounding")
    _require_equal(normalization["pca_solver"], "full_svd", "PCA solver")
    _require_equal(normalization["primary"], "raw_retain_all", "primary normalization")
    _require_equal(
        normalization["sensitivities"],
        ["raw_effective_rank", "pca_effective_rank"],
        "normalization sensitivities",
    )
    _require_equal(
        normalization["fit_split"], "development_train", "normalization fit split"
    )
    scale_floor = _require_finite_real(normalization["scale_floor"], "scale floor")
    block_l2_epsilon = _require_finite_real(
        normalization["block_l2_epsilon"], "block L2 epsilon"
    )
    if scale_floor <= 0.0 or block_l2_epsilon <= 0.0:
        raise ValueError("normalization numerical floors must be positive")
    shortcut = _require_fields(
        config["shortcut"], {"condition_columns", "gait_columns"}, "shortcut"
    )
    condition_columns = list(shortcut["condition_columns"])
    gait_columns = list(shortcut["gait_columns"])
    if not condition_columns or not gait_columns or set(condition_columns) & set(gait_columns):
        raise ValueError("shortcut blocks must contain distinct named columns")
    _require_equal(
        condition_columns,
        [
            "shortcut_horizontal_centroid_drift_signed",
            "shortcut_horizontal_centroid_drift_absolute",
            "shortcut_foreground_area_mean",
            "shortcut_foreground_area_std",
            "shortcut_foreground_area_q25",
            "shortcut_foreground_area_median",
            "shortcut_foreground_area_q75",
        ],
        "condition shortcut columns",
    )
    _require_equal(
        gait_columns,
        ["shortcut_log_frame_count", "shortcut_duration_seconds"],
        "gait shortcut columns",
    )
    _require_equal(config["metrics"], ["top1", "mean_reciprocal_rank", "donor_attraction"], "metrics")
    _require_equal(config["primary_metric"], "top1", "primary metric")
    _require_equal(config["primary_contrast"], "learned_minus_shortcut", "primary contrast")
    bootstrap = _require_fields(
        config["bootstrap"],
        {"unit", "resamples", "seed", "confidence_level", "interval"},
        "bootstrap",
    )
    _require_equal(bootstrap["unit"], "participant", "bootstrap unit")
    _require_equal(bootstrap["interval"], "percentile", "bootstrap interval")
    _require_integer(bootstrap["resamples"], "bootstrap resamples", minimum=1)
    _require_integer(bootstrap["seed"], "bootstrap seed")
    confidence_level = _require_finite_real(
        bootstrap["confidence_level"], "bootstrap confidence level"
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must lie in (0, 1)")
    power = _require_fields(
        config["power"],
        {
            "effect",
            "alpha",
            "minimum_power",
            "alternative",
            "participant_standard_deviation_ddof",
        },
        "power",
    )
    _require_equal(power["alternative"], "two_sided", "power alternative")
    _require_equal(_require_integer(power["participant_standard_deviation_ddof"], "power spread ddof"), 1, "power spread ddof")
    power_effect = _require_finite_real(power["effect"], "power effect")
    power_alpha = _require_finite_real(power["alpha"], "power alpha")
    minimum_power = _require_finite_real(
        power["minimum_power"], "minimum power"
    )
    if power_effect <= 0.0 or not 0.0 < power_alpha < 1.0:
        raise ValueError("power effect and alpha must be valid")
    if not 0.0 < minimum_power < 1.0:
        raise ValueError("minimum power must lie in (0, 1)")



__all__ = ["load_gfc_config", "validate_gfc_config"]
