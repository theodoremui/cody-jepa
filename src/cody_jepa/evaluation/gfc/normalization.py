"""Training-only normalization and linear adapters for GFC.

All fit functions receive the training rows directly.  They do not inspect a
split manifest or any held-out data.  Callers select the training rows first,
then reuse the returned numerical fit for evaluation rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np

from .core import CANONICAL_CELLS, CLOTHING, DIRECTIONS, SPEEDS, Cell


FLOAT_DTYPE = np.dtype(np.float64)
SCALE_FLOOR = 1e-12
ZERO_NORM_EPSILON = 1e-12
EIGENVALUE_TIE_RTOL = 1e-12

DimensionPolicy = Literal["retain_all", "effective_rank"]
NormalizationMethod = Literal["raw", "pca"]
AdapterKind = Literal["condition", "gait"]


def _finite_matrix(value: Sequence[Sequence[float]] | np.ndarray, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=FLOAT_DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a rectangular numeric matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{label} must have nonempty shape [N, D]")
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    return matrix


def _readonly_array(value: np.ndarray, *, ndim: int, label: str) -> np.ndarray:
    array = np.array(value, dtype=FLOAT_DTYPE, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    array.setflags(write=False)
    return array


def _sorted_rows(matrix: np.ndarray) -> np.ndarray:
    """Remove irrelevant caller row order from floating-point accumulation."""

    order = sorted(
        range(matrix.shape[0]),
        key=lambda index: tuple(float(value).hex() for value in matrix[index]),
    )
    return matrix[np.asarray(order, dtype=int)]


def entropy_effective_rank(eigenvalues: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=FLOAT_DTYPE)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a nonempty vector")
    if not np.isfinite(values).all():
        raise FloatingPointError("eigenvalues contain non-finite values")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values, dtype=FLOAT_DTYPE))
    if total == 0.0:
        return 1.0
    probabilities = values[values > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities), dtype=FLOAT_DTYPE)))


def retained_dimension(effective_rank: float, available_dimension: int) -> int:
    if not math.isfinite(effective_rank) or effective_rank < 0.0:
        raise ValueError("effective_rank must be finite and nonnegative")
    if (
        isinstance(available_dimension, bool)
        or not isinstance(available_dimension, int)
        or available_dimension < 1
    ):
        raise ValueError("available_dimension must be positive")
    return min(available_dimension, max(1, math.floor(effective_rank + 0.5)))


def _stable_l2_normalize(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    scales = np.max(np.abs(matrix), axis=1)
    scaled = np.zeros_like(matrix, dtype=FLOAT_DTYPE)
    nonzero_scale = scales > 0.0
    scaled[nonzero_scale] = matrix[nonzero_scale] / scales[nonzero_scale, None]
    scaled_norms = np.sqrt(np.sum(scaled * scaled, axis=1, dtype=FLOAT_DTYPE))
    result = np.zeros_like(matrix, dtype=FLOAT_DTYPE)
    keep = nonzero_scale & (scales > epsilon / np.maximum(scaled_norms, 1.0))
    result[keep] = scaled[keep] / scaled_norms[keep, None]
    if not np.isfinite(result).all():
        raise FloatingPointError("L2 normalization produced non-finite values")
    return result


def rowwise_l2_normalize(
    rows: Sequence[Sequence[float]] | np.ndarray,
    *,
    epsilon: float = ZERO_NORM_EPSILON,
) -> np.ndarray:
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return _stable_l2_normalize(_finite_matrix(rows, "rows"), epsilon)


def _population_spectrum(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = (centered.T @ centered) / np.float64(centered.shape[0])
    covariance = (covariance + covariance.T) * np.float64(0.5)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    return covariance, eigenvalues


def _modified_gram_schmidt(candidate: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    value = np.array(candidate, dtype=FLOAT_DTYPE, copy=True)
    for _ in range(2):
        for vector in basis:
            value -= np.dot(vector, value) * vector
    return value


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(vector)
    pivot = int(np.flatnonzero(magnitudes == np.max(magnitudes))[0])
    return -vector if vector[pivot] < 0.0 else vector


def _eigenvalues_tied(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right == 0.0
    return abs(left - right) <= EIGENVALUE_TIE_RTOL * max(abs(left), abs(right))


def _canonicalize_component_group(group: np.ndarray) -> np.ndarray:
    projector = (group.T @ group + (group.T @ group).T) * np.float64(0.5)
    basis: list[np.ndarray] = []
    threshold = 64.0 * np.finfo(np.float64).eps
    for axis in range(group.shape[1]):
        candidate = _modified_gram_schmidt(projector[:, axis], basis)
        norm = float(np.linalg.norm(candidate))
        if norm > threshold:
            basis.append(_canonical_sign(candidate / norm))
        if len(basis) == group.shape[0]:
            break
    if len(basis) != group.shape[0]:
        raise RuntimeError("failed to orient a tied PCA subspace")
    return np.stack(basis, axis=0)


def _canonical_components(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=True)
    dimension = centered.shape[1]
    eigenvalues = np.zeros(dimension, dtype=FLOAT_DTYPE)
    eigenvalues[: singular_values.size] = (
        singular_values * singular_values / np.float64(centered.shape[0])
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    components = np.empty((dimension, dimension), dtype=FLOAT_DTYPE)
    start = 0
    while start < dimension:
        stop = start + 1
        while stop < dimension and _eigenvalues_tied(
            float(eigenvalues[start]), float(eigenvalues[stop])
        ):
            stop += 1
        group = vh[start:stop]
        components[start:stop] = (
            _canonical_sign(group[0])[None, :]
            if stop - start == 1
            else _canonicalize_component_group(group)
        )
        start = stop
    return eigenvalues, components


@dataclass(frozen=True)
class NormalizationFit:
    """Numerical parameters for raw-coordinate or PCA normalization."""

    method: NormalizationMethod
    dimension_policy: DimensionPolicy
    fit_row_count: int
    input_dimension: int
    effective_rank: float
    retained_dimension: int
    eigenvalues: np.ndarray = field(repr=False, compare=False)
    coordinate_variances: np.ndarray = field(repr=False, compare=False)
    selected_indices: tuple[int, ...] | None
    input_mean: np.ndarray | None = field(repr=False, compare=False)
    components: np.ndarray | None = field(repr=False, compare=False)
    output_mean: np.ndarray = field(repr=False, compare=False)
    output_scale: np.ndarray = field(repr=False, compare=False)
    scale_floor: float = SCALE_FLOOR
    zero_norm_epsilon: float = ZERO_NORM_EPSILON

    def __post_init__(self) -> None:
        if self.method not in ("raw", "pca"):
            raise ValueError("method must be 'raw' or 'pca'")
        if self.dimension_policy not in ("retain_all", "effective_rank"):
            raise ValueError("invalid dimension_policy")
        if self.fit_row_count < 1 or self.input_dimension < 1:
            raise ValueError("fit dimensions must be positive")
        expected_width = (
            self.input_dimension
            if self.dimension_policy == "retain_all"
            else retained_dimension(self.effective_rank, self.input_dimension)
        )
        if self.retained_dimension != expected_width:
            raise ValueError("retained_dimension disagrees with dimension_policy")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0.0:
            raise ValueError("scale_floor must be finite and positive")
        if not math.isfinite(self.zero_norm_epsilon) or self.zero_norm_epsilon <= 0.0:
            raise ValueError("zero_norm_epsilon must be finite and positive")
        eigenvalues = _readonly_array(self.eigenvalues, ndim=1, label="eigenvalues")
        variances = _readonly_array(
            self.coordinate_variances, ndim=1, label="coordinate_variances"
        )
        output_mean = _readonly_array(self.output_mean, ndim=1, label="output_mean")
        output_scale = _readonly_array(self.output_scale, ndim=1, label="output_scale")
        if eigenvalues.shape != (self.input_dimension,) or variances.shape != (
            self.input_dimension,
        ):
            raise ValueError("spectrum widths do not match input_dimension")
        if output_mean.shape != (self.retained_dimension,) or output_scale.shape != (
            self.retained_dimension,
        ):
            raise ValueError("output statistics do not match retained_dimension")
        if np.any(output_scale < self.scale_floor):
            raise ValueError("output_scale violates scale_floor")
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "coordinate_variances", variances)
        object.__setattr__(self, "output_mean", output_mean)
        object.__setattr__(self, "output_scale", output_scale)
        if self.method == "raw":
            if self.selected_indices is None or len(self.selected_indices) != self.retained_dimension:
                raise ValueError("raw normalization requires selected_indices")
            if self.input_mean is not None or self.components is not None:
                raise ValueError("raw normalization cannot contain PCA parameters")
            ranking = tuple(
                sorted(
                    range(self.input_dimension),
                    key=lambda index: (-float(variances[index]), index),
                )
            )
            expected_indices = (
                tuple(range(self.input_dimension))
                if self.dimension_policy == "retain_all"
                else ranking[: self.retained_dimension]
            )
            if self.selected_indices != expected_indices:
                raise ValueError("selected_indices disagree with the raw dimension policy")
        else:
            if self.selected_indices is not None or self.input_mean is None or self.components is None:
                raise ValueError("PCA normalization has inconsistent parameters")
            input_mean = _readonly_array(self.input_mean, ndim=1, label="input_mean")
            components = _readonly_array(self.components, ndim=2, label="components")
            if input_mean.shape != (self.input_dimension,) or components.shape != (
                self.retained_dimension,
                self.input_dimension,
            ):
                raise ValueError("PCA parameter shapes are inconsistent")
            if not np.allclose(
                components @ components.T,
                np.eye(self.retained_dimension),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError("PCA components must be orthonormal")
            object.__setattr__(self, "input_mean", input_mean)
            object.__setattr__(self, "components", components)

    def transform(self, rows: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        matrix = _finite_matrix(rows, "rows")
        if matrix.shape[1] != self.input_dimension:
            raise ValueError(f"rows have width {matrix.shape[1]}, expected {self.input_dimension}")
        if self.method == "raw":
            assert self.selected_indices is not None
            projected = matrix[:, self.selected_indices]
        else:
            assert self.input_mean is not None and self.components is not None
            projected = (matrix - self.input_mean) @ self.components.T
        standardized = (projected - self.output_mean) / self.output_scale
        return _stable_l2_normalize(standardized, self.zero_norm_epsilon)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": self.method,
            "dimension_policy": self.dimension_policy,
            "fit_row_count": self.fit_row_count,
            "input_dimension": self.input_dimension,
            "effective_rank": self.effective_rank,
            "retained_dimension": self.retained_dimension,
        }


def _fit_normalizer(
    training_rows: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: NormalizationMethod,
    dimension_policy: DimensionPolicy,
    scale_floor: float,
    zero_norm_epsilon: float,
) -> NormalizationFit:
    if dimension_policy not in ("retain_all", "effective_rank"):
        raise ValueError("dimension_policy must be 'retain_all' or 'effective_rank'")
    if not math.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    if not math.isfinite(zero_norm_epsilon) or zero_norm_epsilon <= 0.0:
        raise ValueError("zero_norm_epsilon must be finite and positive")
    training = _sorted_rows(_finite_matrix(training_rows, "training_rows"))
    input_mean = np.mean(training, axis=0, dtype=FLOAT_DTYPE)
    centered = training - input_mean
    covariance, covariance_eigenvalues = _population_spectrum(centered)
    variances = np.diag(covariance)
    if method == "raw":
        eigenvalues = covariance_eigenvalues
        effective_rank = entropy_effective_rank(eigenvalues)
        width = (
            training.shape[1]
            if dimension_policy == "retain_all"
            else retained_dimension(effective_rank, training.shape[1])
        )
        ranking = tuple(
            sorted(range(training.shape[1]), key=lambda index: (-float(variances[index]), index))
        )
        selected = tuple(range(training.shape[1])) if dimension_policy == "retain_all" else ranking[:width]
        projected = training[:, selected]
        pca_mean = None
        components = None
    elif method == "pca":
        eigenvalues, all_components = _canonical_components(centered)
        effective_rank = entropy_effective_rank(eigenvalues)
        width = (
            training.shape[1]
            if dimension_policy == "retain_all"
            else retained_dimension(effective_rank, training.shape[1])
        )
        components = all_components[:width]
        projected = centered @ components.T
        selected = None
        pca_mean = input_mean
    else:
        raise ValueError("method must be 'raw' or 'pca'")
    output_mean = np.mean(projected, axis=0, dtype=FLOAT_DTYPE)
    output_scale = np.maximum(
        np.std(projected, axis=0, ddof=0, dtype=FLOAT_DTYPE), scale_floor
    )
    return NormalizationFit(
        method=method,
        dimension_policy=dimension_policy,
        fit_row_count=training.shape[0],
        input_dimension=training.shape[1],
        effective_rank=effective_rank,
        retained_dimension=width,
        eigenvalues=eigenvalues,
        coordinate_variances=variances,
        selected_indices=selected,
        input_mean=pca_mean,
        components=components,
        output_mean=output_mean,
        output_scale=output_scale,
        scale_floor=scale_floor,
        zero_norm_epsilon=zero_norm_epsilon,
    )


def fit_raw_normalizer(
    training_rows: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension_policy: DimensionPolicy = "effective_rank",
    scale_floor: float = SCALE_FLOOR,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> NormalizationFit:
    return _fit_normalizer(
        training_rows,
        method="raw",
        dimension_policy=dimension_policy,
        scale_floor=scale_floor,
        zero_norm_epsilon=zero_norm_epsilon,
    )


def fit_pca_normalizer(
    training_rows: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension_policy: DimensionPolicy = "effective_rank",
    scale_floor: float = SCALE_FLOOR,
    zero_norm_epsilon: float = ZERO_NORM_EPSILON,
) -> NormalizationFit:
    return _fit_normalizer(
        training_rows,
        method="pca",
        dimension_policy=dimension_policy,
        scale_floor=scale_floor,
        zero_norm_epsilon=zero_norm_epsilon,
    )


def _adapter_targets(cells: Sequence[Cell], kind: AdapterKind) -> np.ndarray:
    if kind == "condition":
        targets = np.zeros((len(cells), 4), dtype=FLOAT_DTYPE)
        for row, cell in enumerate(cells):
            targets[row, CLOTHING.index(cell.clothing)] = 1.0
            targets[row, 2 + DIRECTIONS.index(cell.direction)] = 1.0
        return targets
    if kind == "gait":
        targets = np.zeros((len(cells), 2), dtype=FLOAT_DTYPE)
        for row, cell in enumerate(cells):
            targets[row, SPEEDS.index(cell.speed)] = 1.0
        return targets
    raise ValueError("kind must be 'condition' or 'gait'")


def _validate_training_grid(subject_ids: Sequence[str], cells: Sequence[Cell]) -> None:
    if len(subject_ids) != len(cells):
        raise ValueError("subject_ids and cells must have equal lengths")
    grouped: dict[str, list[Cell]] = {}
    for subject_id, cell in zip(subject_ids, cells):
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_ids must be nonempty strings")
        if not isinstance(cell, Cell):
            raise TypeError("cells must contain Cell values")
        grouped.setdefault(subject_id, []).append(cell)
    expected = set(CANONICAL_CELLS)
    for subject_cells in grouped.values():
        if len(subject_cells) != len(set(subject_cells)):
            raise ValueError("adapter training has a duplicate subject-cell row")
        if len(subject_cells) != len(expected) or set(subject_cells) != expected:
            raise ValueError("every adapter training subject requires all eight cells")


@dataclass(frozen=True)
class RidgeAdapterFit:
    kind: AdapterKind
    alpha: float
    fit_row_count: int
    input_dimension: int
    output_dimension: int
    target_labels: tuple[str, ...]
    feature_mean: np.ndarray = field(repr=False, compare=False)
    feature_scale: np.ndarray = field(repr=False, compare=False)
    coefficients: np.ndarray = field(repr=False, compare=False)
    intercept: np.ndarray = field(repr=False, compare=False)
    scale_floor: float = SCALE_FLOOR

    def __post_init__(self) -> None:
        expected_labels = (
            ("WoJ", "WJ", "R2L", "L2R") if self.kind == "condition" else ("UGS", "FGS")
        )
        if self.kind not in ("condition", "gait") or self.target_labels != expected_labels:
            raise ValueError("adapter kind and target labels are inconsistent")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("ridge alpha must be finite and positive")
        if self.output_dimension != len(expected_labels) or self.input_dimension < 1:
            raise ValueError("adapter dimensions are inconsistent")
        mean = _readonly_array(self.feature_mean, ndim=1, label="feature_mean")
        scale = _readonly_array(self.feature_scale, ndim=1, label="feature_scale")
        coefficients = _readonly_array(self.coefficients, ndim=2, label="coefficients")
        intercept = _readonly_array(self.intercept, ndim=1, label="intercept")
        if mean.shape != (self.input_dimension,) or scale.shape != (self.input_dimension,):
            raise ValueError("adapter input statistics have inconsistent shapes")
        if coefficients.shape != (self.input_dimension, self.output_dimension):
            raise ValueError("adapter coefficient shape is inconsistent")
        if intercept.shape != (self.output_dimension,):
            raise ValueError("adapter intercept shape is inconsistent")
        if np.any(scale < self.scale_floor):
            raise ValueError("adapter feature_scale violates scale_floor")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)

    def transform(self, rows: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        matrix = _finite_matrix(rows, "adapter rows")
        if matrix.shape[1] != self.input_dimension:
            raise ValueError(f"adapter rows have width {matrix.shape[1]}, expected {self.input_dimension}")
        standardized = (matrix - self.feature_mean) / self.feature_scale
        result = (
            np.einsum("nd,dk->nk", standardized, self.coefficients, optimize=False)
            + self.intercept
        )
        if not np.isfinite(result).all():
            raise FloatingPointError("adapter produced non-finite values")
        return result


def fit_ridge_adapter(
    training_rows: Sequence[Sequence[float]] | np.ndarray,
    subject_ids: Sequence[str],
    cells: Sequence[Cell],
    *,
    kind: AdapterKind,
    alpha: float = 1.0,
    scale_floor: float = SCALE_FLOOR,
) -> RidgeAdapterFit:
    matrix = _finite_matrix(training_rows, "training_rows")
    ids = tuple(subject_ids)
    cell_values = tuple(cells)
    if len(ids) != matrix.shape[0] or len(cell_values) != matrix.shape[0]:
        raise ValueError("subject_ids and cells must contain one value per training row")
    _validate_training_grid(ids, cell_values)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("ridge alpha must be finite and positive")
    if not math.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    targets = _adapter_targets(cell_values, kind)
    order = sorted(
        range(matrix.shape[0]),
        key=lambda index: (
            ids[index],
            cell_values[index].canonical_index,
            *(float(value).hex() for value in matrix[index]),
        ),
    )
    order_array = np.asarray(order, dtype=int)
    training = matrix[order_array]
    targets = targets[order_array]
    feature_mean = np.mean(training, axis=0, dtype=FLOAT_DTYPE)
    feature_scale = np.maximum(
        np.std(training, axis=0, ddof=0, dtype=FLOAT_DTYPE), scale_floor
    )
    standardized = (training - feature_mean) / feature_scale
    x_mean = np.mean(standardized, axis=0, dtype=FLOAT_DTYPE)
    y_mean = np.mean(targets, axis=0, dtype=FLOAT_DTYPE)
    centered_x = standardized - x_mean
    centered_y = targets - y_mean
    gram = np.einsum("ni,nj->ij", centered_x, centered_x, dtype=FLOAT_DTYPE)
    right_hand_side = np.einsum(
        "ni,nj->ij", centered_x, centered_y, dtype=FLOAT_DTYPE
    )
    if not np.isfinite(gram).all() or not np.isfinite(right_hand_side).all():
        raise FloatingPointError("ridge normal equations are non-finite")
    regularized = gram + float(alpha) * np.eye(training.shape[1], dtype=FLOAT_DTYPE)
    coefficients = np.linalg.solve(regularized, right_hand_side)
    intercept = y_mean - x_mean @ coefficients
    labels = ("WoJ", "WJ", "R2L", "L2R") if kind == "condition" else ("UGS", "FGS")
    return RidgeAdapterFit(
        kind=kind,
        alpha=float(alpha),
        fit_row_count=training.shape[0],
        input_dimension=training.shape[1],
        output_dimension=len(labels),
        target_labels=labels,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
        scale_floor=scale_floor,
    )


def fit_condition_adapter(
    training_rows: Sequence[Sequence[float]] | np.ndarray,
    subject_ids: Sequence[str],
    cells: Sequence[Cell],
    *,
    alpha: float = 1.0,
) -> RidgeAdapterFit:
    return fit_ridge_adapter(training_rows, subject_ids, cells, kind="condition", alpha=alpha)


def fit_gait_adapter(
    training_rows: Sequence[Sequence[float]] | np.ndarray,
    subject_ids: Sequence[str],
    cells: Sequence[Cell],
    *,
    alpha: float = 1.0,
) -> RidgeAdapterFit:
    return fit_ridge_adapter(training_rows, subject_ids, cells, kind="gait", alpha=alpha)


__all__ = [
    "SCALE_FLOOR",
    "ZERO_NORM_EPSILON",
    "NormalizationFit",
    "RidgeAdapterFit",
    "entropy_effective_rank",
    "fit_condition_adapter",
    "fit_gait_adapter",
    "fit_pca_normalizer",
    "fit_raw_normalizer",
    "fit_ridge_adapter",
    "retained_dimension",
    "rowwise_l2_normalize",
]
