"""Exact symbolic oracle spectra for factorial retrieval protocols.

The oracle in this module is deliberately independent of learned features and the
floating-point GFC scorer.  A protocol compiler constructs queries from factorial
cells and donor roles; the enumerator then asks what score a solver that recovers a
declared subset of factors exactly must receive.

This separation makes protocol geometry executable.  In particular, it exposes how
gallery filtering changes partial-factor ceilings without treating those ceilings as
empirical results about a representation.  The current compiler intentionally supports
the binary complementary donor rule, the retain-all and exclude-donors galleries, and
the fractional-top-1/average-rank tie rule required by GFC-v2.  Exhaustive coverage is
tested for designs with two through five binary factors; larger designs have
exponentially more cells, queries, and recovered-factor subsets.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
import itertools


Cell = tuple[str, ...]

RETAIN_ALL = "retain_all"
EXCLUDE_DONORS = "exclude_donors"
BINARY_COMPLEMENTARY_TWO_DONOR = "binary_complementary_two_donor"
FRACTIONAL_AVERAGE_RANK = "fractional_top1_average_occupied_rank"


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be nonempty text without outer whitespace")
    return value


@dataclass(frozen=True, order=True)
class Factor:
    """One named categorical factor and its canonical value order."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "factor name"))
        if isinstance(self.values, (str, bytes)):
            raise TypeError("factor values must be a sequence of labels, not text")
        values = tuple(_nonempty_text(value, f"{self.name} value") for value in self.values)
        if len(values) < 2:
            raise ValueError(f"factor {self.name!r} must have at least two values")
        if len(set(values)) != len(values):
            raise ValueError(f"factor {self.name!r} values must be unique")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class FactorialDesign:
    """A finite Cartesian product of ordered categorical factors."""

    factors: tuple[Factor, ...]

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        if len(factors) < 2:
            raise ValueError("a factorial oracle design requires at least two factors")
        if not all(isinstance(factor, Factor) for factor in factors):
            raise TypeError("factors must contain only Factor instances")
        names = [factor.name for factor in factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")
        object.__setattr__(self, "factors", factors)

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(factor.name for factor in self.factors)

    @property
    def cells(self) -> tuple[Cell, ...]:
        return tuple(itertools.product(*(factor.values for factor in self.factors)))

    def factor_index(self, name: str) -> int:
        try:
            return self.factor_names.index(name)
        except ValueError as error:
            raise ValueError(f"unknown factor {name!r}") from error

    def validate_cell(self, cell: Sequence[str], *, label: str = "cell") -> Cell:
        if isinstance(cell, (str, bytes)):
            raise TypeError(f"{label} must be a sequence of factor values, not text")
        values = tuple(cell)
        if len(values) != len(self.factors):
            raise ValueError(f"{label} has {len(values)} values, expected {len(self.factors)}")
        for factor, value in zip(self.factors, values):
            if value not in factor.values:
                raise ValueError(
                    f"{label} has unknown {factor.name} value {value!r}; "
                    f"expected one of {factor.values}"
                )
        return values

    def cell_dict(self, cell: Sequence[str]) -> dict[str, str]:
        values = self.validate_cell(cell)
        return dict(zip(self.factor_names, values))

    def to_dict(self) -> dict[str, object]:
        return {
            "factors": [
                {"name": factor.name, "values": list(factor.values)} for factor in self.factors
            ],
            "cell_count": len(self.cells),
        }


@dataclass(frozen=True)
class CompiledQuery:
    """A symbolic query whose factor values are copied from declared donors."""

    query_index: int
    target: Cell
    focal_factor: str
    donors: tuple[Cell, ...]
    factor_sources: tuple[int, ...]
    gallery: tuple[Cell, ...]

    def compose(self) -> Cell:
        """Construct the query values from donors instead of assuming the target."""

        return tuple(
            self.donors[source_index][factor_index]
            for factor_index, source_index in enumerate(self.factor_sources)
        )


@dataclass(frozen=True)
class CompiledProtocol:
    """A fully explicit finite protocol ready for symbolic enumeration."""

    name: str
    design: FactorialDesign
    focal_factors: tuple[str, ...]
    donor_rule: str
    gallery_policy: str
    tie_policy: str
    queries: tuple[CompiledQuery, ...]


@dataclass(frozen=True)
class OracleQueryScore:
    """Exact rank quantities for one compiled query and recovered-factor subset."""

    query_index: int
    strictly_closer_count: int
    tie_size: int
    average_rank: Fraction
    top1: Fraction
    reciprocal_rank: Fraction


@dataclass(frozen=True)
class OracleSpectrumEntry:
    """Aggregate exact score for one subset of recovered factors."""

    recovered_factors: tuple[str, ...]
    query_scores: tuple[OracleQueryScore, ...]
    top1: Fraction
    mrr: Fraction

    def to_dict(self) -> dict[str, object]:
        ties = Counter(score.tie_size for score in self.query_scores)
        closer = Counter(score.strictly_closer_count for score in self.query_scores)
        return {
            "recovered_factors": list(self.recovered_factors),
            "recovered_factor_count": len(self.recovered_factors),
            "query_count": len(self.query_scores),
            "top1_fraction": str(self.top1),
            "top1": float(self.top1),
            "mrr_fraction": str(self.mrr),
            "mrr": float(self.mrr),
            "tie_size_histogram": {str(size): count for size, count in sorted(ties.items())},
            "strictly_closer_histogram": {
                str(count): frequency for count, frequency in sorted(closer.items())
            },
        }


@dataclass(frozen=True)
class OracleSpectrum:
    """The exact partial-factor spectrum for one compiled protocol."""

    protocol: CompiledProtocol
    entries: tuple[OracleSpectrumEntry, ...]

    def by_recovered_factors(self) -> dict[tuple[str, ...], OracleSpectrumEntry]:
        return {entry.recovered_factors: entry for entry in self.entries}

    def to_dict(self) -> dict[str, object]:
        spectrum_rows = []
        for entry in self.entries:
            row = entry.to_dict()
            focal_breakdown = []
            for focal in self.protocol.focal_factors:
                scores = tuple(
                    score
                    for query, score in zip(self.protocol.queries, entry.query_scores)
                    if query.focal_factor == focal
                )
                focal_breakdown.append(
                    {
                        "focal_factor": focal,
                        "query_count": len(scores),
                        "top1_fraction": str(_fraction_mean(score.top1 for score in scores)),
                        "top1": float(_fraction_mean(score.top1 for score in scores)),
                        "mrr_fraction": str(
                            _fraction_mean(score.reciprocal_rank for score in scores)
                        ),
                        "mrr": float(_fraction_mean(score.reciprocal_rank for score in scores)),
                    }
                )
            row["focal_breakdown"] = focal_breakdown
            spectrum_rows.append(row)
        return {
            "protocol": self.protocol.name,
            "design": self.protocol.design.to_dict(),
            "focal_factors": list(self.protocol.focal_factors),
            "donor_rule": self.protocol.donor_rule,
            "gallery_policy": self.protocol.gallery_policy,
            "tie_policy": self.protocol.tie_policy,
            "query_count": len(self.protocol.queries),
            "spectrum": spectrum_rows,
        }


def _validated_focal_factors(
    design: FactorialDesign,
    focal_factors: Iterable[str] | None,
) -> tuple[str, ...]:
    if isinstance(focal_factors, (str, bytes)):
        raise TypeError("focal_factors must be a sequence of names, not text")
    values = design.factor_names if focal_factors is None else tuple(focal_factors)
    if not values:
        raise ValueError("focal_factors must not be empty")
    values = tuple(_nonempty_text(value, "focal factor") for value in values)
    if len(set(values)) != len(values):
        raise ValueError("focal_factors must be unique")
    unknown = tuple(value for value in values if value not in design.factor_names)
    if unknown:
        raise ValueError(f"unknown focal factors: {unknown}")
    return values


def _binary_opposite(factor: Factor, value: str) -> str:
    if len(factor.values) != 2:
        raise ValueError(
            "the complementary donor rule requires exactly two values for every factor"
        )
    return factor.values[1 - factor.values.index(value)]


def compile_binary_complement_protocol(
    design: FactorialDesign,
    *,
    focal_factors: Iterable[str] | None = None,
    gallery_policy: str = RETAIN_ALL,
    name: str = "binary_complement",
) -> CompiledProtocol:
    """Compile complementary two-donor queries for a binary factorial design.

    Donor ``u`` matches the target on the focal factor and flips every other
    factor.  Donor ``v`` flips the focal factor and matches every other factor.
    The query takes the focal value from ``u`` and all remaining values from
    ``v``.
    """

    if not isinstance(design, FactorialDesign):
        raise TypeError("design must be a FactorialDesign")
    protocol_name = _nonempty_text(name, "protocol name")
    focals = _validated_focal_factors(design, focal_factors)
    if gallery_policy not in {RETAIN_ALL, EXCLUDE_DONORS}:
        raise ValueError(f"gallery_policy must be {RETAIN_ALL!r} or {EXCLUDE_DONORS!r}")
    if any(len(factor.values) != 2 for factor in design.factors):
        raise ValueError(
            "the complementary donor rule requires exactly two values for every factor"
        )

    queries: list[CompiledQuery] = []
    cells = design.cells
    for target in cells:
        for focal_factor in focals:
            focal_index = design.factor_index(focal_factor)
            donor_u = tuple(
                value if index == focal_index else _binary_opposite(factor, value)
                for index, (factor, value) in enumerate(zip(design.factors, target))
            )
            donor_v = tuple(
                _binary_opposite(factor, value) if index == focal_index else value
                for index, (factor, value) in enumerate(zip(design.factors, target))
            )
            donors = (donor_u, donor_v)
            factor_sources = tuple(
                0 if index == focal_index else 1 for index in range(len(design.factors))
            )
            if gallery_policy == RETAIN_ALL:
                gallery = cells
            else:
                excluded = set(donors)
                gallery = tuple(cell for cell in cells if cell not in excluded)
            queries.append(
                CompiledQuery(
                    query_index=len(queries),
                    target=target,
                    focal_factor=focal_factor,
                    donors=donors,
                    factor_sources=factor_sources,
                    gallery=gallery,
                )
            )

    protocol = CompiledProtocol(
        name=protocol_name,
        design=design,
        focal_factors=focals,
        donor_rule=BINARY_COMPLEMENTARY_TWO_DONOR,
        gallery_policy=gallery_policy,
        tie_policy=FRACTIONAL_AVERAGE_RANK,
        queries=tuple(queries),
    )
    _validate_protocol(protocol)
    return protocol


def compile_healthgait_gfc_v2_protocol(*, gallery_policy: str = RETAIN_ALL) -> CompiledProtocol:
    """Compile the canonical 16-query Health&Gait GFC-v2 oracle protocol.

    Direction is intentionally not a focal factor: the opposite-direction clip
    for fixed speed and clothing comes from the same physical walk.  Dataset-level
    ``source_video_id`` assertions remain the responsibility of the production
    evaluator; a cell-level oracle cannot establish session independence.
    """

    design = FactorialDesign(
        (
            Factor("speed", ("UGS", "FGS")),
            Factor("clothing", ("WoJ", "WJ")),
            Factor("direction", ("R2L", "L2R")),
        )
    )
    return compile_binary_complement_protocol(
        design,
        focal_factors=("speed", "clothing"),
        gallery_policy=gallery_policy,
        name="gfc_v2",
    )


def _validate_protocol(protocol: CompiledProtocol) -> None:
    if not isinstance(protocol, CompiledProtocol):
        raise TypeError("protocol must be a CompiledProtocol")
    _nonempty_text(protocol.name, "protocol name")
    design = protocol.design
    if not isinstance(design, FactorialDesign):
        raise TypeError("protocol design must be a FactorialDesign")
    if not isinstance(protocol.focal_factors, tuple):
        raise TypeError("protocol focal_factors must be an immutable tuple")
    _validated_focal_factors(design, protocol.focal_factors)
    if not isinstance(protocol.donor_rule, str):
        raise TypeError("protocol donor_rule must be text")
    if protocol.donor_rule != BINARY_COMPLEMENTARY_TWO_DONOR:
        raise ValueError("protocol has an unsupported donor rule")
    if not isinstance(protocol.gallery_policy, str):
        raise TypeError("protocol gallery_policy must be text")
    if protocol.gallery_policy not in {RETAIN_ALL, EXCLUDE_DONORS}:
        raise ValueError("protocol has an unsupported gallery policy")
    if not isinstance(protocol.tie_policy, str):
        raise TypeError("protocol tie_policy must be text")
    if protocol.tie_policy != FRACTIONAL_AVERAGE_RANK:
        raise ValueError("protocol has an unsupported tie policy")
    if not isinstance(protocol.queries, tuple):
        raise TypeError("protocol queries must be an immutable tuple")
    if not protocol.queries:
        raise ValueError("protocol must contain at least one query")
    if not all(isinstance(query, CompiledQuery) for query in protocol.queries):
        raise TypeError("protocol queries must contain only CompiledQuery instances")

    design_cells = set(design.cells)
    indices = [query.query_index for query in protocol.queries]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("query indices must be integers, not booleans")
    if indices != list(range(len(protocol.queries))):
        raise ValueError("query indices must be consecutive and canonical")
    target_focal_pairs = [(query.target, query.focal_factor) for query in protocol.queries]
    expected_pairs = [(cell, focal) for cell in design.cells for focal in protocol.focal_factors]
    if target_focal_pairs != expected_pairs:
        raise ValueError(
            "queries must cover every target and focal factor exactly once in canonical order"
        )
    for query in protocol.queries:
        if not isinstance(query.target, tuple):
            raise TypeError("query target must be an immutable tuple")
        target = design.validate_cell(query.target, label="query target")
        _nonempty_text(query.focal_factor, "query focal factor")
        if query.focal_factor not in protocol.focal_factors:
            raise ValueError("query focal factor is not declared by the protocol")
        if not isinstance(query.donors, tuple):
            raise TypeError("query donors must be an immutable tuple")
        if not query.donors:
            raise ValueError("query must contain at least one donor")
        donors = tuple(design.validate_cell(donor, label="query donor") for donor in query.donors)
        if len(set(donors)) != len(donors):
            raise ValueError("query donors must be distinct")
        if target in donors:
            raise ValueError("query target must not be one of its donors")
        if not isinstance(query.factor_sources, tuple):
            raise TypeError("query factor_sources must be an immutable tuple")
        if len(query.factor_sources) != len(design.factors):
            raise ValueError("query must declare one donor source per factor")
        if any(
            isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < len(donors)
            for source in query.factor_sources
        ):
            raise ValueError("query factor source index is invalid")
        focal_index = design.factor_index(query.focal_factor)
        expected_u = tuple(
            value if index == focal_index else _binary_opposite(factor, value)
            for index, (factor, value) in enumerate(zip(design.factors, target))
        )
        expected_v = tuple(
            _binary_opposite(factor, value) if index == focal_index else value
            for index, (factor, value) in enumerate(zip(design.factors, target))
        )
        expected_sources = tuple(
            0 if index == focal_index else 1 for index in range(len(design.factors))
        )
        if donors != (expected_u, expected_v) or query.factor_sources != expected_sources:
            raise ValueError("query does not obey the declared complementary donor rule")
        if not isinstance(query.gallery, tuple):
            raise TypeError("query gallery must be an immutable tuple")
        gallery = tuple(
            design.validate_cell(cell, label="query gallery cell") for cell in query.gallery
        )
        if not gallery:
            raise ValueError("query gallery must not be empty")
        if len(set(gallery)) != len(gallery):
            raise ValueError("query gallery cells must be unique")
        if not set(gallery).issubset(design_cells):
            raise ValueError("query gallery contains a cell outside the design")
        if target not in gallery:
            raise ValueError("query target must remain in its gallery")
        if protocol.gallery_policy == RETAIN_ALL and set(gallery) != design_cells:
            raise ValueError("retain-all gallery must contain every factorial cell")
        if protocol.gallery_policy == EXCLUDE_DONORS:
            if any(donor in gallery for donor in donors):
                raise ValueError("exclude-donors gallery contains a donor")
            if set(gallery) != design_cells.difference(donors):
                raise ValueError("exclude-donors gallery removed a non-donor cell")
        composed = design.validate_cell(query.compose(), label="composed query")
        if composed != target:
            raise ValueError("declared donor sources do not compose the target cell")


def _powerset(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset for size in range(len(values) + 1) for subset in itertools.combinations(values, size)
    )


def _fraction_mean(values: Iterable[Fraction]) -> Fraction:
    items = tuple(values)
    if not items:
        raise ValueError("cannot average an empty collection")
    return sum(items, Fraction(0, 1)) / len(items)


def _validated_recovered_factors(
    design: FactorialDesign,
    recovered_factors: Iterable[str],
) -> tuple[str, ...]:
    if isinstance(recovered_factors, (str, bytes)):
        raise TypeError("recovered_factors must be a sequence of names, not text")
    recovered = tuple(_nonempty_text(value, "recovered factor") for value in recovered_factors)
    if len(set(recovered)) != len(recovered):
        raise ValueError("recovered_factors must be unique")
    unknown = tuple(name for name in recovered if name not in design.factor_names)
    if unknown:
        raise ValueError(f"unknown recovered factors: {unknown}")
    return recovered


def _score_validated_oracle_query(
    protocol: CompiledProtocol,
    query: CompiledQuery,
    recovered: tuple[str, ...],
) -> OracleQueryScore:
    recovered_indices = tuple(protocol.design.factor_index(name) for name in recovered)
    composed = query.compose()
    distances = {
        cell: sum(composed[index] != cell[index] for index in recovered_indices)
        for cell in query.gallery
    }
    target_distance = distances[query.target]
    closer = sum(distance < target_distance for distance in distances.values())
    tie_size = sum(distance == target_distance for distance in distances.values())
    average_rank = Fraction(closer, 1) + Fraction(tie_size + 1, 2)
    top1 = Fraction(1, tie_size) if closer == 0 else Fraction(0, 1)
    return OracleQueryScore(
        query_index=query.query_index,
        strictly_closer_count=closer,
        tie_size=tie_size,
        average_rank=average_rank,
        top1=top1,
        reciprocal_rank=Fraction(1, 1) / average_rank,
    )


def score_oracle_query(
    protocol: CompiledProtocol,
    query: CompiledQuery,
    recovered_factors: Iterable[str],
) -> OracleQueryScore:
    """Score one query exactly for a solver that recovers selected factors."""

    _validate_protocol(protocol)
    recovered = _validated_recovered_factors(protocol.design, recovered_factors)
    if query not in protocol.queries:
        raise ValueError("query is not part of the compiled protocol")
    return _score_validated_oracle_query(protocol, query, recovered)


def enumerate_oracle_spectrum(protocol: CompiledProtocol) -> OracleSpectrum:
    """Enumerate exact top-1 and MRR for every recovered-factor subset."""

    _validate_protocol(protocol)
    entries: list[OracleSpectrumEntry] = []
    for recovered in _powerset(protocol.design.factor_names):
        query_scores = tuple(
            _score_validated_oracle_query(protocol, query, recovered) for query in protocol.queries
        )
        entries.append(
            OracleSpectrumEntry(
                recovered_factors=recovered,
                query_scores=query_scores,
                top1=_fraction_mean(score.top1 for score in query_scores),
                mrr=_fraction_mean(score.reciprocal_rank for score in query_scores),
            )
        )
    return OracleSpectrum(protocol=protocol, entries=tuple(entries))


__all__ = [
    "BINARY_COMPLEMENTARY_TWO_DONOR",
    "EXCLUDE_DONORS",
    "FRACTIONAL_AVERAGE_RANK",
    "RETAIN_ALL",
    "Cell",
    "CompiledProtocol",
    "CompiledQuery",
    "Factor",
    "FactorialDesign",
    "OracleQueryScore",
    "OracleSpectrum",
    "OracleSpectrumEntry",
    "compile_binary_complement_protocol",
    "compile_healthgait_gfc_v2_protocol",
    "enumerate_oracle_spectrum",
    "score_oracle_query",
]
