"""Grounded Factorial Completion evaluation pipeline."""

from .oracle import (
    BINARY_COMPLEMENTARY_TWO_DONOR,
    EXCLUDE_DONORS,
    FRACTIONAL_AVERAGE_RANK,
    RETAIN_ALL,
    CompiledProtocol,
    CompiledQuery,
    Factor,
    FactorialDesign,
    OracleQueryScore,
    OracleSpectrum,
    OracleSpectrumEntry,
    compile_binary_complement_protocol,
    compile_healthgait_gfc_v2_protocol,
    enumerate_oracle_spectrum,
    score_oracle_query,
)


def run_gfc(*args, **kwargs):
    """Run GFC without eagerly importing the command and I/O layer."""
    from .runner import run_gfc as _run_gfc

    return _run_gfc(*args, **kwargs)


__all__ = [
    "BINARY_COMPLEMENTARY_TWO_DONOR",
    "EXCLUDE_DONORS",
    "FRACTIONAL_AVERAGE_RANK",
    "RETAIN_ALL",
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
    "run_gfc",
    "score_oracle_query",
]
