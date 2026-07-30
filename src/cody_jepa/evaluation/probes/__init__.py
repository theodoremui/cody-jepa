"""Simple evaluation protocols for frozen representations."""

from .gait import evaluate_gait_system
from .identity import (
    closed_set_masks,
    evaluate_identity_closed_set,
    evaluate_identity_heldout_retrieval,
)
from .pipeline import PROBE_SUMMARY_COLUMNS, evaluate_all_probes, write_probe_results

__all__ = [
    "PROBE_SUMMARY_COLUMNS",
    "closed_set_masks",
    "evaluate_all_probes",
    "evaluate_gait_system",
    "evaluate_identity_closed_set",
    "evaluate_identity_heldout_retrieval",
    "write_probe_results",
]
