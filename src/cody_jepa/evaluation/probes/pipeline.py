"""Probe orchestration and result serialization."""

from collections.abc import Mapping
from pathlib import Path
import os

import pandas as pd

from ..features import FEATURE_SOURCE, _atomic_path, _write_json_atomic
from .gait import evaluate_gait_system
from .identity import evaluate_identity_closed_set, evaluate_identity_heldout_retrieval


PROBE_SUMMARY_COLUMNS = (
    "task",
    "protocol",
    "feature_source",
    "train_examples",
    "val_examples",
    "num_classes",
    "majority_baseline",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
)


def evaluate_all_probes(
    table,
    feature_source=FEATURE_SOURCE,
    validation_fraction=0.25,
    enrollment_sources=1,
    max_iter=2000,
    seed=0,
):
    return [
        evaluate_identity_closed_set(
            table, feature_source, validation_fraction, max_iter, seed
        ),
        evaluate_identity_heldout_retrieval(
            table, feature_source, enrollment_sources, seed
        ),
        evaluate_gait_system(table, feature_source, max_iter, seed),
    ]


def write_probe_results(results, output_dir, run_metadata=None):
    """Write detailed JSON and a one-row-per-task CSV summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "probe_metrics.json"
    csv_path = output_dir / "probe_metrics.csv"
    payload = {
        **(dict(run_metadata) if run_metadata is not None else {}),
        "results": list(results),
    }
    _write_json_atomic(payload, json_path)
    csv_rows = []
    for result in payload["results"]:
        if not isinstance(result, Mapping):
            raise TypeError("each probe result must be a mapping")
        missing = [column for column in PROBE_SUMMARY_COLUMNS if column not in result]
        if missing:
            raise ValueError(
                "probe result is missing summary columns: " + ", ".join(missing)
            )
        csv_rows.append({column: result[column] for column in PROBE_SUMMARY_COLUMNS})
    temporary = _atomic_path(csv_path)
    try:
        pd.DataFrame(csv_rows, columns=PROBE_SUMMARY_COLUMNS).to_csv(
            temporary, index=False, float_format="%.9g"
        )
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"json": json_path, "csv": csv_path}


__all__ = ["PROBE_SUMMARY_COLUMNS", "evaluate_all_probes", "write_probe_results"]
