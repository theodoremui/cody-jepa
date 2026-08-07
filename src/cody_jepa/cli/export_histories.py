#!/usr/bin/env python3
"""Export compact, deterministic training histories from JEPA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from cody_jepa.single_stream_jepa import load_checkpoint


EXPORT_SCHEMA = 1
IDENTITY_COLUMNS = ("run_id", "phase", "history_index")


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    value = _json_value(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def flatten_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested evaluation dictionaries into stable CSV column names."""

    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = _csv_value(nested_value)
        elif key not in {"val", "train_eval"}:
            flattened[key] = _csv_value(value)
    return flattened


def checkpoint_record(
    checkpoint_path: Path,
    *,
    run_id: str,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    state = load_checkpoint(checkpoint_path)
    history = state.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{checkpoint_path} does not contain a non-empty history")
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError(f"{checkpoint_path} history epochs are not contiguous from one")
    completed_epochs = int(state["completed_epochs"])
    if completed_epochs != len(history):
        raise ValueError(
            f"{checkpoint_path} completed_epochs={completed_epochs} but has "
            f"{len(history)} history rows"
        )

    rows = []
    for index, history_row in enumerate(history):
        rows.append(
            {
                "run_id": run_id,
                "phase": phase,
                "history_index": index,
                **flatten_history_row(history_row),
            }
        )

    metadata = {
        "run_id": run_id,
        "phase": phase,
        "architecture": state.get("architecture"),
        "completed_epochs": completed_epochs,
        "global_step": int(state["global_step"]),
        "history_rows": len(history),
        "evaluation_epochs": [int(row["epoch"]) for row in history if row.get("val") is not None],
        "best_val_loss": _json_value(state.get("best_val_loss")),
        "best_epoch": state.get("best_epoch"),
        "best_healthy_val_loss": _json_value(state.get("best_healthy_val_loss")),
        "best_healthy_epoch": state.get("best_healthy_epoch"),
        "config": _json_value(state.get("config", {})),
        "mask_groups": _json_value(state.get("mask_groups", [])),
    }
    return rows, metadata


def export_histories(
    sources: list[tuple[str, str, Path]],
    csv_path: Path,
    metadata_path: Path,
) -> tuple[Path, Path]:
    """Export ordered checkpoint sources to a CSV history and JSON manifest."""

    all_rows: list[dict[str, Any]] = []
    runs = []
    seen = set()
    for run_id, phase, checkpoint_path in sources:
        if run_id in seen:
            raise ValueError(f"duplicate run_id {run_id!r}")
        seen.add(run_id)
        rows, metadata = checkpoint_record(checkpoint_path, run_id=run_id, phase=phase)
        all_rows.extend(rows)
        runs.append(metadata)

    observed = set().union(*(row.keys() for row in all_rows))
    metric_columns = sorted(observed - set(IDENTITY_COLUMNS))
    columns = [*IDENTITY_COLUMNS, *metric_columns]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    manifest = {
        "schema": EXPORT_SCHEMA,
        "history_csv": str(csv_path.name),
        "columns": columns,
        "row_count": len(all_rows),
        "runs": runs,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, metadata_path


def default_sources(repo_root: Path) -> list[tuple[str, str, Path]]:
    """Return the maintained Phase 0 and Phase 1 checkpoint set."""

    sources = [("phase0-job-91108", "phase0", repo_root / "outputs/jepa-v4/latest.pt")]
    phase1_root = repo_root / "outputs" / "phase1"
    for checkpoint in sorted(phase1_root.glob("*/latest.pt")):
        run_id = checkpoint.parent.name
        if run_id.startswith(("a", "b")):
            sources.append((run_id, "phase1", checkpoint))
    expected = {
        *(f"a{index:02d}" for index in range(8)),
        *(f"b{index:02d}" for index in range(3)),
    }
    observed = {run_id.split("-", 1)[0] for run_id, phase, _ in sources if phase == "phase1"}
    if observed != expected:
        raise FileNotFoundError(
            "expected Phase 1 runs a00-a07 and b00-b02; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--csv", type=Path, default=Path("results/checkpoint_histories.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("results/checkpoint_histories.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    csv_path, metadata_path = export_histories(
        default_sources(repo_root),
        (repo_root / args.csv).resolve(),
        (repo_root / args.metadata).resolve(),
    )
    print(json.dumps([str(csv_path), str(metadata_path)], indent=2))


if __name__ == "__main__":
    main()
