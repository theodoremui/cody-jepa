#!/usr/bin/env python3
"""Run one indexed row of the frozen 5x4 GaitLU scaling study."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from cody_jepa.cli.train import _read_config, parse_args as parse_train_args, run_training
from cody_jepa.data.gaitlu_prepare import TRAINING_REGISTRY_COLUMNS
from cody_jepa.evaluation.gfc.study import CHECKPOINT_METADATA_VERSION


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args(argv)


def _read_registry_row(path: Path, index: int):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRAINING_REGISTRY_COLUMNS:
            raise ValueError(
                "training registry must have exactly these columns in order: "
                + ",".join(TRAINING_REGISTRY_COLUMNS)
            )
        rows = list(reader)
    if len(rows) != 20:
        raise ValueError(f"training registry must contain exactly 20 rows, found {len(rows)}")
    if not 0 <= index < len(rows):
        raise IndexError(f"run index {index} is outside [0, {len(rows) - 1}]")
    if len({row["model_label"] for row in rows}) != 20:
        raise ValueError("training registry model_label values must be unique")
    return rows[index]


def main(argv=None):
    args = parse_args(argv)
    run_index = args.run_index
    if run_index is None:
        try:
            run_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                "provide --run-index or launch as a Slurm array with SLURM_ARRAY_TASK_ID"
            ) from error
    registry = args.registry.expanduser().resolve()
    row = _read_registry_row(registry, run_index)
    config, _ = _read_config(args.config.expanduser().resolve())
    exposure = (
        int(config["steps"])
        * int(config["batch_size"])
        * int(config["accumulation_steps"])
    )
    declared_exposure = int(row["training_exposure"])
    if exposure != declared_exposure:
        raise ValueError(
            f"config processes {exposure} examples but registry declares {declared_exposure}"
        )
    epoch_examples = int(config.get("loader_epoch_examples", 65_536))
    effective_batch = int(config["batch_size"]) * int(config["accumulation_steps"])
    if epoch_examples % effective_batch:
        raise ValueError("loader_epoch_examples must be divisible by effective batch size")
    updates_per_epoch = epoch_examples // effective_batch
    if int(config["steps"]) % updates_per_epoch:
        raise ValueError("steps must be divisible by virtual-epoch updates")
    required_epochs = int(config["steps"]) // updates_per_epoch
    if int(config["num_epochs"]) != required_epochs:
        raise ValueError(
            f"num_epochs must be exactly {required_epochs} for the declared exposure"
        )

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / row["model_label"]
    latest = output_dir / "latest.pt"
    if output_dir.exists() and not args.resume_existing:
        raise FileExistsError(
            f"output exists: {output_dir}; pass --resume-existing only to resume latest.pt"
        )
    if args.resume_existing and not latest.is_file():
        raise FileNotFoundError(f"cannot resume because latest.pt is absent: {latest}")

    checkpoint_id = f"{row['model_label']}-final-step"
    study_metadata = {
        "version": CHECKPOINT_METADATA_VERSION,
        "training_dataset": "GaitLU-1M",
        "checkpoint_kind": "final_step",
        "model_label": row["model_label"],
        "checkpoint_id": checkpoint_id,
        "pool_seed": int(row["pool_seed"]),
        "optimization_seed": int(row["optimization_seed"]),
        "unique_sequences": int(row["unique_sequences"]),
        "training_exposure": declared_exposure,
    }
    train_argv = [
        "--config", str(args.config),
        "--dataset", "gaitlu",
        "--train-manifest", str(data_root / row["train_manifest"]),
        "--val-manifest", str(data_root / row["val_manifest"]),
        "--data-root", str(data_root),
        "--output-dir", str(output_dir),
        "--repo-root", str(args.repo_root),
        "--device", args.device,
        "--num-workers", str(args.num_workers),
        "--eval-windows", "1",
    ]
    if args.resume_existing:
        train_argv.extend(("--resume", str(latest)))
    result = run_training(
        parse_train_args(train_argv),
        config_updates={
            "run_id": row["model_label"],
            "seed": int(row["optimization_seed"]),
            "study_metadata": study_metadata,
        },
    )
    if result["global_step"] != int(config["steps"]):
        raise RuntimeError("training returned before the declared final step")
    run_record = {
        **study_metadata,
        "run_index": run_index,
        "ladder": row["ladder"],
        "rung": row["rung"],
        "checkpoint_path": str(latest),
        "completed_epochs": result["completed_epochs"],
        "examples_per_second": result["examples_per_second"],
    }
    (output_dir / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    return run_record


if __name__ == "__main__":
    main()
