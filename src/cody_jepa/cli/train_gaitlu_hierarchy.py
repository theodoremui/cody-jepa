#!/usr/bin/env python3
"""Train one cell from a complete GaitLU hierarchy registry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cody_jepa.cli.train import _read_config, parse_args as parse_train_args, run_training
from cody_jepa.data.gaitlu_hierarchy import read_hierarchy_registry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _resolved_run_index(value):
    if value is not None:
        return int(value)
    try:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "provide --run-index or launch as a Slurm array with SLURM_ARRAY_TASK_ID"
        ) from error


def _validate_exposure(config, declared_exposure):
    effective_batch = int(config["batch_size"]) * int(config["accumulation_steps"])
    exposure = int(config["steps"]) * effective_batch
    if exposure != int(declared_exposure):
        raise ValueError(
            f"config processes {exposure} examples but registry declares "
            f"{int(declared_exposure)}"
        )
    epoch_examples = int(config.get("loader_epoch_examples", 65_536))
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
    return exposure


def run_registered_training(args):
    run_index = _resolved_run_index(args.run_index)
    registry = args.registry.expanduser().resolve()
    rows = read_hierarchy_registry(registry)
    if not 0 <= run_index < len(rows):
        raise IndexError(f"run index {run_index} is outside [0, {len(rows) - 1}]")
    row = rows[run_index]
    config_path = args.config.expanduser().resolve()
    config, _ = _read_config(config_path)
    _validate_exposure(config, row["training_exposure"])

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / row["model_label"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")

    registry_root = registry.parent
    train_argv = [
        "--config",
        str(config_path),
        "--dataset",
        "gaitlu",
        "--train-manifest",
        str(registry_root / row["train_manifest"]),
        "--val-manifest",
        str(registry_root / row["val_manifest"]),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--repo-root",
        str(args.repo_root),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--eval-windows",
        "1",
    ]
    result = run_training(
        parse_train_args(train_argv),
        config_updates={
            "run_id": row["model_label"],
            "seed": row["optimization_seed"],
            "train_window_policy": row["window_policy"],
            "anchor_spacing": row["anchor_spacing"],
            "replicate_seed": row["replicate_seed"],
        },
    )
    if result["global_step"] != int(config["steps"]):
        raise RuntimeError("training returned before the declared final step")
    latest = output_dir / "latest.pt"
    if not latest.is_file():
        raise RuntimeError(f"training reached the final step without writing {latest}")
    return {
        **row,
        "run_index": run_index,
        "checkpoint_path": str(latest),
        "completed_epochs": result["completed_epochs"],
        "examples_per_second": result["examples_per_second"],
    }


def main(argv=None):
    run_registered_training(parse_args(argv))


if __name__ == "__main__":
    main()
