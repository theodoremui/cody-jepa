#!/usr/bin/env python3
"""Train the single-stream JEPA baseline from a readable JSON config.

Example:
    uv run cody-jepa-train --config configs/train/healthgait_baseline.json \
        --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
        --output-dir outputs/training-baseline
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch

from cody_jepa.data import (
    GAITLU_SEED_SCHEME,
    GaitLULoaderConfig,
    HealthGaitLoaderConfig,
    build_gaitlu_datasets_from_config,
    build_gaitlu_loaders_from_config,
    build_healthgait_datasets_from_config,
    build_healthgait_loaders_from_config,
    gaitlu_manifest_pair_sha256,
)
from cody_jepa.masks import MaskGroupConfig
from cody_jepa.training import load_checkpoint, train_jepa


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("healthgait", "gaitlu"), default="healthgait")
    parser.add_argument("--manifest", type=Path, help="Health&Gait manifest")
    parser.add_argument("--train-manifest", type=Path, help="GaitLU training-pool manifest")
    parser.add_argument("--val-manifest", type=Path, help="GaitLU common-holdout manifest")
    parser.add_argument("--data-root", type=Path, help="root of prepared GaitLU shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", help="Override required_device from the config")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-windows", type=int)
    parser.add_argument("--image-verify-mode", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--skip-drop-last", action="store_true")
    return parser.parse_args(argv)


def _read_config(path):
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise TypeError("training config must be a JSON object")
    preset_name = str(config.get("mask_preset", "medium"))
    presets = config.get("mask_presets")
    if not isinstance(presets, dict) or preset_name not in presets:
        raise ValueError(f"mask_preset {preset_name!r} is absent from mask_presets")
    groups = tuple(
        MaskGroupConfig(
            label=str(group["label"]),
            num_blocks=int(group["num_blocks"]),
            spatial_scale=float(group["spatial_scale"]),
            aspect_ratio=tuple(float(value) for value in group.get("aspect_ratio", (0.75, 1.5))),
        )
        for group in presets[preset_name]
    )
    return config, groups


def run_training(args, *, config_updates=None):
    repo_root = args.repo_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config, mask_groups = _read_config(config_path)
    if config_updates:
        config.update(config_updates)
    if args.device is not None:
        config["required_device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    pin_memory = config.get("required_device") == "cuda" or torch.cuda.is_available()
    gaitlu_manifest_digest = None
    if args.dataset == "healthgait":
        if args.manifest is None:
            raise ValueError("--manifest is required for --dataset healthgait")
        if any(value is not None for value in (args.train_manifest, args.val_manifest, args.data_root)):
            raise ValueError("GaitLU manifest/data-root arguments cannot be used with Health&Gait")
        loader_config = HealthGaitLoaderConfig(
            manifest_csv=args.manifest.expanduser(),
            repo_root=repo_root,
            split="train",
            clip_length=int(config["num_frames"]),
            image_size=(int(config["img_size"]), int(config["img_size"])),
            channels=int(config["in_channels"]),
            seed=int(config.get("seed", 0)),
            batch_size=int(config["batch_size"]),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=1,
            train_crop_scale=(0.90, 1.0),
            train_horizontal_flip_prob=float(config["train_horizontal_flip_prob"]),
            strict_frame_sequence=True,
            image_verify_mode=args.image_verify_mode,
            allowed_data_root=repo_root / "data" / "healthgait",
            eval_windows=3 if args.eval_windows is None else args.eval_windows,
            drop_last_train=not args.skip_drop_last,
        )
        datasets = build_healthgait_datasets_from_config(loader_config)
        train_loader, val_loader = build_healthgait_loaders_from_config(
            loader_config, datasets=datasets
        )
    else:
        if args.manifest is not None:
            raise ValueError("--manifest cannot be used with --dataset gaitlu")
        if args.train_manifest is None or args.val_manifest is None or args.data_root is None:
            raise ValueError(
                "--train-manifest, --val-manifest, and --data-root are required for GaitLU"
            )
        if args.skip_drop_last:
            raise ValueError("GaitLU fixed-exposure training always uses drop_last=True")
        if int(config["in_channels"]) != 1:
            raise ValueError("the GaitLU silhouette loader requires in_channels=1")
        loader_config = GaitLULoaderConfig(
            train_manifest_csv=args.train_manifest,
            val_manifest_csv=args.val_manifest,
            data_root=args.data_root,
            clip_length=int(config["num_frames"]),
            image_size=(int(config["img_size"]), int(config["img_size"])),
            seed=int(config.get("seed", 0)),
            batch_size=int(config["batch_size"]),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=1,
            train_crop_scale=tuple(config.get("train_crop_scale", (0.90, 1.0))),
            train_horizontal_flip_prob=float(config["train_horizontal_flip_prob"]),
            eval_windows=1 if args.eval_windows is None else args.eval_windows,
            epoch_examples=int(config.get("loader_epoch_examples", 65_536)),
            train_window_policy=config.get("train_window_policy"),
            anchor_spacing=int(config.get("anchor_spacing", 8)),
            replicate_seed=int(config.get("replicate_seed", 0)),
        )
        gaitlu_manifest_digest = gaitlu_manifest_pair_sha256(
            loader_config.train_manifest_csv, loader_config.val_manifest_csv
        )
        datasets = build_gaitlu_datasets_from_config(loader_config)
        if gaitlu_manifest_pair_sha256(
            loader_config.train_manifest_csv, loader_config.val_manifest_csv
        ) != gaitlu_manifest_digest:
            raise RuntimeError("GaitLU manifests changed while datasets were loading")
        train_loader, val_loader = build_gaitlu_loaders_from_config(
            loader_config, datasets=datasets
        )
    data_contract = {
        "loader_config": {
            key: value
            for key, value in loader_config.as_dict().items()
            if key not in {"manifest_csv", "repo_root", "allowed_data_root"}
        },
        "train_dataset": datasets[0].description(),
        "val_dataset": datasets[1].description(),
    }
    if args.dataset == "gaitlu":
        data_contract.update(
            {
                "dataset": "gaitlu",
                "manifest_sha256": gaitlu_manifest_digest,
                "seed_scheme": GAITLU_SEED_SCHEME,
            }
        )
    resume_state = load_checkpoint(args.resume.expanduser().resolve()) if args.resume else None
    result = train_jepa(
        config,
        train_loader,
        val_loader,
        data_contract,
        checkpoint_dir=output_dir,
        resume_state=resume_state,
        device=config.get("required_device", "auto"),
        mask_groups=mask_groups,
    )
    summary = {
        "output_dir": str(output_dir),
        "global_step": result["global_step"],
        "completed_epochs": result["completed_epochs"],
        "best_epoch": result["best_epoch"],
        "best_healthy_epoch": result["best_healthy_epoch"],
        "termination_reason": result["termination_reason"],
        "elapsed_seconds": result["elapsed_seconds"],
        "examples_per_second": result["examples_per_second"],
    }
    print(json.dumps(summary, indent=2))
    return result


def main(argv=None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
