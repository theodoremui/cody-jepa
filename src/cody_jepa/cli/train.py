#!/usr/bin/env python3
"""Train the single-stream JEPA baseline from a readable JSON config.

Example:
    uv run python scripts/train.py --config configs/train/healthgait_baseline.json \
        --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
        --output-dir outputs/training-baseline
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch

from cody_jepa.data import (
    HealthGaitLoaderConfig,
    build_healthgait_datasets_from_config,
    build_healthgait_loaders_from_config,
)
from cody_jepa.single_stream_jepa import MaskGroupConfig, load_checkpoint, train_jepa


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", help="Override required_device from the config")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-windows", type=int, default=3)
    parser.add_argument("--image-verify-mode", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--skip-drop-last", action="store_true")
    return parser.parse_args()


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


def main():
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config, mask_groups = _read_config(config_path)
    if args.device is not None:
        config["required_device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

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
        pin_memory=(config.get("required_device") == "cuda" or torch.cuda.is_available()),
        prefetch_factor=1,
        train_crop_scale=(0.90, 1.0),
        train_horizontal_flip_prob=float(config["train_horizontal_flip_prob"]),
        strict_frame_sequence=True,
        image_verify_mode=args.image_verify_mode,
        allowed_data_root=repo_root / "data" / "healthgait",
        eval_windows=args.eval_windows,
        drop_last_train=not args.skip_drop_last,
    )
    datasets = build_healthgait_datasets_from_config(loader_config)
    train_loader, val_loader = build_healthgait_loaders_from_config(
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
    print(json.dumps({
        "output_dir": str(output_dir),
        "global_step": result["global_step"],
        "completed_epochs": result["completed_epochs"],
        "best_epoch": result["best_epoch"],
        "best_healthy_epoch": result["best_healthy_epoch"],
        "termination_reason": result["termination_reason"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
