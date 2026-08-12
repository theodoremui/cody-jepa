#!/usr/bin/env python3
"""Export frozen EMA-target features from a trained model.

Example:
    uv run cody-jepa-export-features --checkpoint outputs/run/best.pt \
        --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
        --output outputs/features.npz
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from cody_jepa.data import (
    HealthGaitLoaderConfig,
    build_healthgait_datasets_from_config,
    build_sequential_healthgait_loaders,
)
from cody_jepa.evaluation import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    build_frozen_target_encoder,
    export_frozen_features,
    write_feature_table,
)
from cody_jepa.training import load_checkpoint, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A .csv or .npz path")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--windows-per-sequence", type=int)
    parser.add_argument("--image-verify-mode", choices=("none", "sample", "all"), default="none")
    return parser.parse_args()


def _probe_loader_config(args, checkpoint):
    model_cfg = checkpoint["config"]
    saved = checkpoint.get("data_contract", {}).get("loader_config", {})
    windows = (
        int(saved.get("eval_windows", 3))
        if args.windows_per_sequence is None
        else int(args.windows_per_sequence)
    )
    if windows <= 0:
        raise ValueError("windows-per-sequence must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    manifest = args.manifest.expanduser()
    return HealthGaitLoaderConfig(
        manifest_csv=manifest,
        repo_root=repo_root,
        split="train",
        clip_length=int(model_cfg["num_frames"]),
        image_size=(int(model_cfg["img_size"]), int(model_cfg["img_size"])),
        channels=int(model_cfg["in_channels"]),
        seed=int(saved.get("seed", model_cfg.get("seed", 0))),
        window_policy="center",
        batch_size=(
            int(saved.get("batch_size", model_cfg.get("batch_size", 16)))
            if args.batch_size is None
            else int(args.batch_size)
        ),
        num_workers=(
            int(saved.get("num_workers", 0))
            if args.num_workers is None
            else int(args.num_workers)
        ),
        pin_memory=False,
        prefetch_factor=int(saved.get("prefetch_factor", 2)),
        train_crop_scale=(1.0, 1.0),
        train_horizontal_flip_prob=0.0,
        expected_modality=str(saved.get("expected_modality", "silhouette")),
        strict_frame_sequence=bool(saved.get("strict_frame_sequence", True)),
        image_verify_mode=args.image_verify_mode,
        allowed_data_root=repo_root / "data" / "healthgait",
        eval_windows=windows,
        drop_last_train=False,
    )


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = _probe_loader_config(args, checkpoint)
    datasets = build_healthgait_datasets_from_config(config)
    device = resolve_device(args.device)
    encoder = build_frozen_target_encoder(checkpoint, device)
    table = export_frozen_features(
        encoder,
        build_sequential_healthgait_loaders(
            config, datasets, pin_memory=device.type == "cuda"
        ),
        checkpoint["config"],
        device,
        show_progress=True,
    )
    paths = write_feature_table(
        table,
        output_path,
        {
            "checkpoint": str(checkpoint_path),
            "run_id": checkpoint.get("config", {}).get("run_id"),
            "completed_epochs": checkpoint.get("completed_epochs"),
            "device": str(device),
            "feature_source": FEATURE_SOURCE,
            "feature_formula": FEATURE_FORMULA,
            "encoder": "ema_target_encoder",
            "token_stage": "pre_final_layer_norm",
            "pooling_axis": "token",
            "output_dtype": "float32",
            "clip_length": int(checkpoint["config"]["num_frames"]),
            "image_size": int(checkpoint["config"]["img_size"]),
            "windows_per_sequence": config.eval_windows,
            "window_policy": "deterministic_evenly_spaced_no_augmentation",
            "datasets": {dataset.split: dataset.description() for dataset in datasets},
        },
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
