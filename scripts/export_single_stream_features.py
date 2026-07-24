#!/usr/bin/env python3
"""Export deterministic frozen EMA-target features from a training checkpoint."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch
from torch.utils.data import DataLoader

from cody_jepa.phase0 import (
    checkpoint_record,
    guard_research_path,
    load_protocol,
    portable_path,
    require_unchanged_hash,
)
from cody_jepa.data import (
    HealthGaitLoaderConfig,
    build_healthgait_datasets_from_config,
    healthgait_manifest_path,
)
from cody_jepa.probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    build_frozen_target_encoder,
    checkpoint_sha256,
    export_frozen_features,
    write_feature_table,
)
from cody_jepa.single_stream_jepa import load_checkpoint, resolve_device
from cody_jepa.single_stream_jepa import CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locked-phase0-legacy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True, help="A .csv or .npz path")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--windows-per-sequence",
        type=int,
        help="Default: checkpoint loader's eval_windows (usually 3)",
    )
    parser.add_argument(
        "--image-verify-mode", choices=("none", "sample", "all"), default="none"
    )
    return parser.parse_args()


def _probe_loader_config(args, checkpoint):
    model_cfg = checkpoint["config"]
    saved = checkpoint.get("data_contract", {}).get("loader_config", {})
    repo_root = args.repo_root.expanduser().resolve()
    manifest = (
        healthgait_manifest_path(repo_root)
        if args.manifest is None
        else args.manifest.expanduser()
    )
    windows = (
        int(saved.get("eval_windows", 3))
        if args.windows_per_sequence is None
        else args.windows_per_sequence
    )
    if windows <= 0:
        raise ValueError("windows-per-sequence must be positive")
    return HealthGaitLoaderConfig(
        manifest_csv=manifest,
        repo_root=repo_root,
        split="train",
        clip_length=int(model_cfg["num_frames"]),
        image_size=(int(model_cfg["img_size"]), int(model_cfg["img_size"])),
        channels=int(model_cfg["in_channels"]),
        seed=int(saved.get("seed", model_cfg.get("seed", 0))),
        # Frozen probes use deterministic, unaugmented temporal windows in both
        # splits. Training-time random crops/flips would make the table unstable.
        window_policy="center",
        batch_size=(
            int(saved.get("batch_size", model_cfg.get("batch_size", 16)))
            if args.batch_size is None
            else args.batch_size
        ),
        num_workers=(
            int(saved.get("num_workers", 0))
            if args.num_workers is None
            else args.num_workers
        ),
        pin_memory=False,
        prefetch_factor=int(saved.get("prefetch_factor", 2)),
        train_crop_scale=(1.0, 1.0),
        train_horizontal_flip_prob=0.0,
        expected_modality=str(saved.get("expected_modality", "silhouette")),
        strict_frame_sequence=bool(saved.get("strict_frame_sequence", True)),
        image_verify_mode=args.image_verify_mode,
        inventory_hash_mode=str(saved.get("inventory_hash_mode", "sample")),
        allowed_data_root=repo_root / "data" / "healthgait",
        eval_windows=windows,
        drop_last_train=False,
    )


def _validate_checkpoint_data(checkpoint, datasets):
    saved_contract = checkpoint.get("data_contract")
    if not isinstance(saved_contract, dict):
        raise ValueError("checkpoint has no data_contract to validate feature provenance")
    for dataset in datasets:
        saved = saved_contract.get(f"{dataset.split}_dataset")
        if not isinstance(saved, dict):
            raise ValueError(f"checkpoint has no {dataset.split}_dataset signature")
        current = dataset.signature()
        for key in ("manifest_sha256", "inventory_sha256", "sequence_count"):
            if saved.get(key) != current.get(key):
                raise ValueError(
                    f"{dataset.split} dataset {key} differs from checkpoint: "
                    f"checkpoint={saved.get(key)!r}, current={current.get(key)!r}"
                )


def _sequential_loaders(config, datasets, device):
    worker_options = (
        {"prefetch_factor": config.prefetch_factor} if config.num_workers > 0 else {}
    )
    return {
        dataset.split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            **worker_options,
        )
        for dataset in datasets
    }


def main():
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_path = guard_research_path(checkpoint_path, repo_root, write=False)
    output_path = guard_research_path(args.output, repo_root, write=True)
    if output_path.exists() or output_path.with_suffix(
        output_path.suffix + ".metadata.json"
    ).exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {output_path}")
    checkpoint_hash = checkpoint_sha256(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if args.locked_phase0_legacy:
        protocol = load_protocol(repo_root)
        checkpoint_record(checkpoint_path, protocol, checkpoint_path.name)
    config = _probe_loader_config(args, checkpoint)
    datasets = build_healthgait_datasets_from_config(config)
    _validate_checkpoint_data(checkpoint, datasets)

    device = resolve_device(args.device)
    expected_schema = (
        LEGACY_CHECKPOINT_SCHEMA if args.locked_phase0_legacy else CHECKPOINT_SCHEMA
    )
    encoder = build_frozen_target_encoder(
        checkpoint, device, expected_schema=expected_schema
    )
    table = export_frozen_features(
        encoder,
        _sequential_loaders(config, datasets, device),
        checkpoint["config"],
        device,
        show_progress=True,
    )
    require_unchanged_hash(checkpoint_path, checkpoint_hash, "checkpoint")
    paths = write_feature_table(
        table,
        output_path,
        {
            "checkpoint": portable_path(checkpoint_path, repo_root),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_schema": checkpoint["schema"],
            "checkpoint_architecture": checkpoint["architecture"],
            "checkpoint_completed_epochs": checkpoint.get("completed_epochs"),
            "device": str(device),
            "inference_mode": True,
            "encoder_eval": True,
            "encoder_frozen": True,
            "feature_source": FEATURE_SOURCE,
            "feature_formula": FEATURE_FORMULA,
            "preprocessing": {
                "channels": int(checkpoint["config"]["in_channels"]),
                "clip_length": int(checkpoint["config"]["num_frames"]),
                "image_size": [
                    int(checkpoint["config"]["img_size"]),
                    int(checkpoint["config"]["img_size"]),
                ],
                "resize_interpolation": "bilinear",
                "decoded_range": [0.0, 1.0],
                "input_mean": float(checkpoint["config"]["input_mean"]),
                "input_std": float(checkpoint["config"]["input_std"]),
                "encoder": "ema_target_encoder",
                "token_stage": "pre_final_layer_norm",
                "pooling_axis": "token",
                "output_dtype": "float32",
            },
            "windows_per_sequence": config.eval_windows,
            "window_policy": "deterministic_evenly_spaced_no_augmentation",
            "dataset_signatures": {
                dataset.split: dataset.signature() for dataset in datasets
            },
        },
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
