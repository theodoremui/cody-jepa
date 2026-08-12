#!/usr/bin/env python3
"""Export frozen EMA-target features from a trained checkpoint.

    python export_features.py --checkpoint outputs/run-01/best.pt \
        --manifest data/healthgait/manifest.csv --output outputs/run-01/features.csv
"""

import argparse
from pathlib import Path

from cody_jepa.data import HealthGaitDataset
from cody_jepa.engine import load_checkpoint, resolve_device
from cody_jepa.evaluation import build_target_encoder, export_features
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    encoder = build_target_encoder(checkpoint, device)

    loaders = [
        DataLoader(
            HealthGaitDataset(
                args.manifest,
                split=split,
                root=args.root,
                clip_length=int(config["num_frames"]),
                image_size=int(config["img_size"]),
                windows=args.windows,
            ),
            batch_size=int(config["batch_size"]),
            num_workers=args.num_workers,
        )
        for split in ("train", "val")
    ]
    table = export_features(encoder, loaders, config, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"wrote {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
