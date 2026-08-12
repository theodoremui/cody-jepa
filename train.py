#!/usr/bin/env python3
"""Train a single-stream masked JEPA.

    python train.py --config configs/healthgait.json \
        --manifest data/healthgait/manifest.csv --output-dir outputs/run-01
"""

import argparse
import json
from pathlib import Path

from cody_jepa.data import build_loaders
from cody_jepa.engine import load_checkpoint, resolve_device, train_jepa
from cody_jepa.masks import DEFAULT_MASK_GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    mask_groups = config.get("masks", DEFAULT_MASK_GROUPS)
    train_loader, tune_loader = build_loaders(
        config, args.manifest, args.root, args.num_workers
    )
    print(
        f"train: {len(train_loader.dataset)} clips | "
        f"tune: {len(tune_loader.dataset)} clips"
    )

    history = train_jepa(
        config,
        train_loader,
        tune_loader,
        output_dir=args.output_dir,
        device=resolve_device(args.device),
        mask_groups=mask_groups,
        resume=load_checkpoint(args.resume) if args.resume else None,
    )
    if args.output_dir is not None:
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
