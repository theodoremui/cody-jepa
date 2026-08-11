#!/usr/bin/env python3
"""Create nested GaitLU manifests for the hierarchical-diversity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cody_jepa.data.gaitlu_hierarchy import finalize_gaitlu_hierarchy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="prepared GaitLU root containing finalized inventory.csv",
    )
    parser.add_argument("--training-exposure", type=int, required=True)
    parser.add_argument("--holdout-target", type=int, default=10_000)
    parser.add_argument("--holdout-seed", type=int, default=20_260_806)
    parser.add_argument("--pool-seeds", type=int, nargs=8, default=tuple(range(8)))
    parser.add_argument("--low-target", type=int, default=2_500)
    parser.add_argument("--high-target", type=int, default=250_000)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--anchor-spacing", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = finalize_gaitlu_hierarchy(
        args.prepared_root,
        training_exposure=args.training_exposure,
        holdout_target=args.holdout_target,
        holdout_seed=args.holdout_seed,
        pool_seeds=args.pool_seeds,
        low_target=args.low_target,
        high_target=args.high_target,
        clip_length=args.clip_length,
        anchor_spacing=args.anchor_spacing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
