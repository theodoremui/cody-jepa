#!/usr/bin/env python3
"""Prepare trusted GaitLU-1M pickle shards for the fixed-exposure scaling study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cody_jepa.data.gaitlu_prepare import finalize_gaitlu_study, pack_gaitlu_shard


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser(
        "pack-shard", help="stream one .tar.gz and write one seekable bit-packed .tar"
    )
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--prepared-root", type=Path, required=True)
    pack.add_argument(
        "--trust-pickles",
        action="store_true",
        help="confirm that the input is the official trusted release",
    )
    pack.add_argument("--min-frames", type=int, default=16)
    pack.add_argument("--max-empty-frame-fraction", type=float, default=0.20)
    pack.add_argument("--min-foreground-fraction", type=float, default=1e-4)
    pack.add_argument("--max-intermediate-fraction", type=float, default=0.05)
    pack.add_argument("--max-pickle-mib", type=int, default=256)

    finalize = subparsers.add_parser(
        "finalize", help="deduplicate and create holdout, five ladders, and run registry"
    )
    finalize.add_argument("--prepared-root", type=Path, required=True)
    finalize.add_argument("--holdout-size", type=int, default=10_000)
    finalize.add_argument("--holdout-seed", type=int, default=20_260_806)
    finalize.add_argument("--pool-seeds", type=int, nargs=5, default=(0, 1, 2, 3, 4))
    finalize.add_argument("--pool-sizes", type=int, nargs=3, default=(2_500, 25_000, 250_000))
    finalize.add_argument("--training-exposure", type=int, default=8_192_000)
    finalize.add_argument("--expected-shards", type=int, default=100)
    finalize.add_argument(
        "--source-groups",
        type=Path,
        help="optional exact sequence_id,source_group CSV supplied by the distributor",
    )
    return parser.parse_args(argv)


def run_preparation(args):
    if args.command == "pack-shard":
        return pack_gaitlu_shard(
            args.input,
            args.prepared_root,
            trust_pickles=args.trust_pickles,
            min_frames=args.min_frames,
            max_empty_frame_fraction=args.max_empty_frame_fraction,
            min_foreground_fraction=args.min_foreground_fraction,
            max_intermediate_fraction=args.max_intermediate_fraction,
            max_pickle_bytes=args.max_pickle_mib * 1024 * 1024,
        )
    return finalize_gaitlu_study(
        args.prepared_root,
        holdout_size=args.holdout_size,
        holdout_seed=args.holdout_seed,
        pool_seeds=args.pool_seeds,
        pool_sizes=args.pool_sizes,
        training_exposure=args.training_exposure,
        source_groups_csv=args.source_groups,
        expected_shards=args.expected_shards,
    )


def main(argv=None) -> None:
    result = run_preparation(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
