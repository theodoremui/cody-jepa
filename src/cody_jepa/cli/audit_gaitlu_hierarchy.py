#!/usr/bin/env python3
"""Audit whether GaitLU can support the hierarchical-diversity intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cody_jepa.data.gaitlu_hierarchy_audit import audit_hierarchical_support


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="prepared GaitLU root containing inventory.csv",
    )
    parser.add_argument("--exposure", type=int, default=4_096_000)
    parser.add_argument("--holdout-size", type=int, default=10_000)
    parser.add_argument("--holdout-seed", type=int, default=20_260_806)
    parser.add_argument("--pool-seeds", type=int, nargs=8, default=tuple(range(8)))
    parser.add_argument("--low-target", type=int, default=2_500)
    parser.add_argument("--high-target", type=int, default=250_000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = audit_hierarchical_support(
        args.prepared_root / "inventory.csv",
        draws=args.exposure,
        holdout_size=args.holdout_size,
        holdout_seed=args.holdout_seed,
        pool_seeds=args.pool_seeds,
        low_target=args.low_target,
        high_target=args.high_target,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
