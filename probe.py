#!/usr/bin/env python3
"""Run linear probes on an exported feature table.

    python probe.py --features outputs/run-01/features.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from cody_jepa.evaluation import run_probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--random-init-features",
        type=Path,
        help="optional same-geometry random-init control features from export_features.py",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_probes(pd.read_csv(args.features), args.seed)
    if args.random_init_features is not None:
        control = run_probes(
            pd.read_csv(args.random_init_features), args.seed, model="random_init"
        )
        results = pd.concat([results, control], ignore_index=True)
    print(results.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
