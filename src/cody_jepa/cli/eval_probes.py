#!/usr/bin/env python3
"""Evaluate the paper's linear probes from an exported feature table.

Example:
    uv run python scripts/eval_probes.py \
        --features outputs/features.npz --output-dir outputs/probes
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from cody_jepa.probes import (
    FEATURE_SOURCE,
    evaluate_all_probes,
    read_feature_table,
    write_probe_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--identity-validation-fraction", type=float, default=0.25)
    parser.add_argument("--retrieval-enrollment-sources", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    feature_path = args.features.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    table, feature_metadata = read_feature_table(feature_path)
    feature_source = str(feature_metadata.get("feature_source", FEATURE_SOURCE))
    results = evaluate_all_probes(
        table,
        feature_source=feature_source,
        validation_fraction=args.identity_validation_fraction,
        enrollment_sources=args.retrieval_enrollment_sources,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    paths = write_probe_results(
        results,
        output_dir,
        {
            "feature_table": str(feature_path),
            "feature_source": feature_source,
            "feature_formula": feature_metadata.get("feature_formula"),
            "seed": args.seed,
            "max_iter": args.max_iter,
            "identity_validation_fraction": args.identity_validation_fraction,
            "retrieval_enrollment_sources": args.retrieval_enrollment_sources,
        },
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
