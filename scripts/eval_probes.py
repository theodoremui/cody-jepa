#!/usr/bin/env python3
"""Evaluate linear-recoverability probes from an exported feature table."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from cody_jepa.probes import (
    FEATURE_SOURCE,
    checkpoint_sha256,
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
    parser.add_argument("--retrieval-enrollment-sequences", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    feature_path = args.features.expanduser().resolve()
    table, feature_metadata = read_feature_table(feature_path)
    feature_source = str(feature_metadata.get("feature_source", FEATURE_SOURCE))
    results = evaluate_all_probes(
        table,
        feature_source=feature_source,
        validation_fraction=args.identity_validation_fraction,
        enrollment_sequences=args.retrieval_enrollment_sequences,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    paths = write_probe_results(
        results,
        args.output_dir.expanduser(),
        {
            "feature_table": str(feature_path),
            "feature_table_sha256": checkpoint_sha256(feature_path),
            "feature_source": feature_source,
            "seed": args.seed,
        },
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
