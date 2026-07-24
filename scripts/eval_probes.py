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
    validate_feature_metadata,
    write_probe_results,
)
from cody_jepa.phase0 import (
    guard_research_path,
    load_protocol,
    portable_path,
    require_unchanged_hash,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--identity-validation-fraction", type=float, default=0.25)
    parser.add_argument("--retrieval-enrollment-sequences", type=int, default=1)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--locked-phase0-provenance-label",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _feature_table_provenance(
    args, repo_root, feature_path, feature_metadata, feature_hash, sidecar_hash
):
    if args.locked_phase0_provenance_label is None:
        return portable_path(feature_path, repo_root)
    protocol = load_protocol(repo_root)
    checkpoint_hash = feature_metadata.get("checkpoint_sha256")
    matches = [
        filename
        for filename, contract in protocol["candidate_checkpoints"].items()
        if contract["sha256"] == checkpoint_hash
    ]
    if len(matches) != 1:
        raise ValueError("locked Phase-0 provenance requires one protocol checkpoint")
    filename = matches[0]
    expected_label = (
        f"outputs/phase0/job-{protocol['baseline_job_id']}/"
        f"{Path(filename).stem}/features.npz"
    )
    if args.locked_phase0_provenance_label != expected_label:
        raise ValueError("locked Phase-0 provenance label does not match the protocol")
    artifacts = protocol["candidate_checkpoints"][filename]["artifacts"]
    if feature_hash != artifacts["feature_sha256"]:
        raise ValueError("locked Phase-0 feature hash does not match the protocol")
    if sidecar_hash != artifacts["feature_metadata_sha256"]:
        raise ValueError("locked Phase-0 feature metadata hash does not match the protocol")
    return expected_label


def main():
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    feature_path = guard_research_path(args.features, repo_root, write=False)
    output_dir = guard_research_path(args.output_dir, repo_root, write=True)
    for name in ("probe_metrics.json", "probe_metrics.csv"):
        if (output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite probe artifact: {output_dir / name}")
    feature_sidecar = feature_path.with_suffix(feature_path.suffix + ".metadata.json")
    feature_hash = checkpoint_sha256(feature_path)
    sidecar_hash = checkpoint_sha256(feature_sidecar)
    table, feature_metadata = read_feature_table(feature_path)
    validate_feature_metadata(table, feature_path, feature_metadata)
    feature_source = str(feature_metadata["feature_source"])
    results = evaluate_all_probes(
        table,
        feature_source=feature_source,
        validation_fraction=args.identity_validation_fraction,
        enrollment_sequences=args.retrieval_enrollment_sequences,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    require_unchanged_hash(feature_path, feature_hash, "feature table")
    require_unchanged_hash(feature_sidecar, sidecar_hash, "feature metadata")
    feature_table_provenance = _feature_table_provenance(
        args, repo_root, feature_path, feature_metadata, feature_hash, sidecar_hash
    )
    paths = write_probe_results(
        results,
        output_dir,
        {
            "feature_table": feature_table_provenance,
            "feature_table_sha256": feature_hash,
            "feature_metadata_sha256": sidecar_hash,
            "feature_source": feature_source,
            "feature_formula": feature_metadata["feature_formula"],
            "checkpoint": feature_metadata["checkpoint"],
            "checkpoint_sha256": feature_metadata["checkpoint_sha256"],
            "seed": args.seed,
            "max_iter": args.max_iter,
            "identity_validation_fraction": args.identity_validation_fraction,
            "retrieval_enrollment_sequences": args.retrieval_enrollment_sequences,
        },
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
