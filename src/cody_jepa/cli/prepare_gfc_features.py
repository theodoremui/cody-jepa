#!/usr/bin/env python3
"""Upgrade a legacy deterministic feature export with GFC manifest metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cody_jepa.evaluation.features import METADATA_COLUMNS, write_feature_table


JOIN_COLUMNS = ("subject_id", "gait_system", "trial", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.legacy_features.open("rb") as handle:
        archive = np.load(handle, allow_pickle=False)
        required = {
            "sequence_id",
            "split",
            "subject_id",
            "gait_system",
            "trial",
            "window_start",
            "features",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("legacy feature archive is missing arrays: " + ", ".join(missing))
        features = np.asarray(archive["features"])
        if features.dtype != np.float32:
            raise ValueError("legacy features must use float32 to preserve values exactly")
        legacy = pd.DataFrame(
            {column: archive[column] for column in sorted(required - {"features"})}
        )
    if features.ndim != 2 or len(legacy) != len(features):
        raise ValueError("legacy features must have shape [rows, dimensions]")

    with args.manifest.open(newline="") as handle:
        manifest = pd.DataFrame(list(csv.DictReader(handle)))
    missing_manifest = sorted(
        (set(METADATA_COLUMNS) - {"sequence_id", "window_start"})
        - set(manifest.columns)
    )
    if missing_manifest:
        raise ValueError("GFC manifest is missing columns: " + ", ".join(missing_manifest))
    if manifest.duplicated(list(JOIN_COLUMNS)).any():
        raise ValueError("GFC manifest has duplicate recording join keys")

    merged = legacy.merge(
        manifest,
        on=list(JOIN_COLUMNS),
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_manifest"),
    )
    if not (merged["_merge"] == "both").all():
        bad = merged.loc[merged["_merge"] != "both", list(JOIN_COLUMNS)].head()
        raise ValueError(f"legacy feature rows do not match the GFC manifest:\n{bad}")
    merged = merged.drop(columns="_merge")
    window_counts = merged.groupby("recording_id")["window_start"].agg(
        rows="size", distinct_starts="nunique"
    )
    if window_counts["rows"].ne(3).any() or window_counts["distinct_starts"].ne(3).any():
        raise ValueError(
            "every recording must contain exactly three rows with distinct window starts"
        )

    feature_frame = pd.DataFrame(
        features,
        columns=[f"feature_{index}" for index in range(features.shape[1])],
    )
    table = pd.concat(
        [merged.loc[:, METADATA_COLUMNS].reset_index(drop=True), feature_frame], axis=1
    )
    paths = write_feature_table(
        table,
        args.output,
        {
            "legacy_features": args.legacy_features.name,
            "gfc_manifest": args.manifest.name,
            "upgrade_method": (
                "exact subject/gait_system/trial/split join; feature values unchanged"
            ),
        },
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
