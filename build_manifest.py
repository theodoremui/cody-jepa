#!/usr/bin/env python3
"""Build a subject-disjoint Health&Gait manifest by walking the raw frame tree.

Expects <raw-root>/<modality>/<participant>/<speed>/<clothing>_<direction>/*.png,
e.g. silhouette/PA001/UGS/WoJ_1. Writes one CSV row per recording.

    python build_manifest.py --raw-root data/healthgait/raw/Health_Gait
"""

import argparse
import csv
import random
import re
from pathlib import Path

from cody_jepa.data import MANIFEST_COLUMNS, list_frame_paths


TRIAL_PATTERN = re.compile(r"^(WoJ|WJ)_([12])(?:_|$)")
DIRECTIONS = {"1": "R2L", "2": "L2R"}


def build_rows(raw_root, modality, root, clip_length):
    modality_root = Path(raw_root) / modality
    rows = []
    for frame_dir in sorted(modality_root.glob("PA*/**/*")):
        if not frame_dir.is_dir():
            continue
        relative = frame_dir.relative_to(modality_root)
        if len(relative.parts) < 3:
            continue
        subject_id, speed, trial = relative.parts[0], relative.parts[1], relative.parts[-1]
        match = TRIAL_PATTERN.match(trial)
        if speed not in {"UGS", "FGS"} or match is None:
            continue
        frames = list_frame_paths(frame_dir)
        if len(frames) < clip_length:
            continue
        clothing, direction_suffix = match.groups()
        rows.append(
            {
                "subject_id": subject_id,
                "split": "",
                "frame_dir": frame_dir.resolve().relative_to(Path(root).resolve()).as_posix(),
                "num_frames": len(frames),
                "gait_system": speed,
                "speed": speed,
                "clothing": clothing,
                "direction": DIRECTIONS[direction_suffix],
                "recording_id": relative.as_posix(),
                # Both directions of one walk share a source video; probes hold
                # this out so identity cannot be matched on the same recording.
                "source_video_id": Path(*relative.parts[:-1], clothing).as_posix(),
            }
        )
    if not rows:
        raise SystemExit(f"no recordings found under {modality_root}")
    return rows


def assign_splits(rows, val_fraction, seed, tune_fraction=None):
    """Make subject-disjoint train, tuning, and test participant splits."""
    subjects = sorted({row["subject_id"] for row in rows})
    if len(subjects) < 3:
        raise ValueError("three subject-disjoint splits require at least three participants")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if tune_fraction is None:
        tune_fraction = val_fraction / 2.0
    if not 0.0 < tune_fraction < val_fraction:
        raise ValueError("tune_fraction must be positive and smaller than val_fraction")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    heldout_count = max(2, round(val_fraction * len(subjects)))
    if heldout_count >= len(subjects):
        raise ValueError("val_fraction leaves no participants for training")
    tune_count = round(heldout_count * tune_fraction / val_fraction)
    tune_count = min(heldout_count - 1, max(1, tune_count))
    tune_subjects = set(subjects[:tune_count])
    test_subjects = set(subjects[tune_count:heldout_count])
    for row in rows:
        if row["subject_id"] in tune_subjects:
            row["split"] = "tune"
        elif row["subject_id"] in test_subjects:
            row["split"] = "test"
        else:
            row["split"] = "train"
    return len(subjects) - heldout_count, tune_count, heldout_count - tune_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/healthgait/raw/Health_Gait"))
    parser.add_argument("--output", type=Path, default=Path("data/healthgait/manifest.csv"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--modality", default="silhouette")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument(
        "--heldout-fraction",
        "--val-fraction",
        dest="heldout_fraction",
        type=float,
        default=0.2,
        help="total participant fraction reserved for tuning and test",
    )
    parser.add_argument(
        "--tune-fraction",
        type=float,
        help="participant fraction reserved for tuning; defaults to half of heldout-fraction",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(args.raw_root, args.modality, args.root, args.clip_length)
    train_subjects, tune_subjects, test_subjects = assign_splits(
        rows, args.heldout_fraction, args.seed, args.tune_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} recordings to {args.output} "
        f"({train_subjects} train / {tune_subjects} tune / {test_subjects} test participants)"
    )


if __name__ == "__main__":
    main()
