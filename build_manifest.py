#!/usr/bin/env python3
"""Build a subject-disjoint Health&Gait manifest by walking the raw frame tree.

Expects <raw-root>/<modality>/<participant>/<speed>/<clothing>_<direction>/*.png,
e.g. silhouette/PA001/UGS/WoJ_1. Writes one CSV row per recording.

    python build_manifest.py --raw-root data/healthgait/raw/Health_Gait --fps 30
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


def assign_splits(rows, val_fraction, seed):
    """Hold out whole participants so train and val never share a subject."""
    subjects = sorted({row["subject_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    val_count = max(1, round(val_fraction * len(subjects)))
    val_subjects = set(subjects[:val_count])
    for row in rows:
        row["split"] = "val" if row["subject_id"] in val_subjects else "train"
    return len(subjects) - val_count, val_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/healthgait/raw/Health_Gait"))
    parser.add_argument("--output", type=Path, default=Path("data/healthgait/manifest.csv"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--modality", default="silhouette")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(args.raw_root, args.modality, args.root, args.clip_length)
    train_subjects, val_subjects = assign_splits(rows, args.val_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} recordings to {args.output} "
        f"({train_subjects} train / {val_subjects} val participants)"
    )


if __name__ == "__main__":
    main()
