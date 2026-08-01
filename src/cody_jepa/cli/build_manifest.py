#!/usr/bin/env python3
"""Build a subject-disjoint Health&Gait manifest with GFC metadata.

The shortcut measurements are computed from every decoded silhouette frame in
each recording. Raw participant data remains under ``data/`` and is ignored by
Git; the generated manifest stores only repository-relative paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from cody_jepa.data import (
    GFC_FACTOR_COLUMNS,
    RECORDING_HIERARCHY_COLUMNS,
    SHORTCUT_FEATURE_COLUMNS,
    summarize_healthgait_manifest,
    write_healthgait_metadata_summary,
)
from cody_jepa.data.frames import contiguous_window_starts, list_frame_paths


PROJECT_ROOT = Path.cwd()
GFC_WINDOWS_PER_RECORDING = 3
TRIAL_PATTERN = re.compile(r"^(WoJ|WJ)_([12])(?:_|$)")
DIRECTION_BY_SUFFIX = {"1": "R2L", "2": "L2R"}
MANIFEST_COLUMNS = (
    "subject_id",
    "modality",
    "gait_system",
    "trial",
    *GFC_FACTOR_COLUMNS,
    *RECORDING_HIERARCHY_COLUMNS,
    "frame_dir",
    "num_frames",
    "fps",
    *SHORTCUT_FEATURE_COLUMNS,
    "split",
)


def compute_shortcut_features(
    frame_paths: list[Path], *, fps: float, foreground_threshold: float
) -> dict[str, float]:
    """Compute the declared non-learned shortcut vector from full silhouettes."""

    if not frame_paths:
        raise ValueError("shortcut features require at least one silhouette frame")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold must lie in [0, 1]")

    centroids: list[float | None] = []
    foreground_areas: list[float] = []
    expected_shape: tuple[int, int] | None = None
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        if pixels.ndim != 2 or pixels.size == 0:
            raise ValueError(f"invalid silhouette frame: {frame_path}")
        if expected_shape is None:
            expected_shape = pixels.shape
        elif pixels.shape != expected_shape:
            raise ValueError(
                f"inconsistent silhouette dimensions: {frame_path} has "
                f"{pixels.shape}, expected {expected_shape}"
            )
        foreground = pixels > foreground_threshold
        _, columns = np.nonzero(foreground)
        width = pixels.shape[1]
        centroid = (
            (
                float(np.mean(columns, dtype=np.float64)) / float(width - 1)
                if width > 1
                else 0.0
            )
            if columns.size
            else None
        )
        centroids.append(centroid)
        foreground_areas.append(float(np.mean(foreground, dtype=np.float64)))

    areas = np.asarray(foreground_areas, dtype=np.float64)
    q25, median, q75 = np.quantile(areas, (0.25, 0.5, 0.75), method="linear")
    valid_centroids = [value for value in centroids if value is not None]
    if not valid_centroids:
        raise ValueError("silhouette recording contains no foreground pixels")
    # Segmentation occasionally emits an empty boundary frame. Preserve that
    # frame in the area and duration controls, but measure displacement across
    # the first and last frames for which a silhouette is actually observed.
    signed_drift = valid_centroids[-1] - valid_centroids[0]
    frame_count = len(frame_paths)
    return {
        "shortcut_log_frame_count": math.log(frame_count),
        "shortcut_duration_seconds": frame_count / fps,
        "shortcut_horizontal_centroid_drift_signed": signed_drift,
        "shortcut_horizontal_centroid_drift_absolute": abs(signed_drift),
        "shortcut_foreground_area_mean": float(np.mean(areas, dtype=np.float64)),
        "shortcut_foreground_area_std": float(np.std(areas, ddof=0, dtype=np.float64)),
        "shortcut_foreground_area_q25": float(q25),
        "shortcut_foreground_area_median": float(median),
        "shortcut_foreground_area_q75": float(q75),
    }


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Health&Gait inputs must remain below the repository root: {path}"
        ) from error


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    raw_root = args.raw_root.expanduser().resolve()
    modality_root = raw_root / args.modality
    rows: list[dict[str, object]] = []
    for trial_dir in sorted(modality_root.glob("PA*/**/*")):
        if not trial_dir.is_dir():
            continue
        frames = list_frame_paths(trial_dir)
        if len(contiguous_window_starts(frames, args.clip_length)) < (
            GFC_WINDOWS_PER_RECORDING
        ):
            continue
        relative = trial_dir.relative_to(modality_root)
        if len(relative.parts) < 3:
            continue
        subject_id, speed, trial = relative.parts[0], relative.parts[1], relative.parts[-1]
        if speed not in {"UGS", "FGS"}:
            continue
        match = TRIAL_PATTERN.match(trial)
        if match is None:
            continue
        clothing, direction_suffix = match.groups()
        recording_id = relative.as_posix()
        source_video_id = Path(*relative.parts[:-1], clothing).as_posix()
        rows.append(
            {
                "subject_id": subject_id,
                "modality": args.modality,
                "gait_system": speed,
                "trial": trial,
                "recording_id": recording_id,
                "source_video_id": source_video_id,
                "direction_clip_id": recording_id,
                "speed": speed,
                "clothing": clothing,
                "direction": DIRECTION_BY_SUFFIX[direction_suffix],
                "frame_dir": _portable_path(trial_dir, args.repo_root),
                "num_frames": len(frames),
                "fps": args.fps,
                **compute_shortcut_features(
                    frames,
                    fps=args.fps,
                    foreground_threshold=args.foreground_threshold,
                ),
            }
        )

    subjects = sorted({str(row["subject_id"]) for row in rows})
    if len(subjects) < 2:
        raise ValueError("manifest construction requires at least two participants")
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    validation_count = min(
        len(subjects) - 1,
        max(1, round(args.val_fraction * len(subjects))),
    )
    validation_subjects = set(subjects[:validation_count])
    for row in rows:
        row["split"] = "val" if row["subject_id"] in validation_subjects else "train"
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/healthgait/raw/Health_Gait"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/healthgait/manifests/silhouette_subject_split_seed0.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--modality", default="silhouette")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.clip_length <= 0:
        raise ValueError("clip-length must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must lie strictly between zero and one")
    rows = build_rows(args)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = args.repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    train_subjects = {row["subject_id"] for row in rows if row["split"] == "train"}
    validation_subjects = {row["subject_id"] for row in rows if row["split"] == "val"}
    summary = summarize_healthgait_manifest(
        output,
        repo_root=args.repo_root,
        clip_length=args.clip_length,
    )
    summary_paths = write_healthgait_metadata_summary(
        summary,
        args.repo_root / "data" / "healthgait" / "diagnostics",
        "healthgait_manifest_summary",
    )
    print(
        json.dumps(
            {
                "manifest": str(output),
                "recordings": len(rows),
                "train_subjects": len(train_subjects),
                "validation_subjects": len(validation_subjects),
                "subject_overlap": sorted(train_subjects & validation_subjects),
                "summary": {key: str(value) for key, value in summary_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
