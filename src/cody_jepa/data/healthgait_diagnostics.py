from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import sys

import numpy as np
from PIL import Image

from .dataset import REQUIRED_MANIFEST_COLUMNS


IMAGE_SUFFIXES = frozenset({".jpg", ".png"})
MOTION_ARTIFACT_STEM = "healthgait_motion_diagnostics"


def _pyplot():
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _resolve_frame_dir(frame_dir, repo_root):
    path = Path(frame_dir)
    return path if path.is_absolute() else Path(repo_root) / path


def _list_image_paths(frame_dir):
    if not frame_dir.is_dir():
        return []
    return sorted(
        (path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.name,
    )


def summarize_healthgait_manifest(
    manifest_csv,
    repo_root,
    clip_length,
    check_corrupt_frames=True,
):
    """Summarize manifest metadata and filesystem-level frame health."""
    manifest_csv = Path(manifest_csv)
    split_counts = Counter()
    split_subjects = defaultdict(set)
    gait_system_counts = Counter()
    trial_counts = Counter()
    declared_frame_counts = []
    dropped_short_clips = 0
    missing_frame_count = 0
    corrupt_frame_count = 0
    row_count = 0

    with manifest_csv.open(newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"Invalid Health&Gait manifest {manifest_csv}: missing required columns: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            row_count += 1
            split = row["split"].strip()
            subject_id = row["subject_id"].strip()
            split_counts[split] += 1
            if subject_id:
                split_subjects[split].add(subject_id)
            gait_system_counts[row["gait_system"].strip()] += 1
            trial_counts[row["trial"].strip()] += 1

            try:
                declared_num_frames = int(row["num_frames"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid num_frames value {row['num_frames']!r} in {manifest_csv}"
                ) from error

            declared_frame_counts.append(declared_num_frames)
            frame_dir = _resolve_frame_dir(row["frame_dir"].strip(), repo_root)
            frame_paths = _list_image_paths(frame_dir)
            actual_num_frames = len(frame_paths)

            if actual_num_frames < int(clip_length):
                dropped_short_clips += 1
            missing_frame_count += max(0, declared_num_frames - actual_num_frames)

            if check_corrupt_frames:
                for frame_path in frame_paths:
                    try:
                        with Image.open(frame_path) as image:
                            image.verify()
                    except (OSError, SyntaxError, ValueError):
                        corrupt_frame_count += 1

    all_splits = sorted(split_counts)
    overlap = set()
    for split_index, split in enumerate(all_splits):
        for other_split in all_splits[split_index + 1 :]:
            overlap.update(split_subjects[split] & split_subjects[other_split])

    if declared_frame_counts:
        frame_count = {
            "min": min(declared_frame_counts),
            "mean": sum(declared_frame_counts) / len(declared_frame_counts),
            "max": max(declared_frame_counts),
        }
    else:
        frame_count = {"min": None, "mean": None, "max": None}

    return {
        "row_count": row_count,
        "split_counts": dict(sorted(split_counts.items())),
        "subject_count_by_split": {
            split: len(subjects) for split, subjects in sorted(split_subjects.items())
        },
        "subject_overlap": sorted(overlap),
        "gait_system_counts": dict(sorted(gait_system_counts.items())),
        "trial_counts": dict(sorted(trial_counts.items())),
        "frame_count": frame_count,
        "dropped_short_clips": dropped_short_clips,
        "missing_frame_count": missing_frame_count,
        "corrupt_frame_count": corrupt_frame_count,
    }


def _flatten_summary(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_summary(value[key], child_prefix)
    else:
        yield {"metric": prefix, "value": json.dumps(value, sort_keys=True)}


def write_healthgait_metadata_summary(summary, output_dir, stem):
    """Write a deterministic JSON summary and flattened CSV companion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(_flatten_summary(summary))

    return {"json": json_path, "csv": csv_path}


def _dataset_map(datasets):
    if isinstance(datasets, dict):
        return dict(datasets)
    return {dataset.split: dataset for dataset in datasets}


def _sample_indices(split, dataset_length, sample_count, seed, epoch):
    key = f"{seed}\0{epoch}\0{split}".encode("utf-8")
    split_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    rng = random.Random(split_seed)
    return rng.sample(range(dataset_length), min(sample_count, dataset_length))


def _to_numpy_video(video):
    if hasattr(video, "detach"):
        video = video.detach().cpu().numpy()
    video = np.asarray(video, dtype=np.float32)
    if video.ndim != 4:
        raise ValueError(f"Expected video shaped [T, C, H, W], got {video.shape}")
    return video.mean(axis=1)


def _diagnostic_record(sample):
    video = _to_numpy_video(sample["video"])
    temporal_diff = np.abs(np.diff(video, axis=0))
    diff_map = temporal_diff.mean(axis=0) if len(temporal_diff) else np.zeros_like(video[0])
    frame_max = video.reshape(video.shape[0], -1).max(axis=1)
    record = {
        "sequence_id": str(sample["sequence_id"]),
        "split": str(sample["split"]),
        "modality": str(sample["modality"]),
        "subject_id": str(sample["subject_id"]),
        "gait_system": str(sample["gait_system"]),
        "trial": str(sample["trial"]),
        "window_start": int(sample["window_start"]),
        "frame_indices": [int(index) for index in sample["frame_indices"]],
        "motion_energy": float(temporal_diff.mean()) if temporal_diff.size else 0.0,
        "blank_frame_count": int(np.count_nonzero(frame_max <= 1e-6)),
        "mean_intensity": float(video.mean()),
        "std_intensity": float(video.std()),
        "min_intensity": float(video.min()),
        "max_intensity": float(video.max()),
    }
    return record, video, diff_map


def _save_contact_sheet(records, videos, output_path):
    plt = _pyplot()
    columns = 3
    rows = max(1, len(records))
    figure, axes = plt.subplots(rows, columns, figsize=(9, 2.7 * rows), squeeze=False)
    for row_index in range(rows):
        for column_index in range(columns):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if row_index >= len(records):
                continue
            video = videos[row_index]
            frame_index = [0, video.shape[0] // 2, video.shape[0] - 1][column_index]
            axis.imshow(video[frame_index], cmap="gray", vmin=0, vmax=1)
            axis.set_title(
                f"{records[row_index]['split']} | {records[row_index]['subject_id']}\n"
                f"frame {records[row_index]['frame_indices'][frame_index]}",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_diff_maps(records, diff_maps, output_path):
    plt = _pyplot()
    columns = min(4, max(1, len(records)))
    rows = max(1, math.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 2.8 * rows), squeeze=False)
    vmax = max((float(diff_map.max()) for diff_map in diff_maps), default=1.0) or 1.0
    for plot_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if plot_index >= len(records):
            continue
        axis.imshow(diff_maps[plot_index], cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(
            f"{records[plot_index]['split']} | E={records[plot_index]['motion_energy']:.4f}",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_motion_histogram(records, output_path):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4))
    for split in sorted({record["split"] for record in records}):
        energies = [record["motion_energy"] for record in records if record["split"] == split]
        axis.hist(energies, bins=min(10, max(1, len(energies))), alpha=0.6, label=split)
    axis.set_xlabel("Mean absolute frame difference")
    axis.set_ylabel("Sample count")
    axis.set_title("Health&Gait motion energy")
    if records:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_healthgait_motion_diagnostics(
    datasets,
    output_dir,
    samples_per_split=8,
    seed=0,
    epoch=0,
):
    """Sample clips deterministically and write visual and tabular motion diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_by_split = _dataset_map(datasets)
    records = []
    videos = []
    diff_maps = []

    for split, dataset in sorted(dataset_by_split.items()):
        previous_epoch = getattr(dataset, "epoch", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        try:
            for index in _sample_indices(split, len(dataset), samples_per_split, seed, epoch):
                record, video, diff_map = _diagnostic_record(dataset[index])
                records.append(record)
                videos.append(video)
                diff_maps.append(diff_map)
        finally:
            if previous_epoch is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(previous_epoch)

    contact_sheet_path = output_dir / "healthgait_motion_contact_sheet.png"
    diff_maps_path = output_dir / "healthgait_motion_diff_maps.png"
    histogram_path = output_dir / "healthgait_motion_energy_histogram.png"
    csv_path = output_dir / f"{MOTION_ARTIFACT_STEM}.csv"
    json_path = output_dir / f"{MOTION_ARTIFACT_STEM}.json"

    _save_contact_sheet(records, videos, contact_sheet_path)
    _save_diff_maps(records, diff_maps, diff_maps_path)
    _save_motion_histogram(records, histogram_path)

    csv_fieldnames = [
        "sequence_id",
        "split",
        "modality",
        "subject_id",
        "gait_system",
        "trial",
        "window_start",
        "frame_indices",
        "motion_energy",
        "blank_frame_count",
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "frame_indices": json.dumps(record["frame_indices"])})

    ranked_records = sorted(
        records,
        key=lambda record: (record["motion_energy"], record["sequence_id"]),
    )
    example_count = min(3, len(ranked_records))
    compact_keys = ("sequence_id", "split", "subject_id", "trial", "motion_energy")

    def compact(record):
        return {key: record[key] for key in compact_keys}

    summary = {
        "sample_count": len(records),
        "samples_per_split": {
            split: sum(record["split"] == split for record in records)
            for split in sorted(dataset_by_split)
        },
        "low_motion_examples": [compact(record) for record in ranked_records[:example_count]],
        "high_motion_examples": [
            compact(record) for record in reversed(ranked_records[-example_count:])
        ],
    }
    with json_path.open("w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")

    return {
        **summary,
        "artifacts": {
            "contact_sheet": contact_sheet_path,
            "diff_maps": diff_maps_path,
            "histogram": histogram_path,
            "csv": csv_path,
            "json": json_path,
        },
    }


__all__ = [
    "run_healthgait_motion_diagnostics",
    "summarize_healthgait_manifest",
    "write_healthgait_metadata_summary",
]
