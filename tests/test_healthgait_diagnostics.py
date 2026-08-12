import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cody_jepa.data import (
    FACTOR_COLUMNS,
    RECORDING_HIERARCHY_COLUMNS,
    SHORTCUT_FEATURE_COLUMNS,
    run_healthgait_motion_diagnostics,
    summarize_healthgait_manifest,
    write_healthgait_metadata_summary,
)


MANIFEST_FIELDNAMES = [
    "subject_id",
    "modality",
    "gait_system",
    "trial",
    "frame_dir",
    "num_frames",
    "split",
    "fps",
    *FACTOR_COLUMNS,
    *RECORDING_HIERARCHY_COLUMNS,
    *SHORTCUT_FEATURE_COLUMNS,
]


def scientific_fields(recording_id, frame_count, *, speed="FGS", direction="R2L"):
    return {
        "recording_id": recording_id,
        "source_video_id": str(Path(recording_id).parent / "WoJ"),
        "direction_clip_id": recording_id,
        "speed": speed,
        "clothing": "WoJ",
        "direction": direction,
        "fps": "30",
        "shortcut_log_frame_count": str(math.log(frame_count)),
        "shortcut_duration_seconds": str(frame_count / 30),
        "shortcut_horizontal_centroid_drift_signed": "0",
        "shortcut_horizontal_centroid_drift_absolute": "0",
        "shortcut_foreground_area_mean": "0.25",
        "shortcut_foreground_area_std": "0",
        "shortcut_foreground_area_q25": "0.25",
        "shortcut_foreground_area_median": "0.25",
        "shortcut_foreground_area_q75": "0.25",
    }


class SyntheticMotionDataset:
    def __init__(self, split, samples):
        self.split = split
        self.samples = samples
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, name = self.samples[index]
        return {
            "video": video,
            "sequence_id": f"{self.split}-{name}",
            "split": self.split,
            "modality": "silhouette",
            "subject_id": f"subject-{name}",
            "gait_system": "FGS",
            "trial": name,
            "window_start": self.epoch,
            "frame_indices": list(range(1, video.shape[0] + 1)),
        }


class HealthGaitMetadataSummaryTest(unittest.TestCase):
    def _write_images(self, root, relative_dir, values):
        frame_dir = root / relative_dir
        frame_dir.mkdir(parents=True)
        for index, value in enumerate(values, start=1):
            Image.new("L", (8, 8), color=value).save(frame_dir / f"{index:03d}.png")
        return frame_dir

    def test_summary_counts_frame_health_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write_images(root, "frames/train/p1", [0, 32, 64, 96])
            self._write_images(root, "frames/val/p1", [0, 32, 64, 96, 128])
            short_dir = self._write_images(root, "frames/train/p2", [0, 32])
            (short_dir / "003.png").write_bytes(b"not a valid image")

            rows = [
                {
                    "subject_id": "P1",
                    "modality": "silhouette",
                    "gait_system": "FGS",
                    "trial": "T0",
                    "frame_dir": "frames/train/p1",
                    "num_frames": "4",
                    "split": "train",
                    **scientific_fields("frames/train/p1", 4),
                },
                {
                    "subject_id": "P1",
                    "modality": "silhouette",
                    "gait_system": "FGS",
                    "trial": "T0",
                    "frame_dir": "frames/val/p1",
                    "num_frames": "5",
                    "split": "val",
                    **scientific_fields("frames/val/p1", 5),
                },
                {
                    "subject_id": "P2",
                    "modality": "silhouette",
                    "gait_system": "VGS",
                    "trial": "T1",
                    "frame_dir": "frames/train/p2",
                    "num_frames": "6",
                    "split": "train",
                    **scientific_fields("frames/train/p2", 6, speed="UGS", direction="L2R"),
                },
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as manifest_file:
                writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            summary = summarize_healthgait_manifest(manifest, root, clip_length=4)

            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(summary["split_counts"], {"train": 2, "val": 1})
            self.assertEqual(summary["subject_count_by_split"], {"train": 2, "val": 1})
            self.assertEqual(summary["subject_overlap"], ["P1"])
            self.assertEqual(summary["gait_system_counts"], {"FGS": 2, "VGS": 1})
            self.assertEqual(summary["trial_counts"], {"T0": 2, "T1": 1})
            self.assertEqual(summary["speed_counts"], {"FGS": 2, "UGS": 1})
            self.assertEqual(summary["clothing_counts"], {"WoJ": 3})
            self.assertEqual(summary["direction_counts"], {"L2R": 1, "R2L": 2})
            self.assertEqual(summary["frame_count"], {"min": 4, "mean": 5.0, "max": 6})
            self.assertEqual(summary["dropped_short_clips"], 1)
            self.assertEqual(summary["missing_frame_count"], 3)
            self.assertEqual(summary["corrupt_frame_count"], 1)

            paths = write_healthgait_metadata_summary(summary, root / "diagnostics", "summary")
            self.assertEqual(json.loads(paths["json"].read_text()), summary)
            with paths["csv"].open(newline="") as csv_file:
                csv_rows = list(csv.DictReader(csv_file))
            self.assertIn(
                {"metric": "frame_count.mean", "value": "5.0"},
                csv_rows,
            )

    def test_summary_counts_jpeg_extension_accepted_by_loader(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_dir = root / "frames" / "train" / "p1"
            frame_dir.mkdir(parents=True)
            for index in range(2):
                Image.new("L", (8, 8), color=32).save(
                    frame_dir / f"{index + 1:03d}.jpeg"
                )
            row = {
                "subject_id": "P1",
                "modality": "silhouette",
                "gait_system": "FGS",
                "trial": "T0",
                "frame_dir": "frames/train/p1",
                "num_frames": "2",
                "split": "train",
                **scientific_fields("frames/train/p1", 2),
            }
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
                writer.writeheader()
                writer.writerow(row)
            summary = summarize_healthgait_manifest(manifest, root, clip_length=2)
            self.assertEqual(summary["missing_frame_count"], 0)
            self.assertEqual(summary["dropped_short_clips"], 0)


class HealthGaitMotionDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _constant_video(value):
        return torch.full((4, 1, 8, 8), value, dtype=torch.float32)

    @staticmethod
    def _alternating_video():
        frames = [torch.full((1, 8, 8), value) for value in (0.0, 1.0, 0.0, 1.0)]
        return torch.stack(frames)

    def test_motion_diagnostics_are_deterministic_and_rank_motion(self):
        train = SyntheticMotionDataset(
            "train",
            [
                (self._constant_video(0.0), "repeated"),
                (self._alternating_video(), "alternating"),
                (self._constant_video(0.5), "mid"),
            ],
        )
        val = SyntheticMotionDataset(
            "val",
            [
                (self._constant_video(0.25), "steady"),
                (self._alternating_video() * 0.5, "moving"),
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = run_healthgait_motion_diagnostics(
                [train, val], root / "first", samples_per_split=8, seed=7, epoch=3
            )
            second = run_healthgait_motion_diagnostics(
                [train, val], root / "second", samples_per_split=8, seed=7, epoch=3
            )

            self.assertEqual(first["sample_count"], 5)
            self.assertEqual(first["samples_per_split"], {"train": 3, "val": 2})
            self.assertEqual(first["low_motion_examples"], second["low_motion_examples"])
            self.assertEqual(first["high_motion_examples"], second["high_motion_examples"])
            self.assertEqual(
                first["outputs"]["csv"].read_text(),
                second["outputs"]["csv"].read_text(),
            )
            self.assertIn(
                "train-repeated",
                [example["sequence_id"] for example in first["low_motion_examples"]],
            )
            self.assertEqual(
                first["high_motion_examples"][0]["sequence_id"],
                "train-alternating",
            )
            self.assertEqual(train.epoch, 0)
            self.assertEqual(val.epoch, 0)

            for output_path in first["outputs"].values():
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

            compact_summary = json.loads(first["outputs"]["json"].read_text())
            self.assertEqual(compact_summary["sample_count"], 5)


if __name__ == "__main__":
    unittest.main()
