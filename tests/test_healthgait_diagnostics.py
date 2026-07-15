import csv
import json
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
    run_healthgait_motion_diagnostics,
    summarize_healthgait_manifest,
    write_healthgait_dummy_probe_exports,
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
]


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

    def test_summary_counts_frame_health_and_writes_artifacts(self):
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
                },
                {
                    "subject_id": "P1",
                    "modality": "silhouette",
                    "gait_system": "FGS",
                    "trial": "T0",
                    "frame_dir": "frames/val/p1",
                    "num_frames": "5",
                    "split": "val",
                },
                {
                    "subject_id": "P2",
                    "modality": "silhouette",
                    "gait_system": "VGS",
                    "trial": "T1",
                    "frame_dir": "frames/train/p2",
                    "num_frames": "6",
                    "split": "train",
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
                first["artifacts"]["csv"].read_text(),
                second["artifacts"]["csv"].read_text(),
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

            for artifact_path in first["artifacts"].values():
                self.assertTrue(artifact_path.exists())
                self.assertGreater(artifact_path.stat().st_size, 0)

            compact_summary = json.loads(first["artifacts"]["json"].read_text())
            self.assertEqual(compact_summary["sample_count"], 5)


class HealthGaitDummyProbeExportTest(unittest.TestCase):
    @staticmethod
    def _ramp_video():
        frames = [torch.full((1, 2, 2), value) for value in (0.0, 0.25, 0.5, 0.75)]
        return torch.stack(frames)

    @staticmethod
    def _steady_video():
        return torch.full((4, 1, 2, 2), 0.5, dtype=torch.float32)

    def test_probe_export_locks_schema_and_uses_deterministic_clip_stats(self):
        train = SyntheticMotionDataset("train", [(self._ramp_video(), "ramp")])
        val = SyntheticMotionDataset("val", [(self._steady_video(), "steady")])
        train.set_epoch(11)
        val.set_epoch(13)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = write_healthgait_dummy_probe_exports(
                [train, val], root / "first", latent_dim=4, epoch=2
            )
            second = write_healthgait_dummy_probe_exports(
                [train, val], root / "second", latent_dim=4, epoch=2
            )

            expected_fieldnames = [
                "sequence_id",
                "split",
                "subject_id",
                "gait_system",
                "trial",
                "window_start",
                "s_attr_0",
                "s_attr_1",
                "s_attr_2",
                "s_attr_3",
                "s_dyn_0",
                "s_dyn_1",
                "s_dyn_2",
                "s_dyn_3",
            ]
            self.assertEqual(first["fieldnames"], expected_fieldnames)
            self.assertEqual(first["row_count"], 2)
            self.assertEqual(first["csv"].read_text(), second["csv"].read_text())
            self.assertEqual(train.epoch, 11)
            self.assertEqual(val.epoch, 13)

            with first["csv"].open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(rows[0]["sequence_id"], "train-ramp")
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(rows[0]["window_start"], "2")
            self.assertEqual(rows[0]["s_attr_0"], "0.37500000")
            self.assertEqual(rows[0]["s_attr_1"], "0.27950850")
            self.assertEqual(rows[0]["s_attr_2"], "0.00000000")
            self.assertEqual(rows[0]["s_attr_3"], "0.75000000")
            self.assertEqual(rows[0]["s_dyn_0"], "0.25000000")
            self.assertEqual(rows[0]["s_dyn_1"], "0.00000000")
            self.assertEqual(rows[0]["s_dyn_2"], "0.25000000")
            self.assertEqual(rows[0]["s_dyn_3"], "0.25000000")
            self.assertEqual(rows[1]["sequence_id"], "val-steady")
            self.assertEqual(rows[1]["s_dyn_0"], "0.00000000")


if __name__ == "__main__":
    unittest.main()
