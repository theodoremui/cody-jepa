import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from cody_jepa.evaluation.features import METADATA_COLUMNS, read_feature_table
from cody_jepa.cli.prepare_gfc_features import main


class PrepareGFCFeaturesTest(unittest.TestCase):
    def _write_inputs(
        self,
        root: Path,
        *,
        subject_id: str = "P1",
        feature_dtype=np.float32,
        window_starts=(0, 7, 14),
    ):
        row_count = len(window_starts)
        legacy_path = root / "legacy.npz"
        np.savez_compressed(
            legacy_path,
            sequence_id=np.asarray([f"clip-{index}" for index in range(row_count)]),
            split=np.asarray(["train"] * row_count),
            subject_id=np.asarray([subject_id] * row_count),
            gait_system=np.asarray(["UGS-WoJ"] * row_count),
            trial=np.asarray(["1"] * row_count),
            window_start=np.asarray(window_starts, dtype=np.int64),
            features=np.asarray(
                [[2.0 * index + 1.0, 2.0 * index + 2.0] for index in range(row_count)],
                dtype=feature_dtype,
            ),
        )
        manifest_path = root / "manifest.csv"
        row = {
            "recording_id": "recording-1",
            "source_video_id": "source-1",
            "direction_clip_id": "direction-1",
            "split": "train",
            "subject_id": "P1",
            "gait_system": "UGS-WoJ",
            "trial": "1",
            "speed": "UGS",
            "clothing": "WoJ",
            "direction": "R2L",
            "num_frames": 30,
            "fps": 30.0,
            "shortcut_log_frame_count": math.log(30),
            "shortcut_duration_seconds": 1.0,
            "shortcut_horizontal_centroid_drift_signed": 0.25,
            "shortcut_horizontal_centroid_drift_absolute": 0.25,
            "shortcut_foreground_area_mean": 0.2,
            "shortcut_foreground_area_std": 0.1,
            "shortcut_foreground_area_q25": 0.1,
            "shortcut_foreground_area_median": 0.2,
            "shortcut_foreground_area_q75": 0.3,
        }
        fieldnames = [
            column
            for column in METADATA_COLUMNS
            if column not in {"sequence_id", "window_start"}
        ]
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return legacy_path, manifest_path

    def _run(self, legacy_path: Path, manifest_path: Path, output_path: Path):
        arguments = [
            "prepare_gfc_features.py",
            "--legacy-features",
            str(legacy_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ]
        stdout = io.StringIO()
        with mock.patch("sys.argv", arguments), contextlib.redirect_stdout(stdout):
            main()
        return json.loads(stdout.getvalue())

    def test_upgrades_legacy_features_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path, manifest_path = self._write_inputs(root)
            output_path = root / "features.npz"

            paths = self._run(legacy_path, manifest_path, output_path)
            table, metadata = read_feature_table(output_path)

            self.assertEqual(paths["features"], str(output_path))
            self.assertEqual(list(table["recording_id"]), ["recording-1"] * 3)
            np.testing.assert_array_equal(
                table[["feature_0", "feature_1"]].to_numpy(),
                np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            )
            self.assertIn("exact subject/gait_system/trial/split join", metadata["upgrade_method"])
            self.assertEqual(metadata["legacy_features"], "legacy.npz")
            self.assertEqual(metadata["gfc_manifest"], "manifest.csv")
            self.assertEqual(len(metadata["legacy_features_sha256"]), 64)
            self.assertEqual(len(metadata["gfc_manifest_sha256"]), 64)
            self.assertNotIn(str(root), json.dumps(metadata))

    def test_rejects_legacy_rows_without_a_manifest_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path, manifest_path = self._write_inputs(root, subject_id="P2")
            with self.assertRaisesRegex(ValueError, "do not match the GFC manifest"):
                self._run(legacy_path, manifest_path, root / "features.npz")

    def test_rejects_non_float32_features_instead_of_rounding_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path, manifest_path = self._write_inputs(
                root, feature_dtype=np.float64
            )
            with self.assertRaisesRegex(ValueError, "must use float32"):
                self._run(legacy_path, manifest_path, root / "features.npz")

    def test_rejects_more_than_three_rows_even_with_three_distinct_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path, manifest_path = self._write_inputs(
                root, window_starts=(0, 7, 14, 14)
            )
            with self.assertRaisesRegex(ValueError, "exactly three rows"):
                self._run(legacy_path, manifest_path, root / "features.npz")


if __name__ == "__main__":
    unittest.main()
