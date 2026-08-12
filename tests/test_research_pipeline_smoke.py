import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from cody_jepa.cli.build_manifest import MANIFEST_COLUMNS, build_rows
from cody_jepa.data import HealthGaitManifestDataset
from cody_jepa.evaluation.features import (
    export_frozen_features,
    read_feature_table,
    write_feature_table,
)
from cody_jepa.evaluation.probes import evaluate_all_probes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FactorPatternEncoder(nn.Module):
    def forward(self, video, return_pre_norm=False):
        pooled = video.mean(dim=1).flatten(start_dim=1).unsqueeze(1)
        return (pooled, pooled) if return_pre_norm else pooled


class ResearchPipelineSmokeTest(unittest.TestCase):
    """Raw frames -> manifest -> dataset -> frozen features -> probes."""

    def _build_corpus(self, root: Path) -> Path:
        raw_root = root / "data" / "healthgait" / "raw" / "Health_Gait"
        for subject_index in range(10):
            subject = f"PA{subject_index:03d}"
            for speed in ("UGS", "FGS"):
                for clothing in ("WoJ", "WJ"):
                    for suffix, direction in (("1", "R2L"), ("2", "L2R")):
                        frame_dir = (
                            raw_root / "silhouette" / subject / speed / f"{clothing}_{suffix}"
                        )
                        frame_dir.mkdir(parents=True)
                        pattern = np.zeros((8, 8), dtype=np.uint8)
                        pattern[0, 0] = 255
                        pattern[1, 1 if speed == "UGS" else 2] = 255
                        pattern[2, 1 if clothing == "WoJ" else 2] = 255
                        pattern[3, 1 if direction == "R2L" else 2] = 255
                        for frame_index in range(6):
                            Image.fromarray(pattern, mode="L").save(
                                frame_dir / f"{frame_index + 1:03d}.png"
                            )
        return raw_root

    def _export_features(self, root: Path, raw_root: Path, manifest: Path):
        datasets = {
            split: HealthGaitManifestDataset(
                manifest,
                split,
                root,
                clip_length=4,
                image_size=(8, 8),
                deterministic_windows=3,
                allowed_data_root=raw_root,
            )
            for split in ("train", "val")
        }
        loaders = {
            split: DataLoader(dataset, batch_size=32, shuffle=False)
            for split, dataset in datasets.items()
        }
        return export_frozen_features(
            FactorPatternEncoder(),
            loaders,
            {"num_frames": 4, "in_channels": 1, "img_size": 8},
            torch.device("cpu"),
        )

    def test_manifest_to_deterministic_feature_export_to_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = self._build_corpus(root)

            rows = build_rows(
                SimpleNamespace(
                    raw_root=raw_root,
                    modality="silhouette",
                    clip_length=4,
                    fps=30.0,
                    foreground_threshold=0.5,
                    repo_root=root,
                    seed=0,
                    val_fraction=0.2,
                )
            )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            # Every recording contributes exactly the requested number of
            # deterministic windows, so exports are reproducible across runs.
            table = self._export_features(root, raw_root, manifest)
            starts_per_recording = table.groupby("recording_id")["window_start"].nunique()
            self.assertTrue((starts_per_recording == 3).all())

            repeated = self._export_features(root, raw_root, manifest)
            self.assertTrue(table.equals(repeated))

            # CSV and NPZ are interchangeable serializations of the same table.
            csv_path = root / "features.csv"
            npz_path = root / "features.npz"
            write_feature_table(table, csv_path)
            write_feature_table(table, npz_path)
            csv_table, _ = read_feature_table(csv_path)
            npz_table, _ = read_feature_table(npz_path)
            # CSV re-inferates dtypes on read, so compare values, not dtypes.
            pd.testing.assert_frame_equal(csv_table, npz_table, check_dtype=False)

            # The encoder writes speed into a dedicated pixel, so the gait-system
            # probe must clear its majority-class baseline.
            results = evaluate_all_probes(csv_table)
            gait = next(r for r in results if r["task"] == "gait_system")
            self.assertGreater(
                float(gait["accuracy"]), float(gait["majority_baseline"])
            )


if __name__ == "__main__":
    unittest.main()
