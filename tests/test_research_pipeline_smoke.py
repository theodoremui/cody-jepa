import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from cody_jepa.data import HealthGaitManifestDataset
from cody_jepa.probes import export_frozen_features, write_feature_table
from scripts.build_healthgait_manifest import MANIFEST_COLUMNS, build_rows
from scripts.run_gfc import run_gfc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FactorPatternEncoder(nn.Module):
    def forward(self, video, return_pre_norm=False):
        pooled = video.mean(dim=1).flatten(start_dim=1).unsqueeze(1)
        return (pooled, pooled) if return_pre_norm else pooled


class ResearchPipelineSmokeTest(unittest.TestCase):
    def test_manifest_to_deterministic_export_to_gfc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "data" / "healthgait" / "raw" / "Health_Gait"
            for subject_index in range(10):
                subject = f"PA{subject_index:03d}"
                for speed in ("UGS", "FGS"):
                    for clothing in ("WoJ", "WJ"):
                        for suffix, direction in (("1", "R2L"), ("2", "L2R")):
                            frame_dir = (
                                raw_root
                                / "silhouette"
                                / subject
                                / speed
                                / f"{clothing}_{suffix}"
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
            table = export_frozen_features(
                FactorPatternEncoder(),
                loaders,
                {"num_frames": 4, "in_channels": 1, "img_size": 8},
                torch.device("cpu"),
            )
            starts_per_recording = table.groupby("recording_id")["window_start"].nunique()
            self.assertTrue((starts_per_recording == 3).all())
            features = root / "features.csv"
            write_feature_table(table, features)
            summary = run_gfc(
                features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                root / "gfc",
                model_label="synthetic-factor-pattern-encoder",
            )
            self.assertEqual(summary["evaluation"]["participant_count"], 2)
            self.assertEqual(summary["evaluation"]["excluded_participant_count"], 0)
            self.assertGreater(summary["learned"]["top1"], summary["shortcut"]["top1"])

            npz_features = root / "features.npz"
            write_feature_table(table, npz_features)
            npz_summary = run_gfc(
                npz_features,
                PROJECT_ROOT / "configs" / "eval" / "gfc_healthgait.json",
                "development",
                root / "gfc-npz",
                model_label="synthetic-factor-pattern-encoder",
            )
            self.assertEqual(npz_summary, summary)


if __name__ == "__main__":
    unittest.main()
