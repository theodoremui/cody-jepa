import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cody_jepa.evaluation.features import (
    FEATURE_SOURCE,
    METADATA_COLUMNS,
    export_frozen_features,
    read_feature_table,
    validate_feature_table,
    write_feature_table,
)
from cody_jepa.evaluation.probes import (
    PROBE_SUMMARY_COLUMNS,
    evaluate_all_probes,
    evaluate_gait_system,
    write_probe_results,
)
from cody_jepa.evaluation.probes.identity import closed_set_masks as _closed_set_masks


def clip_metadata(sequence, split, subject, gait, trial, window):
    return {
        "sequence_id": sequence,
        "recording_id": sequence,
        "source_video_id": f"{subject}/{gait}/WoJ",
        "direction_clip_id": sequence,
        "split": split,
        "subject_id": subject,
        "gait_system": gait,
        "trial": trial,
        "speed": gait,
        "clothing": "WoJ",
        "direction": "R2L" if trial.endswith("0") else "L2R",
        "window_start": window,
        "num_frames": 16,
        "fps": 30.0,
        "shortcut_log_frame_count": np.log(16),
        "shortcut_duration_seconds": 16 / 30,
        "shortcut_horizontal_centroid_drift_signed": 0.1,
        "shortcut_horizontal_centroid_drift_absolute": 0.1,
        "shortcut_foreground_area_mean": 0.25,
        "shortcut_foreground_area_std": 0.02,
        "shortcut_foreground_area_q25": 0.23,
        "shortcut_foreground_area_median": 0.25,
        "shortcut_foreground_area_q75": 0.27,
    }


def synthetic_feature_table():
    subjects = [("train", "train-a"), ("train", "train-b"),
                ("val", "val-a"), ("val", "val-b")]
    identity_index = {subject: index for index, (_, subject) in enumerate(subjects)}
    rows = []
    for split, subject in subjects:
        for gait_index, gait in enumerate(("FGS", "UGS")):
            for trial_index in range(2):
                sequence = f"{subject}-{gait}-{trial_index}"
                for window in range(2):
                    identity = np.zeros(len(subjects), dtype=np.float64)
                    identity[identity_index[subject]] = 10.0
                    features = [*identity, -5.0 if gait == "FGS" else 5.0]
                    rows.append({
                        **clip_metadata(
                            sequence, split, subject, gait,
                            f"trial-{trial_index}", window * 4,
                        ),
                        **{
                            f"feature_{index}": value
                            for index, value in enumerate(features)
                        },
                    })
    return pd.DataFrame(rows)


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.saw_inference_mode = False

    def forward(self, video, return_pre_norm=False):
        self.saw_inference_mode = torch.is_inference_mode_enabled()
        if self.training:
            raise AssertionError("export did not put encoder in eval mode")
        pooled = video.mean(dim=(1, 2, 3, 4))
        pre_norm = torch.stack((pooled, pooled * 2), dim=-1).unsqueeze(1).repeat(1, 3, 1)
        return (pre_norm, pre_norm) if return_pre_norm else pre_norm


class ProbeTest(unittest.TestCase):
    def test_frozen_export_uses_inference_eval_pre_norm_mean_and_metadata(self):
        encoder = RecordingEncoder().train()
        batch = {
            "video": torch.stack((torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2))),
            "sequence_id": ["s0", "s1"],
            "recording_id": ["s0", "s1"],
            "source_video_id": ["source0", "source1"],
            "direction_clip_id": ["s0", "s1"],
            "split": ["train", "train"],
            "subject_id": ["p0", "p1"],
            "gait_system": ["FGS", "UGS"],
            "trial": ["t0", "t1"],
            "speed": ["FGS", "UGS"],
            "clothing": ["WoJ", "WoJ"],
            "direction": ["R2L", "L2R"],
            "window_start": torch.tensor([0, 4]),
            "num_frames": torch.tensor([2, 2]),
            "fps": torch.tensor([30.0, 30.0]),
            "shortcut_log_frame_count": torch.tensor([np.log(2), np.log(2)]),
            "shortcut_duration_seconds": torch.tensor([2 / 30, 2 / 30]),
            "shortcut_horizontal_centroid_drift_signed": torch.tensor([0.0, 0.1]),
            "shortcut_horizontal_centroid_drift_absolute": torch.tensor([0.0, 0.1]),
            "shortcut_foreground_area_mean": torch.tensor([0.2, 0.3]),
            "shortcut_foreground_area_std": torch.tensor([0.01, 0.02]),
            "shortcut_foreground_area_q25": torch.tensor([0.18, 0.28]),
            "shortcut_foreground_area_median": torch.tensor([0.2, 0.3]),
            "shortcut_foreground_area_q75": torch.tensor([0.22, 0.32]),
        }
        cfg = {
            "num_frames": 2,
            "in_channels": 1,
            "img_size": 2,
            "input_mean": 0.0,
            "input_std": 1.0,
        }
        table = export_frozen_features(
            encoder, {"train": [batch]}, cfg, torch.device("cpu")
        )

        self.assertTrue(encoder.saw_inference_mode)
        self.assertFalse(encoder.training)
        self.assertFalse(encoder.weight.requires_grad)
        self.assertIsNone(encoder.weight.grad)
        self.assertEqual(
            list(table.columns),
            [*METADATA_COLUMNS, "feature_0", "feature_1"],
        )
        np.testing.assert_allclose(table[["feature_0", "feature_1"]], [[0, 0], [1, 2]])

    def test_feature_table_csv_and_npz_round_trip_without_pickle(self):
        table = synthetic_feature_table()
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".csv", ".npz"):
                path = Path(tmp) / f"features{suffix}"
                paths = write_feature_table(table, path, {"test_marker": "yes"})
                loaded, metadata = read_feature_table(path)
                self.assertEqual(paths["features"], path)
                self.assertEqual(metadata["feature_source"], FEATURE_SOURCE)
                self.assertEqual(metadata["test_marker"], "yes")
                self.assertEqual(validate_feature_table(loaded), [f"feature_{i}" for i in range(5)])
                np.testing.assert_allclose(
                    loaded[[f"feature_{i}" for i in range(5)]],
                    table[[f"feature_{i}" for i in range(5)]],
                )

    def test_feature_sidecar_describes_interpretation_and_shape(self):
        table = synthetic_feature_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            write_feature_table(
                table,
                path,
                {
                    "checkpoint": "checkpoint.pt",
                },
            )
            loaded, metadata = read_feature_table(path)
            self.assertEqual(len(loaded), len(table))
            self.assertEqual(metadata["checkpoint"], "checkpoint.pt")
            self.assertEqual(metadata["row_count"], len(table))
            self.assertEqual(metadata["feature_dim"], 5)
            self.assertEqual(metadata["feature_source"], FEATURE_SOURCE)

    def test_all_protocols_recover_synthetic_linear_signal(self):
        results = evaluate_all_probes(synthetic_feature_table(), seed=9, max_iter=500)
        self.assertEqual(
            [result["task"] for result in results],
            ["identity_closed_set", "identity_heldout_retrieval", "gait_system"],
        )
        for result in results:
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["balanced_accuracy"], 1.0)
            self.assertEqual(result["macro_f1"], 1.0)
            self.assertIn("confusion_matrix", result)
            self.assertGreater(result["accuracy"], result["majority_baseline"])
        self.assertEqual(results[0]["train_sources"] + results[0]["val_sources"], 4)
        self.assertEqual(results[0]["source_split"], "train")
        self.assertEqual(results[1]["enrollment_sources"], 2)
        self.assertEqual(results[1]["query_sources"], 2)
        self.assertEqual(results[2]["protocol"], "subject_heldout_logistic_regression")

    def test_identity_partitions_keep_paired_directions_together(self):
        table = synthetic_feature_table()
        train = table.loc[table["split"] == "train"].copy()
        fit_mask, validation_mask = _closed_set_masks(train, 0.25, seed=3)
        fit_sources = set(train.loc[fit_mask, "source_video_id"])
        validation_sources = set(train.loc[validation_mask, "source_video_id"])
        self.assertFalse(fit_sources & validation_sources)
        for source_id in set(train["source_video_id"]):
            source_rows = train["source_video_id"] == source_id
            self.assertTrue(fit_mask[source_rows].all() or validation_mask[source_rows].all())

    def test_gait_probe_rejects_subject_overlap(self):
        table = synthetic_feature_table()
        table.loc[table["split"] == "val", "subject_id"] = "train-a"
        with self.assertRaisesRegex(ValueError, "subject overlap"):
            evaluate_gait_system(table)

    def test_result_outputs_include_required_metrics(self):
        results = evaluate_all_probes(synthetic_feature_table(), max_iter=500)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_probe_results(results, tmp, {"seed": 0})
            payload = json.loads(paths["json"].read_text())
            csv_table = pd.read_csv(paths["csv"])
        required_json = {
            "task", "feature_source", "train_examples", "val_examples", "num_classes",
            "majority_baseline", "accuracy", "balanced_accuracy", "macro_f1",
            "class_labels", "confusion_matrix",
        }
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(len(csv_table), 3)
        self.assertTrue(required_json.issubset(payload["results"][0]))
        self.assertEqual(list(csv_table.columns), list(PROBE_SUMMARY_COLUMNS))
        self.assertNotIn("class_labels", csv_table.columns)
        self.assertNotIn("confusion_matrix", csv_table.columns)
        self.assertEqual(
            list(csv_table["task"]),
            [result["task"] for result in payload["results"]],
        )
        for row, result in zip(csv_table.to_dict("records"), payload["results"]):
            for column in PROBE_SUMMARY_COLUMNS:
                if isinstance(result[column], float):
                    self.assertAlmostEqual(row[column], result[column])
                else:
                    self.assertEqual(row[column], result[column])

    def test_confusion_matrices_are_complete_and_well_formed(self):
        for result in evaluate_all_probes(
            synthetic_feature_table(), seed=9, max_iter=500
        ):
            labels = result["class_labels"]
            matrix = np.asarray(result["confusion_matrix"])
            self.assertEqual(len(labels), result["num_classes"])
            self.assertEqual(len(set(labels)), len(labels))
            self.assertEqual(matrix.shape, (len(labels), len(labels)))
            self.assertTrue(np.issubdtype(matrix.dtype, np.integer))
            self.assertTrue((matrix >= 0).all())
            self.assertEqual(int(matrix.sum()), result["val_examples"])


if __name__ == "__main__":
    unittest.main()
