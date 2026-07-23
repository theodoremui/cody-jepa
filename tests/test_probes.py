import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cody_jepa.probes import (
    FEATURE_SOURCE,
    evaluate_all_probes,
    evaluate_gait_system,
    export_frozen_features,
    read_feature_table,
    validate_feature_metadata,
    validate_feature_table,
    write_feature_table,
    write_probe_results,
)


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
                        "sequence_id": sequence,
                        "split": split,
                        "subject_id": subject,
                        "gait_system": gait,
                        "trial": f"trial-{trial_index}",
                        "window_start": window * 4,
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
            "split": ["train", "train"],
            "subject_id": ["p0", "p1"],
            "gait_system": ["FGS", "UGS"],
            "trial": ["t0", "t1"],
            "window_start": torch.tensor([0, 4]),
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
            ["sequence_id", "split", "subject_id", "gait_system", "trial",
             "window_start", "feature_0", "feature_1"],
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

    def test_feature_sidecar_reserves_contract_fields_and_detects_table_drift(self):
        table = synthetic_feature_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            write_feature_table(
                table,
                path,
                {
                    "schema_version": 999,
                    "feature_formula": "tampered",
                    "row_count": 0,
                    "checkpoint": "checkpoint.pt",
                    "checkpoint_sha256": "a" * 64,
                },
            )
            loaded, metadata = read_feature_table(path)
            validate_feature_metadata(loaded, path, metadata)
            self.assertEqual(metadata["schema_version"], 1)
            self.assertNotEqual(metadata["feature_formula"], "tampered")
            with path.open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(ValueError, "feature_table_sha256"):
                validate_feature_metadata(loaded, path, metadata)

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
        self.assertEqual(results[0]["train_sequences"] + results[0]["val_sequences"], 8)
        self.assertEqual(results[0]["source_split"], "train")
        self.assertEqual(results[2]["protocol"], "subject_heldout_logistic_regression")

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
        required = {
            "task", "feature_source", "train_examples", "val_examples", "num_classes",
            "majority_baseline", "accuracy", "balanced_accuracy", "macro_f1",
            "confusion_matrix",
        }
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(len(csv_table), 3)
        self.assertTrue(required.issubset(payload["results"][0]))
        self.assertTrue(required.issubset(csv_table.columns))


if __name__ == "__main__":
    unittest.main()
