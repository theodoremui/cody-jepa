import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cody_jepa.phase0 import (
    REPRODUCIBILITY_CODE_PATHS,
    guard_research_path,
    require_unchanged_hash,
    validate_candidate_artifacts,
    validate_completed_run,
    validate_manifest,
)
from cody_jepa.probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    checkpoint_sha256,
    evaluate_all_probes,
    write_feature_table,
    write_probe_results,
)
from cody_jepa.single_stream_jepa import CHECKPOINT_SCHEMA, MODEL_ARCHITECTURE, train_jepa


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _feature_table():
    subjects = [("train", "ta"), ("train", "tb"), ("val", "va"), ("val", "vb")]
    rows = []
    for subject_index, (split, subject) in enumerate(subjects):
        for gait_index, gait in enumerate(("FGS", "UGS")):
            for trial in range(2):
                identity = np.zeros(4)
                identity[subject_index] = 10
                values = [*identity, float(gait_index * 10)]
                rows.append(
                    {
                        "sequence_id": f"{subject}-{gait}-{trial}",
                        "split": split,
                        "subject_id": subject,
                        "gait_system": gait,
                        "trial": str(trial),
                        "window_start": 0,
                        **{f"feature_{index}": value for index, value in enumerate(values)},
                    }
                )
    return pd.DataFrame(rows)


class Phase0TrustBoundaryTest(unittest.TestCase):
    def test_checked_in_report_attests_current_code_and_lockfile(self):
        report = json.loads((PROJECT_ROOT / "reports" / "phase0-baseline.json").read_text())
        reproducibility = report["reproducibility"]
        self.assertEqual(
            set(reproducibility["code_sha256"]), set(REPRODUCIBILITY_CODE_PATHS)
        )
        self.assertEqual(
            reproducibility["uv_lock_sha256"], checkpoint_sha256(PROJECT_ROOT / "uv.lock")
        )
        for relative_path, expected in reproducibility["code_sha256"].items():
            self.assertEqual(
                expected,
                checkpoint_sha256(PROJECT_ROOT / relative_path),
                relative_path,
            )

    def test_path_guard_resolves_aliases_and_rejects_retired_or_baseline_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "outputs" / "jepa-v4").mkdir(parents=True)
            alias = root / "baseline-alias"
            alias.symlink_to(root / "outputs" / "jepa-v4", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "read-only"):
                guard_research_path(alias / "features.npz", root, write=True)
            with self.assertRaisesRegex(ValueError, "retired"):
                guard_research_path(root / "outputs" / "jepa-v3" / "new.pt", root, write=False)
            self.assertEqual(
                guard_research_path(root / "outputs" / "jepa-v4" / "best_loss.pt", root, write=False),
                (root / "outputs" / "jepa-v4" / "best_loss.pt").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "read-only baseline"):
                train_jepa(
                    {},
                    None,
                    None,
                    {},
                    checkpoint_dir=root / "outputs" / "jepa-v4" / "descendant",
                )

    def test_manifest_contract_rejects_schema_and_count_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.csv"
            path.write_text(
                "subject_id,modality,gait_system,trial,frame_dir,num_frames,split\n"
                "a,silhouette,FGS,1,a,16,train\n"
                "b,silhouette,UGS,1,b,16,val\n"
            )
            protocol = {
                "manifest": {
                    "path": "manifest.csv",
                    "sha256": checkpoint_sha256(path),
                    "metadata_schema": [
                        "subject_id", "modality", "gait_system", "trial",
                        "frame_dir", "num_frames", "split",
                    ],
                    "splits": {
                        "train": {"sequences": 1, "subjects": 1},
                        "val": {"sequences": 1, "subjects": 1},
                    },
                }
            }
            self.assertEqual(validate_manifest(protocol, root)["splits"], protocol["manifest"]["splits"])
            protocol["manifest"]["metadata_schema"] = list(reversed(protocol["manifest"]["metadata_schema"]))
            with self.assertRaisesRegex(ValueError, "schema drift"):
                validate_manifest(protocol, root)

    def _artifacts(self, root):
        checkpoint_path = root / "best_loss.pt"
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": MODEL_ARCHITECTURE,
            "completed_epochs": 2,
            "global_step": 4,
            "best_epoch": 2,
            "best_val_loss": 0.5,
            "best_healthy_epoch": None,
            "config": {"num_epochs": 2, "steps": 4},
            "history": [{"epoch": 2}],
            "data_contract": {
                "train_dataset": {
                    "manifest_sha256": "m" * 64,
                    "inventory_sha256": "i" * 64,
                    "sequence_count": 4,
                },
                "val_dataset": {
                    "manifest_sha256": "m" * 64,
                    "inventory_sha256": "i" * 64,
                    "sequence_count": 4,
                },
            },
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_hash = checkpoint_sha256(checkpoint_path)
        table = _feature_table()
        feature_path = root / "features.npz"
        preprocessing = {
            "channels": 1,
            "clip_length": 2,
            "image_size": [2, 2],
            "resize_interpolation": "bilinear",
            "decoded_range": [0.0, 1.0],
            "input_mean": 0.5,
            "input_std": 0.5,
            "encoder": "ema_target_encoder",
            "token_stage": "pre_final_layer_norm",
            "pooling_axis": "token",
            "output_dtype": "float32",
        }
        write_feature_table(
            table,
            feature_path,
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "device": "cpu",
                "windows_per_sequence": 1,
                "window_policy": "deterministic_evenly_spaced_no_augmentation",
                "preprocessing": preprocessing,
                "dataset_signatures": {
                    split: {
                        "manifest_sha256": "m" * 64,
                        "inventory_sha256": "i" * 64,
                        "sequence_count": 4,
                    }
                    for split in ("train", "val")
                },
            },
        )
        results = evaluate_all_probes(table, seed=0, max_iter=2000)
        probe_dir = root / "probes"
        sidecar = feature_path.with_suffix(".npz.metadata.json")
        write_probe_results(
            results,
            probe_dir,
            {
                "feature_table": str(feature_path),
                "feature_table_sha256": checkpoint_sha256(feature_path),
                "feature_metadata_sha256": checkpoint_sha256(sidecar),
                "feature_source": FEATURE_SOURCE,
                "feature_formula": FEATURE_FORMULA,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "seed": 0,
                "max_iter": 2000,
                "identity_validation_fraction": 0.25,
                "retrieval_enrollment_sequences": 1,
            },
        )
        protocol = {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_architecture": MODEL_ARCHITECTURE,
            "candidate_checkpoints": {
                "best_loss.pt": {
                    "sha256": checkpoint_hash,
                    "completed_epochs": 2,
                    "global_step": 4,
                }
            },
            "manifest": {
                "sha256": "m" * 64,
                "sampled_inventory_sha256": "i" * 64,
                "splits": {
                    "train": {"sequences": 4, "subjects": 2},
                    "val": {"sequences": 4, "subjects": 2},
                },
            },
            "feature_export": {
                "source": FEATURE_SOURCE,
                "formula": FEATURE_FORMULA,
                "metadata_schema": [
                    "sequence_id", "split", "subject_id", "gait_system", "trial", "window_start"
                ],
                "windows_per_sequence": 1,
                "window_policy": "deterministic_evenly_spaced_no_augmentation",
                "reference_device": "cpu",
                "preprocessing": preprocessing,
                "expected_rows": {"train": 8, "val": 8},
            },
            "probes": {
                "seed": 0,
                "max_iter": 2000,
                "identity_validation_fraction": 0.25,
                "retrieval_enrollment_sequences": 1,
                "tasks": ["identity_closed_set", "identity_heldout_retrieval", "gait_system"],
            },
        }
        return checkpoint_path, feature_path, probe_dir / "probe_metrics.json", protocol

    def test_candidate_validation_recomputes_probes_and_rejects_favorable_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._artifacts(Path(temporary))
            record = validate_candidate_artifacts(*paths, "best_loss.pt")
            self.assertEqual(record["checkpoint"]["completed_epochs"], 2)
            probe_path = paths[2]
            payload = json.loads(probe_path.read_text())
            payload["results"][0]["accuracy"] = 0.123456
            probe_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "recomputed probe results"):
                validate_candidate_artifacts(*paths, "best_loss.pt")

    def test_candidate_validation_rejects_missing_sidecar_and_protocol_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._artifacts(Path(temporary))
            probe_path = paths[2]
            payload = json.loads(probe_path.read_text())
            payload["seed"] = 1
            probe_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "probe seed"):
                validate_candidate_artifacts(*paths, "best_loss.pt")
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._artifacts(Path(temporary))
            paths[1].with_suffix(".npz.metadata.json").unlink()
            with self.assertRaisesRegex(ValueError, "sidecar is required"):
                validate_candidate_artifacts(*paths, "best_loss.pt")

    def test_completed_run_rejects_partial_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path, _, _, _ = self._artifacts(Path(temporary))
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            checkpoint["completed_epochs"] = 1
            checkpoint["global_step"] = 1
            checkpoint["history"] = [{"epoch": 1}]
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "completed run"):
                validate_completed_run(checkpoint_path)

    def test_candidate_validation_rejects_checkpoint_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._artifacts(Path(temporary))
            checkpoint = torch.load(paths[0], weights_only=True)
            checkpoint["global_step"] = 5
            torch.save(checkpoint, paths[0])
            with self.assertRaisesRegex(ValueError, "baseline checkpoint drift"):
                validate_candidate_artifacts(*paths, "best_loss.pt")

    def test_candidate_validation_rejects_protocol_checkpoint_contract_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = list(self._artifacts(Path(temporary)))
            protocol = paths[3]
            protocol["checkpoint_schema"] = CHECKPOINT_SCHEMA + 1
            with self.assertRaisesRegex(ValueError, "protocol checkpoint schema"):
                validate_candidate_artifacts(*paths, "best_loss.pt")
        with tempfile.TemporaryDirectory() as temporary:
            paths = list(self._artifacts(Path(temporary)))
            protocol = paths[3]
            protocol["checkpoint_architecture"] = "incompatible-model"
            with self.assertRaisesRegex(ValueError, "protocol checkpoint architecture"):
                validate_candidate_artifacts(*paths, "best_loss.pt")

    def test_long_running_stage_detects_input_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"before")
            digest = checkpoint_sha256(path)
            path.write_bytes(b"after")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                require_unchanged_hash(path, digest, "checkpoint")


if __name__ == "__main__":
    unittest.main()
