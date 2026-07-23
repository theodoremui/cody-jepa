import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from cody_jepa.probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    checkpoint_sha256,
    evaluate_all_probes,
    read_feature_table,
    write_feature_table,
    write_probe_results,
)
from cody_jepa.single_stream_jepa import CHECKPOINT_SCHEMA, MODEL_ARCHITECTURE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_phase0_pipeline", PROJECT_ROOT / "scripts" / "run_phase0_pipeline.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def _synthetic_feature_table():
    rows = []
    subjects = (("train", "ta"), ("train", "tb"), ("val", "va"), ("val", "vb"))
    for subject_index, (split, subject) in enumerate(subjects):
        for gait_index, gait in enumerate(("FGS", "UGS")):
            for trial in range(2):
                identity = np.zeros(len(subjects))
                identity[subject_index] = 10.0
                features = [*identity, float(gait_index * 10)]
                rows.append({
                    "sequence_id": f"{subject}-{gait}-{trial}",
                    "split": split,
                    "subject_id": subject,
                    "gait_system": gait,
                    "trial": str(trial),
                    "window_start": 0,
                    **{
                        f"feature_{index}": value
                        for index, value in enumerate(features)
                    },
                })
    return pd.DataFrame(rows)


def _synthetic_checkpoint():
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": MODEL_ARCHITECTURE,
        "completed_epochs": 1,
        "global_step": 1,
        "best_epoch": 1,
        "best_val_loss": 0.5,
        "best_healthy_epoch": None,
        "config": {"num_epochs": 1, "steps": 1},
        "history": [{"epoch": 1}],
        "data_contract": {
            split: {
                "manifest_sha256": "m" * 64,
                "inventory_sha256": "i" * 64,
                "sequence_count": 4,
            }
            for split in ("train_dataset", "val_dataset")
        },
    }


class Phase0PipelineTest(unittest.TestCase):
    def test_synthetic_run_pipeline_writes_verified_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            artifact_dir = root / "artifacts"
            report = root / "report.md"
            table = _synthetic_feature_table()

            def fake_run(command, *, cwd, env=None):
                command = [str(part) for part in command]
                if "jupyter" in command:
                    checkpoint = _synthetic_checkpoint()
                    torch.save(checkpoint, run_dir / "latest.pt")
                    torch.save(checkpoint, run_dir / "best_loss.pt")
                    return
                if any(part.endswith("export_single_stream_features.py") for part in command):
                    output = Path(command[command.index("--output") + 1])
                    checkpoint = Path(command[command.index("--checkpoint") + 1])
                    write_feature_table(
                        table,
                        output,
                        {
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha256(checkpoint),
                            "feature_source": FEATURE_SOURCE,
                            "feature_formula": FEATURE_FORMULA,
                        },
                    )
                    return
                if any(part.endswith("eval_probes.py") for part in command):
                    feature_path = Path(command[command.index("--features") + 1])
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    loaded, metadata = read_feature_table(feature_path)
                    sidecar = feature_path.with_suffix(".npz.metadata.json")
                    results = evaluate_all_probes(loaded, seed=0, max_iter=2000)
                    write_probe_results(
                        results,
                        output_dir,
                        {
                            "feature_table": str(feature_path),
                            "feature_table_sha256": checkpoint_sha256(feature_path),
                            "feature_metadata_sha256": checkpoint_sha256(sidecar),
                            "feature_source": FEATURE_SOURCE,
                            "feature_formula": FEATURE_FORMULA,
                            "checkpoint": metadata["checkpoint"],
                            "checkpoint_sha256": metadata["checkpoint_sha256"],
                            "seed": 0,
                            "max_iter": 2000,
                            "identity_validation_fraction": 0.25,
                            "retrieval_enrollment_sequences": 1,
                        },
                    )
                    return
                self.fail(f"unexpected pipeline command: {command}")

            args = SimpleNamespace(
                repo_root=root,
                allow_local_run=True,
                run_dir=run_dir,
                artifact_dir=artifact_dir,
                report=report,
                notebook_dir=root / "notebooks",
                checkpoint_name="best_loss.pt",
                success_criterion="synthetic pipeline succeeds",
                device="cpu",
                batch_size=1,
                num_workers=0,
                windows_per_sequence=1,
                seed=0,
                max_iter=2000,
                identity_validation_fraction=0.25,
                retrieval_enrollment_sequences=1,
            )
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                PIPELINE, "_run", side_effect=fake_run
            ) as run:
                PIPELINE.command_run(args)

            self.assertEqual(run.call_count, 3)
            self.assertTrue(report.is_file())
            payload = json.loads(report.with_suffix(".json").read_text())
            self.assertEqual(payload["success_criterion"], args.success_criterion)
            self.assertEqual(payload["checkpoint"]["completed_epochs"], 1)
            self.assertEqual(
                [result["task"] for result in payload["probe_results"]["results"]],
                ["identity_closed_set", "identity_heldout_retrieval", "gait_system"],
            )

    def test_baseline_defaults_to_unique_ignored_destinations(self):
        args = SimpleNamespace(artifact_dir=None, report=None)
        first = PIPELINE._baseline_destinations(args, PROJECT_ROOT)
        second = PIPELINE._baseline_destinations(args, PROJECT_ROOT)
        self.assertNotEqual(first, second)
        for artifact_dir, report in (first, second):
            self.assertIn("outputs/phase0/regenerations", artifact_dir.as_posix())
            self.assertEqual(report.parent, artifact_dir.parent)

    def test_run_outside_slurm_requires_explicit_local_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo_root=Path(temporary), allow_local_run=False)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
                    PIPELINE.command_run(args)

    def test_failed_export_stage_never_writes_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.md"
            args = SimpleNamespace(
                repo_root=root,
                checkpoint=root / "checkpoint.pt",
                artifact_dir=root / "artifacts",
                report=report,
                device="cpu",
                batch_size=1,
                num_workers=0,
                windows_per_sequence=1,
                seed=0,
                max_iter=1,
                identity_validation_fraction=0.25,
                retrieval_enrollment_sequences=1,
            )
            with mock.patch.object(
                PIPELINE, "_run", side_effect=subprocess.CalledProcessError(1, ["export"])
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    PIPELINE.command_evaluate(args)
            self.assertFalse(report.exists())
            self.assertFalse(report.with_suffix(".json").exists())

    def test_report_writer_refuses_existing_destination_before_reading_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.md"
            report.write_text("locked\n")
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                PIPELINE._write_generic_report(
                    root / "missing.pt",
                    {"features": root / "missing.npz", "probe_json": root / "missing.json"},
                    report,
                    root,
                )

    def test_slurm_worker_invokes_same_uv_pipeline_and_requires_exported_paths(self):
        script = (PROJECT_ROOT / "slurm" / "train-single-stream-jepa.sbatch").read_text()
        self.assertIn("uv run --frozen --no-sync python scripts/run_phase0_pipeline.py run", script)
        for variable in (
            "CODY_JEPA_OUTPUT_DIR",
            "CODY_JEPA_ARTIFACT_DIR",
            "CODY_JEPA_REPORT_PATH",
            "CODY_JEPA_CHECKPOINT_NAME",
            "CODY_JEPA_SUCCESS_CRITERION",
        ):
            self.assertIn(f'${{{variable}:?', script)


if __name__ == "__main__":
    unittest.main()
