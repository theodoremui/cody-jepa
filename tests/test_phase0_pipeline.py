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
from cody_jepa.single_stream_jepa import (
    CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    checkpoint_model_state_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_phase0_pipeline", PROJECT_ROOT / "scripts" / "run_phase0_pipeline.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)
EVAL_SPEC = importlib.util.spec_from_file_location(
    "eval_probes", PROJECT_ROOT / "scripts" / "eval_probes.py"
)
EVAL_PROBES = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(EVAL_PROBES)


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
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": MODEL_ARCHITECTURE,
        "completed_epochs": 1,
        "global_step": 1,
        "best_epoch": 1,
        "best_val_loss": 0.5,
        "best_healthy_epoch": None,
        "best_healthy_val_loss": float("inf"),
        "config": {
            "num_epochs": 1,
            "steps": 1,
            "selection_metric": "subject_balanced_loss",
        },
        "mask_groups": [{"label": "synthetic"}],
        "history": [
            {
                "epoch": 1,
                "step": 1,
                "val": {
                    "subject_balanced_loss": 0.5,
                    "feature_std": 0.0,
                    "near_zero_variance_fraction": 0.0,
                    "effective_rank_ratio": 0.5,
                    "subject_balanced_context_shuffle_loss_gap": 0.1,
                    "context_shuffle_status": "complete",
                    "context_shuffle_pairs": 1,
                    "min_feature_norm": 1.0,
                    "max_feature_norm": 2.0,
                    "representations_healthy": False,
                    "health_issues": ["feature_std_below_threshold"],
                },
            }
        ],
        "data_contract": {
            split: {
                "manifest_sha256": "m" * 64,
                "inventory_sha256": "i" * 64,
                "sequence_count": 4,
            }
            for split in ("train_dataset", "val_dataset")
        },
        "context_encoder": {"weight": torch.tensor([1.0])},
        "target_encoder": {"weight": torch.tensor([2.0])},
        "predictor": {"weight": torch.tensor([3.0])},
    }
    fingerprint = checkpoint_model_state_sha256(checkpoint)
    checkpoint.update(
        model_state_sha256=fingerprint,
        best_loss_model_state_sha256=fingerprint,
        best_healthy_model_state_sha256=None,
    )
    return checkpoint


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
            self.assertEqual(payload["checkpoint"]["selection_role"], "best_loss")
            self.assertEqual(payload["completed_run_checkpoint"]["completed_epochs"], 1)
            self.assertEqual(payload["checkpoint"]["path"], "run/best_loss.pt")
            self.assertEqual(
                payload["completed_run_checkpoint"]["path"], "run/latest.pt"
            )
            self.assertEqual(
                payload["completed_run_checkpoint"][
                    "best_loss_model_state_sha256"
                ],
                payload["checkpoint"]["model_state_sha256"],
            )
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

    def test_baseline_evaluation_uses_locked_canonical_probe_provenance_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "outputs" / "jepa-v4" / "best_loss.pt"
            latest = checkpoint.with_name("latest.pt")
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"selected")
            latest.write_bytes(b"latest")
            commands = []
            protocol = {"baseline_job_id": 91108}
            with (
                mock.patch.object(
                    PIPELINE, "validate_checkpoint_from_completed_run"
                ),
                mock.patch.object(
                    PIPELINE,
                    "prepare_empty_directory",
                    return_value=root / "artifacts",
                ),
                mock.patch.object(
                    PIPELINE,
                    "_run",
                    side_effect=lambda command, **_: commands.append(
                        [str(part) for part in command]
                    ),
                ),
            ):
                PIPELINE._evaluate_checkpoint(
                    checkpoint=checkpoint,
                    completed_run_checkpoint=latest,
                    artifact_dir=root / "artifacts",
                    repo_root=root,
                    device="cpu",
                    batch_size=1,
                    num_workers=0,
                    windows_per_sequence=1,
                    seed=0,
                    max_iter=1,
                    identity_validation_fraction=0.25,
                    retrieval_enrollment_sequences=1,
                    protocol=protocol,
                )
            probe_command = commands[1]
            option = probe_command.index("--locked-phase0-provenance-label")
            self.assertEqual(
                probe_command[option + 1],
                "outputs/phase0/job-91108/best_loss/features.npz",
            )

    def test_locked_probe_provenance_label_reproduces_protocol_json_hash(self):
        protocol = json.loads(
            (PROJECT_ROOT / "protocols" / "phase0-baseline.json").read_text()
        )
        for candidate in ("best_loss.pt", "latest.pt"):
            with self.subTest(candidate=candidate):
                label_name = Path(candidate).stem
                feature = (
                    PROJECT_ROOT
                    / f"outputs/phase0/job-91108/{label_name}/features.npz"
                )
                table, metadata = read_feature_table(feature)
                del table
                sidecar = feature.with_suffix(".npz.metadata.json")
                feature_hash = checkpoint_sha256(feature)
                sidecar_hash = checkpoint_sha256(sidecar)
                args = SimpleNamespace(
                    features=feature,
                    locked_phase0_provenance_label=(
                        f"outputs/phase0/job-91108/{label_name}/features.npz"
                    ),
                )
                label = EVAL_PROBES._feature_table_provenance(
                    args,
                    PROJECT_ROOT,
                    feature,
                    metadata,
                    feature_hash,
                    sidecar_hash,
                )
                locked_probe = json.loads(
                    (
                        PROJECT_ROOT
                        / f"outputs/phase0/job-91108/{label_name}/probes/probe_metrics.json"
                    ).read_text()
                )
                with tempfile.TemporaryDirectory() as temporary:
                    paths = write_probe_results(
                        locked_probe["results"],
                        Path(temporary),
                        {
                            "feature_table": label,
                            "feature_table_sha256": feature_hash,
                            "feature_metadata_sha256": sidecar_hash,
                            "feature_source": metadata["feature_source"],
                            "feature_formula": metadata["feature_formula"],
                            "checkpoint": metadata["checkpoint"],
                            "checkpoint_sha256": metadata["checkpoint_sha256"],
                            "seed": locked_probe["seed"],
                            "max_iter": locked_probe["max_iter"],
                            "identity_validation_fraction": locked_probe[
                                "identity_validation_fraction"
                            ],
                            "retrieval_enrollment_sequences": locked_probe[
                                "retrieval_enrollment_sequences"
                            ],
                        },
                    )
                    self.assertEqual(
                        checkpoint_sha256(paths["json"]),
                        protocol["candidate_checkpoints"][candidate]["artifacts"][
                            "probe_json_sha256"
                        ],
                    )

    def test_baseline_checks_evidence_before_exports_and_after_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                repo_root=root,
                artifact_dir=root / "artifacts",
                report=root / "report.md",
                protocol=root / "protocol.json",
                device="cpu",
                batch_size=1,
                num_workers=0,
            )
            protocol = {
                "read_only_baseline_directory": "outputs/jepa-v4",
                "candidate_checkpoints": {"best_loss.pt": {}},
                "probes": {
                    "seed": 0,
                    "max_iter": 1,
                    "identity_validation_fraction": 0.25,
                    "retrieval_enrollment_sequences": 1,
                },
                "feature_export": {
                    "reference_device": "cpu",
                    "reference_batch_size": 1,
                    "reference_num_workers": 0,
                    "windows_per_sequence": 1,
                },
            }
            events = []

            with (
                mock.patch.object(PIPELINE, "load_protocol", return_value=protocol),
                mock.patch.object(PIPELINE, "validate_manifest"),
                mock.patch.object(
                    PIPELINE,
                    "validate_read_only_evidence",
                    side_effect=lambda *_: events.append("evidence"),
                ),
                mock.patch.object(PIPELINE, "validate_completed_run"),
                mock.patch.object(PIPELINE, "validate_checkpoint_from_completed_run"),
                mock.patch.object(PIPELINE, "checkpoint_record"),
                mock.patch.object(
                    PIPELINE,
                    "_evaluate_checkpoint",
                    side_effect=lambda **_: events.append("export"),
                ),
                mock.patch.object(
                    PIPELINE,
                    "build_baseline_report",
                    side_effect=lambda *_: events.append("report") or {},
                ),
            ):
                PIPELINE.command_baseline(args)

            self.assertEqual(events, ["evidence", "export", "report", "evidence"])

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
                checkpoint=root / "best_loss.pt",
                completed_run_checkpoint=root / "latest.pt",
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
            checkpoint = _synthetic_checkpoint()
            torch.save(checkpoint, args.checkpoint)
            torch.save(checkpoint, args.completed_run_checkpoint)
            with mock.patch.object(
                PIPELINE, "_run", side_effect=subprocess.CalledProcessError(1, ["export"])
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    PIPELINE.command_evaluate(args)
            self.assertFalse(report.exists())
            self.assertFalse(report.with_suffix(".json").exists())

    def test_evaluate_rejects_partial_run_before_creating_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "best_loss.pt"
            latest = root / "latest.pt"
            checkpoint = _synthetic_checkpoint()
            torch.save(checkpoint, selected)
            checkpoint["config"]["num_epochs"] = 2
            checkpoint["config"]["steps"] = 2
            torch.save(checkpoint, latest)
            args = SimpleNamespace(
                repo_root=root,
                checkpoint=selected,
                completed_run_checkpoint=latest,
                artifact_dir=root / "artifacts",
                report=root / "report.md",
                device="cpu",
                batch_size=1,
                num_workers=0,
                windows_per_sequence=1,
                seed=0,
                max_iter=1,
                identity_validation_fraction=0.25,
                retrieval_enrollment_sequences=1,
            )
            with mock.patch.object(PIPELINE, "_run") as run:
                with self.assertRaisesRegex(ValueError, "completed run"):
                    PIPELINE.command_evaluate(args)
            run.assert_not_called()
            self.assertFalse(args.artifact_dir.exists())

    def test_report_writer_refuses_existing_destination_before_reading_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.md"
            report.write_text("locked\n")
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                PIPELINE._write_generic_report(
                    root / "missing.pt",
                    root / "latest.pt",
                    {"features": root / "missing.npz", "probe_json": root / "missing.json"},
                    report,
                    root,
                )

    def test_evaluate_preflights_report_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.txt"
            args = SimpleNamespace(
                repo_root=root,
                checkpoint=root / "best_loss.pt",
                completed_run_checkpoint=root / "latest.pt",
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
            with mock.patch.object(PIPELINE, "_evaluate_checkpoint") as evaluate:
                with self.assertRaisesRegex(ValueError, "must end in .md"):
                    PIPELINE.command_evaluate(args)
            evaluate.assert_not_called()

    def test_generic_report_claim_serializes_cooperative_writers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.md"
            with PIPELINE._claim_generic_report(report, root):
                with self.assertRaisesRegex(FileExistsError, "already claimed"):
                    with PIPELINE._claim_generic_report(report, root):
                        self.fail("a second report writer acquired the same claim")
            self.assertFalse((root / ".report.md.pipeline-claim").exists())

    def test_generic_report_rejects_artifact_mutation_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "best_loss.pt"
            latest = root / "latest.pt"
            feature = root / "features.npz"
            sidecar = root / "features.npz.metadata.json"
            probe = root / "probe_metrics.json"
            report = root / "report.md"
            selected.write_bytes(b"selected")
            latest.write_bytes(b"latest")
            feature.write_bytes(b"features")
            sidecar.write_text("{}")
            selected_hash = checkpoint_sha256(selected)
            feature_hash = checkpoint_sha256(feature)
            sidecar_hash = checkpoint_sha256(sidecar)
            probes = {
                "feature_table_sha256": feature_hash,
                "feature_metadata_sha256": sidecar_hash,
                "checkpoint_sha256": selected_hash,
                "feature_source": FEATURE_SOURCE,
                "feature_formula": FEATURE_FORMULA,
                "identity_validation_fraction": 0.25,
                "retrieval_enrollment_sequences": 1,
                "max_iter": 1,
                "seed": 0,
                "results": [],
            }
            probe.write_text(json.dumps(probes))
            checkpoint_validation = {
                "selected_checkpoint": {
                    "sha256": selected_hash,
                    "identifier": f"sha256:{selected_hash}",
                    "completed_epochs": 1,
                },
                "completed_run_checkpoint": {
                    "sha256": checkpoint_sha256(latest),
                    "identifier": f"sha256:{checkpoint_sha256(latest)}",
                    "completed_epochs": 1,
                },
            }

            def mutate_feature(*args, **kwargs):
                feature.write_bytes(b"mutated")
                return []

            with (
                mock.patch.object(
                    PIPELINE,
                    "validate_checkpoint_from_completed_run",
                    return_value=checkpoint_validation,
                ),
                mock.patch.object(
                    PIPELINE,
                    "read_feature_table",
                    return_value=(pd.DataFrame(), {"checkpoint_sha256": selected_hash}),
                ),
                mock.patch.object(PIPELINE, "validate_feature_metadata"),
                mock.patch.object(
                    PIPELINE, "evaluate_all_probes", side_effect=mutate_feature
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during"):
                    PIPELINE._write_generic_report(
                        selected,
                        latest,
                        {"features": feature, "probe_json": probe},
                        report,
                        root,
                    )
            self.assertFalse(report.exists())
            self.assertFalse(report.with_suffix(".json").exists())

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

    def test_submit_exports_worker_contract_to_sbatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                repo_root=root,
                run_dir=root / "outputs" / "run-1",
                artifact_dir=root / "outputs" / "pipeline" / "run-1",
                report=root / "reports" / "run-1.md",
                checkpoint_name="best_loss.pt",
                success_criterion="synthetic Slurm pipeline succeeds",
            )
            completed = subprocess.CompletedProcess(
                args=["sbatch"], returncode=0, stdout="12345\n", stderr=""
            )
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                PIPELINE.subprocess, "run", return_value=completed
            ) as submit:
                PIPELINE.command_submit(args)

            submit.assert_called_once()
            command = submit.call_args.args[0]
            options = submit.call_args.kwargs
            exported = {
                "CODY_JEPA_OUTPUT_DIR": "outputs/run-1",
                "CODY_JEPA_ARTIFACT_DIR": "outputs/pipeline/run-1",
                "CODY_JEPA_REPORT_PATH": "reports/run-1.md",
                "CODY_JEPA_CHECKPOINT_NAME": "best_loss.pt",
                "CODY_JEPA_SUCCESS_CRITERION": args.success_criterion,
            }
            self.assertEqual(command[0:2], ["sbatch", "--parsable"])
            self.assertEqual(
                command[2], "--export=ALL," + ",".join(exported)
            )
            self.assertEqual(
                Path(command[3]),
                root.resolve() / "slurm" / "train-single-stream-jepa.sbatch",
            )
            self.assertEqual(options["cwd"], root.resolve())
            self.assertTrue(options["check"])
            self.assertTrue(options["text"])
            self.assertTrue(options["capture_output"])
            for key, value in exported.items():
                self.assertEqual(options["env"][key], value)
            self.assertTrue((root / "outputs" / ".run-1.pipeline-claim").is_file())

    def test_submit_failure_removes_pipeline_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                repo_root=root,
                run_dir=root / "outputs" / "run-1",
                artifact_dir=root / "outputs" / "pipeline" / "run-1",
                report=root / "reports" / "run-1.md",
                checkpoint_name="best_loss.pt",
                success_criterion="synthetic Slurm pipeline succeeds",
            )
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                PIPELINE.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["sbatch"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    PIPELINE.command_submit(args)

            self.assertFalse((root / "outputs" / ".run-1.pipeline-claim").exists())


if __name__ == "__main__":
    unittest.main()
