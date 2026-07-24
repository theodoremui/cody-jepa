import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import cody_jepa.phase0 as phase0_module

from cody_jepa.phase0 import (
    REQUIRED_READ_ONLY_EVIDENCE_PATHS,
    REPRODUCIBILITY_CODE_PATHS,
    checkpoint_record,
    guard_research_path,
    load_protocol,
    require_unchanged_hash,
    validate_candidate_artifacts,
    validate_checkpoint_from_completed_run,
    validate_completed_run,
    validate_manifest,
    validate_read_only_evidence,
    write_text_atomic,
    write_texts_atomic,
)
from cody_jepa.probes import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    checkpoint_sha256,
    evaluate_all_probes,
    write_feature_table,
    write_probe_results,
)
from cody_jepa.single_stream_jepa import (
    CHECKPOINT_SCHEMA,
    LEGACY_CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    checkpoint_model_state_sha256,
    train_jepa,
)


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
    def _read_only_evidence_protocol(self, root):
        evidence = {}
        for index, relative_path in enumerate(sorted(REQUIRED_READ_ONLY_EVIDENCE_PATHS)):
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"evidence-{index}".encode())
            evidence[relative_path] = {
                "sha256": checkpoint_sha256(path),
                "role": f"test evidence {index}",
            }
        return {
            "schema_version": 2,
            "read_only_baseline_directory": "outputs/jepa-v4",
            "retired_baseline_directories": ["outputs/jepa-v3"],
            "read_only_evidence": evidence,
            "candidate_checkpoints": {
                filename: {"sha256": evidence[f"outputs/jepa-v4/{filename}"]["sha256"]}
                for filename in ("best_loss.pt", "latest.pt")
            },
            "canonical_checkpoint": {"filename": "best_loss.pt"},
        }

    def _completed_run_checkpoints(self, root, *, best_epoch=1, healthy_epoch=None):
        history = []
        for epoch in range(1, 4):
            healthy = epoch == healthy_epoch
            history.append(
                {
                    "epoch": epoch,
                    "step": epoch * 2,
                    "val": {
                        "subject_balanced_loss": float(epoch),
                        "feature_std": 0.1 if healthy else 0.0,
                        "near_zero_variance_fraction": 0.0,
                        "effective_rank_ratio": 0.5,
                        "subject_balanced_context_shuffle_loss_gap": 0.1,
                        "context_shuffle_status": "complete",
                        "context_shuffle_pairs": 1,
                        "min_feature_norm": 1.0,
                        "max_feature_norm": 2.0,
                        "representations_healthy": healthy,
                        "health_issues": (
                            [] if healthy else ["feature_std_below_threshold"]
                        ),
                    },
                }
            )

        def model_state(epoch):
            return {
                component: {"weight": torch.tensor([float(epoch)])}
                for component in ("context_encoder", "target_encoder", "predictor")
            }

        fingerprints = {
            epoch: checkpoint_model_state_sha256(model_state(epoch))
            for epoch in range(1, 4)
        }
        common = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": MODEL_ARCHITECTURE,
            "config": {
                "num_epochs": 3,
                "steps": 6,
                "selection_metric": "subject_balanced_loss",
            },
            "mask_groups": [{"label": "small", "context": 0.5}],
            "data_contract": {
                "train_dataset": {
                    "manifest_sha256": "m" * 64,
                    "sequence_count": 4,
                },
                "val_dataset": {
                    "manifest_sha256": "m" * 64,
                    "sequence_count": 4,
                },
            },
            "best_epoch": best_epoch,
            "best_val_loss": float(best_epoch),
            "best_healthy_epoch": healthy_epoch,
            "best_healthy_val_loss": (
                float(healthy_epoch) if healthy_epoch is not None else float("inf")
            ),
            "best_loss_model_state_sha256": fingerprints[best_epoch],
            "best_healthy_model_state_sha256": (
                fingerprints[healthy_epoch] if healthy_epoch is not None else None
            ),
        }
        latest = {
            **common,
            **model_state(3),
            "completed_epochs": 3,
            "global_step": 6,
            "history": history,
            "model_state_sha256": fingerprints[3],
        }
        latest_path = root / "latest.pt"
        torch.save(latest, latest_path)
        selected_epoch = healthy_epoch if healthy_epoch is not None else best_epoch
        selected = {
            **common,
            **model_state(selected_epoch),
            "completed_epochs": selected_epoch,
            "global_step": selected_epoch * 2,
            "history": history[:selected_epoch],
            "model_state_sha256": fingerprints[selected_epoch],
        }
        selected_path = root / (
            "best_healthy.pt" if healthy_epoch is not None else "best_loss.pt"
        )
        torch.save(selected, selected_path)
        return selected_path, latest_path

    def test_checked_in_report_attests_current_code_and_lockfile(self):
        protocol_path = PROJECT_ROOT / "protocols" / "phase0-baseline.json"
        protocol = load_protocol(PROJECT_ROOT, protocol_path)
        report = json.loads((PROJECT_ROOT / "reports" / "phase0-baseline.json").read_text())
        reproducibility = report["reproducibility"]
        self.assertEqual(
            reproducibility["protocol_path"], "protocols/phase0-baseline.json"
        )
        self.assertEqual(
            reproducibility["protocol_sha256"], checkpoint_sha256(protocol_path)
        )
        self.assertEqual(reproducibility["protocol_payload"], protocol)
        self.assertEqual(report["baseline_job_id"], protocol["baseline_job_id"])
        self.assertEqual(
            report["read_only_evidence"], protocol["read_only_evidence"]
        )
        self.assertEqual(
            report["retired_baseline_directories"],
            protocol["retired_baseline_directories"],
        )
        self.assertEqual(
            report["canonical_checkpoint"], protocol["canonical_checkpoint"]
        )
        self.assertEqual(report["manifest"], protocol["manifest"])
        self.assertEqual(report["feature_export"], protocol["feature_export"])
        self.assertEqual(report["probe_protocol"], protocol["probes"])
        for filename, contract in protocol["candidate_checkpoints"].items():
            candidate = report["candidates"][filename]
            self.assertEqual(candidate["checkpoint"]["sha256"], contract["sha256"])
            self.assertEqual(
                candidate["features"]["sha256"],
                contract["artifacts"]["feature_sha256"],
            )
            self.assertEqual(
                candidate["features"]["metadata_sha256"],
                contract["artifacts"]["feature_metadata_sha256"],
            )
            self.assertEqual(
                candidate["probes"]["json_sha256"],
                contract["artifacts"]["probe_json_sha256"],
            )
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
            retained_notebook = root / "haic-results" / "job_91108.ipynb"
            retained_notebook.parent.mkdir()
            retained_notebook.write_text("retained")
            notebook_alias = root / "notebook-alias.ipynb"
            notebook_alias.symlink_to(retained_notebook)
            for protected_path in (retained_notebook, notebook_alias):
                with self.assertRaisesRegex(ValueError, "read-only evidence"):
                    guard_research_path(protected_path, root, write=True)
            self.assertEqual(
                guard_research_path(retained_notebook, root, write=False),
                retained_notebook.resolve(),
            )
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

    def test_checked_in_read_only_evidence_is_complete_and_hash_locked(self):
        protocol = load_protocol(PROJECT_ROOT)
        self.assertEqual(
            set(protocol["read_only_evidence"]),
            set(REQUIRED_READ_ONLY_EVIDENCE_PATHS),
        )
        verified = validate_read_only_evidence(protocol, PROJECT_ROOT)
        self.assertEqual(verified, protocol["read_only_evidence"])
        self.assertEqual(
            protocol["retired_baseline_directories"], ["outputs/jepa-v3"]
        )

    def test_read_only_evidence_rejects_drift_and_symlink_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self._read_only_evidence_protocol(root)
            verified = validate_read_only_evidence(protocol, root)
            self.assertEqual(verified, protocol["read_only_evidence"])
            drifted = root / "outputs" / "jepa-v4" / "probe_metrics.json"
            drifted.write_text("drifted")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                validate_read_only_evidence(protocol, root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self._read_only_evidence_protocol(root)
            retained = root / "haic-results" / "job_91108.ipynb"
            outside = root / "outside.ipynb"
            outside.write_bytes(retained.read_bytes())
            retained.unlink()
            retained.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "must not use symlinks"):
                validate_read_only_evidence(protocol, root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self._read_only_evidence_protocol(root)
            retired_target = root / "retired-target"
            retired_target.mkdir()
            (root / "outputs" / "jepa-v3").symlink_to(
                retired_target, target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "lexically absent"):
                validate_read_only_evidence(protocol, root)

    def test_protocol_schema_requires_evidence_and_retired_declaration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self._read_only_evidence_protocol(root)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            self.assertEqual(load_protocol(root, protocol_path), protocol)
            protocol["schema_version"] = 1
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_protocol(root, protocol_path)
            protocol["schema_version"] = 2
            protocol["retired_baseline_directories"] = []
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "retired baseline"):
                load_protocol(root, protocol_path)
            protocol["retired_baseline_directories"] = ["outputs/jepa-v3"]
            protocol["read_only_evidence"]["extra.txt"] = {
                "sha256": "0" * 64,
                "role": "unexpected",
            }
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_protocol(root, protocol_path)

    def test_protocol_rejects_candidate_path_escape_and_canonical_drift(self):
        mutations = (
            lambda protocol: protocol["candidate_checkpoints"].update(
                {"../best_loss.pt": protocol["candidate_checkpoints"].pop("best_loss.pt")}
            ),
            lambda protocol: protocol["candidate_checkpoints"].update(
                {"extra.pt": {"sha256": "0" * 64}}
            ),
            lambda protocol: protocol["canonical_checkpoint"].update(
                filename="latest.pt"
            ),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                protocol = self._read_only_evidence_protocol(root)
                mutate(protocol)
                protocol_path = root / "protocol.json"
                protocol_path.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(ValueError, "candidate|canonical"):
                    load_protocol(root, protocol_path)

    def test_atomic_text_pair_rolls_back_and_rejects_symlink_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            markdown = root / "report.md"
            report_json = root / "report.json"
            markdown.write_text("old markdown")
            report_json.write_text("old json")
            real_replace = os.replace
            replacements = 0

            def fail_second_publish(*args, **kwargs):
                nonlocal replacements
                replacements += 1
                if replacements == 4:
                    raise OSError("injected pair publication failure")
                return real_replace(*args, **kwargs)

            with mock.patch.object(
                phase0_module.os, "replace", side_effect=fail_second_publish
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    write_texts_atomic(
                        {markdown: "new markdown", report_json: "new json"}
                    )
            self.assertEqual(markdown.read_text(), "old markdown")
            self.assertEqual(report_json.read_text(), "old json")

            with self.assertRaisesRegex(RuntimeError, "post-publication drift"):
                write_texts_atomic(
                    {markdown: "new markdown", report_json: "new json"},
                    validate_after=lambda: (_ for _ in ()).throw(
                        RuntimeError("post-publication drift")
                    ),
                )
            self.assertEqual(markdown.read_text(), "old markdown")
            self.assertEqual(report_json.read_text(), "old json")

            fresh_markdown = root / "fresh.md"
            fresh_json = root / "fresh.json"
            fresh_markdown.write_text("concurrent writer")
            with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                write_texts_atomic(
                    {fresh_markdown: "ours", fresh_json: "ours"},
                    require_absent=True,
                )
            self.assertEqual(fresh_markdown.read_text(), "concurrent writer")
            self.assertFalse(fresh_json.exists())

            race_markdown = root / "race.md"
            race_json = root / "race.json"
            real_link = os.link

            def create_second_target_before_link(source, destination, **kwargs):
                if destination == race_json.name and not race_json.exists():
                    race_json.write_text("concurrent writer")
                return real_link(source, destination, **kwargs)

            with mock.patch.object(
                phase0_module.os,
                "link",
                side_effect=create_second_target_before_link,
            ):
                with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                    write_texts_atomic(
                        {race_markdown: "ours", race_json: "ours"},
                        require_absent=True,
                    )
            self.assertFalse(race_markdown.exists())
            self.assertEqual(race_json.read_text(), "concurrent writer")

            replaced_markdown = root / "replaced.md"
            replaced_json = root / "replaced.json"

            def replace_first_before_second_link(source, destination, **kwargs):
                if destination == replaced_json.name and not replaced_json.exists():
                    replaced_markdown.unlink()
                    replaced_markdown.write_text("concurrent first")
                    replaced_json.write_text("concurrent second")
                return real_link(source, destination, **kwargs)

            with mock.patch.object(
                phase0_module.os,
                "link",
                side_effect=replace_first_before_second_link,
            ):
                with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                    write_texts_atomic(
                        {replaced_markdown: "ours", replaced_json: "ours"},
                        require_absent=True,
                    )
            self.assertEqual(replaced_markdown.read_text(), "concurrent first")
            self.assertEqual(replaced_json.read_text(), "concurrent second")

            protected = root / "protected"
            protected.mkdir()
            alias = root / "alias"
            alias.symlink_to(protected, target_is_directory=True)
            with self.assertRaises(OSError):
                write_text_atomic(alias / "forbidden.txt", "no")
            self.assertFalse((protected / "forbidden.txt").exists())

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
            "context_encoder": {"weight": torch.tensor([1.0])},
            "target_encoder": {"weight": torch.tensor([2.0])},
            "predictor": {"weight": torch.tensor([3.0])},
        }
        checkpoint["model_state_sha256"] = checkpoint_model_state_sha256(checkpoint)
        checkpoint["best_loss_model_state_sha256"] = checkpoint[
            "model_state_sha256"
        ]
        checkpoint["best_healthy_model_state_sha256"] = None
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
            _, checkpoint_path = self._completed_run_checkpoints(Path(temporary))
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            checkpoint["completed_epochs"] = 2
            checkpoint["global_step"] = 4
            checkpoint["history"] = checkpoint["history"][:2]
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "completed run"):
                validate_completed_run(checkpoint_path)

    def test_completed_run_rejects_step_limited_partial_final_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, latest = self._completed_run_checkpoints(Path(temporary))
            checkpoint = torch.load(latest, weights_only=True)
            checkpoint["config"]["steps"] = 5
            checkpoint["global_step"] = 5
            checkpoint["history"][-1]["step"] = 5
            torch.save(checkpoint, latest)
            with self.assertRaisesRegex(ValueError, "epoch-boundary exact"):
                validate_completed_run(latest)

    def test_completed_run_detects_mutation_during_semantic_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, latest = self._completed_run_checkpoints(Path(temporary))
            real_validate = phase0_module._validate_history_and_selection

            def mutate_after_validation(checkpoint, *, require_completion):
                result = real_validate(
                    checkpoint, require_completion=require_completion
                )
                drifted = torch.load(latest, weights_only=True)
                drifted["global_step"] = 999
                torch.save(drifted, latest)
                return result

            with mock.patch.object(
                phase0_module,
                "_validate_history_and_selection",
                side_effect=mutate_after_validation,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during"):
                    validate_completed_run(latest)

    def test_selected_checkpoint_is_proven_from_terminal_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            result = validate_checkpoint_from_completed_run(selected, latest)
            self.assertEqual(result["selected_checkpoint"]["selection_role"], "best_loss")
            self.assertEqual(result["selected_checkpoint"]["completed_epochs"], 1)
            self.assertEqual(result["completed_run_checkpoint"]["completed_epochs"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(
                Path(temporary), healthy_epoch=2
            )
            result = validate_checkpoint_from_completed_run(selected, latest)
            self.assertEqual(result["selected_checkpoint"]["selection_role"], "best_healthy")

    def test_selected_checkpoint_rejects_selection_and_contract_drift(self):
        mutations = {
            "best loss epoch": lambda state: state.update(best_epoch=2),
            "configuration": lambda state: state["config"].update(seed=99),
            "mask groups": lambda state: state.update(mask_groups=[{"label": "other"}]),
            "architecture": lambda state: state.update(architecture="other-model"),
            "data contract": lambda state: state["data_contract"]["train_dataset"].update(
                sequence_count=5
            ),
        }
        for expected, mutate in mutations.items():
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as temporary,
            ):
                selected, latest = self._completed_run_checkpoints(Path(temporary))
                state = torch.load(latest, weights_only=True)
                mutate(state)
                torch.save(state, latest)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_checkpoint_from_completed_run(selected, latest)

    def test_selected_checkpoint_rejects_partial_or_unrelated_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            terminal = torch.load(latest, weights_only=True)
            terminal["completed_epochs"] = 2
            terminal["global_step"] = 4
            terminal["history"] = terminal["history"][:2]
            torch.save(terminal, latest)
            with self.assertRaisesRegex(ValueError, "completed run"):
                validate_checkpoint_from_completed_run(selected, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(selected, weights_only=True)
            state["history"][0]["val"]["subject_balanced_loss"] = 999.0
            torch.save(state, selected)
            with self.assertRaisesRegex(ValueError, "not a prefix"):
                validate_checkpoint_from_completed_run(selected, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            selected.write_bytes(latest.read_bytes())
            with self.assertRaisesRegex(ValueError, "best loss epoch"):
                validate_checkpoint_from_completed_run(selected, latest)

    def test_selected_checkpoint_detects_mutation_during_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            original_load = phase0_module.load_checkpoint
            mutated = False

            def racing_load(path):
                nonlocal mutated
                state = original_load(path)
                if Path(path).resolve() == selected.resolve() and not mutated:
                    drifted = dict(state)
                    drifted["global_step"] += 1
                    torch.save(drifted, selected)
                    mutated = True
                return state

            with mock.patch.object(phase0_module, "load_checkpoint", side_effect=racing_load):
                with self.assertRaisesRegex(RuntimeError, "changed during"):
                    validate_checkpoint_from_completed_run(selected, latest)

    def test_new_checkpoint_fingerprints_reject_weight_and_commitment_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(selected, weights_only=True)
            state["target_encoder"]["weight"] += 1
            torch.save(state, selected)
            with self.assertRaisesRegex(ValueError, "model-state fingerprint"):
                validate_checkpoint_from_completed_run(selected, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(selected, weights_only=True)
            state["target_encoder"]["weight"] += 1
            state["model_state_sha256"] = checkpoint_model_state_sha256(state)
            torch.save(state, selected)
            with self.assertRaisesRegex(
                ValueError, "model-state commitment|does not commit current state"
            ):
                validate_checkpoint_from_completed_run(selected, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(selected, weights_only=True)
            state["best_loss_model_state_sha256"] = "0" * 64
            torch.save(state, selected)
            with self.assertRaisesRegex(ValueError, "does not commit current state"):
                validate_checkpoint_from_completed_run(selected, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(latest, weights_only=True)
            state["history"][2]["val"]["subject_balanced_loss"] = 0.5
            state["best_epoch"] = 3
            state["best_val_loss"] = 0.5
            torch.save(state, latest)
            with self.assertRaisesRegex(ValueError, "does not commit current state"):
                validate_checkpoint_from_completed_run(latest, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(latest, weights_only=True)
            state.pop("best_loss_model_state_sha256")
            torch.save(state, latest)
            with self.assertRaisesRegex(
                ValueError, "best-model commitment|best_loss_model_state_sha256 is invalid"
            ):
                validate_checkpoint_from_completed_run(selected, latest)

    def test_terminal_recomputes_first_minimum_and_health_formula(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(latest, weights_only=True)
            state["history"][1]["val"]["subject_balanced_loss"] = 1.0
            state["best_epoch"] = 2
            state["best_val_loss"] = 1.0
            torch.save(state, latest)
            with self.assertRaisesRegex(ValueError, "best loss epoch"):
                validate_checkpoint_from_completed_run(selected, latest)
        with tempfile.TemporaryDirectory() as temporary:
            selected, latest = self._completed_run_checkpoints(Path(temporary))
            state = torch.load(latest, weights_only=True)
            state["history"][0]["val"].update(
                representations_healthy=True,
                health_issues=[],
            )
            state["best_healthy_epoch"] = 1
            state["best_healthy_val_loss"] = 1.0
            state["best_healthy_model_state_sha256"] = state[
                "best_loss_model_state_sha256"
            ]
            torch.save(state, latest)
            with self.assertRaisesRegex(ValueError, "representation health"):
                validate_checkpoint_from_completed_run(selected, latest)

    def test_completed_run_enforces_paths_counters_and_legacy_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, latest = self._completed_run_checkpoints(root)
            renamed = latest.with_name("terminal.pt")
            renamed.write_bytes(latest.read_bytes())
            with self.assertRaisesRegex(ValueError, "named latest.pt"):
                validate_completed_run(renamed)
            unsupported = selected.with_name("winner.pt")
            unsupported.write_bytes(selected.read_bytes())
            with self.assertRaisesRegex(ValueError, "unsupported filename"):
                validate_checkpoint_from_completed_run(unsupported, latest)
            alias = root / "best_loss-alias"
            alias.mkdir()
            (alias / "latest.pt").write_bytes(latest.read_bytes())
            with self.assertRaisesRegex(ValueError, "completed-run checkpoint path"):
                validate_checkpoint_from_completed_run(alias / "latest.pt", latest)
            selected.unlink()
            selected.symlink_to(latest)
            with self.assertRaisesRegex(ValueError, "distinct from latest"):
                validate_checkpoint_from_completed_run(selected, latest)
            selected.unlink()
            state = torch.load(latest, weights_only=True)
            state["global_step"] = 7
            torch.save(state, latest)
            with self.assertRaisesRegex(ValueError, "exceeds configured steps"):
                validate_completed_run(latest)
        protocol = load_protocol(PROJECT_ROOT)
        selected = PROJECT_ROOT / "outputs" / "jepa-v4" / "best_loss.pt"
        latest = PROJECT_ROOT / "outputs" / "jepa-v4" / "latest.pt"
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint schema"):
            validate_checkpoint_from_completed_run(selected, latest)
        result = validate_checkpoint_from_completed_run(selected, latest, protocol)
        self.assertEqual(result["selected_checkpoint"]["schema"], LEGACY_CHECKPOINT_SCHEMA)
        with tempfile.TemporaryDirectory() as temporary:
            forged_path = Path(temporary) / "best_loss.pt"
            torch.save(phase0_module.load_checkpoint(selected), forged_path)
            forged_protocol = json.loads(json.dumps(protocol))
            forged_protocol["candidate_checkpoints"]["best_loss.pt"]["sha256"] = (
                checkpoint_sha256(forged_path)
            )
            with self.assertRaisesRegex(ValueError, "locked legacy checkpoint hash"):
                checkpoint_record(forged_path, forged_protocol, "best_loss.pt")

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
