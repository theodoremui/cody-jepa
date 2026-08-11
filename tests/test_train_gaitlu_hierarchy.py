import csv
import io
from pathlib import Path
import pickle
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cody_jepa.cli.train_gaitlu_hierarchy import parse_args, run_registered_training
from cody_jepa.data.gaitlu import FixedExposureSampler, GaitLULoaderConfig
from cody_jepa.data.gaitlu import build_gaitlu_datasets_from_config
from cody_jepa.data.gaitlu_hierarchy import (
    HIERARCHY_REGISTRY_COLUMNS,
    finalize_gaitlu_hierarchy,
    read_hierarchy_registry,
)
from cody_jepa.data.gaitlu_prepare import (
    FINAL_INVENTORY_COLUMNS,
    TRAINING_REGISTRY_COLUMNS,
    pack_gaitlu_shard,
)
from cody_jepa.training.checkpoint import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "train" / "gaitlu_hierarchy_smoke.json"


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _synthetic_sequence(seed, frames=40, height=8, width=8):
    rng = np.random.default_rng(seed)
    return (rng.random((frames, height, width)) > 0.72).astype(np.uint8) * 255


def _write_raw_shard(path, count=12):
    with tarfile.open(path, "w:gz") as archive:
        for index in range(count):
            payload = pickle.dumps([_synthetic_sequence(index)], protocol=pickle.HIGHEST_PROTOCOL)
            member = tarfile.TarInfo(f"{index:03d}/030/000/000.pkl")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def _prepared_hierarchy(root):
    raw = root / "gaitlu-000.tar.gz"
    prepared = root / "prepared"
    _write_raw_shard(raw)
    pack_gaitlu_shard(raw, prepared, trust_pickles=True)
    inventory = _read_csv(prepared / "inventories" / "gaitlu-000.csv")
    finalized = []
    for row in inventory:
        finalized.append(
            {
                **row,
                "source_group": f"group-{row['sequence_id']}",
                "duplicate_of": "",
                "eligible": "true",
                "cohort": "training",
            }
        )
    _write_csv(prepared / "inventory.csv", FINAL_INVENTORY_COLUMNS, finalized)
    finalize_gaitlu_hierarchy(
        prepared,
        training_exposure=64,
        holdout_target=2,
        holdout_seed=91,
        pool_seeds=tuple(range(8)),
        low_target=2,
        high_target=6,
    )
    return prepared, prepared / "hierarchy" / "training_registry.csv"


def _args(registry, prepared, output_root, *, run_index=0, config=SMOKE_CONFIG):
    return parse_args(
        [
            "--registry",
            str(registry),
            "--run-index",
            str(run_index),
            "--config",
            str(config),
            "--data-root",
            str(prepared),
            "--output-root",
            str(output_root),
            "--repo-root",
            str(REPO_ROOT),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )


class TrainGaitLUHierarchyCLITest(unittest.TestCase):
    def test_valid_row_selection_and_loader_seed_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, registry = _prepared_hierarchy(root)
            rows = read_hierarchy_registry(registry)
            captured = {}

            def fake_run(train_args, *, config_updates):
                captured["args"] = train_args
                captured["updates"] = config_updates
                train_args.output_dir.mkdir(parents=True)
                (train_args.output_dir / "latest.pt").write_bytes(b"checkpoint")
                return {
                    "global_step": 1,
                    "completed_epochs": 1,
                    "examples_per_second": 1.0,
                }

            with patch(
                "cody_jepa.cli.train_gaitlu_hierarchy.run_training",
                side_effect=fake_run,
            ):
                result = run_registered_training(
                    _args(registry, prepared, root / "outputs", run_index=2)
                )

            row = rows[2]
            self.assertEqual(result["model_label"], row["model_label"])
            self.assertEqual(captured["updates"]["seed"], row["optimization_seed"])
            self.assertEqual(
                captured["updates"]["train_window_policy"], row["window_policy"]
            )
            self.assertEqual(captured["updates"]["anchor_spacing"], row["anchor_spacing"])
            self.assertEqual(captured["updates"]["replicate_seed"], row["replicate_seed"])
            self.assertEqual(
                captured["args"].train_manifest.resolve(),
                (registry.parent / row["train_manifest"]).resolve(),
            )

    def test_rejects_old_incomplete_and_duplicate_registries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, registry = _prepared_hierarchy(root)
            old = root / "old-registry.csv"
            _write_csv(old, TRAINING_REGISTRY_COLUMNS, [])
            with self.assertRaisesRegex(ValueError, "hierarchy registry must have exactly"):
                run_registered_training(_args(old, prepared, root / "old-output"))

            for mutation, message in (("incomplete", "exactly 32"), ("duplicate", "duplicate cell")):
                with self.subTest(mutation=mutation):
                    rows = _read_csv(registry)
                    if mutation == "incomplete":
                        rows.pop()
                    else:
                        rows[-1].update(
                            {
                                "replicate": rows[0]["replicate"],
                                "sequence_support": rows[0]["sequence_support"],
                                "window_policy": rows[0]["window_policy"],
                            }
                        )
                    mutated = root / f"{mutation}.csv"
                    _write_csv(mutated, HIERARCHY_REGISTRY_COLUMNS, rows)
                    with self.assertRaisesRegex(ValueError, message):
                        run_registered_training(
                            _args(mutated, prepared, root / f"{mutation}-output")
                        )

    def test_rejects_exposure_mismatch_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, registry = _prepared_hierarchy(root)
            full_config = REPO_ROOT / "configs" / "train" / "gaitlu_hierarchy_full.json"
            with self.assertRaisesRegex(ValueError, "registry declares"):
                run_registered_training(
                    _args(registry, prepared, root / "mismatch", config=full_config)
                )

            first = read_hierarchy_registry(registry)[0]
            existing = root / "existing" / first["model_label"]
            existing.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_registered_training(_args(registry, prepared, root / "existing"))

    def test_rejects_early_training_return_and_missing_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, registry = _prepared_hierarchy(root)
            incomplete_result = {
                "global_step": 0,
                "completed_epochs": 0,
                "examples_per_second": 0.0,
            }
            with (
                patch(
                    "cody_jepa.cli.train_gaitlu_hierarchy.run_training",
                    return_value=incomplete_result,
                ),
                self.assertRaisesRegex(RuntimeError, "before the declared final step"),
            ):
                run_registered_training(_args(registry, prepared, root / "early"))

            complete_without_checkpoint = {
                "global_step": 1,
                "completed_epochs": 1,
                "examples_per_second": 1.0,
            }
            with (
                patch(
                    "cody_jepa.cli.train_gaitlu_hierarchy.run_training",
                    return_value=complete_without_checkpoint,
                ),
                self.assertRaisesRegex(RuntimeError, "without writing"),
            ):
                run_registered_training(_args(registry, prepared, root / "missing"))


class GaitLUHierarchyFourCellSmokeTest(unittest.TestCase):
    def test_four_cells_finish_one_real_cpu_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, registry = _prepared_hierarchy(root)
            rows = read_hierarchy_registry(registry)[:4]
            output_root = root / "smoke-runs"
            checkpoints = {}

            for run_index, row in enumerate(rows):
                with self.subTest(cell=(row["sequence_support"], row["window_policy"])):
                    result = run_registered_training(
                        _args(registry, prepared, output_root, run_index=run_index)
                    )
                    checkpoint_path = Path(result["checkpoint_path"])
                    self.assertTrue(checkpoint_path.is_file())
                    state = load_checkpoint(checkpoint_path)
                    self.assertEqual(state["global_step"], 1)
                    loader_contract = state["data_contract"]["loader_config"]
                    self.assertEqual(
                        loader_contract["train_window_policy"], row["window_policy"]
                    )
                    self.assertEqual(loader_contract["anchor_spacing"], 8)
                    self.assertEqual(loader_contract["replicate_seed"], row["replicate_seed"])
                    self.assertEqual(
                        state["data_contract"]["train_dataset"]["manifest_name"],
                        Path(row["train_manifest"]).name,
                    )
                    checkpoints[(row["sequence_support"], row["window_policy"])] = state

            for support in ("low", "high"):
                frozen_state = checkpoints[(support, "frozen_random")]
                resampled_state = checkpoints[(support, "resampled_anchor")]
                self.assertEqual(frozen_state["mask_rng_state"], resampled_state["mask_rng_state"])

                paired_rows = [row for row in rows if row["sequence_support"] == support]
                datasets = []
                samplers = []
                for row in paired_rows:
                    config = GaitLULoaderConfig(
                        train_manifest_csv=registry.parent / row["train_manifest"],
                        val_manifest_csv=registry.parent / row["val_manifest"],
                        data_root=prepared,
                        clip_length=16,
                        image_size=(16, 16),
                        seed=row["optimization_seed"],
                        batch_size=16,
                        num_workers=0,
                        pin_memory=False,
                        prefetch_factor=1,
                        epoch_examples=64,
                        train_window_policy=row["window_policy"],
                        anchor_spacing=row["anchor_spacing"],
                        replicate_seed=row["replicate_seed"],
                    )
                    train, _ = build_gaitlu_datasets_from_config(config)
                    datasets.append(train)
                    samplers.append(list(FixedExposureSampler(train, 64, config.seed)))
                self.assertEqual(samplers[0], samplers[1])
                self.assertEqual(
                    [datasets[0][index]["sequence_id"] for index in samplers[0]],
                    [datasets[1][index]["sequence_id"] for index in samplers[1]],
                )
                probe = torch.arange(16 * 8 * 8, dtype=torch.float32).reshape(16, 1, 8, 8)
                for sample_index, draw_index in samplers[0][:8]:
                    self.assertTrue(
                        torch.equal(
                            datasets[0]._transform(
                                probe.clone(), sample_index, draw_index
                            ),
                            datasets[1]._transform(
                                probe.clone(), sample_index, draw_index
                            ),
                        )
                    )

                frozen, resampled = datasets
                frozen_starts = {frozen[(0, draw)]["window_start"] for draw in range(32)}
                resampled_starts = {
                    resampled[(0, draw)]["window_start"] for draw in range(64)
                }
                self.assertEqual(len(frozen_starts), 1)
                self.assertLessEqual(resampled_starts, {0, 8, 16, 24})
                self.assertGreater(len(resampled_starts), 1)


if __name__ == "__main__":
    unittest.main()
