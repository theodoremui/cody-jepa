import copy
from contextlib import redirect_stderr
import csv
from dataclasses import fields
import io
import inspect
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cody_jepa.cli.train import parse_args as parse_train_args, run_training
from cody_jepa.cli.train_gaitlu_study import parse_args as parse_study_args
from cody_jepa.data import (
    GaitLUIndexedDataset,
    GaitLULoaderConfig,
    ManifestValidationError,
    build_gaitlu_datasets_from_config,
    build_gaitlu_loaders_from_config,
)
from cody_jepa.data import gaitlu
from cody_jepa.data.gaitlu import (
    FixedExposureSampler,
    GAITLU_MANIFEST_COLUMNS,
    GAITLU_MANIFEST_VERSION,
    gaitlu_manifest_pair_sha256,
)
from cody_jepa.data.gaitlu_prepare import (
    INVENTORY_COLUMNS,
    TRAINING_REGISTRY_COLUMNS,
    finalize_gaitlu_study,
    pack_gaitlu_shard,
    validate_and_pack_sequence,
)
from cody_jepa.masks import DEFAULT_MASK_GROUPS, multiblock_mask
from cody_jepa.training.checkpoint import (
    CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    validate_resume_state,
)


def synthetic_sequence(seed, frames=20, height=8, width=6):
    rng = np.random.default_rng(seed)
    sequence = (rng.random((frames, height, width)) > 0.72).astype(np.uint8) * 255
    return sequence


def write_raw_shard(path, shard_index, sequences):
    with tarfile.open(path, "w:gz") as archive:
        for index, sequence in enumerate(sequences):
            view = shard_index * 100 + index
            member_name = f"{shard_index:03d}/030/{view:03d}/{view:03d}.pkl"
            payload = pickle.dumps([sequence], protocol=pickle.HIGHEST_PROTOCOL)
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_sequence_ids(path):
    return {row["sequence_id"] for row in read_csv(path)}


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_one_shard(root, sequences=None):
    sequences = sequences or [synthetic_sequence(index) for index in range(4)]
    raw = root / "gaitlu-000.tar.gz"
    prepared = root / "prepared"
    write_raw_shard(raw, 0, sequences)
    summary = pack_gaitlu_shard(raw, prepared, trust_pickles=True)
    inventory = read_csv(Path(summary["inventory"]))
    rows = []
    for row in inventory:
        rows.append(
            {
                column: (
                    row["sequence_id"]
                    if column == "source_group"
                    else "train"
                    if column == "split"
                    else row[column]
                )
                for column in GAITLU_MANIFEST_COLUMNS
            }
        )
    manifest = prepared / "train.csv"
    write_csv(manifest, GAITLU_MANIFEST_COLUMNS, rows)
    return prepared, manifest, inventory


def prepare_study(root):
    raw = root / "raw"
    prepared = root / "prepared"
    raw.mkdir()
    sequences = [synthetic_sequence(index) for index in range(12)]
    sequences[1] = sequences[0].copy()  # within-shard storage reuse
    sequences[-1] = sequences[2].copy()  # cross-shard finalization deduplication
    summaries = []
    for shard_index in range(2):
        path = raw / f"gaitlu-{shard_index:03d}.tar.gz"
        write_raw_shard(
            path,
            shard_index,
            sequences[shard_index * 6 : (shard_index + 1) * 6],
        )
        summaries.append(pack_gaitlu_shard(path, prepared, trust_pickles=True))
    study = finalize_gaitlu_study(
        prepared,
        holdout_size=2,
        holdout_seed=9,
        pool_seeds=(10, 11, 12, 13, 14),
        pool_sizes=(2, 4, 6),
        training_exposure=64,
        expected_shards=2,
    )
    return prepared, summaries, study


def loader_config(prepared, *, num_workers):
    return GaitLULoaderConfig(
        train_manifest_csv="manifests/ladder-0-small.csv",
        val_manifest_csv="manifests/common-holdout.csv",
        data_root=prepared,
        clip_length=16,
        image_size=(12, 12),
        seed=1234,
        batch_size=2,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=1,
        train_crop_scale=(0.65, 1.0),
        train_horizontal_flip_prob=0.5,
        epoch_examples=12,
    )


def collect_batches(loader):
    return [
        {
            "video": batch["video"].clone(),
            "sequence_id": list(batch["sequence_id"]),
            "window_start": batch["window_start"].clone(),
        }
        for batch in loader
    ]


def assert_batches_equal(test_case, first, second):
    test_case.assertEqual(len(first), len(second))
    for left, right in zip(first, second):
        test_case.assertEqual(left["sequence_id"], right["sequence_id"])
        test_case.assertTrue(torch.equal(left["window_start"], right["window_start"]))
        test_case.assertTrue(torch.equal(left["video"], right["video"]))


def count_key(value, wanted):
    if isinstance(value, dict):
        return sum(key == wanted for key in value) + sum(
            count_key(child, wanted) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(count_key(child, wanted) for child in value)
    return 0


class GaitLUPreparationTest(unittest.TestCase):
    def test_validation_retains_preparation_content_hashes(self):
        packed = validate_and_pack_sequence([synthetic_sequence(1)])
        self.assertEqual(packed["shape"], (20, 8, 6))
        self.assertRegex(packed["content_sha256"], r"^[0-9a-f]{64}$")

        with self.assertRaisesRegex(ValueError, "at least 16"):
            validate_and_pack_sequence(synthetic_sequence(2, frames=15))
        with self.assertRaisesRegex(ValueError, "empty frames"):
            validate_and_pack_sequence(np.zeros((20, 8, 6), dtype=np.uint8))
        grayscale = np.linspace(0, 255, 20 * 8 * 6).reshape(20, 8, 6)
        with self.assertRaisesRegex(ValueError, "not binary-looking"):
            validate_and_pack_sequence(grayscale)

    def test_pack_requires_explicit_pickle_trust(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "gaitlu-000.tar.gz"
            write_raw_shard(raw, 0, [synthetic_sequence(0)])
            with self.assertRaisesRegex(PermissionError, "pickle loading is unsafe"):
                pack_gaitlu_shard(raw, root / "prepared")

    def test_finalize_rejects_source_groups_with_outer_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "gaitlu-000.tar.gz"
            prepared = root / "prepared"
            write_raw_shard(
                raw, 0, [synthetic_sequence(index) for index in range(8)]
            )
            pack_gaitlu_shard(raw, prepared, trust_pickles=True)
            inventory = read_csv(prepared / "inventories" / "gaitlu-000.csv")
            groups = root / "source-groups.csv"
            rows = [
                {
                    "sequence_id": row["sequence_id"],
                    "source_group": (
                        f" {row['sequence_id']} " if index == 0 else row["sequence_id"]
                    ),
                }
                for index, row in enumerate(inventory)
            ]
            write_csv(groups, ("sequence_id", "source_group"), rows)

            with self.assertRaisesRegex(ValueError, "source-group mapping on row 2"):
                finalize_gaitlu_study(
                    prepared,
                    holdout_size=2,
                    pool_seeds=(10, 11, 12, 13, 14),
                    pool_sizes=(1, 2, 3),
                    source_groups_csv=groups,
                    expected_shards=1,
                )

    def test_finalize_requires_strictly_increasing_pool_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "gaitlu-000.tar.gz"
            prepared = root / "prepared"
            write_raw_shard(
                raw, 0, [synthetic_sequence(index) for index in range(8)]
            )
            pack_gaitlu_shard(raw, prepared, trust_pickles=True)

            for pool_sizes in ((2, 2, 6), (0, 2, 6)):
                with (
                    self.subTest(pool_sizes=pool_sizes),
                    self.assertRaisesRegex(ValueError, "positive, strictly increasing"),
                ):
                    finalize_gaitlu_study(
                        prepared,
                        pool_seeds=(10, 11, 12, 13, 14),
                        pool_sizes=pool_sizes,
                        expected_shards=1,
                    )

    def test_finalize_rejects_duplicate_rungs_from_coarse_source_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "gaitlu-000.tar.gz"
            prepared = root / "prepared"
            write_raw_shard(
                raw, 0, [synthetic_sequence(index) for index in range(8)]
            )
            pack_gaitlu_shard(raw, prepared, trust_pickles=True)
            inventory = read_csv(prepared / "inventories" / "gaitlu-000.csv")
            groups = root / "source-groups.csv"
            write_csv(
                groups,
                ("sequence_id", "source_group"),
                [
                    {
                        "sequence_id": row["sequence_id"],
                        "source_group": f"group-{index // 2}",
                    }
                    for index, row in enumerate(inventory)
                ],
            )

            with self.assertRaisesRegex(ValueError, "strictly nested pool rungs"):
                finalize_gaitlu_study(
                    prepared,
                    holdout_size=2,
                    pool_seeds=(10, 11, 12, 13, 14),
                    pool_sizes=(1, 2, 3),
                    source_groups_csv=groups,
                    expected_shards=1,
                )

    def test_v2_finalize_reuses_and_deduplicates_content_without_runtime_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, summaries, study = prepare_study(Path(directory))

            first_inventory = read_csv(Path(summaries[0]["inventory"]))
            self.assertEqual(tuple(first_inventory[0]), INVENTORY_COLUMNS)
            self.assertRegex(first_inventory[0]["content_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                first_inventory[0]["content_sha256"],
                first_inventory[1]["content_sha256"],
            )
            self.assertEqual(
                (first_inventory[0]["record_offset"], first_inventory[0]["record_size"]),
                (first_inventory[1]["record_offset"], first_inventory[1]["record_size"]),
            )
            self.assertEqual(summaries[0]["unique_records_in_shard"], 5)

            self.assertEqual(study["version"], "gaitlu-scaling-pools-v2")
            self.assertEqual(study["exact_duplicates"], 2)
            self.assertEqual(study["holdout_sequences"], 2)
            self.assertEqual(len(study["pools"]), 20)
            self.assertEqual(count_key(study, "manifest_sha256"), 0)
            self.assertFalse(
                any(key.endswith("_sha256") for key in study),
                json.dumps(study, indent=2),
            )
            on_disk_summary = json.loads((prepared / "study_pools.json").read_text())
            self.assertEqual(on_disk_summary, study)
            self.assertNotIn("_sha256", (prepared / "study_pools.json").read_text())

            final_inventory = read_csv(prepared / "inventory.csv")
            duplicates = [row for row in final_inventory if row["duplicate_of"]]
            self.assertEqual(len(duplicates), 2)
            self.assertTrue(all(row["content_sha256"] for row in final_inventory))

            for manifest in (prepared / "manifests").glob("*.csv"):
                with manifest.open(newline="") as handle:
                    fieldnames = tuple(csv.DictReader(handle).fieldnames or ())
                self.assertEqual(fieldnames, GAITLU_MANIFEST_COLUMNS)
                self.assertNotIn("content_sha256", fieldnames)
            self.assertEqual(GAITLU_MANIFEST_VERSION, "gaitlu-indexed-bitpack-v2")

            registry_path = prepared / "training_registry.csv"
            registry = read_csv(registry_path)
            self.assertEqual(tuple(registry[0]), TRAINING_REGISTRY_COLUMNS)
            self.assertEqual(len(registry), 20)
            for row in registry:
                self.assertEqual(
                    row["manifest_sha256"],
                    gaitlu_manifest_pair_sha256(
                        prepared / row["train_manifest"],
                        prepared / row["val_manifest"],
                    ),
                )
            full_digests = {
                row["manifest_sha256"] for row in registry if row["rung"] == "full"
            }
            self.assertEqual(len(full_digests), 1)

            for ladder in range(5):
                small = read_sequence_ids(
                    prepared / "manifests" / f"ladder-{ladder}-small.csv"
                )
                medium = read_sequence_ids(
                    prepared / "manifests" / f"ladder-{ladder}-medium.csv"
                )
                large = read_sequence_ids(
                    prepared / "manifests" / f"ladder-{ladder}-large.csv"
                )
                full = read_sequence_ids(
                    prepared / "manifests" / f"ladder-{ladder}-full.csv"
                )
                self.assertTrue(small < medium < large <= full)


class GaitLURuntimeValidationTest(unittest.TestCase):
    def test_runtime_api_and_clis_have_no_record_verification_surface(self):
        self.assertNotIn("verify_records", inspect.signature(GaitLUIndexedDataset).parameters)
        self.assertNotIn("verify_records", {field.name for field in fields(GaitLULoaderConfig)})
        self.assertNotIn("verify_records", GaitLULoaderConfig.__annotations__)
        self.assertFalse(hasattr(gaitlu, "packed_sequence_sha256"))

        train_required = [
            "--config", "config.json", "--output-dir", "output",
        ]
        train_args = parse_train_args(train_required)
        self.assertFalse(hasattr(train_args, "record_verify_mode"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_train_args([*train_required, "--record-verify-mode", "all"])

        study_required = [
            "--registry", "registry.csv", "--config", "config.json",
            "--data-root", "data", "--output-root", "output",
        ]
        study_args = parse_study_args(study_required)
        self.assertFalse(hasattr(study_args, "record_verify_mode"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_study_args([*study_required, "--record-verify-mode", "all"])

    def test_loader_preserves_structural_manifest_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(Path(directory))
            valid_rows = read_csv(manifest)
            shard = prepared / valid_rows[0]["shard_path"]
            mutations = {
                "unsafe relative path": ("shard_path", "../escape.tar", "safe relative"),
                "missing shard": ("shard_path", "shards/missing.tar", "missing"),
                "invalid dimension": ("width", "0", "must be positive"),
                "invalid record size": (
                    "record_size", str(int(valid_rows[0]["record_size"]) + 1), "expected"
                ),
                "out-of-bounds offset": (
                    "record_offset", str(shard.stat().st_size), "extends beyond"
                ),
            }
            for label, (key, value, message) in mutations.items():
                with self.subTest(label=label):
                    rows = copy.deepcopy(valid_rows)
                    rows[0][key] = value
                    write_csv(manifest, GAITLU_MANIFEST_COLUMNS, rows)
                    with self.assertRaisesRegex(ManifestValidationError, message):
                        GaitLUIndexedDataset(
                            manifest, data_root=prepared, split="train", clip_length=16
                        )

            rows = copy.deepcopy(valid_rows)
            rows[0]["content_sha256"] = "0" * 64
            wrong_columns = (*GAITLU_MANIFEST_COLUMNS, "content_sha256")
            write_csv(manifest, wrong_columns, rows)
            with self.assertRaisesRegex(ManifestValidationError, "exactly these columns"):
                GaitLUIndexedDataset(
                    manifest, data_root=prepared, split="train", clip_length=16
                )

    def test_short_reads_still_fail_after_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(Path(directory))
            row = read_csv(manifest)[0]
            dataset = GaitLUIndexedDataset(
                manifest, data_root=prepared, split="train", clip_length=16
            )
            shard = prepared / row["shard_path"]
            with shard.open("r+b") as handle:
                handle.truncate(int(row["record_offset"]) + int(row["record_size"]) - 1)
            with self.assertRaisesRegex(ManifestValidationError, "short read"):
                dataset[0]

    def test_same_length_content_corruption_is_not_hashed_at_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(Path(directory))
            row = read_csv(manifest)[0]
            shard = prepared / row["shard_path"]
            with shard.open("r+b") as handle:
                handle.seek(int(row["record_offset"]))
                byte = handle.read(1)
                handle.seek(int(row["record_offset"]))
                handle.write(bytes([byte[0] ^ 1]))
            dataset = GaitLUIndexedDataset(
                manifest, data_root=prepared, split="train", clip_length=16
            )
            self.assertEqual(tuple(dataset[0]["video"].shape), (16, 1, 112, 112))
            self.assertFalse(
                any("sha256" in key or "hash" in key or "path" in key for key in dataset.samples[0]),
                dataset.samples[0],
            )


class GaitLUDeterminismTest(unittest.TestCase):
    def test_splitmix64_seed_golden_vectors_and_coordinate_separation(self):
        vectors = {
            (0, 0, 0, 0, 1): 12_916_091_066_660_601_997,
            (42, 3, 7, 11, 2): 12_196_335_617_894_468_679,
            (-1, 2**63, 999, 4, 3): 2_563_875_528_026_691_382,
            (42, 3, 7, 11, 4): 18_140_374_009_462_029_608,
        }
        actual = {coordinates: gaitlu._gaitlu_seed(*coordinates) for coordinates in vectors}
        self.assertEqual(actual, vectors)

        base = (123, 4, 5, 6, 2)
        seeds = {gaitlu._gaitlu_seed(*base)}
        for coordinate in range(1, 5):
            changed = list(base)
            changed[coordinate] += 1
            seeds.add(gaitlu._gaitlu_seed(*changed))
        self.assertEqual(len(seeds), 5)
        self.assertTrue(all(0 <= seed <= 2**64 - 1 for seed in seeds))

    def test_sampler_windows_and_transforms_ignore_global_rng_state(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, _, _ = prepare_study(Path(directory))
            config = loader_config(prepared, num_workers=0)
            first, _ = build_gaitlu_datasets_from_config(config)
            first.set_epoch(3)
            first_draws = list(FixedExposureSampler(first, 20, config.seed))
            first_examples = [first[index] for index in first_draws]

            random.seed(999)
            np.random.seed(888)
            torch.manual_seed(777)
            second, _ = build_gaitlu_datasets_from_config(config)
            second.set_epoch(3)
            second_draws = list(FixedExposureSampler(second, 20, config.seed))
            second_examples = [second[index] for index in second_draws]

            self.assertEqual(first_draws, second_draws)
            self.assertGreater(len({item["window_start"] for item in first_examples}), 1)
            for left, right in zip(first_examples, second_examples):
                self.assertEqual(left["window_start"], right["window_start"])
                self.assertTrue(torch.equal(left["video"], right["video"]))

    def test_loaders_match_with_workers_and_parent_preread(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, _, _ = prepare_study(Path(directory))
            single_config = loader_config(prepared, num_workers=0)
            single_datasets = build_gaitlu_datasets_from_config(single_config)
            single_train, single_val = build_gaitlu_loaders_from_config(
                single_config, datasets=single_datasets
            )
            expected_train = collect_batches(single_train)
            expected_val = collect_batches(single_val)

            worker_config = loader_config(prepared, num_workers=2)
            worker_datasets = build_gaitlu_datasets_from_config(worker_config)
            _ = worker_datasets[0][(0, 91)]  # open/read in parent before fork
            _ = worker_datasets[1][0]
            worker_train, worker_val = build_gaitlu_loaders_from_config(
                worker_config, datasets=worker_datasets
            )
            assert_batches_equal(self, expected_train, collect_batches(worker_train))
            assert_batches_equal(self, expected_val, collect_batches(worker_val))


class GaitLUHierarchyWindowTest(unittest.TestCase):
    def _dataset(self, manifest, prepared, policy, *, seed=31, replicate_seed=47):
        return GaitLUIndexedDataset(
            manifest,
            data_root=prepared,
            split="train",
            clip_length=16,
            image_size=(12, 12),
            random_windows=True,
            base_seed=seed,
            crop_scale=(0.65, 1.0),
            horizontal_flip_prob=0.5,
            window_policy=policy,
            anchor_spacing=8,
            replicate_seed=replicate_seed,
        )

    def test_anchor_grid_and_hierarchy_option_validation(self):
        expected = {
            16: (0,),
            24: (0, 8),
            32: (0, 8, 16),
            40: (0, 8, 16, 24),
        }
        self.assertEqual(
            {
                frames: gaitlu._allowed_temporal_anchors(frames, 16, 8)
                for frames in expected
            },
            expected,
        )

        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(
                Path(directory),
                [synthetic_sequence(index, frames=frames) for index, frames in enumerate(expected)],
            )
            with self.assertRaisesRegex(
                ManifestValidationError, "at least two allowed temporal anchors"
            ):
                self._dataset(manifest, prepared, "frozen_random")

            with self.assertRaisesRegex(ValueError, "anchor_spacing must be positive"):
                GaitLULoaderConfig(
                    train_manifest_csv=manifest,
                    val_manifest_csv=manifest,
                    data_root=prepared,
                    anchor_spacing=0,
                )
            with self.assertRaisesRegex(ValueError, "unknown hierarchy window policy"):
                GaitLULoaderConfig(
                    train_manifest_csv=manifest,
                    val_manifest_csv=manifest,
                    data_root=prepared,
                    train_window_policy="random_anchor",
                )

            val_manifest = prepared / "val.csv"
            val_rows = read_csv(manifest)
            for row in val_rows:
                row["split"] = "val"
            write_csv(val_manifest, GAITLU_MANIFEST_COLUMNS, val_rows)
            with self.assertRaisesRegex(ValueError, "accepted only for training"):
                GaitLUIndexedDataset(
                    val_manifest,
                    data_root=prepared,
                    split="val",
                    window_policy="frozen_random",
                )

    def test_frozen_anchor_is_stable_across_draws_epochs_workers_and_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(
                Path(directory),
                [synthetic_sequence(index, frames=40) for index in range(3)],
            )
            rows = read_csv(manifest)
            target_id = rows[0]["sequence_id"]
            low_manifest = prepared / "low.csv"
            high_manifest = prepared / "high.csv"
            write_csv(low_manifest, GAITLU_MANIFEST_COLUMNS, rows[:2])
            write_csv(high_manifest, GAITLU_MANIFEST_COLUMNS, list(reversed(rows)))
            low = self._dataset(low_manifest, prepared, "frozen_random")
            high = self._dataset(high_manifest, prepared, "frozen_random")
            low_index = next(
                index
                for index, sample in enumerate(low.samples)
                if sample["sequence_id"] == target_id
            )
            high_index = next(
                index
                for index, sample in enumerate(high.samples)
                if sample["sequence_id"] == target_id
            )

            starts = set()
            for epoch in (0, 3, 11):
                low.set_epoch(epoch)
                starts.update(low[(low_index, draw)]["window_start"] for draw in range(12))
            high.set_epoch(19)
            starts.update(high[(high_index, draw)]["window_start"] for draw in range(12))
            self.assertEqual(len(starts), 1)

            low.set_epoch(23)
            worker_copy = pickle.loads(pickle.dumps(low))
            worker_starts = {
                worker_copy[(low_index, draw)]["window_start"] for draw in range(8)
            }
            self.assertEqual(worker_starts, starts)

            worker_loader = torch.utils.data.DataLoader(
                low,
                batch_size=4,
                sampler=[(low_index, draw) for draw in range(8)],
                num_workers=2,
            )
            loader_starts = {
                int(value)
                for batch in worker_loader
                for value in batch["window_start"]
            }
            self.assertEqual(loader_starts, starts)

            starts_by_seed = {
                self._dataset(
                    low_manifest,
                    prepared,
                    "frozen_random",
                    replicate_seed=replicate_seed,
                )[(low_index, 0)]["window_start"]
                for replicate_seed in range(32)
            }
            self.assertGreater(len(starts_by_seed), 1)

    def test_resampled_anchors_are_allowed_variable_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(
                Path(directory), [synthetic_sequence(0, frames=40)]
            )
            first = self._dataset(manifest, prepared, "resampled_anchor")
            first.set_epoch(5)
            starts = [first[(0, draw)]["window_start"] for draw in range(64)]
            self.assertTrue(set(starts) <= {0, 8, 16, 24})
            self.assertGreater(len(set(starts)), 1)

            random.seed(901)
            np.random.seed(902)
            torch.manual_seed(903)
            repeated = self._dataset(manifest, prepared, "resampled_anchor")
            repeated.set_epoch(5)
            self.assertEqual(
                starts,
                [repeated[(0, draw)]["window_start"] for draw in range(64)],
            )
            repeated.set_epoch(6)
            next_epoch = [repeated[(0, draw)]["window_start"] for draw in range(64)]
            self.assertTrue(set(next_epoch) <= {0, 8, 16, 24})
            self.assertNotEqual(starts, next_epoch)

    def test_paired_policies_preserve_sequence_spatial_and_mask_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, manifest, _ = prepare_one_shard(
                Path(directory),
                [synthetic_sequence(index, frames=40) for index in range(3)],
            )
            frozen = self._dataset(manifest, prepared, "frozen_random")
            resampled = self._dataset(manifest, prepared, "resampled_anchor")
            frozen.set_epoch(4)
            resampled.set_epoch(4)
            frozen_draws = list(FixedExposureSampler(frozen, 24, seed=31))
            resampled_draws = list(FixedExposureSampler(resampled, 24, seed=31))
            self.assertEqual(frozen_draws, resampled_draws)
            self.assertEqual(
                [frozen[index]["sequence_id"] for index in frozen_draws],
                [resampled[index]["sequence_id"] for index in resampled_draws],
            )

            probe = torch.arange(16 * 8 * 6, dtype=torch.float32).reshape(16, 1, 8, 6)
            for sample_index, draw_index in frozen_draws:
                self.assertTrue(
                    torch.equal(
                        frozen._transform(probe.clone(), sample_index, draw_index),
                        resampled._transform(probe.clone(), sample_index, draw_index),
                    )
                )

            mask_config = {
                "tubelet_size": 2,
                "patch_size": 8,
                "num_frames": 16,
                "img_size": 16,
                "num_tokens": 32,
                "min_context_tokens": 8,
            }
            frozen_mask_rng = random.Random(31)
            resampled_mask_rng = random.Random(31)
            for frozen_index, resampled_index in zip(frozen_draws, resampled_draws):
                _ = frozen[frozen_index]
                frozen_masks = multiblock_mask(mask_config, 2, frozen_mask_rng)
                _ = resampled[resampled_index]
                resampled_masks = multiblock_mask(mask_config, 2, resampled_mask_rng)
                for left, right in zip(frozen_masks, resampled_masks):
                    self.assertTrue(torch.equal(left["ctx"], right["ctx"]))
                    self.assertTrue(torch.equal(left["pred"], right["pred"]))


class GaitLUManifestContractTest(unittest.TestCase):
    def test_combined_manifest_digest_is_streaming_role_sensitive_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            val = root / "val.csv"
            train_bytes = b"train-manifest\n" * 257
            val_bytes = b"val-manifest\n" * 193
            train.write_bytes(train_bytes)
            val.write_bytes(val_bytes)
            digest = gaitlu_manifest_pair_sha256(train, val)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(digest, gaitlu_manifest_pair_sha256(train, val))
            self.assertNotEqual(digest, gaitlu_manifest_pair_sha256(val, train))
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must stream")):
                self.assertEqual(digest, gaitlu_manifest_pair_sha256(train, val))

            train.write_bytes(train_bytes + b"changed")
            self.assertNotEqual(digest, gaitlu_manifest_pair_sha256(train, val))
            train.write_bytes(train_bytes)
            val.write_bytes(val_bytes + b"changed")
            self.assertNotEqual(digest, gaitlu_manifest_pair_sha256(train, val))

    def test_gaitlu_training_contract_contains_exactly_one_manifest_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            val = root / "val.csv"
            train.write_text("train")
            val.write_text("val")
            argv = [
                "--config", str(root / "config.json"),
                "--dataset", "gaitlu",
                "--train-manifest", str(train),
                "--val-manifest", str(val),
                "--data-root", str(root),
                "--output-dir", str(root / "output"),
                "--num-workers", "0",
            ]
            config = {
                "num_frames": 16,
                "img_size": 12,
                "in_channels": 1,
                "batch_size": 2,
                "seed": 7,
                "train_horizontal_flip_prob": 0.0,
            }
            fake_datasets = (
                SimpleNamespace(description=lambda: {"manifest_name": "train.csv", "split": "train"}),
                SimpleNamespace(description=lambda: {"manifest_name": "val.csv", "split": "val"}),
            )
            result = {
                "global_step": 0,
                "completed_epochs": 0,
                "best_epoch": None,
                "best_healthy_epoch": None,
                "termination_reason": "test",
                "elapsed_seconds": 0.0,
                "examples_per_second": 0.0,
            }
            captured = {}

            def fake_train(_config, _train, _val, data_contract, **_kwargs):
                captured["data_contract"] = data_contract
                return result

            with (
                patch("cody_jepa.cli.train._read_config", return_value=(config, DEFAULT_MASK_GROUPS)),
                patch("cody_jepa.cli.train.build_gaitlu_datasets_from_config", return_value=fake_datasets),
                patch("cody_jepa.cli.train.build_gaitlu_loaders_from_config", return_value=(object(), object())),
                patch("cody_jepa.cli.train.train_jepa", side_effect=fake_train),
            ):
                run_training(parse_train_args(argv))

            contract = captured["data_contract"]
            self.assertEqual(count_key(contract, "manifest_sha256"), 1)
            self.assertEqual(
                contract["manifest_sha256"], gaitlu_manifest_pair_sha256(train, val)
            )
            self.assertEqual(contract["seed_scheme"], "splitmix64-v1")
            self.assertNotIn("manifest_sha256", contract["train_dataset"])
            self.assertNotIn("manifest_sha256", contract["val_dataset"])

    def test_training_rejects_manifest_changes_during_dataset_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            val = root / "val.csv"
            train.write_text("train-before")
            val.write_text("val")
            argv = [
                "--config", str(root / "config.json"),
                "--dataset", "gaitlu",
                "--train-manifest", str(train),
                "--val-manifest", str(val),
                "--data-root", str(root),
                "--output-dir", str(root / "output"),
                "--num-workers", "0",
            ]
            config = {
                "num_frames": 16,
                "img_size": 12,
                "in_channels": 1,
                "batch_size": 2,
                "seed": 7,
                "train_horizontal_flip_prob": 0.0,
            }

            def mutate_manifest(_loader_config):
                train.write_text("train-after")
                return (object(), object())

            with (
                patch(
                    "cody_jepa.cli.train._read_config",
                    return_value=(config, DEFAULT_MASK_GROUPS),
                ),
                patch(
                    "cody_jepa.cli.train.build_gaitlu_datasets_from_config",
                    side_effect=mutate_manifest,
                ),
                self.assertRaisesRegex(RuntimeError, "changed while datasets were loading"),
            ):
                run_training(parse_train_args(argv))

    def test_healthgait_training_contract_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_loader = {
                "clip_length": 16,
                "image_size": [12, 12],
                "seed": 7,
                "image_verify_mode": "sample",
            }
            fake_loader_config = SimpleNamespace(
                as_dict=lambda: {
                    **expected_loader,
                    "manifest_csv": "private.csv",
                    "repo_root": "private-root",
                    "allowed_data_root": "private-data-root",
                }
            )
            train_description = {
                "version": "healthgait-test",
                "split": "train",
                "manifest_sha256": "train-hash",
            }
            val_description = {
                "version": "healthgait-test",
                "split": "val",
                "manifest_sha256": "val-hash",
            }
            fake_datasets = (
                SimpleNamespace(description=lambda: train_description),
                SimpleNamespace(description=lambda: val_description),
            )
            config = {
                "num_frames": 16,
                "img_size": 12,
                "in_channels": 1,
                "batch_size": 2,
                "seed": 7,
                "train_horizontal_flip_prob": 0.0,
            }
            result = {
                "global_step": 0,
                "completed_epochs": 0,
                "best_epoch": None,
                "best_healthy_epoch": None,
                "termination_reason": "test",
                "elapsed_seconds": 0.0,
                "examples_per_second": 0.0,
            }
            captured = {}

            def fake_train(_config, _train, _val, data_contract, **_kwargs):
                captured["data_contract"] = data_contract
                return result

            argv = [
                "--config", str(root / "config.json"),
                "--manifest", str(root / "manifest.csv"),
                "--output-dir", str(root / "output"),
                "--repo-root", str(root),
                "--num-workers", "0",
            ]
            with (
                patch("cody_jepa.cli.train._read_config", return_value=(config, DEFAULT_MASK_GROUPS)),
                patch("cody_jepa.cli.train.HealthGaitLoaderConfig", return_value=fake_loader_config),
                patch("cody_jepa.cli.train.build_healthgait_datasets_from_config", return_value=fake_datasets),
                patch("cody_jepa.cli.train.build_healthgait_loaders_from_config", return_value=(object(), object())),
                patch("cody_jepa.cli.train.train_jepa", side_effect=fake_train),
            ):
                run_training(parse_train_args(argv))

            self.assertEqual(
                captured["data_contract"],
                {
                    "loader_config": expected_loader,
                    "train_dataset": train_description,
                    "val_dataset": val_description,
                },
            )

    def test_changed_manifest_and_old_seed_contract_reject_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            val = root / "val.csv"
            train.write_text("train-v1")
            val.write_text("val-v1")
            contract = {
                "dataset": "gaitlu",
                "manifest_sha256": gaitlu_manifest_pair_sha256(train, val),
                "seed_scheme": "splitmix64-v1",
            }
            state = {
                "schema": CHECKPOINT_SCHEMA,
                "architecture": MODEL_ARCHITECTURE,
                "mask_groups": [vars(group) for group in DEFAULT_MASK_GROUPS],
                "data_contract": contract,
                "config": {},
            }
            train.write_text("train-v2")
            changed_train = gaitlu_manifest_pair_sha256(train, val)
            train.write_text("train-v1")
            val.write_text("val-v2")
            changed_val = gaitlu_manifest_pair_sha256(train, val)
            val.write_text("val-v1")
            swapped = gaitlu_manifest_pair_sha256(val, train)
            for changed_digest in (changed_train, changed_val, swapped):
                with self.assertRaisesRegex(ValueError, "dataset/loader contract"):
                    validate_resume_state(
                        state,
                        {},
                        DEFAULT_MASK_GROUPS,
                        {**contract, "manifest_sha256": changed_digest},
                    )

            old = {key: value for key, value in contract.items() if key != "seed_scheme"}
            old_state = {**state, "data_contract": old}
            with self.assertRaisesRegex(ValueError, "dataset/loader contract"):
                validate_resume_state(old_state, {}, DEFAULT_MASK_GROUPS, contract)


if __name__ == "__main__":
    unittest.main()
