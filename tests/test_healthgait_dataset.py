import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cody_jepa.data.dataset import HealthGaitManifestDataset, ManifestValidationError
from cody_jepa.data.healthgait import (
    HealthGaitLoaderConfig,
    audit_healthgait_clip_quality,
    build_healthgait_datasets_from_config,
    build_healthgait_loaders_from_config,
)


MANIFEST_FIELDNAMES = [
    "subject_id",
    "modality",
    "gait_system",
    "trial",
    "frame_dir",
    "num_frames",
    "split",
]


def identity_collate(batch):
    return batch


class WindowOnlyHealthGaitManifestDataset(HealthGaitManifestDataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        stable_id = sample["sequence_id"]
        start = self._choose_window_start(stable_id, len(frame_paths))
        selected_paths = frame_paths[start : start + self.clip_length]
        return self._clip_metadata(sample, start, selected_paths)


class HealthGaitDatasetSeedTest(unittest.TestCase):
    def _make_frames(self, root, relative_dir, frame_count):
        frame_dir = root / relative_dir
        frame_dir.mkdir(parents=True)

        for frame_idx in range(frame_count):
            image = Image.new("L", (6, 6), color=frame_idx % 255)
            image.save(frame_dir / f"{frame_idx + 1:03d}.png")

        return frame_dir

    def _write_manifest(self, root, rows, fieldnames=MANIFEST_FIELDNAMES):
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def _valid_row(self, root, split, subject_id, frame_count=6, num_frames=None):
        relative_dir = Path("frames") / split / subject_id
        self._make_frames(root, relative_dir, frame_count)

        return {
            "subject_id": subject_id,
            "modality": "silhouette",
            "gait_system": "FGS",
            "trial": "T0",
            "frame_dir": str(relative_dir),
            "num_frames": str(frame_count if num_frames is None else num_frames),
            "split": split,
        }

    def _build_dataset(self, root, manifest, clip_length=4):
        return HealthGaitManifestDataset(
            manifest,
            split="train",
            repo_root=root,
            clip_length=clip_length,
            image_size=(6, 6),
            random_windows=True,
            base_seed=17,
        )

    def _write_fixture(self, root):
        manifest = root / "manifest.csv"
        rows = []

        for idx, split in enumerate(["train", "train", "train", "val"]):
            frame_dir = root / "frames" / split / f"sample_{idx}"
            frame_dir.mkdir(parents=True)
            frame_count = 12 + idx

            for frame_idx in range(frame_count):
                image = Image.new("L", (6, 6), color=(idx * 40 + frame_idx) % 255)
                image.save(frame_dir / f"{frame_idx + 1:03d}.png")

            rows.append({
                "subject_id": f"P{idx:03d}",
                "modality": "silhouette",
                "gait_system": "FGS",
                "trial": f"T{idx}",
                "frame_dir": str(frame_dir.relative_to(root)),
                "num_frames": str(frame_count),
                "split": split,
            })

        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "subject_id",
                    "modality",
                    "gait_system",
                    "trial",
                    "frame_dir",
                    "num_frames",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return manifest

    def _dataset(self, root, manifest, base_seed=17):
        return WindowOnlyHealthGaitManifestDataset(
            manifest,
            split="train",
            repo_root=root,
            clip_length=4,
            image_size=(6, 6),
            random_windows=True,
            base_seed=base_seed,
        )

    def _window_trace(self, root, manifest, epoch, num_workers):
        dataset = self._dataset(root, manifest)
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=identity_collate,
        )

        trace = []
        for batch in loader:
            trace.extend(
                (
                    sample["sequence_id"],
                    sample["window_start"],
                    tuple(sample["frame_indices"]),
                )
                for sample in batch
            )
        return trace

    def test_window_sampling_is_independent_of_global_random_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            dataset = self._dataset(root, manifest)
            dataset.set_epoch(3)

            random.seed(1)
            first = [dataset[idx]["window_start"] for idx in range(len(dataset))]

            random.seed(999)
            second = [dataset[idx]["window_start"] for idx in range(len(dataset))]

            self.assertEqual(first, second)

    def test_window_sampling_is_stable_across_dataloader_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)

            single_worker = self._window_trace(root, manifest, epoch=2, num_workers=0)
            multi_worker = self._window_trace(root, manifest, epoch=2, num_workers=2)
            repeated = self._window_trace(root, manifest, epoch=2, num_workers=2)

            self.assertEqual(single_worker, multi_worker)
            self.assertEqual(multi_worker, repeated)

    def test_epoch_participates_in_window_sampling_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)

            epoch_0 = self._window_trace(root, manifest, epoch=0, num_workers=0)
            epoch_1 = self._window_trace(root, manifest, epoch=1, num_workers=0)

            self.assertNotEqual(epoch_0, epoch_1)

    def test_sample_contains_traceable_clip_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            dataset = HealthGaitManifestDataset(
                manifest,
                split="train",
                repo_root=root,
                clip_length=4,
                image_size=(6, 6),
                random_windows=True,
                base_seed=17,
            )
            dataset.set_epoch(2)

            sample = dataset[0]
            start = sample["window_start"]

            self.assertEqual(tuple(sample["video"].shape), (4, 1, 6, 6))
            self.assertEqual(
                {
                    "video",
                    "sequence_id",
                    "split",
                    "modality",
                    "subject_id",
                    "gait_system",
                    "trial",
                    "window_start",
                    "window_index",
                    "frame_indices",
                    "num_frames",
                },
                set(sample) - {"frame_dir"},
            )
            self.assertEqual(sample["sequence_id"], "P000::silhouette::FGS::T0::frames/train/sample_0")
            self.assertEqual(sample["split"], "train")
            self.assertEqual(sample["modality"], "silhouette")
            self.assertEqual(sample["subject_id"], "P000")
            self.assertEqual(sample["gait_system"], "FGS")
            self.assertEqual(sample["trial"], "T0")
            self.assertEqual(sample["num_frames"], 12)
            self.assertEqual(sample["frame_indices"], list(range(start + 1, start + 5)))
            self.assertTrue(sample["frame_dir"].endswith("frames/train/sample_0"))

    def test_dataloader_batch_preserves_traceable_clip_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            dataset = HealthGaitManifestDataset(
                manifest,
                split="train",
                repo_root=root,
                clip_length=4,
                image_size=(6, 6),
                random_windows=True,
                base_seed=17,
            )
            dataset.set_epoch(2)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

            batch = next(iter(loader))
            first_sample = dataset[0]

            self.assertEqual(tuple(batch["video"].shape), (2, 4, 1, 6, 6))
            self.assertEqual(list(batch["sequence_id"]), [
                dataset.samples[0]["sequence_id"],
                dataset.samples[1]["sequence_id"],
            ])
            self.assertEqual(list(batch["split"]), ["train", "train"])
            self.assertEqual(list(batch["modality"]), ["silhouette", "silhouette"])
            self.assertEqual(int(batch["window_start"][0]), first_sample["window_start"])
            self.assertEqual(
                [int(frame[0]) for frame in batch["frame_indices"]],
                first_sample["frame_indices"],
            )

    def test_loader_config_serializes_and_builds_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            config = HealthGaitLoaderConfig(
                manifest_csv=manifest,
                repo_root=root,
                split="train",
                clip_length=4,
                image_size=[6, 6],
                channels=1,
                seed=17,
                window_policy="train_random_val_center",
                strict_validation=True,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )

            self.assertEqual(
                config.as_dict(),
                {
                    "manifest_csv": str(manifest.resolve()),
                    "repo_root": str(root.resolve()),
                    "split": "train",
                    "clip_length": 4,
                    "image_size": [6, 6],
                    "channels": 1,
                    "seed": 17,
                    "window_policy": "train_random_val_center",
                    "strict_validation": True,
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "prefetch_factor": 2,
                    "train_crop_scale": [1.0, 1.0],
                    "train_horizontal_flip_prob": 0.0,
                    "expected_modality": "silhouette",
                    "strict_frame_sequence": True,
                    "image_verify_mode": "none",
                    "inventory_hash_mode": "sample",
                    "allowed_data_root": str(root.resolve()),
                    "eval_windows": 1,
                    "drop_last_train": False,
                },
            )
            self.assertTrue(config.uses_random_windows())
            self.assertFalse(config.for_split("val").uses_random_windows())

            train_ds, val_ds = build_healthgait_datasets_from_config(config)
            self.assertTrue(train_ds.random_windows)
            self.assertFalse(val_ds.random_windows)
            self.assertEqual(tuple(train_ds[0]["video"].shape), (4, 1, 6, 6))

            train_loader, val_loader = build_healthgait_loaders_from_config(config)
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            self.assertEqual(tuple(train_batch["video"].shape), (2, 4, 1, 6, 6))
            self.assertEqual(tuple(val_batch["video"].shape), (1, 4, 1, 6, 6))
            self.assertIn("subject_id", train_batch)

    def test_deterministic_train_config_honors_eval_windows_for_feature_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            config = HealthGaitLoaderConfig(
                manifest_csv=manifest,
                repo_root=root,
                split="train",
                clip_length=4,
                image_size=(6, 6),
                window_policy="center",
                eval_windows=3,
            )

            train_ds, val_ds = build_healthgait_datasets_from_config(config)

            self.assertFalse(train_ds.random_windows)
            self.assertEqual(train_ds.deterministic_windows, 3)
            self.assertEqual(val_ds.deterministic_windows, 3)
            self.assertEqual(len(train_ds), len(train_ds.samples) * 3)
            self.assertEqual(len(val_ds), len(val_ds.samples) * 3)

    def test_loader_config_rejects_unsupported_loss_path_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)

            with self.assertRaisesRegex(ValueError, "one grayscale channel"):
                HealthGaitLoaderConfig(
                    manifest_csv=manifest,
                    repo_root=root,
                    channels=3,
                )

    def test_manifest_validation_requires_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_manifest(
                root,
                rows=[{"subject_id": "P000"}],
                fieldnames=["subject_id"],
            )

            with self.assertRaisesRegex(
                ManifestValidationError,
                r"missing required columns: .*frame_dir.*num_frames.*split",
            ):
                self._build_dataset(root, manifest)

    def test_manifest_validation_rejects_invalid_split_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_row = self._valid_row(root, "train", "P000")
            val_row = self._valid_row(root, "val", "P001")
            bad_row = self._valid_row(root, "train", "P002")
            bad_row["split"] = "test"
            manifest = self._write_manifest(root, [train_row, val_row, bad_row])

            with self.assertRaisesRegex(ManifestValidationError, r"row 4: invalid split 'test'"):
                self._build_dataset(root, manifest)

    def test_manifest_validation_requires_existing_frame_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_row = {
                "subject_id": "P000",
                "modality": "silhouette",
                "gait_system": "FGS",
                "trial": "T0",
                "frame_dir": "frames/train/missing",
                "num_frames": "6",
                "split": "train",
            }
            val_row = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [bad_row, val_row])

            with self.assertRaisesRegex(ManifestValidationError, r"frame_dir does not exist"):
                self._build_dataset(root, manifest)

    def test_manifest_validation_checks_num_frames_against_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_row = self._valid_row(root, "train", "P000", frame_count=6, num_frames=99)
            val_row = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [bad_row, val_row])

            with self.assertRaisesRegex(
                ManifestValidationError,
                r"num_frames=99 but found 6 frame files",
            ):
                self._build_dataset(root, manifest)

    def test_manifest_validation_rejects_short_clip_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_row = self._valid_row(root, "train", "P000", frame_count=3)
            val_row = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [bad_row, val_row])

            with self.assertRaisesRegex(
                ManifestValidationError,
                r"only 3 frame files .* clip_length requires at least 4",
            ):
                self._build_dataset(root, manifest)

    def test_manifest_validation_rejects_empty_train_or_val_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_row = self._valid_row(root, "train", "P000")
            manifest = self._write_manifest(root, [train_row])

            with self.assertRaisesRegex(
                ManifestValidationError,
                r"manifest has no rows for split 'val'",
            ):
                self._build_dataset(root, manifest)

    def test_manifest_validation_rejects_subject_overlap_between_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_row = self._valid_row(root, "train", "P000")
            val_row = self._valid_row(root, "val", "P000")
            manifest = self._write_manifest(root, [train_row, val_row])

            with self.assertRaisesRegex(
                ManifestValidationError,
                r"subject overlap between train and val splits: P000",
            ):
                self._build_dataset(root, manifest)

    def test_manifest_validation_rejects_blank_and_casefolded_subject_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blank = self._valid_row(root, "train", "blank-subject")
            blank["subject_id"] = ""
            val = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [blank, val])
            with self.assertRaisesRegex(ManifestValidationError, "subject_id is empty"):
                self._build_dataset(root, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._valid_row(root, "train", "PA001")
            val = self._valid_row(root, "val", "pa001")
            manifest = self._write_manifest(root, [train, val])
            with self.assertRaisesRegex(ManifestValidationError, "subject overlap"):
                self._build_dataset(root, manifest)

    def test_manifest_validation_rejects_duplicate_canonical_frame_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._valid_row(root, "train", "P000")
            duplicate = dict(train)
            duplicate["subject_id"] = "P001"
            duplicate["frame_dir"] = str(Path(train["frame_dir"]) / ".." / "P000")
            val = self._valid_row(root, "val", "P002")
            manifest = self._write_manifest(root, [train, duplicate, val])
            with self.assertRaisesRegex(ManifestValidationError, "duplicate canonical frame_dir"):
                self._build_dataset(root, manifest)

    def test_gapped_sources_only_yield_contiguous_windows_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._valid_row(root, "train", "P000", frame_count=8)
            frame_dir = root / train["frame_dir"]
            (frame_dir / "005.png").rename(frame_dir / "009.png")
            val = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [train, val])
            dataset = self._build_dataset(root, manifest)
            for epoch in range(10):
                dataset.set_epoch(epoch)
                indices = dataset[0]["frame_indices"]
                self.assertEqual(indices, list(range(indices[0], indices[0] + 4)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._valid_row(root, "train", "P000")
            frame_dir = root / train["frame_dir"]
            Image.new("L", (6, 6)).save(frame_dir / "001.jpg")
            train["num_frames"] = "7"
            val = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [train, val])
            with self.assertRaisesRegex(ManifestValidationError, "duplicate numeric frame indices"):
                self._build_dataset(root, manifest)

    def test_image_preflight_rejects_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._valid_row(root, "train", "P000")
            frame_dir = root / train["frame_dir"]
            (frame_dir / "001.png").write_bytes(b"not an image")
            val = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [train, val])
            with self.assertRaisesRegex(ManifestValidationError, "corrupt image"):
                HealthGaitManifestDataset(
                    manifest,
                    split="train",
                    repo_root=root,
                    clip_length=4,
                    image_size=(6, 6),
                    image_verify_mode="all",
                )

    def test_all_image_preflight_rejects_corrupt_nonsampled_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._valid_row(root, "train", "P000", frame_count=8)
            frame_dir = root / train["frame_dir"]
            (frame_dir / "002.png").write_bytes(b"not an image")
            val = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [train, val])
            with self.assertRaisesRegex(ManifestValidationError, "002.png"):
                HealthGaitManifestDataset(
                    manifest,
                    split="train",
                    repo_root=root,
                    clip_length=4,
                    image_size=(6, 6),
                    image_verify_mode="all",
                )

    def test_non_square_image_size_uses_height_width_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            dataset = HealthGaitManifestDataset(
                manifest,
                split="train",
                repo_root=root,
                clip_length=4,
                image_size=(8, 12),
            )
            self.assertEqual(tuple(dataset[0]["video"].shape), (4, 1, 8, 12))

    def test_clip_consistent_augmentation_is_deterministic_and_epoch_varying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split, subject in (("train", "P000"), ("val", "P001")):
                relative = Path("frames") / split / subject
                frame_dir = root / relative
                frame_dir.mkdir(parents=True)
                image = Image.new("L", (12, 12))
                for y in range(12):
                    for x in range(12):
                        image.putpixel((x, y), x * 10 + y)
                for index in range(6):
                    image.save(frame_dir / f"{index + 1:03d}.png")
                rows.append({
                    "subject_id": subject,
                    "modality": "silhouette",
                    "gait_system": "FGS",
                    "trial": "T0",
                    "frame_dir": str(relative),
                    "num_frames": "6",
                    "split": split,
                })
            manifest = self._write_manifest(root, rows)
            dataset = HealthGaitManifestDataset(
                manifest,
                split="train",
                repo_root=root,
                clip_length=4,
                image_size=(12, 12),
                random_windows=True,
                base_seed=3,
                crop_scale=(0.75, 0.9),
                horizontal_flip_prob=0.5,
            )
            dataset.set_epoch(2)
            first = dataset[0]["video"]
            repeated = dataset[0]["video"]
            self.assertTrue(first.equal(repeated))
            self.assertTrue(all(first[0].equal(frame) for frame in first[1:]))
            dataset.set_epoch(3)
            self.assertFalse(first.equal(dataset[0]["video"]))

    def test_validation_windows_cover_start_center_and_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            dataset = HealthGaitManifestDataset(
                manifest,
                split="val",
                repo_root=root,
                clip_length=4,
                image_size=(6, 6),
                deterministic_windows=3,
            )
            starts = [dataset[index]["window_start"] for index in range(3)]
            max_start = dataset.samples[0]["num_frames"] - 4
            self.assertEqual(starts, [0, round(max_start / 2), max_start])
            self.assertEqual(
                [dataset.subject_id_at(index) for index in range(3)],
                [dataset.samples[0]["subject_id"]] * 3,
            )
            self.assertEqual(dataset.subject_id_at(-1), dataset.samples[-1]["subject_id"])
            with self.assertRaises(IndexError):
                dataset.subject_id_at(len(dataset))

    def test_dataset_signature_changes_with_transform_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            plain = HealthGaitManifestDataset(
                manifest, "train", root, clip_length=4, image_size=(6, 6)
            )
            augmented = HealthGaitManifestDataset(
                manifest,
                "train",
                root,
                clip_length=4,
                image_size=(6, 6),
                crop_scale=(0.9, 1.0),
            )
            self.assertEqual(
                plain.signature()["manifest_sha256"],
                augmented.signature()["manifest_sha256"],
            )
            self.assertNotEqual(
                plain.signature()["dataset_sha256"],
                augmented.signature()["dataset_sha256"],
            )

    def test_inventory_signature_changes_when_sampled_pixels_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            original = HealthGaitManifestDataset(
                manifest, "train", root, clip_length=4, image_size=(6, 6)
            ).signature()
            first_dir = root / "frames" / "train" / "sample_0"
            Image.new("L", (6, 6), color=255).save(first_dir / "001.png")
            changed = HealthGaitManifestDataset(
                manifest, "train", root, clip_length=4, image_size=(6, 6)
            ).signature()
            self.assertNotEqual(
                original["inventory_sha256"], changed["inventory_sha256"]
            )

    def test_full_inventory_hash_detects_same_size_nonsampled_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            original = HealthGaitManifestDataset(
                manifest,
                "train",
                root,
                clip_length=4,
                image_size=(6, 6),
                inventory_hash_mode="full",
            ).signature()
            path = root / "frames" / "train" / "sample_0" / "002.png"
            mutated = bytearray(path.read_bytes())
            mutated[-1] ^= 1
            path.write_bytes(mutated)
            changed = HealthGaitManifestDataset(
                manifest,
                "train",
                root,
                clip_length=4,
                image_size=(6, 6),
                inventory_hash_mode="full",
            ).signature()
            self.assertEqual(path.stat().st_size, len(mutated))
            self.assertNotEqual(
                original["inventory_sha256"], changed["inventory_sha256"]
            )

    def test_manifest_short_row_is_reported_as_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = self._valid_row(root, "val", "P001")
            manifest = self._write_manifest(root, [val])
            with manifest.open("a", newline="") as handle:
                handle.write("P000,silhouette\n")
            with self.assertRaisesRegex(ManifestValidationError, "gait_system is empty"):
                self._build_dataset(root, manifest)

    def test_manifest_rejects_frame_dirs_outside_allowed_root(self):
        for path_kind in ("absolute", "traversal", "symlink"):
            with self.subTest(path_kind=path_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                outside = root.parent / f"{root.name}-outside-{path_kind}"
                outside.mkdir()
                try:
                    outside_frames = self._make_frames(outside, "escaped", 6)
                    if path_kind == "absolute":
                        bad_path = str(outside_frames)
                    elif path_kind == "traversal":
                        bad_path = str(Path("..") / outside.name / "escaped")
                    else:
                        link = root / "escape-link"
                        link.symlink_to(outside_frames, target_is_directory=True)
                        bad_path = str(link.relative_to(root))
                    bad = {
                        "subject_id": "P000",
                        "modality": "silhouette",
                        "gait_system": "FGS",
                        "trial": "T0",
                        "frame_dir": bad_path,
                        "num_frames": "6",
                        "split": "train",
                    }
                    val = self._valid_row(root, "val", "P001")
                    manifest = self._write_manifest(root, [bad, val])
                    with self.assertRaisesRegex(
                        ManifestValidationError, "escapes allowed_data_root"
                    ):
                        self._build_dataset(root, manifest)
                finally:
                    for path in (outside / "escaped").glob("*"):
                        path.unlink()
                    (outside / "escaped").rmdir()
                    outside.rmdir()

    def test_quality_audit_checks_late_sequences_in_both_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._valid_row(root, "train", "P000"),
                self._valid_row(root, "train", "P001"),
                self._valid_row(root, "train", "P002"),
                self._valid_row(root, "val", "P003"),
            ]
            blank_dir = root / rows[2]["frame_dir"]
            for path in blank_dir.glob("*.png"):
                Image.new("L", (6, 6), color=0).save(path)
            manifest = self._write_manifest(root, rows)
            config = HealthGaitLoaderConfig(
                manifest_csv=manifest,
                repo_root=root,
                clip_length=4,
                image_size=(6, 6),
                batch_size=2,
            )
            datasets = build_healthgait_datasets_from_config(config)
            with self.assertRaisesRegex(ManifestValidationError, "P002"):
                audit_healthgait_clip_quality(
                    datasets,
                    min_foreground_fraction=0.1,
                    max_foreground_fraction=0.99,
                    foreground_threshold=0.0,
                )

    def test_direct_dataset_api_rejects_non_boolean_window_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_fixture(root)
            with self.assertRaisesRegex(TypeError, "random_windows"):
                HealthGaitManifestDataset(
                    manifest,
                    "train",
                    root,
                    clip_length=4,
                    image_size=(6, 6),
                    random_windows="false",
                )


if __name__ == "__main__":
    unittest.main()
