"""End-to-end coverage: manifest -> loader -> training -> features -> probes.

Runs on a synthetic silhouette corpus, so it needs no private data.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader

import build_manifest
from cody_jepa.data import HealthGaitDataset, build_loaders
from cody_jepa.engine import (
    accumulation_size,
    collapse_metrics,
    ema_tau_for_step,
    ema_update,
    evaluate,
    learning_rate_for_step,
    load_checkpoint,
    train_jepa,
)
from cody_jepa.evaluation import (
    build_random_target_encoder,
    build_target_encoder,
    export_features,
    run_probes,
)
from cody_jepa.masks import multiblock_mask
from cody_jepa.models import build_models

CONFIG = {
    "seed": 0,
    "batch_size": 4,
    "accumulation_steps": 1,
    "num_epochs": 2,
    "steps": 100,
    "lr": 1e-3,
    "start_lr": 1e-4,
    "min_lr": 1e-5,
    "warmup_steps": 2,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "ema_start": 0.9,
    "ema_end": 1.0,
    "var_coef": 1.0,
    "cov_coef": 0.04,
    "num_frames": 4,
    "img_size": 16,
    "patch_size": 4,
    "tubelet_size": 2,
    "in_channels": 1,
    "min_context_tokens": 4,
    "embed_dim": 24,
    "hidden_dim": 48,
    "num_heads": 2,
    "num_layers": 1,
    "pred_dim": 12,
    "pred_depth": 1,
    "eval_every_epochs": 1,
}


def make_corpus(root, subjects=8):
    """Write a synthetic frame tree where speed/clothing/direction are decodable."""
    raw_root = root / "raw"
    for index in range(subjects):
        subject = f"PA{index:03d}"
        for speed in ("UGS", "FGS"):
            for clothing in ("WoJ", "WJ"):
                for suffix in ("1", "2"):
                    frame_dir = raw_root / "silhouette" / subject / speed / f"{clothing}_{suffix}"
                    frame_dir.mkdir(parents=True)
                    pattern = np.zeros((16, 16), dtype=np.uint8)
                    pattern[0, index] = 255
                    pattern[1, 1 if speed == "UGS" else 2] = 255
                    pattern[2, 1 if clothing == "WoJ" else 2] = 255
                    pattern[3, int(suffix)] = 255
                    for frame in range(8):
                        Image.fromarray(pattern, mode="L").save(frame_dir / f"{frame + 1:03d}.png")
    return raw_root


def write_manifest(root, raw_root):
    rows = build_manifest.build_rows(raw_root, "silhouette", root, clip_length=4)
    build_manifest.assign_splits(rows, val_fraction=0.25, seed=0)
    manifest = root / "manifest.csv"
    import csv

    from cody_jepa.data import MANIFEST_COLUMNS

    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def write_wide_manifest(root):
    frame_dir = root / "wide" / "silhouette" / "PA000" / "UGS" / "WoJ_1"
    frame_dir.mkdir(parents=True)
    frame = np.zeros((32, 64), dtype=np.uint8)
    frame[6:27, 44:53] = 255
    for index in range(8):
        Image.fromarray(frame, mode="L").save(frame_dir / f"{index + 1:03d}.png")
    manifest = root / "wide_manifest.csv"
    import csv

    from cody_jepa.data import MANIFEST_COLUMNS

    row = {
        "subject_id": "PA000",
        "split": "train",
        "frame_dir": "wide/silhouette/PA000/UGS/WoJ_1",
        "num_frames": 8,
        "gait_system": "UGS",
        "speed": "UGS",
        "clothing": "WoJ",
        "direction": "1",
        "recording_id": "PA000_UGS_WoJ_1",
        "source_video_id": "PA000_UGS_WoJ",
    }
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return manifest


def foreground_pixel_fraction(video):
    return video.gt(0.05).float().mean().item()


class ScheduleTest(unittest.TestCase):
    def test_learning_rate_warms_up_then_decays(self):
        config = {"lr": 1.0, "start_lr": 0.0, "min_lr": 0.0, "warmup_steps": 5, "steps": 20}
        self.assertAlmostEqual(learning_rate_for_step(config, 1), 0.0)
        self.assertAlmostEqual(learning_rate_for_step(config, 5), 1.0)
        self.assertLess(learning_rate_for_step(config, 20), 1e-6)
        rates = [learning_rate_for_step(config, step) for step in range(5, 21)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_ema_tau_moves_from_start_to_end(self):
        config = {"ema_start": 0.9, "ema_end": 1.0, "steps": 10}
        self.assertAlmostEqual(ema_tau_for_step(config, 1), 0.9)
        self.assertAlmostEqual(ema_tau_for_step(config, 10), 1.0)

    def test_ema_update_moves_target_toward_online(self):
        config = dict(CONFIG)
        context, target, _ = build_models(config, torch.device("cpu"), 2)
        with torch.no_grad():
            for parameter in context.parameters():
                parameter.add_(1.0)
        before = next(target.parameters()).clone()
        ema_update(target, context, tau=0.5)
        after = next(target.parameters())
        self.assertFalse(torch.allclose(before, after))
        self.assertTrue(torch.allclose(after, before + 0.5))

    def test_final_accumulation_group_uses_its_actual_size(self):
        sizes = [accumulation_size(10, index, 4) for index in range(10)]
        self.assertEqual(sizes, [4, 4, 4, 4, 4, 4, 4, 4, 2, 2])


class MaskTest(unittest.TestCase):
    def test_context_and_target_tubes_are_disjoint(self):
        masks = multiblock_mask(CONFIG, batch_size=3, rng=random.Random(0))
        self.assertEqual(len(masks), 2)
        for group in masks:
            self.assertEqual(group["ctx"].shape[0], 3)
            for row in range(3):
                context = set(group["ctx"][row].tolist())
                target = set(group["pred"][row].tolist())
                self.assertTrue(context)
                self.assertTrue(target)
                self.assertFalse(context & target)

    def test_masks_are_reproducible_for_a_seed(self):
        first = multiblock_mask(CONFIG, 3, random.Random(7))
        second = multiblock_mask(CONFIG, 3, random.Random(7))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left["ctx"], right["ctx"]))
            self.assertTrue(torch.equal(left["pred"], right["pred"]))


class CollapseMetricTest(unittest.TestCase):
    def test_identical_features_have_effective_rank_of_zero(self):
        constant = torch.ones(32, 8)
        self.assertAlmostEqual(collapse_metrics(constant)["effective_rank"], 0.0)

    def test_isotropic_features_have_high_effective_rank(self):
        torch.manual_seed(0)
        metrics = collapse_metrics(torch.randn(512, 8))
        self.assertGreater(metrics["effective_rank"], 6.0)
        self.assertGreater(metrics["feature_std"], 0.5)


class DataTest(unittest.TestCase):
    def test_manifest_splits_hold_out_whole_participants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            import csv

            with manifest.open() as handle:
                rows = list(csv.DictReader(handle))
            subjects = {
                split: {row["subject_id"] for row in rows if row["split"] == split}
                for split in ("train", "tune", "test")
            }
            self.assertTrue(all(subjects.values()))
            self.assertFalse(subjects["train"] & subjects["tune"])
            self.assertFalse(subjects["train"] & subjects["test"])
            self.assertFalse(subjects["tune"] & subjects["test"])

    def test_eval_windows_are_deterministic_and_spread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            dataset = HealthGaitDataset(
                manifest, "tune", root, clip_length=4, image_size=16, windows=3
            )
            self.assertEqual(len(dataset), len(dataset.samples) * 3)
            starts = [dataset[index]["window_start"] for index in range(3)]
            self.assertEqual(starts, [0, 2, 4])
            self.assertEqual(starts, [dataset[index]["window_start"] for index in range(3)])

    def test_clip_shape_and_pixel_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            sample = HealthGaitDataset(manifest, "train", root, clip_length=4, image_size=16)[0]
            self.assertEqual(tuple(sample["video"].shape), (4, 1, 16, 16))
            self.assertGreaterEqual(float(sample["video"].min()), 0.0)
            self.assertLessEqual(float(sample["video"].max()), 1.0)

    def test_foreground_crop_raises_patch_coverage_on_wide_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_wide_manifest(root)
            legacy = HealthGaitDataset(
                manifest,
                "train",
                root,
                clip_length=4,
                image_size=16,
                foreground_crop=False,
            )[0]
            cropped = HealthGaitDataset(
                manifest, "train", root, clip_length=4, image_size=16
            )[0]
            self.assertGreater(foreground_pixel_fraction(cropped["video"]), 0.30)
            self.assertGreater(
                foreground_pixel_fraction(cropped["video"]),
                foreground_pixel_fraction(legacy["video"]),
            )

    def test_train_windows_vary_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            dataset = HealthGaitDataset(
                manifest, "train", root, clip_length=4, image_size=16, train=True
            )
            starts = set()
            for epoch in range(8):
                dataset.set_epoch(epoch)
                starts.add(dataset[0]["window_start"])
            self.assertGreater(len(starts), 1)


class TrainingTest(unittest.TestCase):
    def test_safe_checkpoint_loader_allows_legacy_torch_version_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"torch_version": TorchVersion(torch.__version__)}, path)
            self.assertEqual(load_checkpoint(path)["torch_version"], torch.__version__)

    def test_training_runs_checkpoints_and_exports_probeable_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            output_dir = root / "run"
            train_loader, tune_loader = build_loaders(
                CONFIG, manifest, root, num_workers=0, windows=2
            )

            history = train_jepa(
                CONFIG,
                train_loader,
                tune_loader,
                output_dir=output_dir,
                device=torch.device("cpu"),
            )
            self.assertEqual(len(history), 2)
            self.assertIn("tune", history[-1])
            self.assertTrue(np.isfinite(history[-1]["train_loss"]))
            self.assertTrue((output_dir / "last.pt").is_file())
            self.assertTrue((output_dir / "best.pt").is_file())

            # A checkpoint round-trips into a usable frozen encoder.
            checkpoint = load_checkpoint(output_dir / "best.pt")
            encoder = build_target_encoder(checkpoint, torch.device("cpu"))
            loaders = [
                DataLoader(
                    HealthGaitDataset(
                        manifest, split, root, clip_length=4, image_size=16, windows=2
                    ),
                    batch_size=4,
                )
                for split in ("train", "test")
            ]
            table = export_features(encoder, loaders, CONFIG, torch.device("cpu"))
            self.assertEqual(len(table), sum(len(loader.dataset) for loader in loaders))
            self.assertIn("feature_0", table.columns)
            self.assertFalse(table[["subject_id", "split"]].isna().any().any())

            results = run_probes(table)
            self.assertEqual(set(results["task"]), {"gait_system", "identity"})
            self.assertEqual(set(results["model"]), {"trained"})
            for _, row in results.iterrows():
                self.assertGreaterEqual(row["accuracy"], 0.0)
                self.assertLessEqual(row["accuracy"], 1.0)

            random_encoder = build_random_target_encoder(
                checkpoint["config"], torch.device("cpu"), seed=1
            )
            random_table = export_features(
                random_encoder, loaders, CONFIG, torch.device("cpu")
            )
            random_results = run_probes(random_table, model="random_init")
            self.assertEqual(set(random_results["model"]), {"random_init"})

    def test_resume_continues_from_saved_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            full_dir, resumed_dir = root / "full", root / "resumed"
            full_train, full_tune = build_loaders(CONFIG, manifest, root, num_workers=0)
            full = train_jepa(
                {**CONFIG, "num_epochs": 3}, full_train, full_tune,
                output_dir=full_dir, device=torch.device("cpu"),
            )
            first_train, first_tune = build_loaders(CONFIG, manifest, root, num_workers=0)
            train_jepa(
                CONFIG, first_train, first_tune, output_dir=resumed_dir,
                device=torch.device("cpu"),
            )
            resumed_train, resumed_tune = build_loaders(CONFIG, manifest, root, num_workers=0)
            resumed = train_jepa(
                {**CONFIG, "num_epochs": 3}, resumed_train, resumed_tune,
                output_dir=resumed_dir, device=torch.device("cpu"),
                resume=load_checkpoint(resumed_dir / "last.pt"),
            )
            self.assertEqual([entry["epoch"] for entry in resumed], [3])
            self.assertEqual(resumed[-1]["step"], full[-1]["step"])
            full_checkpoint = load_checkpoint(full_dir / "last.pt")
            resumed_checkpoint = load_checkpoint(resumed_dir / "last.pt")
            self.assertEqual(resumed_checkpoint["global_step"], full_checkpoint["global_step"])
            for name, parameter in full_checkpoint["context_encoder"].items():
                self.assertTrue(torch.equal(parameter, resumed_checkpoint["context_encoder"][name]))
            self.assertEqual(load_checkpoint(resumed_dir / "best.pt")["epoch"], 2)

    def test_training_stops_at_the_configured_step_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            train_loader, tune_loader = build_loaders(CONFIG, manifest, root, num_workers=0)
            history = train_jepa(
                {**CONFIG, "num_epochs": 2, "steps": 1},
                train_loader,
                tune_loader,
                device=torch.device("cpu"),
            )
            self.assertEqual(history[-1]["step"], 1)
            self.assertEqual(len(history), 1)

    def test_evaluate_reports_loss_and_collapse_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, make_corpus(root))
            _, tune_loader = build_loaders(CONFIG, manifest, root, num_workers=0)
            context, target, predictor = build_models(CONFIG, torch.device("cpu"), 2)
            from cody_jepa.masks import DEFAULT_MASK_GROUPS

            metrics = evaluate(
                context, target, predictor, tune_loader, CONFIG,
                torch.device("cpu"), DEFAULT_MASK_GROUPS,
            )
            for key in (
                "loss",
                "cosine",
                "effective_rank",
                "feature_std",
                "clip_effective_rank",
                "token_effective_rank",
                "blank_context_loss_delta",
            ):
                self.assertIn(key, metrics)
                self.assertTrue(np.isfinite(metrics[key]))


class ConfigTest(unittest.TestCase):
    def test_shipped_config_matches_the_model_geometry(self):
        config = json.loads(Path("configs/healthgait.json").read_text())
        self.assertEqual(config["img_size"] % config["patch_size"], 0)
        self.assertEqual(config["num_frames"] % config["tubelet_size"], 0)
        self.assertEqual(config["embed_dim"] % 6, 0)
        self.assertEqual(config["pred_dim"] % 6, 0)
        self.assertEqual(config["embed_dim"] % config["num_heads"], 0)


if __name__ == "__main__":
    unittest.main()
