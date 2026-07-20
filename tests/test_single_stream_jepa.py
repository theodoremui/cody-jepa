import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from cody_jepa.single_stream_jepa import (
    DEFAULT_MASK_GROUPS,
    VisionTransformer,
    build_models,
    ema_tau_for_step,
    learning_rate_for_step,
    multiblock_mask,
    optimizer_param_groups,
    representation_diagnostics,
    resolve_device,
    subject_aware_context_sources,
    train_jepa,
    validate_resume_state,
    video_from_batch,
)
from cody_jepa.single_stream_jepa import _maybe_compile, _prediction_metrics


def tiny_config(num_epochs=2):
    return {
        "seed": 7,
        "batch_size": 2,
        "accumulation_steps": 1,
        "steps": 4,
        "num_epochs": num_epochs,
        "lr": 1e-3,
        "start_lr": 1e-4,
        "warmup_steps": 2,
        "min_lr": 1e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "ema_start": 0.9,
        "ema_end": 1.0,
        "num_frames": 4,
        "img_size": 16,
        "patch_size": 4,
        "tubelet_size": 2,
        "in_channels": 1,
        "num_tokens": 32,
        "min_context_tokens": 4,
        "embed_dim": 12,
        "hidden_dim": 24,
        "num_heads": 3,
        "num_layers": 1,
        "pred_dim": 12,
        "pred_depth": 1,
        "dropout": 0.0,
        "uniform_power": True,
        "norm_eps": 1e-6,
        "loss_exp": 1.0,
        "input_mean": 0.5,
        "input_std": 0.5,
        "amp_dtype": None,
        "compile": False,
        "tf32": False,
        "required_device": "cpu",
        "eval_every_epochs": 1,
        "train_eval_every_epochs": 0,
        "checkpoint_every_epochs": 1,
        "shortcut_diagnostic_batches": 1,
    }


class TinyVideoDataset(Dataset):
    def __init__(self, split, count, seed):
        generator = torch.Generator().manual_seed(seed)
        self.videos = torch.rand(count, 4, 1, 16, 16, generator=generator)
        self.split = split

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        return {
            "video": self.videos[index],
            "split": self.split,
            "sequence_id": f"{self.split}-{index}",
            "subject_id": f"subject-{index}",
        }


def tiny_loaders(seed=7):
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        TinyVideoDataset("train", 4, seed),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    val = DataLoader(
        TinyVideoDataset("val", 2, seed + 1), batch_size=2, shuffle=False
    )
    return train, val


class SingleStreamJEPATest(unittest.TestCase):
    def test_cuda_kernel_preflight_reports_incompatible_torch_build(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.zeros",
                side_effect=RuntimeError(
                    "CUDA error: no kernel image is available for execution on the device"
                ),
            ),
            patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch("torch.cuda.get_arch_list", return_value=["sm_70", "sm_80"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "cuda_compute_capability=sm_90") as caught:
                resolve_device("cuda")
        message = str(caught.exception)
        self.assertIn("torch_cuda_arch_list=['sm_70', 'sm_80']", message)
        self.assertIn("torch_has_required_cuda_arch=False", message)
        self.assertIn("uv sync --frozen --reinstall-package torch", message)
        self.assertIn("PyTorch CUDA 12.8+ build", message)
        self.assertIn("python_executable=", message)

    def test_cuda_compile_requires_working_triton_before_first_batch(self):
        cfg = tiny_config()
        cfg["compile"] = True
        with (
            patch("torch.utils._triton.has_triton", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "TorchInductor cannot use Triton") as caught:
                _maybe_compile(torch.nn.Identity(), cfg, torch.device("cuda"))
        message = str(caught.exception)
        self.assertIn("CONFIG['compile'] = False", message)
        self.assertIn("--reinstall-package triton", message)

    def test_multiblock_masks_are_deterministic_disjoint_and_full_tubes(self):
        cfg = tiny_config()
        first = multiblock_mask(cfg, 3, random.Random(11))
        second = multiblock_mask(cfg, 3, random.Random(11))
        different = multiblock_mask(cfg, 3, random.Random(12))
        cells_per_step = 16
        temporal_grid = 2
        self.assertTrue(
            any(not torch.equal(a["pred"], b["pred"]) for a, b in zip(first, different))
        )
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left["ctx"], right["ctx"]))
            self.assertTrue(torch.equal(left["pred"], right["pred"]))
            for context, target in zip(left["ctx"], left["pred"]):
                self.assertFalse(torch.isin(context, target).any())
                for indices in (context, target):
                    spatial = indices.remainder(cells_per_step).unique()
                    self.assertEqual(indices.numel(), spatial.numel() * temporal_grid)
                    for cell in spatial:
                        expected = torch.arange(temporal_grid) * cells_per_step + cell
                        actual = indices[indices.remainder(cells_per_step) == cell]
                        self.assertTrue(torch.equal(actual, expected))

    def test_masked_pixels_cannot_leak_into_context_encoder(self):
        cfg = tiny_config()
        encoder = VisionTransformer(
            cfg["embed_dim"], cfg["hidden_dim"], cfg["num_heads"], cfg["num_layers"],
            cfg["patch_size"], cfg["tubelet_size"], cfg["num_frames"],
            cfg["img_size"], cfg["in_channels"], dropout=0.0,
        ).eval()
        video = torch.rand(1, 4, 1, 16, 16)
        group = multiblock_mask(cfg, 1, random.Random(13))[0]
        changed = video.clone()
        target_cells = group["pred"][0].remainder(16).unique()
        for cell in target_cells:
            row, column = divmod(int(cell), 4)
            changed[:, :, :, row * 4 : (row + 1) * 4, column * 4 : (column + 1) * 4] = 1.0
        with torch.no_grad():
            original = encoder(video, group["ctx"])
            perturbed = encoder(changed, group["ctx"])
        self.assertTrue(torch.equal(original, perturbed))
        context_changed = video.clone()
        context_cell = int(group["ctx"][0, 0].remainder(16))
        row, column = divmod(context_cell, 4)
        context_changed[
            :, :, :, row * 4 : (row + 1) * 4, column * 4 : (column + 1) * 4
        ] = 1.0
        with torch.no_grad():
            positive_control = encoder(context_changed, group["ctx"])
        self.assertFalse(torch.equal(original, positive_control))

    def test_loss_matches_configured_power_formula(self):
        predicted = torch.tensor([[[1.0, -1.0]]])
        target = torch.zeros_like(predicted)
        l1, _ = _prediction_metrics(predicted, target, 1.0)
        l2, _ = _prediction_metrics(predicted, target, 2.0)
        self.assertEqual(float(l1), 1.0)
        self.assertEqual(float(l2), 0.5)

    def test_optimizer_groups_partition_parameters_and_exclude_bias_norm_masks(self):
        cfg = tiny_config()
        encoder, _, predictor = build_models(cfg, torch.device("cpu"))
        groups = optimizer_param_groups(encoder, predictor, 0.05)
        grouped = [parameter for group in groups for parameter in group["params"]]
        expected = [
            parameter
            for module in (encoder, predictor)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in expected})
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        for name, parameter in predictor.named_parameters():
            if parameter.ndim <= 1 or name.endswith("bias") or "mask_tokens" in name:
                self.assertIn(id(parameter), no_decay_ids)

    def test_lr_and_ema_schedules_have_exact_endpoints(self):
        cfg = tiny_config()
        self.assertEqual(learning_rate_for_step(cfg, 1), cfg["start_lr"])
        self.assertEqual(learning_rate_for_step(cfg, 2), cfg["lr"])
        self.assertAlmostEqual(learning_rate_for_step(cfg, cfg["steps"]), cfg["min_lr"])
        self.assertAlmostEqual(ema_tau_for_step(cfg, 1), cfg["ema_start"])
        self.assertAlmostEqual(ema_tau_for_step(cfg, cfg["steps"]), cfg["ema_end"])

    def test_representation_diagnostics_detect_collapse(self):
        collapsed = representation_diagnostics(torch.ones(8, 12))
        varied = representation_diagnostics(torch.randn(32, 12))
        self.assertEqual(collapsed["effective_rank"], 0.0)
        self.assertEqual(collapsed["near_zero_variance_fraction"], 1.0)
        self.assertGreater(varied["effective_rank"], 1.0)
        self.assertLess(varied["near_zero_variance_fraction"], 1.0)

    def test_batch_boundary_rejects_wrong_split_and_nonfinite_input(self):
        cfg = tiny_config()
        batch = next(iter(tiny_loaders()[0]))
        with self.assertRaisesRegex(ValueError, "split"):
            video_from_batch(batch, torch.device("cpu"), cfg, "val")
        corrupted = copy.deepcopy(batch)
        corrupted["video"][0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            video_from_batch(corrupted, torch.device("cpu"), cfg, "train")

    def test_epoch_boundary_checkpoint_resume_matches_uninterrupted_training(self):
        cfg = tiny_config(num_epochs=2)
        contract = {"loader": "tiny-v1"}
        full_train, full_val = tiny_loaders()
        full = train_jepa(cfg, full_train, full_val, contract, device="cpu")

        first_cfg = tiny_config(num_epochs=1)
        first_train, first_val = tiny_loaders()
        first = train_jepa(first_cfg, first_train, first_val, contract, device="cpu")
        resumed_train, resumed_val = tiny_loaders()
        resumed = train_jepa(
            cfg,
            resumed_train,
            resumed_val,
            contract,
            resume_state=first["checkpoint_state"],
            device="cpu",
        )
        self.assertEqual(full["global_step"], resumed["global_step"])
        self.assertEqual(full["completed_epochs"], resumed["completed_epochs"])
        for full_parameter, resumed_parameter in zip(
            full["context_encoder"].parameters(), resumed["context_encoder"].parameters()
        ):
            self.assertTrue(torch.equal(full_parameter, resumed_parameter))
        for full_parameter, resumed_parameter in zip(
            full["target_encoder"].parameters(), resumed["target_encoder"].parameters()
        ):
            self.assertTrue(torch.equal(full_parameter, resumed_parameter))
        for full_parameter, resumed_parameter in zip(
            full["predictor"].parameters(), resumed["predictor"].parameters()
        ):
            self.assertTrue(torch.equal(full_parameter, resumed_parameter))
        self.assertEqual(
            full["optimizer"].state_dict()["param_groups"],
            resumed["optimizer"].state_dict()["param_groups"],
        )
        full_optimizer_state = full["optimizer"].state_dict()["state"]
        resumed_optimizer_state = resumed["optimizer"].state_dict()["state"]
        self.assertEqual(full_optimizer_state.keys(), resumed_optimizer_state.keys())
        for parameter_id in full_optimizer_state:
            for key, value in full_optimizer_state[parameter_id].items():
                resumed_value = resumed_optimizer_state[parameter_id][key]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, resumed_value))
                else:
                    self.assertEqual(value, resumed_value)
        self.assertEqual(full["best_epoch"], resumed["best_epoch"])
        self.assertEqual(
            full["checkpoint_state"]["mask_rng_state"],
            resumed["checkpoint_state"]["mask_rng_state"],
        )
        self.assertTrue(torch.equal(
            full["checkpoint_state"]["loader_rng_state"],
            resumed["checkpoint_state"]["loader_rng_state"],
        ))
        semantic_keys = {
            "epoch", "step", "lr", "ema_tau", "grad_norm", "train_loss",
            "train_cosine", "train_examples", "val", "train_eval",
        }
        self.assertEqual(
            [{key: row[key] for key in semantic_keys} for row in full["history"]],
            [{key: row[key] for key in semantic_keys} for row in resumed["history"]],
        )
        self.assertEqual(
            full["checkpoint_state"]["scaler"],
            resumed["checkpoint_state"]["scaler"],
        )

    def test_resume_rejects_changed_data_contract(self):
        cfg = tiny_config(num_epochs=1)
        train, val = tiny_loaders()
        result = train_jepa(cfg, train, val, {"dataset": "a"}, device="cpu")
        with self.assertRaisesRegex(ValueError, "dataset/loader"):
            validate_resume_state(
                result["checkpoint_state"], cfg, DEFAULT_MASK_GROUPS, {"dataset": "b"}
            )

    def test_atomic_latest_and_best_checkpoints_are_written(self):
        cfg = tiny_config(num_epochs=1)
        train, val = tiny_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            train_jepa(cfg, train, val, {"dataset": "a"}, checkpoint_dir=tmp, device="cpu")
            self.assertTrue((Path(tmp) / "latest.pt").is_file())
            self.assertTrue((Path(tmp) / "best_loss.pt").is_file())
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_subject_aware_context_sources_never_pair_the_same_subject(self):
        subjects = ["A", "A", "A", "B", "B", "C"]
        sources = subject_aware_context_sources(subjects)
        self.assertIsNotNone(sources)
        self.assertTrue(
            all(subjects[index] != subjects[source] for index, source in enumerate(sources))
        )
        self.assertIsNone(subject_aware_context_sources(["A", "A"]))

    def test_aligned_completion_reports_max_steps(self):
        cfg = tiny_config(num_epochs=1)
        cfg["steps"] = 2
        train, val = tiny_loaders()
        result = train_jepa(cfg, train, val, {"dataset": "a"}, device="cpu")
        self.assertEqual(result["termination_reason"], "max_steps_at_epoch_boundary")

    def test_final_latest_is_saved_even_when_checkpoint_cadence_misses(self):
        cfg = tiny_config(num_epochs=1)
        cfg["checkpoint_every_epochs"] = 2
        train, val = tiny_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            train_jepa(cfg, train, val, {"dataset": "a"}, checkpoint_dir=tmp, device="cpu")
            self.assertTrue((Path(tmp) / "latest.pt").is_file())

    def test_invalid_accumulation_and_cadence_fail_before_training(self):
        cfg = tiny_config()
        cfg["accumulation_steps"] = 3
        train, val = tiny_loaders()
        with self.assertRaisesRegex(ValueError, "not divisible"):
            train_jepa(cfg, train, val, {"dataset": "a"}, device="cpu")
        cfg = tiny_config()
        cfg["eval_every_epochs"] = 0
        train, val = tiny_loaders()
        with self.assertRaisesRegex(ValueError, "eval_every_epochs"):
            train_jepa(cfg, train, val, {"dataset": "a"}, device="cpu")

    def test_multi_microbatch_accumulation_uses_exact_update_count(self):
        cfg = tiny_config(num_epochs=1)
        cfg.update({
            "batch_size": 1,
            "accumulation_steps": 2,
            "steps": 2,
            "warmup_steps": 1,
        })
        train = DataLoader(
            TinyVideoDataset("train", 4, cfg["seed"]),
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
        val = DataLoader(
            TinyVideoDataset("val", 2, cfg["seed"] + 1),
            batch_size=2,
            shuffle=False,
        )
        result = train_jepa(cfg, train, val, {"dataset": "a"}, device="cpu")
        self.assertEqual(result["global_step"], 2)
        self.assertEqual(result["history"][0]["train_examples"], 4)
        self.assertAlmostEqual(result["history"][0]["ema_tau"], cfg["ema_end"])

    def test_tiny_end_to_end_run_learns(self):
        cfg = tiny_config(num_epochs=10)
        cfg.update({"steps": 20, "warmup_steps": 2})
        train, val = tiny_loaders()
        result = train_jepa(cfg, train, val, {"dataset": "learnability"}, device="cpu")
        initial_loss = result["history"][0]["val"]["loss"]
        final_loss = result["history"][-1]["val"]["loss"]
        self.assertLess(final_loss, initial_loss - 0.02)


if __name__ == "__main__":
    unittest.main()
