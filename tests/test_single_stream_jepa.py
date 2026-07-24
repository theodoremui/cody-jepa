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
    CHECKPOINT_SCHEMA,
    DEFAULT_MASK_GROUPS,
    VisionTransformer,
    balanced_wrong_subject_permutation,
    build_models,
    checkpoint_model_state_sha256,
    ema_tau_for_step,
    encode_targets,
    evaluate_jepa,
    healthy_checkpoint_path,
    learning_rate_for_step,
    load_checkpoint,
    multiblock_mask,
    optimizer_param_groups,
    representation_diagnostics,
    resolve_device,
    train_jepa,
    validate_resume_state,
    vicreg_regularization,
    video_from_batch,
)
from cody_jepa.single_stream_jepa import (
    _context_shuffle_plan,
    _context_source_loader,
    _maybe_compile,
    _prediction_metrics,
    _subject_balanced_mean,
)


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
    }


class TinyVideoDataset(Dataset):
    def __init__(self, split, count, seed, subjects=None):
        generator = torch.Generator().manual_seed(seed)
        self.videos = torch.rand(count, 4, 1, 16, 16, generator=generator)
        self.split = split
        self.subjects = (
            list(subjects)
            if subjects is not None
            else [f"subject-{index}" for index in range(count)]
        )
        if len(self.subjects) != count:
            raise ValueError("subjects must match count")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        return {
            "video": self.videos[index],
            "split": self.split,
            "sequence_id": f"{self.split}-{index}",
            "subject_id": self.subjects[index],
        }

    def subject_id_at(self, index):
        return self.subjects[index]


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

    def test_evaluation_diagnostics_use_normalized_full_view_context_features(self):
        cfg = tiny_config(num_epochs=1)
        context, target, predictor = build_models(cfg, torch.device("cpu"))
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()
        _, val = tiny_loaders()
        metrics = evaluate_jepa(
            context,
            target,
            predictor,
            val,
            cfg,
            torch.device("cpu"),
            "val",
            mask_seed=17,
            context_shuffle=False,
        )

        pooled = []
        for batch in val:
            video = video_from_batch(batch, torch.device("cpu"), cfg, "val")
            with torch.no_grad():
                pooled.append(context(video).mean(dim=1))
        expected = representation_diagnostics(torch.cat(pooled, dim=0))
        for key in (
            "feature_std",
            "near_zero_variance_fraction",
            "effective_rank",
            "effective_rank_ratio",
        ):
            self.assertAlmostEqual(metrics[key], expected[key], places=6)
        self.assertEqual(
            metrics["representation_source"],
            "context_encoder_final_norm_full_view_mean_pool",
        )

    def test_global_wrong_subject_permutation_is_seeded_balanced_and_complete(self):
        subjects = ["A"] * 4 + ["b"] * 3 + ["C"] * 2
        first = balanced_wrong_subject_permutation(subjects, seed=19)
        repeated = balanced_wrong_subject_permutation(subjects, seed=19)
        self.assertEqual(first, repeated)
        self.assertGreater(len({
            tuple(balanced_wrong_subject_permutation(subjects, seed=seed))
            for seed in range(10)
        }), 1)
        self.assertEqual(sorted(first), list(range(len(subjects))))
        self.assertTrue(all(
            subjects[target].casefold() != subjects[source].casefold()
            for target, source in enumerate(first)
        ))
        self.assertIsNone(
            balanced_wrong_subject_permutation(["A"] * 5 + ["B"] * 3, seed=1)
        )
        boundary = ["A", "a", "B", "b"]
        boundary_sources = balanced_wrong_subject_permutation(boundary, seed=3)
        self.assertEqual(sorted(boundary_sources), list(range(len(boundary))))
        self.assertTrue(all(
            boundary[target].casefold() != boundary[source].casefold()
            for target, source in enumerate(boundary_sources)
        ))

    def test_subject_balanced_mean_weights_subjects_equally(self):
        values = {"frequent": [1.0, 3.0, 5.0], "rare": [9.0]}
        self.assertEqual(_subject_balanced_mean(values), 6.0)
        self.assertNotEqual(_subject_balanced_mean(values), 4.5)

    def test_context_shuffle_diagnostic_covers_the_full_ordered_validation_set(self):
        cfg = tiny_config(num_epochs=1)
        subjects = ["A"] * 4 + ["B"] * 3 + ["C"] * 2
        dataset = TinyVideoDataset("val", len(subjects), cfg["seed"], subjects)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        context, target, predictor = build_models(cfg, torch.device("cpu"))
        metrics = evaluate_jepa(
            context,
            target,
            predictor,
            loader,
            cfg,
            torch.device("cpu"),
            "val",
            mask_seed=23,
            context_shuffle=True,
            context_seed=29,
        )
        self.assertEqual(metrics["context_shuffle_pairs"], len(dataset))
        self.assertEqual(metrics["context_shuffle_unique_sources"], len(dataset))
        self.assertEqual(metrics["context_shuffle_batches"], len(loader))
        self.assertEqual(metrics["context_shuffle_subjects"], 3)
        self.assertTrue(math.isfinite(metrics["context_shuffle_loss_gap"]))
        self.assertTrue(math.isfinite(
            metrics["subject_balanced_context_shuffle_loss_gap"]
        ))

    def test_context_shuffle_uses_planned_sources_and_subject_balanced_gap(self):
        cfg = tiny_config(num_epochs=1)
        subjects = ["A"] * 3 + ["B"] * 2 + ["C"]
        values = [0.05, 0.15, 0.25, 0.55, 0.70, 0.95]
        dataset = TinyVideoDataset("val", len(subjects), cfg["seed"], subjects)
        for video, value in zip(dataset.videos, values):
            video.fill_(value)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        class ScalarContext(torch.nn.Module):
            def forward(self, video, token_indices=None):
                scalar = video.mean(dim=(1, 2, 3, 4), keepdim=True)
                count = cfg["num_tokens"] if token_indices is None else token_indices.size(1)
                return scalar.reshape(video.size(0), 1, 1).expand(
                    -1, count, cfg["embed_dim"]
                )

        class ZeroTarget(torch.nn.Module):
            def forward(self, video, return_pre_norm=False):
                output = torch.zeros(
                    video.size(0), cfg["num_tokens"], cfg["embed_dim"]
                )
                return (output, output) if return_pre_norm else output

        class ContextValuePredictor(torch.nn.Module):
            def forward(
                self, context_tokens, context_indices, target_indices, mask_index
            ):
                value = context_tokens[:, :1, :]
                return value.expand(-1, target_indices.size(1), -1)

        context_seed = 53
        metrics = evaluate_jepa(
            ScalarContext(),
            ZeroTarget(),
            ContextValuePredictor(),
            loader,
            cfg,
            torch.device("cpu"),
            "val",
            mask_seed=47,
            context_shuffle=True,
            context_seed=context_seed,
        )
        sources = balanced_wrong_subject_permutation(subjects, context_seed)
        normalized = [abs((value - cfg["input_mean"]) / cfg["input_std"]) for value in values]
        gaps_by_subject = {}
        all_gaps = []
        for target, source in enumerate(sources):
            gap = normalized[source] - normalized[target]
            all_gaps.append(gap)
            gaps_by_subject.setdefault(subjects[target], []).append(gap)
        expected_example = sum(all_gaps) / len(all_gaps)
        expected_subject = _subject_balanced_mean(gaps_by_subject)
        self.assertAlmostEqual(metrics["context_shuffle_loss_gap"], expected_example, places=6)
        self.assertAlmostEqual(
            metrics["subject_balanced_context_shuffle_loss_gap"],
            expected_subject,
            places=6,
        )
        self.assertNotAlmostEqual(expected_example, expected_subject, places=6)

    def test_context_shuffle_fails_closed_for_infeasible_subject_distribution(self):
        cfg = tiny_config(num_epochs=1)
        subjects = ["A"] * 5 + ["B"] * 3
        dataset = TinyVideoDataset("val", len(subjects), cfg["seed"], subjects)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        context, target, predictor = build_models(cfg, torch.device("cpu"))
        metrics = evaluate_jepa(
            context,
            target,
            predictor,
            loader,
            cfg,
            torch.device("cpu"),
            "val",
            mask_seed=31,
            context_shuffle=True,
            context_seed=37,
        )
        self.assertEqual(metrics["context_shuffle_status"], "infeasible_subject_distribution")
        self.assertEqual(metrics["context_shuffle_pairs"], 0)
        self.assertEqual(metrics["context_shuffle_batches"], 0)
        self.assertEqual(metrics["context_shuffle_unique_sources"], 0)
        self.assertTrue(math.isnan(metrics["context_shuffle_loss_gap"]))
        self.assertTrue(math.isnan(
            metrics["subject_balanced_context_shuffle_loss_gap"]
        ))
        self.assertFalse(metrics["representations_healthy"])
        self.assertIn("context_shuffle_unavailable", metrics["health_issues"])

    def test_context_shuffle_rejects_nonsequential_or_incomplete_loaders(self):
        cfg = tiny_config(num_epochs=1)
        dataset = TinyVideoDataset("val", 4, cfg["seed"])
        context, target, predictor = build_models(cfg, torch.device("cpu"))
        for loader, message in (
            (DataLoader(dataset, batch_size=2, shuffle=True), "sequential"),
            (DataLoader(dataset, batch_size=3, shuffle=False, drop_last=True), "drop_last"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                evaluate_jepa(
                    context,
                    target,
                    predictor,
                    loader,
                    cfg,
                    torch.device("cpu"),
                    "val",
                    mask_seed=41,
                    context_shuffle=True,
                    context_seed=43,
                )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        with self.assertRaisesRegex(ValueError, "context_seed"):
            evaluate_jepa(
                context,
                target,
                predictor,
                loader,
                cfg,
                torch.device("cpu"),
                "val",
                mask_seed=41,
                context_shuffle=True,
            )
        out_of_order = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=1, in_order=False
        )
        with self.assertRaisesRegex(ValueError, "in_order"):
            evaluate_jepa(
                context,
                target,
                predictor,
                out_of_order,
                cfg,
                torch.device("cpu"),
                "val",
                mask_seed=41,
                context_shuffle=True,
                context_seed=43,
            )

    def test_context_source_loader_preserves_worker_loading_contract(self):
        dataset = TinyVideoDataset("val", 6, 7)
        target_generator = torch.Generator().manual_seed(11)
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=1,
            prefetch_factor=3, pin_memory=True, persistent_workers=False,
            generator=target_generator,
        )
        plan = _context_shuffle_plan(loader, seed=61)
        source_loader = _context_source_loader(loader, plan, seed=67)

        self.assertEqual(plan["status"], "complete")
        self.assertIs(source_loader.dataset, loader.dataset)
        self.assertIs(source_loader.collate_fn, loader.collate_fn)
        self.assertEqual(source_loader.num_workers, 1)
        self.assertEqual(source_loader.prefetch_factor, 3)
        self.assertTrue(source_loader.pin_memory)
        self.assertFalse(source_loader.persistent_workers)
        self.assertIsNot(source_loader.generator, loader.generator)
        self.assertEqual(
            list(source_loader.batch_sampler), plan["source_index_batches"]
        )

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
        self.assertEqual(CHECKPOINT_SCHEMA, 4)
        self.assertEqual(result["checkpoint_state"]["schema"], CHECKPOINT_SCHEMA)
        self.assertEqual(
            result["checkpoint_state"]["model_state_sha256"],
            checkpoint_model_state_sha256(result["checkpoint_state"]),
        )
        self.assertEqual(
            result["checkpoint_state"]["best_loss_model_state_sha256"],
            result["checkpoint_state"]["model_state_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "dataset/loader"):
            validate_resume_state(
                result["checkpoint_state"], cfg, DEFAULT_MASK_GROUPS, {"dataset": "b"}
            )
        legacy = copy.deepcopy(result["checkpoint_state"])
        legacy["schema"] = 3
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_resume_state(
                legacy, cfg, DEFAULT_MASK_GROUPS, {"dataset": "a"}
            )
        tampered = copy.deepcopy(result["checkpoint_state"])
        first = next(iter(tampered["target_encoder"].values()))
        first.view(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_resume_state(
                tampered, cfg, DEFAULT_MASK_GROUPS, {"dataset": "a"}
            )

    def test_healthy_checkpoint_path_is_truthful_about_selection_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_healthy.pt"
            path.write_bytes(b"stale")
            self.assertIsNone(healthy_checkpoint_path(tmp, None))
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "healthy epoch 7"):
                healthy_checkpoint_path(tmp, 7)
            path.write_bytes(b"checkpoint")
            self.assertEqual(healthy_checkpoint_path(tmp, 7), path)

    def test_atomic_latest_and_best_checkpoints_are_written(self):
        cfg = tiny_config(num_epochs=1)
        train, val = tiny_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            train_jepa(cfg, train, val, {"dataset": "a"}, checkpoint_dir=tmp, device="cpu")
            self.assertTrue((Path(tmp) / "latest.pt").is_file())
            self.assertTrue((Path(tmp) / "best_loss.pt").is_file())
            latest = load_checkpoint(Path(tmp) / "latest.pt")
            best = load_checkpoint(Path(tmp) / "best_loss.pt")
            self.assertEqual(
                latest["best_loss_model_state_sha256"], best["model_state_sha256"]
            )
            self.assertEqual(
                best["model_state_sha256"], checkpoint_model_state_sha256(best)
            )
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

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


class AntiCollapseSafeguardTest(unittest.TestCase):
    def test_vicreg_flags_collapsed_and_redundant_features(self):
        collapsed = torch.ones(64, 8)
        variance_loss, covariance_loss = vicreg_regularization(collapsed)
        self.assertGreater(float(variance_loss), 0.9)
        self.assertAlmostEqual(float(covariance_loss), 0.0, places=6)

        generator = torch.Generator().manual_seed(0)
        base = torch.randn(4096, 1, generator=generator)
        base = (base - base.mean()) / base.std(unbiased=False)
        redundant = torch.cat([base, base], dim=1)
        variance_loss, covariance_loss = vicreg_regularization(redundant)
        self.assertLess(float(variance_loss), 0.01)
        self.assertGreater(float(covariance_loss), 0.9)

    def test_vicreg_is_small_for_healthy_features(self):
        generator = torch.Generator().manual_seed(1)
        healthy = torch.randn(8192, 6, generator=generator)
        healthy = (healthy - healthy.mean(dim=0)) / healthy.std(
            dim=0, unbiased=False
        )
        variance_loss, covariance_loss = vicreg_regularization(healthy)
        self.assertLess(float(variance_loss), 0.01)
        self.assertLess(float(covariance_loss), 0.01)

    def test_vicreg_flattens_token_features_and_validates_inputs(self):
        generator = torch.Generator().manual_seed(2)
        tokens = torch.randn(4, 8, 6, generator=generator)
        from_tokens = vicreg_regularization(tokens)
        from_flat = vicreg_regularization(tokens.reshape(-1, 6))
        self.assertAlmostEqual(
            float(from_tokens[0]), float(from_flat[0]), places=6
        )
        self.assertAlmostEqual(
            float(from_tokens[1]), float(from_flat[1]), places=6
        )
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            vicreg_regularization(torch.zeros(1, 4))
        with self.assertRaisesRegex(ValueError, "must be \\[N, D\\]"):
            vicreg_regularization(torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "gamma"):
            vicreg_regularization(torch.zeros(4, 4), gamma=0.0)

    def test_vicreg_gradients_reach_features(self):
        features = torch.randn(32, 4, requires_grad=True)
        variance_loss, covariance_loss = vicreg_regularization(features)
        (variance_loss + covariance_loss).backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_encode_targets_batch_standardization(self):
        cfg = tiny_config()
        _, target_encoder, _ = build_models(cfg, torch.device("cpu"))
        video = torch.rand(4, 4, 1, 16, 16)
        video = (video - 0.5) / 0.5
        default_targets, _ = encode_targets(target_encoder, video)
        standardized, _ = encode_targets(
            target_encoder, video, batch_standardize=True
        )
        self.assertEqual(default_targets.shape, standardized.shape)
        flat = standardized.reshape(-1, standardized.size(-1))
        self.assertTrue(torch.allclose(flat.mean(dim=0), torch.zeros(flat.size(1)), atol=1e-4))
        self.assertTrue(
            torch.allclose(
                flat.std(dim=0, unbiased=False), torch.ones(flat.size(1)), atol=1e-3
            )
        )
        self.assertFalse(torch.allclose(default_targets, standardized))

    def test_invalid_safeguard_config_fails_before_training(self):
        for key, value, message in (
            ("var_coef", -0.1, "var_coef"),
            ("cov_coef", float("nan"), "cov_coef"),
            ("var_gamma", 0.0, "var_gamma"),
            ("target_batch_standardize", "yes", "target_batch_standardize"),
        ):
            cfg = tiny_config()
            cfg[key] = value
            train, val = tiny_loaders()
            with self.assertRaisesRegex(ValueError, message):
                train_jepa(cfg, train, val, {"dataset": "a"}, device="cpu")

    def test_training_with_safeguards_runs_and_logs_regularization(self):
        cfg = tiny_config(num_epochs=2)
        cfg.update({
            "var_coef": 1.0,
            "cov_coef": 0.04,
            "var_gamma": 1.0,
            "target_batch_standardize": True,
        })
        train, val = tiny_loaders()
        result = train_jepa(cfg, train, val, {"dataset": "safeguards"}, device="cpu")
        for epoch_metrics in result["history"]:
            self.assertTrue(math.isfinite(epoch_metrics["train_variance_loss"]))
            self.assertTrue(math.isfinite(epoch_metrics["train_covariance_loss"]))
            self.assertGreaterEqual(epoch_metrics["train_variance_loss"], 0.0)
            self.assertGreaterEqual(epoch_metrics["train_covariance_loss"], 0.0)
            self.assertTrue(math.isfinite(epoch_metrics["train_loss"]))

    def test_training_without_safeguards_keeps_metrics_disabled(self):
        cfg = tiny_config(num_epochs=1)
        train, val = tiny_loaders()
        result = train_jepa(cfg, train, val, {"dataset": "baseline"}, device="cpu")
        self.assertIsNone(result["history"][0]["train_variance_loss"])
        self.assertIsNone(result["history"][0]["train_covariance_loss"])


if __name__ == "__main__":
    unittest.main()
