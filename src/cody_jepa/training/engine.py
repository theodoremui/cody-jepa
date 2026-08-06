"""Single-stream JEPA training engine."""

import math
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
from torch import optim

from cody_jepa.masks import DEFAULT_MASK_GROUPS, multiblock_mask
from cody_jepa.models.factory import build_models

from .batches import video_from_batch
from .checkpoint import atomic_torch_save, checkpoint_payload, validate_resume_state
from .config import validate_training_config
from .evaluation import evaluate_jepa
from .losses import encode_targets, group_forward, vicreg
from .optimization import (
    ema_tau_for_step,
    ema_update,
    learning_rate_for_step,
    optimizer_param_groups,
    scale_gradients,
    set_optimizer_lr,
)
from .runtime import autocast_context, make_scaler, maybe_compile, resolve_device


def train_jepa(
    config,
    train_loader,
    val_loader,
    data_contract,
    checkpoint_dir=None,
    resume_state=None,
    device=None,
    mask_groups=DEFAULT_MASK_GROUPS,
    train_eval_loader=None,
):
    updates_per_epoch = validate_training_config(config, train_loader)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = resolve_device(
        config.get("required_device", "auto") if device is None else device
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(config.get("tf32", False))

    context_encoder, target_encoder, predictor = build_models(
        config, device, mask_groups
    )
    param_groups = optimizer_param_groups(
        context_encoder, predictor, config["weight_decay"]
    )
    optimizer = optim.AdamW(param_groups, lr=float(config["lr"]))
    scaler = make_scaler(config, device)
    trainable_parameters = [
        parameter
        for module in (context_encoder, predictor)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    loader_generator = getattr(train_loader, "generator", None)
    if isinstance(loader_generator, torch.Generator):
        loader_generator.manual_seed(seed)
    mask_rng = random.Random(seed)
    global_step = 0
    completed_epochs = 0
    history = []
    best_val_loss = math.inf
    best_epoch = None
    best_healthy_val_loss = math.inf
    best_healthy_epoch = None

    if resume_state is not None:
        validate_resume_state(resume_state, config, mask_groups, data_contract)
        context_encoder.load_state_dict(resume_state["context_encoder"], strict=True)
        target_encoder.load_state_dict(resume_state["target_encoder"], strict=True)
        predictor.load_state_dict(resume_state["predictor"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        if scaler.is_enabled() and resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        history = list(resume_state.get("history", []))
        global_step = int(resume_state["global_step"])
        completed_epochs = int(resume_state["completed_epochs"])
        best_val_loss = float(resume_state.get("best_val_loss", math.inf))
        best_epoch = resume_state.get("best_epoch")
        best_healthy_val_loss = float(
            resume_state.get("best_healthy_val_loss", math.inf)
        )
        best_healthy_epoch = resume_state.get("best_healthy_epoch")
        mask_rng.setstate(resume_state["mask_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if (
            isinstance(loader_generator, torch.Generator)
            and resume_state.get("loader_rng_state") is not None
        ):
            loader_generator.set_state(resume_state["loader_rng_state"])
        if device.type == "cuda" and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])

    context_runner = maybe_compile(context_encoder, config, device)
    target_runner = maybe_compile(target_encoder, config, device)
    predictor_runner = maybe_compile(predictor, config, device)
    accumulation_steps = int(config["accumulation_steps"])
    max_steps = int(config["steps"])
    var_coef = float(config.get("var_coef", 0.0))
    cov_coef = float(config.get("cov_coef", 0.0))
    clip_var_coef = float(config.get("clip_var_coef", 0.0))
    clip_cov_coef = float(config.get("clip_cov_coef", 0.0))
    var_gamma = float(config.get("var_gamma", 1.0))
    token_regularization_active = var_coef > 0.0 or cov_coef > 0.0
    clip_regularization_active = clip_var_coef > 0.0 or clip_cov_coef > 0.0
    if int(config["num_epochs"]) * updates_per_epoch < max_steps:
        print(
            "warning: num_epochs can produce only "
            f"{int(config['num_epochs']) * updates_per_epoch} of {max_steps} requested steps"
        )
    start_time = time.perf_counter()
    processed_examples = 0
    termination_reason = "num_epochs"

    for epoch in range(completed_epochs, int(config["num_epochs"])):
        if global_step >= max_steps:
            termination_reason = "max_steps_at_epoch_boundary"
            break
        dataset = getattr(train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        pending_examples = 0
        pending_microbatches = 0
        epoch_loss_sum = epoch_cosine_sum = 0.0
        epoch_variance_sum = epoch_covariance_sum = 0.0
        epoch_clip_variance_sum = epoch_clip_covariance_sum = 0.0
        epoch_examples = 0
        grad_norm_sum = 0.0
        grad_updates = 0
        last_lr = None
        last_tau = None
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch_index, batch in enumerate(train_loader):
            next_step = global_step + 1
            if pending_microbatches == 0:
                last_lr = learning_rate_for_step(config, next_step)
                set_optimizer_lr(optimizer, last_lr)
            video = video_from_batch(batch, device, config, "train")
            masks = multiblock_mask(
                config,
                video.size(0),
                mask_rng,
                device=device,
                mask_groups=mask_groups,
            )
            with autocast_context(config, device):
                targets, _ = encode_targets(
                    target_runner,
                    video,
                    batch_standardize=bool(
                        config.get("target_batch_standardize", False)
                    ),
                )
            batch_loss = batch_cosine = 0.0
            for mask_index, mask_group in enumerate(masks):
                with autocast_context(config, device):
                    losses, cosines, context_tokens = group_forward(
                        context_runner,
                        predictor_runner,
                        video,
                        targets,
                        mask_group,
                        mask_index,
                        float(config.get("loss_exp", 1.0)),
                        return_context_tokens=True,
                    )
                    total_group_loss = losses.mean()
                float_context_tokens = (
                    context_tokens.float()
                    if token_regularization_active or clip_regularization_active
                    else None
                )
                if token_regularization_active:
                    # Compute covariance in float32 outside autocast; gradients
                    # still flow into the encoder.
                    variance_loss, covariance_loss = vicreg(
                        float_context_tokens, var_gamma
                    )
                    total_group_loss = (
                        total_group_loss
                        + var_coef * variance_loss
                        + cov_coef * covariance_loss
                    )
                    epoch_variance_sum += (
                        float(variance_loss.detach()) * video.size(0) / len(masks)
                    )
                    epoch_covariance_sum += (
                        float(covariance_loss.detach()) * video.size(0) / len(masks)
                    )
                if clip_regularization_active:
                    # Pool one masked-context representation per clip, matching
                    # the axis used by the representation-health gate.
                    if video.size(0) < 2:
                        raise ValueError(
                            "clip-level VICReg requires at least 2 examples in "
                            "every training microbatch; use drop_last=True"
                        )
                    clip_variance_loss, clip_covariance_loss = vicreg(
                        float_context_tokens.mean(dim=1), var_gamma
                    )
                    total_group_loss = (
                        total_group_loss
                        + clip_var_coef * clip_variance_loss
                        + clip_cov_coef * clip_covariance_loss
                    )
                    epoch_clip_variance_sum += (
                        float(clip_variance_loss.detach()) * video.size(0) / len(masks)
                    )
                    epoch_clip_covariance_sum += (
                        float(clip_covariance_loss.detach()) * video.size(0) / len(masks)
                    )
                if not torch.isfinite(total_group_loss):
                    raise FloatingPointError("non-finite JEPA loss before backward")
                scaler.scale(total_group_loss * video.size(0)).backward()
                batch_loss += float(losses.detach().sum()) / len(masks)
                batch_cosine += float(cosines.detach().sum()) / len(masks)
            pending_examples += video.size(0)
            pending_microbatches += 1
            epoch_examples += video.size(0)
            processed_examples += video.size(0)
            epoch_loss_sum += batch_loss
            epoch_cosine_sum += batch_cosine
            is_boundary = (
                pending_microbatches == accumulation_steps
                or batch_index + 1 == len(train_loader)
            )
            if is_boundary:
                scaler.unscale_(optimizer)
                scale_gradients(
                    trainable_parameters, pending_examples * len(mask_groups)
                )
                max_grad_norm = float(config.get("grad_clip", 0.0))
                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_grad_norm if max_grad_norm > 0 else math.inf,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                last_tau = ema_tau_for_step(config, global_step)
                ema_update(target_encoder, context_encoder, last_tau)
                optimizer.zero_grad(set_to_none=True)
                pending_examples = pending_microbatches = 0
                grad_norm_sum += float(grad_norm)
                grad_updates += 1

        if epoch_examples == 0:
            raise RuntimeError("train_loader produced no batches")
        completed_epochs = epoch + 1
        should_eval = (
            completed_epochs % int(config.get("eval_every_epochs", 1)) == 0
            or global_step >= max_steps
            or completed_epochs == int(config["num_epochs"])
        )
        val_metrics = None
        if should_eval:
            val_metrics = evaluate_jepa(
                context_runner,
                target_runner,
                predictor_runner,
                val_loader,
                config,
                device,
                "val",
                mask_seed=seed + 1,
                mask_groups=mask_groups,
                context_shuffle=True,
                context_seed=seed + 2,
            )
        train_eval_metrics = None
        train_eval_every = int(config.get("train_eval_every_epochs", 0))
        if train_eval_loader is not None and train_eval_every > 0 and (
            completed_epochs % train_eval_every == 0 or global_step >= max_steps
        ):
            train_eval_metrics = evaluate_jepa(
                context_runner,
                target_runner,
                predictor_runner,
                train_eval_loader,
                config,
                device,
                "train",
                mask_seed=seed + 2,
                mask_groups=mask_groups,
                context_shuffle=False,
            )
        epoch_seconds = time.perf_counter() - epoch_started
        metrics = {
            "epoch": completed_epochs,
            "step": global_step,
            "lr": last_lr,
            "ema_tau": last_tau,
            "grad_norm": grad_norm_sum / max(1, grad_updates),
            "train_loss": epoch_loss_sum / epoch_examples,
            "train_cosine": epoch_cosine_sum / epoch_examples,
            "train_variance_loss": (
                epoch_variance_sum / epoch_examples
                if token_regularization_active
                else None
            ),
            "train_covariance_loss": (
                epoch_covariance_sum / epoch_examples
                if token_regularization_active
                else None
            ),
            "train_clip_variance_loss": (
                epoch_clip_variance_sum / epoch_examples
                if clip_regularization_active
                else None
            ),
            "train_clip_covariance_loss": (
                epoch_clip_covariance_sum / epoch_examples
                if clip_regularization_active
                else None
            ),
            "train_examples": epoch_examples,
            "epoch_seconds": epoch_seconds,
            "examples_per_second": epoch_examples / max(epoch_seconds, 1e-12),
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
            "val": val_metrics,
            "train_eval": train_eval_metrics,
        }
        history.append(metrics)
        selection_metric = str(
            config.get("selection_metric", "subject_balanced_loss")
        )
        if val_metrics is not None and selection_metric not in val_metrics:
            raise KeyError(f"unknown selection_metric {selection_metric!r}")
        if val_metrics is not None and val_metrics[selection_metric] < best_val_loss:
            best_val_loss = val_metrics[selection_metric]
            best_epoch = completed_epochs
        if (
            val_metrics is not None
            and val_metrics["representations_healthy"]
            and val_metrics[selection_metric] < best_healthy_val_loss
        ):
            best_healthy_val_loss = val_metrics[selection_metric]
            best_healthy_epoch = completed_epochs

        payload = checkpoint_payload(
            config,
            mask_groups,
            data_contract,
            context_encoder,
            target_encoder,
            predictor,
            optimizer,
            scaler,
            history,
            global_step,
            completed_epochs,
            mask_rng,
            loader_generator,
            best_val_loss,
            best_epoch,
            best_healthy_val_loss,
            best_healthy_epoch,
            study_metadata=config.get("study_metadata"),
        )
        if checkpoint_dir is not None and (
            completed_epochs % int(config.get("checkpoint_every_epochs", 1)) == 0
            or global_step >= max_steps
            or completed_epochs == int(config["num_epochs"])
        ):
            atomic_torch_save(payload, Path(checkpoint_dir) / "latest.pt")
        if checkpoint_dir is not None and val_metrics is not None:
            if best_epoch == completed_epochs:
                atomic_torch_save(payload, Path(checkpoint_dir) / "best_loss.pt")
            if best_healthy_epoch == completed_epochs:
                atomic_torch_save(payload, Path(checkpoint_dir) / "best_healthy.pt")

        val_text = (
            f" | val_loss={val_metrics['loss']:.4f}, "
            f"effective_rank={val_metrics['effective_rank']:.1f}, "
            "subject_balanced_context_shuffle_loss_gap="
            f"{val_metrics['subject_balanced_context_shuffle_loss_gap']:.4f}"
            if val_metrics is not None
            else ""
        )
        print(
            f"epoch={completed_epochs:03d} | step={global_step:05d} | "
            f"lr={last_lr:.2e} | train_loss={metrics['train_loss']:.4f}, "
            f"train_cosine={metrics['train_cosine']:.4f}{val_text}"
        )

    if global_step >= max_steps:
        termination_reason = "max_steps_at_epoch_boundary"
    elif completed_epochs >= int(config["num_epochs"]):
        termination_reason = "num_epochs_before_max_steps"
        print(
            f"warning: num_epochs ended at step {global_step}, below configured {max_steps}"
        )
    elapsed = time.perf_counter() - start_time
    return {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
        "optimizer": optimizer,
        "history": history,
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_healthy_val_loss": best_healthy_val_loss,
        "best_healthy_epoch": best_healthy_epoch,
        "termination_reason": termination_reason,
        "elapsed_seconds": elapsed,
        "examples_per_second": processed_examples / elapsed if elapsed > 0 else float("nan"),
        "checkpoint_state": checkpoint_payload(
            config,
            mask_groups,
            data_contract,
            context_encoder,
            target_encoder,
            predictor,
            optimizer,
            scaler,
            history,
            global_step,
            completed_epochs,
            mask_rng,
            loader_generator,
            best_val_loss,
            best_epoch,
            best_healthy_val_loss,
            best_healthy_epoch,
            study_metadata=config.get("study_metadata"),
        ),
    }


__all__ = ["train_jepa"]
