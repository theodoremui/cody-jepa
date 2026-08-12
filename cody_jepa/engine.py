"""JEPA training loop, schedules, and validation."""

import contextlib
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.torch_version import TorchVersion

from .losses import encode_targets, group_forward, prediction_loss, vicreg
from .masks import DEFAULT_MASK_GROUPS, multiblock_mask
from .models import build_models


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(config, device):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        config.get("amp_dtype")
    )
    if dtype is None or device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def learning_rate_for_step(config, step):
    """Linear warmup to `lr`, then cosine decay to `min_lr`."""
    base_lr = float(config["lr"])
    min_lr = float(config.get("min_lr", 0.0))
    warmup = int(config.get("warmup_steps", 0))
    if warmup and step <= warmup:
        start_lr = float(config.get("start_lr", base_lr))
        return start_lr + (step - 1) / max(1, warmup - 1) * (base_lr - start_lr)
    total = max(1, int(config["steps"]) - warmup)
    progress = min(1.0, (step - warmup) / total)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_tau_for_step(config, step):
    """Linear EMA momentum schedule from `ema_start` to `ema_end`."""
    start = float(config.get("ema_start", 0.998))
    end = float(config.get("ema_end", 1.0))
    total = max(1, int(config["steps"]) - 1)
    return start + (end - start) * min(1.0, (step - 1) / total)


@torch.no_grad()
def ema_update(target, online, tau):
    online_parameters = dict(online.named_parameters())
    for name, parameter in target.named_parameters():
        parameter.mul_(tau).add_(online_parameters[name], alpha=1.0 - tau)
    online_buffers = dict(online.named_buffers())
    for name, buffer in target.named_buffers():
        buffer.copy_(online_buffers[name])


def optimizer_param_groups(modules, weight_decay):
    """Weight-decay everything except biases, norms, and mask tokens."""
    decay, no_decay = [], []
    for module in modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            skip = parameter.ndim <= 1 or "mask_tokens" in name
            (no_decay if skip else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def accumulation_size(num_batches, batch_index, accumulation_steps):
    """Number of microbatches contributing to this optimizer update."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least one")
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, num_batches - group_start)


def video_from_batch(batch, device, config):
    video = batch["video"].to(device, non_blocking=True)
    mean = float(config.get("input_mean", 0.5))
    std = float(config.get("input_std", 0.5))
    return (video - mean) / std


def collapse_metrics(features):
    """Effective rank and feature spread — the signals that reveal collapse.

    Effective rank is exp(entropy of the normalized covariance eigenvalues): it
    equals D for isotropic features and 1 when every clip maps to one direction.
    """
    features = torch.as_tensor(features, dtype=torch.float32)
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, features.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        return {"effective_rank": 0.0, "feature_std": 0.0}
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0]
    return {
        "effective_rank": float(torch.exp(-(probabilities * probabilities.log()).sum())),
        "feature_std": float(centered.pow(2).mean(dim=0).sqrt().mean()),
    }


def prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mask_loss_with_context(predictor, context_tokens, targets, mask_group, mask_index, loss_exp):
    target_indices = mask_group["pred"]
    gather = target_indices.unsqueeze(-1).expand(-1, -1, targets.size(-1))
    target_tokens = torch.gather(targets, 1, gather)
    predicted = predictor(
        context_tokens, mask_group["ctx"], target_indices, mask_index
    )
    return prediction_loss(predicted, target_tokens, loss_exp)[0]


@torch.inference_mode()
def evaluate(context_encoder, target_encoder, predictor, loader, config, device, mask_groups):
    context_was_training = context_encoder.training
    target_was_training = target_encoder.training
    predictor_was_training = predictor.training
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()
    rng = random.Random(int(config.get("seed", 0)) + 1)
    total_loss = total_cosine = 0.0
    total_blank_delta = 0.0
    blank_examples = 0
    examples = 0
    pooled_clips, pooled_tokens = [], []

    for batch in loader:
        video = video_from_batch(batch, device, config)
        masks = multiblock_mask(config, video.size(0), rng, device, mask_groups)
        with autocast(config, device):
            _, target_pre_norm = target_encoder(video, return_pre_norm=True)
            pooled_clips.append(target_pre_norm.float().mean(dim=1).cpu())
            pooled_tokens.append(target_pre_norm.float().reshape(-1, target_pre_norm.size(-1)).cpu())
            targets = encode_targets(
                target_encoder,
                video,
                bool(config.get("target_batch_standardize", False)),
            )
            blank_video = torch.zeros_like(video)
            for mask_index, mask_group in enumerate(masks):
                loss, cosine, context_tokens = group_forward(
                    context_encoder,
                    predictor,
                    video,
                    targets,
                    mask_group,
                    mask_index,
                    float(config.get("loss_exp", 1.0)),
                )
                total_loss += float(loss.sum()) / len(masks)
                total_cosine += float(cosine.sum()) / len(masks)
                blank_context_tokens = context_encoder(blank_video, mask_group["ctx"])
                blank_loss = _mask_loss_with_context(
                    predictor,
                    blank_context_tokens,
                    targets,
                    mask_group,
                    mask_index,
                    float(config.get("loss_exp", 1.0)),
                )
                total_blank_delta += float((blank_loss - loss).sum()) / len(masks)
        blank_examples += video.size(0)
        examples += video.size(0)

    context_encoder.train(context_was_training)
    target_encoder.train(target_was_training)
    predictor.train(predictor_was_training)
    clip_metrics = collapse_metrics(torch.cat(pooled_clips))
    token_metrics = collapse_metrics(torch.cat(pooled_tokens))
    return {
        "loss": total_loss / examples,
        "cosine": total_cosine / examples,
        "blank_context_loss_delta": total_blank_delta / blank_examples,
        "effective_rank": clip_metrics["effective_rank"],
        "feature_std": clip_metrics["feature_std"],
        **prefix_metrics("clip", clip_metrics),
        **prefix_metrics("token", token_metrics),
    }


def save_checkpoint(
    path,
    config,
    context_encoder,
    target_encoder,
    predictor,
    optimizer,
    epoch,
    global_step,
    best_loss,
    mask_rng,
    scaler,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "context_encoder": context_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "mask_rng_state": mask_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "scaler": scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    # Older CoDy-JEPA checkpoints store torch.__version__ as TorchVersion. It is
    # a known PyTorch value type, so allowlisting only it preserves safe loading
    # without enabling arbitrary pickle execution.
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(Path(path), map_location=device, weights_only=True)


def train_jepa(
    config,
    train_loader,
    tune_loader,
    output_dir=None,
    device=None,
    mask_groups=DEFAULT_MASK_GROUPS,
    resume=None,
):
    """Train a single-stream masked JEPA. Returns the run history."""
    device = resolve_device(config.get("device", "auto")) if device is None else device
    torch.manual_seed(int(config.get("seed", 0)))

    context_encoder, target_encoder, predictor = build_models(
        config, device, len(mask_groups)
    )
    optimizer = optim.AdamW(
        optimizer_param_groups(
            (context_encoder, predictor), config.get("weight_decay", 0.0)
        ),
        lr=float(config["lr"]),
    )
    start_epoch = 0
    if resume is not None:
        context_encoder.load_state_dict(resume["context_encoder"])
        target_encoder.load_state_dict(resume["target_encoder"])
        predictor.load_state_dict(resume["predictor"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume.get("epoch", resume.get("completed_epochs", 0)))

    trainable = [
        parameter
        for module in (context_encoder, predictor)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.get("amp_dtype") == "float16"
    )
    mask_rng = random.Random(int(config.get("seed", 0)))
    accumulation = int(config.get("accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("accumulation_steps must be at least one")
    max_steps = int(config["steps"])
    clip_var_coef = float(config.get("clip_var_coef", config.get("var_coef", 0.0)))
    clip_cov_coef = float(config.get("clip_cov_coef", config.get("cov_coef", 0.0)))
    token_var_coef = float(config.get("token_var_coef", 0.0))
    token_cov_coef = float(config.get("token_cov_coef", 0.0))
    var_gamma = float(config.get("var_gamma", 1.0))
    regularize_tokens = token_var_coef > 0.0 or token_cov_coef > 0.0
    regularize_clips = clip_var_coef > 0.0 or clip_cov_coef > 0.0

    history = []
    best_loss = math.inf
    global_step = 0
    if resume is not None:
        global_step = int(resume.get("global_step", 0))
        best_loss = float(resume.get("best_loss", resume.get("best_val_loss", math.inf)))
        if "mask_rng_state" in resume:
            mask_rng.setstate(resume["mask_rng_state"])
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])

    for epoch in range(start_epoch, int(config["num_epochs"])):
        if global_step >= max_steps:
            break
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        context_encoder.train()
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = epoch_cosine = 0.0
        epoch_examples = pending = 0
        epoch_started = time.perf_counter()

        for batch_index, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break
            if pending == 0:
                learning_rate = learning_rate_for_step(config, global_step + 1)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            current_accumulation = accumulation_size(
                len(train_loader), batch_index, accumulation
            )
            video = video_from_batch(batch, device, config)
            masks = multiblock_mask(config, video.size(0), mask_rng, device, mask_groups)
            clip_regularizer = video.new_zeros(())
            with autocast(config, device):
                if regularize_clips:
                    _, pre_norm_tokens = context_encoder(video, return_pre_norm=True)
                    clip_features = pre_norm_tokens.float().mean(dim=1)
                    variance, covariance = vicreg(clip_features, var_gamma)
                    clip_regularizer = (
                        clip_var_coef * variance + clip_cov_coef * covariance
                    )
                targets = encode_targets(
                    target_encoder,
                    video,
                    bool(config.get("target_batch_standardize", False)),
                )
            for mask_index, mask_group in enumerate(masks):
                with autocast(config, device):
                    loss, cosine, context_tokens = group_forward(
                        context_encoder,
                        predictor,
                        video,
                        targets,
                        mask_group,
                        mask_index,
                        float(config.get("loss_exp", 1.0)),
                    )
                    total = loss.mean()
                if regularize_tokens:
                    # Computed in float32 outside autocast; gradients still reach
                    # the encoder.
                    variance, covariance = vicreg(context_tokens.float(), var_gamma)
                    total = total + token_var_coef * variance + token_cov_coef * covariance
                if mask_index == 0:
                    total = total + clip_regularizer * len(masks)
                scaler.scale(total / len(masks) / current_accumulation).backward()
                epoch_loss += float(loss.detach().sum()) / len(masks)
                epoch_cosine += float(cosine.detach().sum()) / len(masks)
            epoch_examples += video.size(0)
            pending += 1

            if pending == accumulation or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                grad_clip = float(config.get("grad_clip", 0.0))
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                ema_update(
                    target_encoder, context_encoder, ema_tau_for_step(config, global_step)
                )
                pending = 0

        metrics = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": epoch_loss / epoch_examples,
            "train_cosine": epoch_cosine / epoch_examples,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if (epoch + 1) % int(config.get("eval_every_epochs", 1)) == 0 or global_step >= max_steps:
            metrics["tune"] = evaluate(
                context_encoder, target_encoder, predictor, tune_loader, config, device, mask_groups
            )
        history.append(metrics)

        tune = metrics.get("tune")
        summary = (
            f" | tune_loss={tune['loss']:.4f}"
            f" clip_rank={tune['clip_effective_rank']:.1f}"
            f" blank_delta={tune['blank_context_loss_delta']:.2e}"
            if tune
            else ""
        )
        print(
            f"epoch {metrics['epoch']:3d} | step {global_step:5d} | "
            f"train_loss={metrics['train_loss']:.4f}{summary}"
        )

        if output_dir is not None:
            is_best = tune is not None and tune["loss"] < best_loss
            if is_best:
                best_loss = tune["loss"]
            arguments = (
                config,
                context_encoder,
                target_encoder,
                predictor,
                optimizer,
                epoch + 1,
                global_step,
                best_loss,
                mask_rng,
                scaler,
            )
            save_checkpoint(Path(output_dir) / "last.pt", *arguments)
            if is_best:
                save_checkpoint(Path(output_dir) / "best.pt", *arguments)

    return history
