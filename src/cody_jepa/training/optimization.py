"""Optimizer partitioning, learning-rate scheduling, and EMA updates."""

import math

import torch


def optimizer_param_groups(context_encoder, predictor, weight_decay):
    decay, no_decay = [], []
    seen = set()
    for module_name, module in (("encoder", context_encoder), ("predictor", predictor)):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise RuntimeError(f"duplicate trainable parameter: {module_name}.{name}")
            seen.add(id(parameter))
            destination = (
                no_decay
                if parameter.ndim <= 1 or name.endswith("bias") or "mask_tokens" in name
                else decay
            )
            destination.append(parameter)
    expected = sum(
        parameter.requires_grad
        for module in (context_encoder, predictor)
        for parameter in module.parameters()
    )
    if len(seen) != expected:
        raise RuntimeError("optimizer parameter partition is incomplete")
    return [
        {"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]


@torch.no_grad()
def ema_update(target, online, tau):
    if not 0.0 <= tau <= 1.0:
        raise ValueError("EMA tau must be in [0, 1]")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("target and online encoder structures differ")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(tau).add_(online_parameters[name], alpha=1.0 - tau)
    online_buffers = dict(online.named_buffers())
    for name, target_buffer in target.named_buffers():
        target_buffer.copy_(online_buffers[name])


def learning_rate_for_step(config, step):
    start_lr = float(config.get("start_lr", config["lr"]))
    base_lr = float(config["lr"])
    min_lr = float(config.get("min_lr", 0.0))
    warmup_steps = max(0, int(config.get("warmup_steps", 0)))
    max_steps = max(1, int(config["steps"]))
    step = max(1, int(step))
    if warmup_steps and step <= warmup_steps:
        if warmup_steps == 1:
            return base_lr
        fraction = (step - 1) / (warmup_steps - 1)
        return start_lr + fraction * (base_lr - start_lr)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def ema_tau_for_step(config, step):
    start = float(config.get("ema_start", 0.998))
    end = float(config.get("ema_end", 1.0))
    max_steps = max(1, int(config["steps"]))
    progress = 1.0 if max_steps == 1 else min(1.0, max(0.0, (step - 1) / (max_steps - 1)))
    return start + (end - start) * progress


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def scale_gradients(parameters, denominator):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


__all__ = [
    "ema_tau_for_step",
    "ema_update",
    "learning_rate_for_step",
    "optimizer_param_groups",
    "scale_gradients",
    "set_optimizer_lr",
]
