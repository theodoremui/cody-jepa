"""Training configuration and loader-contract validation."""

import math


def validate_training_config(config, train_loader):
    """Validate the resolved run once, before any model or optimizer is created."""
    positive_integers = (
        "steps",
        "num_epochs",
        "batch_size",
        "accumulation_steps",
        "eval_every_epochs",
        "checkpoint_every_epochs",
    )
    for key in positive_integers:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
    train_eval_every = config.get("train_eval_every_epochs", 0)
    if not isinstance(train_eval_every, int) or train_eval_every < 0:
        raise ValueError("train_eval_every_epochs must be a non-negative integer")
    for key in ("lr", "start_lr", "min_lr"):
        value = float(config.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if float(config["lr"]) <= 0:
        raise ValueError("lr must be positive")
    warmup_steps = config.get("warmup_steps", 0)
    if not isinstance(warmup_steps, int) or not 0 <= warmup_steps <= config["steps"]:
        raise ValueError("warmup_steps must be an integer in [0, steps]")
    ema_start = float(config.get("ema_start", 0.998))
    ema_end = float(config.get("ema_end", 1.0))
    if not 0.0 <= ema_start <= ema_end <= 1.0:
        raise ValueError("EMA schedule must satisfy 0 <= ema_start <= ema_end <= 1")
    loss_exp = float(config.get("loss_exp", 1.0))
    if not math.isfinite(loss_exp) or loss_exp < 1.0:
        raise ValueError("loss_exp must be finite and at least 1")
    for key in ("var_coef", "cov_coef", "clip_var_coef", "clip_cov_coef"):
        value = float(config.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    clip_regularization = any(
        float(config.get(key, 0.0)) > 0.0
        for key in ("clip_var_coef", "clip_cov_coef")
    )
    if clip_regularization and config["batch_size"] < 2:
        raise ValueError("clip-level VICReg requires batch_size >= 2")
    var_gamma = float(config.get("var_gamma", 1.0))
    if not math.isfinite(var_gamma) or var_gamma <= 0:
        raise ValueError("var_gamma must be finite and positive")
    if not isinstance(config.get("target_batch_standardize", False), bool):
        raise ValueError("target_batch_standardize must be a bool")
    input_mean = float(config.get("input_mean", 0.5))
    input_std = float(config.get("input_std", 0.5))
    if not math.isfinite(input_mean) or not math.isfinite(input_std) or input_std <= 0:
        raise ValueError("input normalization must be finite with input_std > 0")
    if float(config.get("weight_decay", 0.0)) < 0:
        raise ValueError("weight_decay must be non-negative")
    if float(config.get("grad_clip", 0.0)) < 0:
        raise ValueError("grad_clip must be non-negative")
    if config.get("selection_metric", "subject_balanced_loss") not in {
        "loss",
        "subject_balanced_loss",
    }:
        raise ValueError("selection_metric must be a loss metric that is minimized")
    if not isinstance(config.get("compile", False), bool):
        raise ValueError("compile must be a bool")

    microbatches = len(train_loader)
    if microbatches <= 0:
        raise ValueError("train loader must contain at least one microbatch")
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and int(loader_batch_size) != config["batch_size"]:
        raise ValueError(
            f"config batch_size={config['batch_size']} differs from loader "
            f"batch_size={loader_batch_size}"
        )
    accumulation_steps = config["accumulation_steps"]
    if microbatches % accumulation_steps:
        raise ValueError(
            f"train loader has {microbatches} microbatches, not divisible by "
            f"accumulation_steps={accumulation_steps}"
        )
    updates_per_epoch = microbatches // accumulation_steps
    if config["steps"] % updates_per_epoch:
        raise ValueError(
            f"steps={config['steps']} must be divisible by "
            f"updates_per_epoch={updates_per_epoch} for exact epoch-boundary checkpoints"
        )
    return updates_per_epoch


__all__ = ["validate_training_config"]
