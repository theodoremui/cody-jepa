"""Versioned, exact-resume training checkpoints."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch


# Change the architecture for incompatible model graphs and the schema for
# incompatible training-state payloads.
MODEL_ARCHITECTURE = "cody-jepa-single-stream-masked-v3"
CHECKPOINT_SCHEMA = 5


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_config(config):
    return {
        key: value
        for key, value in config.items()
        if isinstance(value, (str, int, float, bool, type(None), tuple, list, dict))
    }


def checkpoint_payload(
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
):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": MODEL_ARCHITECTURE,
        "config": checkpoint_config(config),
        "mask_groups": [asdict(group) for group in mask_groups],
        "data_contract": data_contract,
        "context_encoder": context_encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "history": history,
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_healthy_val_loss": best_healthy_val_loss,
        "best_healthy_epoch": best_healthy_epoch,
        "mask_rng_state": mask_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": (
            loader_generator.get_state()
            if isinstance(loader_generator, torch.Generator)
            else None
        ),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }


def validate_resume_state(state, config, mask_groups, data_contract):
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is incompatible")
    if state.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError("checkpoint architecture is incompatible")
    if state.get("mask_groups") != [asdict(group) for group in mask_groups]:
        raise ValueError("checkpoint mask groups differ from this run")
    if state.get("data_contract") != data_contract:
        raise ValueError("checkpoint dataset/loader contract differs from this run")
    mutable = {
        "num_epochs",
        "eval_every_epochs",
        "train_eval_every_epochs",
        "checkpoint_every_epochs",
        # Each effective command-line override is compared independently.
        "config_overrides",
    }
    saved_config = state.get("config", {})
    current_config = checkpoint_config(config)
    optional_zero_defaults = {"clip_var_coef", "clip_cov_coef"}
    for key in set(saved_config) | set(current_config):
        default = 0.0 if key in optional_zero_defaults else None
        if key not in mutable and saved_config.get(key, default) != current_config.get(key, default):
            raise ValueError(f"checkpoint config mismatch for {key!r}")


def load_checkpoint(path):
    """Load tensor and RNG state without permitting arbitrary pickle execution."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def healthy_checkpoint_path(checkpoint_dir, best_healthy_epoch):
    if best_healthy_epoch is None:
        return None
    path = Path(checkpoint_dir) / "best_healthy.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"healthy epoch {best_healthy_epoch} was selected but {path} was not written"
        )
    return path


__all__ = [
    "CHECKPOINT_SCHEMA",
    "MODEL_ARCHITECTURE",
    "atomic_torch_save",
    "checkpoint_payload",
    "healthy_checkpoint_path",
    "load_checkpoint",
    "validate_resume_state",
]
