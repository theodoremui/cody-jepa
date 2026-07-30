"""Training engine and runtime utilities."""

from .checkpoint import (
    CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    healthy_checkpoint_path,
    load_checkpoint,
    validate_resume_state,
)
from .config import validate_training_config
from .diagnostics import (
    balanced_wrong_subject_permutation,
    representation_diagnostics,
    representation_health,
)
from .engine import train_jepa
from .evaluation import evaluate_jepa
from .losses import encode_targets, group_forward, vicreg
from .optimization import (
    ema_tau_for_step,
    ema_update,
    learning_rate_for_step,
    optimizer_param_groups,
)
from .runtime import resolve_device

__all__ = [
    "CHECKPOINT_SCHEMA",
    "MODEL_ARCHITECTURE",
    "balanced_wrong_subject_permutation",
    "ema_tau_for_step",
    "ema_update",
    "encode_targets",
    "evaluate_jepa",
    "group_forward",
    "healthy_checkpoint_path",
    "learning_rate_for_step",
    "load_checkpoint",
    "optimizer_param_groups",
    "representation_diagnostics",
    "representation_health",
    "resolve_device",
    "train_jepa",
    "validate_resume_state",
    "validate_training_config",
    "vicreg",
]
