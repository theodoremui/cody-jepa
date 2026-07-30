"""Compatibility surface for the decomposed single-stream JEPA implementation.

New code should import from :mod:`cody_jepa.models`, :mod:`cody_jepa.masks`,
or :mod:`cody_jepa.training`. This module preserves the original research API
and checkpoint-facing names while downstream callers migrate.
"""

from .masks import DEFAULT_MASK_GROUPS, MaskGroupConfig, multiblock_mask
from .models import AttentionBlock, Predictor, VisionTransformer, build_encoder
from .models.factory import build_models
from .models.position import sincos_3d_position_embedding
from .training.batches import video_from_batch
from .training.checkpoint import (
    CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    atomic_torch_save as _atomic_torch_save,
    checkpoint_config as _checkpoint_config,
    checkpoint_payload as _checkpoint_payload,
    healthy_checkpoint_path,
    load_checkpoint,
    validate_resume_state,
)
from .training.config import validate_training_config
from .training.diagnostics import (
    balanced_wrong_subject_permutation,
    context_shuffle_plan as _context_shuffle_plan,
    context_source_loader as _context_source_loader,
    representation_diagnostics,
    representation_health,
    subject_balanced_mean as _subject_balanced_mean,
)
from .training.engine import train_jepa
from .training.evaluation import evaluate_jepa
from .training.losses import (
    encode_targets,
    group_forward,
    prediction_metrics as _prediction_metrics,
    vicreg,
)
from .training.optimization import (
    ema_tau_for_step,
    ema_update,
    learning_rate_for_step,
    optimizer_param_groups,
    scale_gradients as _scale_gradients,
    set_optimizer_lr as _set_optimizer_lr,
)
from .training.runtime import (
    amp_dtype as _amp_dtype,
    autocast_context as _autocast_context,
    make_scaler as _make_scaler,
    maybe_compile as _maybe_compile,
    resolve_device,
    validate_compile_runtime as _validate_compile_runtime,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEFAULT_MASK_GROUPS",
    "MODEL_ARCHITECTURE",
    "AttentionBlock",
    "MaskGroupConfig",
    "Predictor",
    "VisionTransformer",
    "balanced_wrong_subject_permutation",
    "build_encoder",
    "build_models",
    "ema_tau_for_step",
    "ema_update",
    "encode_targets",
    "evaluate_jepa",
    "group_forward",
    "healthy_checkpoint_path",
    "learning_rate_for_step",
    "load_checkpoint",
    "multiblock_mask",
    "optimizer_param_groups",
    "representation_diagnostics",
    "resolve_device",
    "sincos_3d_position_embedding",
    "train_jepa",
    "validate_resume_state",
    "validate_training_config",
    "vicreg",
    "video_from_batch",
]
