"""Canonical construction path for trainable JEPA components."""

from .encoder import build_encoder
from .predictor import Predictor
from cody_jepa.masks import DEFAULT_MASK_GROUPS


def build_models(config, device, mask_groups=DEFAULT_MASK_GROUPS):
    context_encoder = build_encoder(config, device)
    target_encoder = build_encoder(config, device)
    target_encoder.load_state_dict(context_encoder.state_dict())
    target_encoder.requires_grad_(False).eval()
    predictor = Predictor(
        config["embed_dim"],
        config["pred_dim"],
        config["pred_depth"],
        config["num_heads"],
        context_encoder.grid_size,
        len(mask_groups),
        config.get("dropout", 0.0),
        config.get("uniform_power", True),
        config.get("norm_eps", 1e-6),
    ).to(device)
    return context_encoder, target_encoder, predictor


__all__ = ["build_models"]
