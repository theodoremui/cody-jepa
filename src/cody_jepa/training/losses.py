"""JEPA prediction and anti-collapse objectives."""

import math

import torch
import torch.nn.functional as F


def prediction_metrics(predicted, target, loss_exp):
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target shapes differ: {predicted.shape} vs {target.shape}"
        )
    if loss_exp < 1:
        raise ValueError("loss_exp must be at least 1")
    error = torch.abs(predicted - target).pow(loss_exp) / loss_exp
    per_example_loss = error.mean(dim=(1, 2))
    per_example_cosine = F.cosine_similarity(predicted, target, dim=-1).mean(dim=1)
    return per_example_loss, per_example_cosine


def vicreg(features, gamma=1.0):
    """Compute VICReg variance and covariance penalties over feature rows."""
    if features.ndim == 3:
        features = features.reshape(-1, features.size(-1))
    if features.ndim != 2:
        raise ValueError("features must be [N, D] or [B, K, D]")
    if features.size(0) < 2:
        raise ValueError("variance/covariance penalties need at least 2 samples")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    features = features.float()
    centered = features - features.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    standard_deviation = torch.sqrt(variance + 1e-4)
    variance_loss = F.relu(gamma - standard_deviation).mean()
    covariance = centered.T @ centered / (features.size(0) - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    covariance_loss = off_diagonal.pow(2).sum() / features.size(1)
    return variance_loss, covariance_loss


@torch.no_grad()
def encode_targets(target_encoder, video, return_pre_norm=False, batch_standardize=False):
    """Encode EMA targets and optionally standardize each feature dimension."""
    encoded = target_encoder(video, return_pre_norm=return_pre_norm)
    if return_pre_norm:
        normalized, pre_norm = encoded
    else:
        normalized, pre_norm = encoded, None
    normalized = F.layer_norm(normalized, (normalized.size(-1),))
    if batch_standardize:
        flat = normalized.reshape(-1, normalized.size(-1)).float()
        if flat.size(0) < 2:
            raise ValueError("batch standardization needs at least 2 tokens")
        mean = flat.mean(dim=0)
        standard_deviation = flat.std(dim=0, unbiased=False).clamp_min(1e-4)
        normalized = ((normalized - mean) / standard_deviation).to(normalized.dtype)
    return normalized, pre_norm


def group_forward(
    context_encoder,
    predictor,
    video,
    all_target_tokens,
    mask_group,
    mask_index,
    loss_exp,
    return_context_tokens=False,
    return_token_losses=False,
):
    context_indices = mask_group["ctx"]
    target_indices = mask_group["pred"]
    context_tokens = context_encoder(video, context_indices)
    gather = target_indices.unsqueeze(-1).expand(-1, -1, all_target_tokens.size(-1))
    target_tokens = torch.gather(all_target_tokens, 1, gather)
    predicted = predictor(
        context_tokens, context_indices, target_indices, mask_index=mask_index
    )
    loss, cosine = prediction_metrics(predicted, target_tokens, loss_exp)
    if return_token_losses:
        token_losses = (
            torch.abs(predicted - target_tokens).pow(loss_exp).mean(dim=-1) / loss_exp
        )
    if return_context_tokens and return_token_losses:
        return loss, cosine, context_tokens, token_losses
    if return_context_tokens:
        return loss, cosine, context_tokens
    if return_token_losses:
        return loss, cosine, token_losses
    return loss, cosine


__all__ = ["encode_targets", "group_forward", "prediction_metrics", "vicreg"]
