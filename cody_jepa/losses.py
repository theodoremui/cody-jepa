"""JEPA prediction loss and the VICReg anti-collapse terms."""

import torch
import torch.nn.functional as F


def prediction_loss(predicted, target, loss_exp=1.0):
    """Per-example prediction error and cosine similarity to the target."""
    error = torch.abs(predicted - target).pow(loss_exp) / loss_exp
    loss = error.mean(dim=(1, 2))
    cosine = F.cosine_similarity(predicted, target, dim=-1).mean(dim=1)
    return loss, cosine


def vicreg(features, gamma=1.0):
    """Variance and covariance penalties that keep embeddings from collapsing."""
    if features.ndim == 3:
        features = features.reshape(-1, features.size(-1))
    features = features.float()
    centered = features - features.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.pow(2).mean(dim=0) + 1e-4)
    variance_loss = F.relu(gamma - standard_deviation).mean()
    covariance = centered.T @ centered / (features.size(0) - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    covariance_loss = off_diagonal.pow(2).sum() / features.size(1)
    return variance_loss, covariance_loss


@torch.no_grad()
def encode_targets(target_encoder, video, batch_standardize=False):
    """Encode EMA targets, layer-normed and optionally standardized per dimension."""
    tokens = target_encoder(video)
    tokens = F.layer_norm(tokens, (tokens.size(-1),))
    if batch_standardize:
        flat = tokens.reshape(-1, tokens.size(-1)).float()
        mean = flat.mean(dim=0)
        standard_deviation = flat.std(dim=0, unbiased=False).clamp_min(1e-4)
        tokens = ((tokens - mean) / standard_deviation).to(tokens.dtype)
    return tokens


def group_forward(
    context_encoder, predictor, video, targets, mask_group, mask_index, loss_exp=1.0
):
    """Encode the context tubes, predict the target tubes, and score them."""
    context_indices = mask_group["ctx"]
    target_indices = mask_group["pred"]
    context_tokens = context_encoder(video, context_indices)
    gather = target_indices.unsqueeze(-1).expand(-1, -1, targets.size(-1))
    target_tokens = torch.gather(targets, 1, gather)
    predicted = predictor(context_tokens, context_indices, target_indices, mask_index)
    loss, cosine = prediction_loss(predicted, target_tokens, loss_exp)
    return loss, cosine, context_tokens
