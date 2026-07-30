"""Video transformer encoder."""

import math

import torch
import torch.nn as nn

from .blocks import AttentionBlock
from .position import sincos_3d_position_embedding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        patch_size,
        tubelet_size,
        num_frames,
        image_size,
        in_channels,
        dropout=0.0,
        uniform_power=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        if self.num_frames % self.tubelet_size or self.image_size % self.patch_size:
            raise ValueError("tubelet/patch sizes must divide the configured input")
        self.grid_size = (
            self.num_frames // self.tubelet_size,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
        )
        self.num_patches = math.prod(self.grid_size)
        self.patch_embed = nn.Conv3d(
            self.in_channels,
            embed_dim,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            stride=(self.tubelet_size, self.patch_size, self.patch_size),
        )
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        self.register_buffer(
            "pos_embedding",
            sincos_3d_position_embedding(
                *self.grid_size, embed_dim, uniform_power=uniform_power
            ),
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [
                AttentionBlock(
                    embed_dim,
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    @staticmethod
    def _batched_indices(indices, batch_size, num_tokens, device):
        indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        if indices.ndim != 2 or indices.size(0) != batch_size:
            raise ValueError("token indices must have shape [B, K] or [K]")
        if indices.numel() and (indices.min() < 0 or indices.max() >= num_tokens):
            raise IndexError(f"token indices must be in [0, {num_tokens})")
        return indices

    def forward(self, video, token_indices=None, return_pre_norm=False):
        if video.ndim != 5:
            raise ValueError("video must be [B, T, C, H, W]")
        batch, frames, channels, height, width = video.shape
        expected = (self.num_frames, self.in_channels, self.image_size, self.image_size)
        if (frames, channels, height, width) != expected:
            raise ValueError(f"expected [T,C,H,W]={expected}, got {video.shape[1:]}")
        tokens = self.patch_embed(video.permute(0, 2, 1, 3, 4).contiguous())
        tokens = tokens.flatten(2).transpose(1, 2)
        positions = self.pos_embedding.expand(batch, -1, -1)
        if token_indices is not None:
            token_indices = self._batched_indices(
                token_indices, batch, self.num_patches, video.device
            )
            gather = token_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
            tokens = torch.gather(tokens, 1, gather)
            positions = torch.gather(positions, 1, gather)
        tokens = self.dropout(tokens + positions)
        for block in self.transformer:
            tokens = block(tokens)
        normalized = self.norm(tokens)
        return (normalized, tokens) if return_pre_norm else normalized


def build_encoder(config, device):
    """Build an encoder from a resolved training configuration."""
    return VisionTransformer(
        config["embed_dim"],
        config["hidden_dim"],
        config["num_heads"],
        config["num_layers"],
        config["patch_size"],
        config["tubelet_size"],
        config["num_frames"],
        config["img_size"],
        config["in_channels"],
        config.get("dropout", 0.0),
        config.get("uniform_power", True),
        config.get("norm_eps", 1e-6),
    ).to(device)


__all__ = ["AttentionBlock", "VisionTransformer", "build_encoder"]
