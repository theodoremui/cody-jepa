"""Masked-token predictor."""

import math

import torch
import torch.nn as nn

from .blocks import AttentionBlock, init_linear
from .encoder import VisionTransformer
from .position import sincos_3d_position_embedding


class Predictor(nn.Module):
    def __init__(
        self,
        dim,
        pred_dim,
        depth,
        num_heads,
        grid_size,
        num_mask_tokens,
        dropout=0.0,
        uniform_power=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.num_tokens = math.prod(grid_size)
        self.embed = nn.Linear(dim, pred_dim)
        init_linear(self.embed)
        self.register_buffer(
            "pos",
            sincos_3d_position_embedding(
                *grid_size, pred_dim, uniform_power=uniform_power
            ),
        )
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, pred_dim)) for _ in range(num_mask_tokens)]
        )
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    pred_dim,
                    pred_dim * 4,
                    num_heads,
                    dropout=dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(pred_dim, eps=norm_eps)
        self.out = nn.Linear(pred_dim, dim)
        init_linear(self.out)

    def _position_tokens(self, indices, batch_size):
        indices = VisionTransformer._batched_indices(
            indices, batch_size, self.num_tokens, self.pos.device
        )
        positions = self.pos.expand(batch_size, -1, -1)
        gather = indices.unsqueeze(-1).expand(-1, -1, positions.size(-1))
        return torch.gather(positions, 1, gather), indices

    def forward(self, context_tokens, context_indices, target_indices, mask_index):
        batch = context_tokens.size(0)
        context_pos, context_indices = self._position_tokens(context_indices, batch)
        target_pos, target_indices = self._position_tokens(target_indices, batch)
        if context_tokens.size(1) != context_indices.size(1):
            raise ValueError("context token/index lengths differ")
        if target_indices.size(1) == 0:
            raise ValueError("target mask is empty")
        context = self.embed(context_tokens) + context_pos
        masked = self.mask_tokens[mask_index].expand(
            batch, target_indices.size(1), -1
        ) + target_pos
        hidden = torch.cat([context, masked], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden[:, -target_indices.size(1) :, :])
        return self.out(hidden)


__all__ = ["Predictor"]
