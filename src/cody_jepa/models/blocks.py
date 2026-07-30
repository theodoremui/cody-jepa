"""Transformer building blocks shared by the encoder and predictor."""

import torch.nn as nn
import torch.nn.functional as F


def init_linear(linear):
    nn.init.trunc_normal_(linear.weight, std=0.02)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, norm_eps=1e-6):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.attn.in_proj_weight, std=0.02)
        nn.init.zeros_(self.attn.in_proj_bias)
        init_linear(self.attn.out_proj)
        init_linear(self.linear_1)
        init_linear(self.linear_2)

    def forward(self, inputs):
        normalized = self.layer_norm_1(inputs)
        inputs = inputs + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        hidden = self.linear_1(self.layer_norm_2(inputs))
        hidden = self.dropout(F.gelu(hidden))
        hidden = self.dropout(self.linear_2(hidden))
        return inputs + hidden


__all__ = ["AttentionBlock", "init_linear"]
