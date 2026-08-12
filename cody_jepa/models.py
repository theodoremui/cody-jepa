"""Vision transformer encoder and masked-token predictor."""

import math

import torch
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


def _sincos_1d(coordinates, embed_dim):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    angles = coordinates.reshape(-1, 1).float() * omega.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def sincos_3d_position_embedding(temporal_grid, height_grid, width_grid, embed_dim):
    """Fixed 3D sin-cos position embedding, one third of the channels per axis."""
    if embed_dim <= 0 or embed_dim % 6:
        raise ValueError("3D position embed_dim must be positive and divisible by 6")
    per_axis = embed_dim // 3
    time, height, width = torch.meshgrid(
        torch.arange(temporal_grid),
        torch.arange(height_grid),
        torch.arange(width_grid),
        indexing="ij",
    )
    embedding = torch.cat(
        [
            _sincos_1d(time, per_axis),
            _sincos_1d(height, per_axis),
            _sincos_1d(width, per_axis),
        ],
        dim=1,
    )
    return embedding.unsqueeze(0)


def _gather_tokens(tokens, indices):
    gather = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, 1, gather)


def _batched_indices(indices, batch_size, device):
    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(batch_size, -1)
    return indices


class VisionTransformer(nn.Module):
    """Encodes [B, T, C, H, W] video into patch tokens, optionally a subset."""

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
            "pos_embedding", sincos_3d_position_embedding(*self.grid_size, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout, norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(self, video, token_indices=None, return_pre_norm=False):
        batch = video.size(0)
        tokens = self.patch_embed(video.permute(0, 2, 1, 3, 4).contiguous())
        tokens = tokens.flatten(2).transpose(1, 2)
        positions = self.pos_embedding.expand(batch, -1, -1)
        if token_indices is not None:
            token_indices = _batched_indices(token_indices, batch, video.device)
            tokens = _gather_tokens(tokens, token_indices)
            positions = _gather_tokens(positions, token_indices)
        tokens = self.dropout(tokens + positions)
        for block in self.transformer:
            tokens = block(tokens)
        normalized = self.norm(tokens)
        return (normalized, tokens) if return_pre_norm else normalized


class Predictor(nn.Module):
    """Predicts target-block embeddings from encoded context tokens."""

    def __init__(
        self,
        dim,
        pred_dim,
        depth,
        num_heads,
        grid_size,
        num_mask_tokens,
        dropout=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.embed = nn.Linear(dim, pred_dim)
        init_linear(self.embed)
        self.register_buffer("pos", sincos_3d_position_embedding(*grid_size, pred_dim))
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, pred_dim)) for _ in range(num_mask_tokens)]
        )
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(pred_dim, pred_dim * 4, num_heads, dropout, norm_eps)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(pred_dim, eps=norm_eps)
        self.out = nn.Linear(pred_dim, dim)
        init_linear(self.out)

    def forward(self, context_tokens, context_indices, target_indices, mask_index):
        batch = context_tokens.size(0)
        positions = self.pos.expand(batch, -1, -1)
        context_pos = _gather_tokens(
            positions, _batched_indices(context_indices, batch, self.pos.device)
        )
        target_indices = _batched_indices(target_indices, batch, self.pos.device)
        target_pos = _gather_tokens(positions, target_indices)
        num_targets = target_indices.size(1)

        context = self.embed(context_tokens) + context_pos
        masked = self.mask_tokens[mask_index].expand(batch, num_targets, -1) + target_pos
        hidden = torch.cat([context, masked], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(self.norm(hidden[:, -num_targets:, :]))


def build_encoder(config, device):
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
        config.get("norm_eps", 1e-6),
    ).to(device)


def build_models(config, device, num_mask_groups):
    """Build the context encoder, its frozen EMA target copy, and the predictor."""
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
        num_mask_groups,
        config.get("dropout", 0.0),
        config.get("norm_eps", 1e-6),
    ).to(device)
    return context_encoder, target_encoder, predictor
