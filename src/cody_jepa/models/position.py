"""Deterministic sinusoidal position embeddings for video tokens."""

import torch


def _sincos_1d(coordinates, embed_dim):
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError("1D position dimension must be positive and even")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    angles = coordinates.reshape(-1, 1).float() * omega.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def sincos_3d_position_embedding(
    temporal_grid, height_grid, width_grid, embed_dim, uniform_power=True
):
    if uniform_power:
        if embed_dim <= 0 or embed_dim % 6:
            raise ValueError("uniform 3D position embed_dim must be divisible by 6")
        dimensions = (embed_dim // 3,) * 3
    else:
        if embed_dim <= 0 or embed_dim % 8:
            raise ValueError("nonuniform 3D position embed_dim must be divisible by 8")
        dimensions = (embed_dim // 2, embed_dim // 4, embed_dim // 4)
    time, height, width = torch.meshgrid(
        torch.arange(temporal_grid),
        torch.arange(height_grid),
        torch.arange(width_grid),
        indexing="ij",
    )
    embedding = torch.cat(
        [
            _sincos_1d(time, dimensions[0]),
            _sincos_1d(height, dimensions[1]),
            _sincos_1d(width, dimensions[2]),
        ],
        dim=1,
    )
    return embedding.unsqueeze(0)


__all__ = ["sincos_3d_position_embedding"]
