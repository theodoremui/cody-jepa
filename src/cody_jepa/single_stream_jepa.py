"""Robust single-stream masked JEPA prototype used by the training notebook.

This module intentionally implements a V-JEPA-style masked latent prediction
baseline. It does not implement CoDy-JEPA's later counterfactual or explicit
future-dynamics objective. Production defaults to BF16. FP16 is supported with
gradient scaling, but non-finite unscaled gradients intentionally fail fast
instead of silently skipping examples and changing the exact-resume trajectory.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import contextlib
import importlib.util
import math
import os
import random
import sys
import time
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SequentialSampler


MODEL_ARCHITECTURE = "cody-jepa-single-stream-masked-v3"
CHECKPOINT_SCHEMA = 3


@dataclass(frozen=True)
class MaskGroupConfig:
    label: str
    num_blocks: int
    spatial_scale: float
    aspect_ratio: tuple[float, float] = (0.75, 1.5)

    def __post_init__(self):
        if not self.label:
            raise ValueError("mask label must be nonempty")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not 0.0 < self.spatial_scale < 1.0:
            raise ValueError("spatial_scale must be in (0, 1)")
        if (
            len(self.aspect_ratio) != 2
            or not 0.0 < self.aspect_ratio[0] <= self.aspect_ratio[1]
        ):
            raise ValueError("aspect_ratio must be an ordered positive pair")


DEFAULT_MASK_GROUPS = (
    MaskGroupConfig("small_blocks", num_blocks=8, spatial_scale=0.15),
    # Adapted downward from the large-scale reference's 0.70 because this
    # prototype uses only a 14x14 spatial grid and must retain useful context.
    MaskGroupConfig("large_blocks", num_blocks=2, spatial_scale=0.55),
)


def _block_size(spatial_grid, scale, aspect_ratio):
    area = max(1, round(spatial_grid * spatial_grid * scale))
    height = min(spatial_grid, max(1, round(math.sqrt(area * aspect_ratio))))
    width = min(spatial_grid, max(1, round(math.sqrt(area / aspect_ratio))))
    return height, width


def _sample_block_union(n_blocks, block_shape, spatial_grid, rng):
    height, width = block_shape
    cells = set()
    for _ in range(n_blocks):
        top = rng.randrange(spatial_grid - height + 1)
        left = rng.randrange(spatial_grid - width + 1)
        cells.update(
            row * spatial_grid + column
            for row in range(top, top + height)
            for column in range(left, left + width)
        )
    return cells


def _sample_mask_sets(group, spatial_grid, min_context_cells, batch_size, rng):
    aspect_ratio = rng.uniform(*group.aspect_ratio)
    block_shape = _block_size(spatial_grid, group.spatial_scale, aspect_ratio)
    all_cells = set(range(spatial_grid * spatial_grid))
    target_sets = []
    context_sets = []
    for _ in range(batch_size):
        for _attempt in range(256):
            target = _sample_block_union(
                group.num_blocks, block_shape, spatial_grid, rng
            )
            context = all_cells - target
            if target and len(context) >= min_context_cells:
                target_sets.append(target)
                context_sets.append(context)
                break
        else:
            raise RuntimeError(
                f"could not sample {group.label!r} mask with at least "
                f"{min_context_cells} context cells; reduce spatial_scale or num_blocks"
            )
    return target_sets, context_sets, block_shape


def _subsample_cells(cells, keep, rng):
    if len(cells) == keep:
        return sorted(cells)
    return sorted(rng.sample(sorted(cells), keep))


def _expand_tubes(spatial_cells, temporal_grid, spatial_grid):
    cells_per_step = spatial_grid * spatial_grid
    return [
        time_index * cells_per_step + cell
        for time_index in range(temporal_grid)
        for cell in spatial_cells
    ]


def multiblock_mask(cfg, batch_size, rng, device=None, mask_groups=DEFAULT_MASK_GROUPS):
    batch_size = int(batch_size)
    tubelet_size = int(cfg["tubelet_size"])
    patch_size = int(cfg["patch_size"])
    num_frames = int(cfg["num_frames"])
    image_size = int(cfg["img_size"])
    min_context_tokens = int(cfg.get("min_context_tokens", 1))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tubelet_size <= 0 or num_frames % tubelet_size:
        raise ValueError("tubelet_size must be positive and divide num_frames")
    if patch_size <= 0 or image_size % patch_size:
        raise ValueError("patch_size must be positive and divide img_size")
    temporal_grid = num_frames // tubelet_size
    spatial_grid = image_size // patch_size
    num_tokens = temporal_grid * spatial_grid * spatial_grid
    if int(cfg["num_tokens"]) != num_tokens:
        raise ValueError(
            f"num_tokens={cfg['num_tokens']} but geometry produces {num_tokens}"
        )
    min_context_cells = max(1, math.ceil(min_context_tokens / temporal_grid))

    output = []
    for group in mask_groups:
        target_sets, context_sets, block_shape = _sample_mask_sets(
            group, spatial_grid, min_context_cells, batch_size, rng
        )
        # Ragged masks are reduced only to this batch's minimum. Subsampling
        # happens at whole-tube granularity, so no time-major truncation bias is
        # introduced and every selected target still came from a sampled block.
        target_keep = min(len(cells) for cells in target_sets)
        context_keep = min(len(cells) for cells in context_sets)
        targets = [
            _expand_tubes(
                _subsample_cells(cells, target_keep, rng), temporal_grid, spatial_grid
            )
            for cells in target_sets
        ]
        contexts = [
            _expand_tubes(
                _subsample_cells(cells, context_keep, rng), temporal_grid, spatial_grid
            )
            for cells in context_sets
        ]
        output.append({
            "label": group.label,
            "block_shape": block_shape,
            "target_cells": target_keep,
            "context_cells": context_keep,
            "target_ratio": target_keep / (spatial_grid * spatial_grid),
            "context_ratio": context_keep / (spatial_grid * spatial_grid),
            "ctx": torch.tensor(contexts, dtype=torch.long, device=device),
            "pred": torch.tensor(targets, dtype=torch.long, device=device),
        })
    return output


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
    embedding = torch.cat([
        _sincos_1d(time, dimensions[0]),
        _sincos_1d(height, dimensions[1]),
        _sincos_1d(width, dimensions[2]),
    ], dim=1)
    return embedding.unsqueeze(0)


def _init_linear(linear):
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
        _init_linear(self.attn.out_proj)
        _init_linear(self.linear_1)
        _init_linear(self.linear_2)

    def forward(self, inputs):
        normalized = self.layer_norm_1(inputs)
        inputs = inputs + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        hidden = self.linear_1(self.layer_norm_2(inputs))
        hidden = self.dropout(F.gelu(hidden))
        hidden = self.dropout(self.linear_2(hidden))
        return inputs + hidden


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
        self.transformer = nn.ModuleList([
            AttentionBlock(
                embed_dim, hidden_dim, num_heads, dropout=dropout, norm_eps=norm_eps
            )
            for _ in range(num_layers)
        ])
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
        _init_linear(self.embed)
        self.register_buffer(
            "pos",
            sincos_3d_position_embedding(
                *grid_size, pred_dim, uniform_power=uniform_power
            ),
        )
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, pred_dim))
            for _ in range(num_mask_tokens)
        ])
        self.blocks = nn.ModuleList([
            AttentionBlock(
                pred_dim, pred_dim * 4, num_heads, dropout=dropout, norm_eps=norm_eps
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_dim, eps=norm_eps)
        self.out = nn.Linear(pred_dim, dim)
        _init_linear(self.out)

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


def build_encoder(cfg, device):
    """Build one encoder from a saved single-stream configuration."""
    encoder_args = (
        cfg["embed_dim"],
        cfg["hidden_dim"],
        cfg["num_heads"],
        cfg["num_layers"],
        cfg["patch_size"],
        cfg["tubelet_size"],
        cfg["num_frames"],
        cfg["img_size"],
        cfg["in_channels"],
        cfg.get("dropout", 0.0),
        cfg.get("uniform_power", True),
        cfg.get("norm_eps", 1e-6),
    )
    return VisionTransformer(*encoder_args).to(device)


def build_models(cfg, device, mask_groups=DEFAULT_MASK_GROUPS):
    context_encoder = build_encoder(cfg, device)
    target_encoder = build_encoder(cfg, device)
    target_encoder.load_state_dict(context_encoder.state_dict())
    target_encoder.requires_grad_(False).eval()
    predictor = Predictor(
        cfg["embed_dim"],
        cfg["pred_dim"],
        cfg["pred_depth"],
        cfg["num_heads"],
        context_encoder.grid_size,
        len(mask_groups),
        cfg.get("dropout", 0.0),
        cfg.get("uniform_power", True),
        cfg.get("norm_eps", 1e-6),
    ).to(device)
    return context_encoder, target_encoder, predictor


def optimizer_param_groups(context_encoder, predictor, weight_decay):
    decay, no_decay = [], []
    seen = set()
    for module_name, module in (("encoder", context_encoder), ("predictor", predictor)):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise RuntimeError(f"duplicate trainable parameter: {module_name}.{name}")
            seen.add(id(parameter))
            if parameter.ndim <= 1 or name.endswith("bias") or "mask_tokens" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    expected = sum(
        parameter.requires_grad
        for module in (context_encoder, predictor)
        for parameter in module.parameters()
    )
    if len(seen) != expected:
        raise RuntimeError("optimizer parameter partition is incomplete")
    return [
        {"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]


@torch.no_grad()
def ema_update(target, online, tau):
    if not 0.0 <= tau <= 1.0:
        raise ValueError("EMA tau must be in [0, 1]")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("target and online encoder structures differ")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(tau).add_(online_parameters[name], alpha=1.0 - tau)
    for name, target_buffer in target.named_buffers():
        target_buffer.copy_(dict(online.named_buffers())[name])


def learning_rate_for_step(cfg, step):
    start_lr = float(cfg.get("start_lr", cfg["lr"]))
    base_lr = float(cfg["lr"])
    min_lr = float(cfg.get("min_lr", 0.0))
    warmup_steps = max(0, int(cfg.get("warmup_steps", 0)))
    max_steps = max(1, int(cfg["steps"]))
    step = max(1, int(step))
    if warmup_steps and step <= warmup_steps:
        if warmup_steps == 1:
            return base_lr
        fraction = (step - 1) / (warmup_steps - 1)
        return start_lr + fraction * (base_lr - start_lr)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def ema_tau_for_step(cfg, step):
    start = float(cfg.get("ema_start", 0.998))
    end = float(cfg.get("ema_end", 1.0))
    max_steps = max(1, int(cfg["steps"]))
    progress = (
        1.0
        if max_steps == 1
        else min(1.0, max(0.0, (step - 1) / (max_steps - 1)))
    )
    return start + (end - start) * progress


def _amp_dtype(cfg, device):
    name = cfg.get("amp_dtype")
    if name in (None, "none", "float32"):
        return None
    if name == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured bfloat16 but the CUDA device lacks BF16 support")
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported amp_dtype {name!r}")


def _autocast_context(cfg, device):
    dtype = _amp_dtype(cfg, device)
    if dtype is not None and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def _make_scaler(cfg, device):
    return torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and _amp_dtype(cfg, device) is torch.float16
    )


def _validate_cuda_runtime(device):
    try:
        probe = torch.zeros(1, device=device)
        probe.add_(1)
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        device_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        architectures = (
            torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        )
        torch_cuda = getattr(torch.version, "cuda", None)
        original_error = str(error)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        has_required_arch = required_arch in architectures
        incompatible_build = any(
            signature in original_error.lower()
            for signature in (
                "no kernel image is available",
                "invalid device function",
                "not compatible with the current pytorch installation",
            )
        )
        remediation = (
            "This usually means the notebook kernel loaded a PyTorch build that "
            f"does not include a runnable CUDA kernel for {required_arch}. If "
            "HAIC allocated an H100/Hopper GPU, reinstall the locked project "
            "environment with `uv sync --frozen --reinstall-package torch "
            "--reinstall-package torchvision`, then launch the notebook through "
            "`uv run --no-sync jupyter ...` and restart the kernel. If HAIC "
            "allocated a newer Blackwell GPU, use a PyTorch CUDA 12.8+ build "
            "or request a Hopper/H100 node."
            if incompatible_build
            else "The CUDA runtime failed before training; inspect the original "
            "error below and the active notebook interpreter."
        )
        raise RuntimeError(
            "CUDA is visible, but this PyTorch build cannot execute a kernel on "
            f"{device_name} (cuda_compute_capability=sm_{capability[0]}{capability[1]}). "
            f"torch_version={torch.__version__}, torch_cuda_version={torch_cuda}, "
            f"torch_cuda_arch_list={architectures or 'unknown'}, "
            f"torch_has_required_cuda_arch={has_required_arch}, "
            f"python_executable={sys.executable}. {remediation} "
            f"original_cuda_error={original_error}"
        ) from error


def resolve_device(required_device="auto"):
    if required_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("this full run requires CUDA, but CUDA is unavailable")
        device = torch.device("cuda")
    elif required_device == "cpu":
        device = torch.device("cpu")
    elif required_device != "auto":
        device = torch.device(required_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable")
        _validate_cuda_runtime(device)
    return device


def validate_training_config(cfg, train_loader):
    positive_integers = (
        "steps",
        "num_epochs",
        "batch_size",
        "accumulation_steps",
        "eval_every_epochs",
        "checkpoint_every_epochs",
    )
    for key in positive_integers:
        value = cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
    train_eval_every = cfg.get("train_eval_every_epochs", 0)
    if not isinstance(train_eval_every, int) or train_eval_every < 0:
        raise ValueError("train_eval_every_epochs must be a non-negative integer")
    for key in ("lr", "start_lr", "min_lr"):
        value = float(cfg.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if float(cfg["lr"]) <= 0:
        raise ValueError("lr must be positive")
    warmup_steps = cfg.get("warmup_steps", 0)
    if not isinstance(warmup_steps, int) or not 0 <= warmup_steps <= cfg["steps"]:
        raise ValueError("warmup_steps must be an integer in [0, steps]")
    ema_start = float(cfg.get("ema_start", 0.998))
    ema_end = float(cfg.get("ema_end", 1.0))
    if not 0.0 <= ema_start <= ema_end <= 1.0:
        raise ValueError("EMA schedule must satisfy 0 <= ema_start <= ema_end <= 1")
    loss_exp = float(cfg.get("loss_exp", 1.0))
    if not math.isfinite(loss_exp) or loss_exp < 1.0:
        raise ValueError("loss_exp must be finite and at least 1")
    input_mean = float(cfg.get("input_mean", 0.5))
    input_std = float(cfg.get("input_std", 0.5))
    if not math.isfinite(input_mean) or not math.isfinite(input_std) or input_std <= 0:
        raise ValueError("input normalization must be finite with input_std > 0")
    if float(cfg.get("weight_decay", 0.0)) < 0:
        raise ValueError("weight_decay must be non-negative")
    if float(cfg.get("grad_clip", 0.0)) < 0:
        raise ValueError("grad_clip must be non-negative")
    if cfg.get("selection_metric", "subject_balanced_loss") not in {
        "loss",
        "subject_balanced_loss",
    }:
        raise ValueError("selection_metric must be a loss metric that is minimized")
    if not isinstance(cfg.get("compile", False), bool):
        raise ValueError("compile must be a bool")
    microbatches = len(train_loader)
    if microbatches <= 0:
        raise ValueError("train loader must contain at least one microbatch")
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and int(loader_batch_size) != cfg["batch_size"]:
        raise ValueError(
            f"CONFIG batch_size={cfg['batch_size']} differs from loader batch_size="
            f"{loader_batch_size}"
        )
    accumulation_steps = cfg["accumulation_steps"]
    if microbatches % accumulation_steps:
        raise ValueError(
            f"train loader has {microbatches} microbatches, not divisible by "
            f"accumulation_steps={accumulation_steps}; use drop_last or adjust batching"
        )
    updates_per_epoch = microbatches // accumulation_steps
    if cfg["steps"] % updates_per_epoch:
        raise ValueError(
            f"steps={cfg['steps']} must be divisible by updates_per_epoch="
            f"{updates_per_epoch}; checkpoints are intentionally epoch-boundary exact"
        )
    return updates_per_epoch


def video_from_batch(batch, device, cfg, expected_split):
    if not isinstance(batch, Mapping) or "video" not in batch:
        raise TypeError("DataLoader batch must be a mapping containing 'video'")
    video = batch["video"]
    if not isinstance(video, torch.Tensor) or not video.is_floating_point():
        raise TypeError("batch['video'] must be a floating-point tensor")
    expected_shape = (
        int(cfg["num_frames"]),
        int(cfg["in_channels"]),
        int(cfg["img_size"]),
        int(cfg["img_size"]),
    )
    if video.ndim != 5 or tuple(video.shape[1:]) != expected_shape:
        raise ValueError(
            f"batch video must be [B,T,C,H,W] with tail {expected_shape}; "
            f"got {tuple(video.shape)}"
        )
    splits = batch.get("split")
    if splits is None or any(str(split) != expected_split for split in splits):
        raise ValueError(f"batch does not contain only split={expected_split!r}")
    if "sequence_id" not in batch or "subject_id" not in batch:
        raise KeyError("batch must retain sequence_id and subject_id metadata")
    if not torch.isfinite(video).all():
        raise FloatingPointError("input video contains non-finite values")
    minimum, maximum = float(video.min()), float(video.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"input pixels must be in [0,1], got [{minimum}, {maximum}]")
    video = video.to(device, non_blocking=True)
    mean = float(cfg.get("input_mean", 0.5))
    std = float(cfg.get("input_std", 0.5))
    if std <= 0:
        raise ValueError("input_std must be positive")
    return (video - mean) / std


def _prediction_metrics(predicted, target, loss_exp):
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


@torch.no_grad()
def encode_targets(target_encoder, video, return_pre_norm=False):
    encoded = target_encoder(video, return_pre_norm=return_pre_norm)
    if return_pre_norm:
        normalized, pre_norm = encoded
    else:
        normalized, pre_norm = encoded, None
    normalized = F.layer_norm(normalized, (normalized.size(-1),))
    return normalized, pre_norm


def group_forward(
    context_encoder,
    predictor,
    video,
    all_target_tokens,
    mask_group,
    mask_index,
    loss_exp,
):
    context_indices = mask_group["ctx"]
    target_indices = mask_group["pred"]
    context_tokens = context_encoder(video, context_indices)
    gather = target_indices.unsqueeze(-1).expand(
        -1, -1, all_target_tokens.size(-1)
    )
    target_tokens = torch.gather(all_target_tokens, 1, gather)
    predicted = predictor(
        context_tokens, context_indices, target_indices, mask_index=mask_index
    )
    loss, cosine = _prediction_metrics(predicted, target_tokens, loss_exp)
    return loss, cosine


def representation_diagnostics(clip_features):
    features = torch.as_tensor(clip_features, dtype=torch.float32)
    if features.ndim != 2 or features.size(0) < 2:
        return {
            "mean_feature_norm": float("nan"),
            "min_feature_norm": float("nan"),
            "max_feature_norm": float("nan"),
            "feature_std": float("nan"),
            "near_zero_variance_fraction": float("nan"),
            "effective_rank": float("nan"),
            "effective_rank_ratio": float("nan"),
        }
    centered = features - features.mean(dim=0, keepdim=True)
    norms = features.norm(dim=1)
    variance = centered.pow(2).mean(dim=0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / (features.size(0) - 1))
    eigenvalues = eigenvalues.clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    return {
        "mean_feature_norm": float(norms.mean()),
        "min_feature_norm": float(norms.min()),
        "max_feature_norm": float(norms.max()),
        "feature_std": float(variance.sqrt().mean()),
        "near_zero_variance_fraction": float((variance < 1e-6).float().mean()),
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / features.size(1),
    }


def _subject_balanced_mean(values_by_subject):
    if not values_by_subject:
        return float("nan")
    return sum(
        sum(values) / len(values) for values in values_by_subject.values()
    ) / len(values_by_subject)


def balanced_wrong_subject_permutation(subject_ids, seed):
    """Return a seeded one-to-one cross-subject source permutation.

    Every row is used exactly once as a target and once as a source. A perfect
    cross-subject permutation exists exactly when no subject owns more than half
    of the rows. Returning ``None`` makes an infeasible diagnostic explicit
    instead of silently reusing a small number of source clips.
    """
    subjects = [str(subject).casefold() for subject in subject_ids]
    if len(set(subjects)) < 2:
        return None
    rows_by_subject = defaultdict(list)
    for index, subject in enumerate(subjects):
        rows_by_subject[subject].append(index)
    maximum_subject_rows = max(len(rows) for rows in rows_by_subject.values())
    if maximum_subject_rows > len(subjects) - maximum_subject_rows:
        return None

    rng = random.Random(seed)
    groups = list(rows_by_subject.values())
    rng.shuffle(groups)
    for rows in groups:
        rng.shuffle(rows)
    ordered_targets = [index for rows in groups for index in rows]
    ordered_sources = (
        ordered_targets[maximum_subject_rows:]
        + ordered_targets[:maximum_subject_rows]
    )
    sources = [None] * len(subjects)
    for target, source in zip(ordered_targets, ordered_sources):
        sources[target] = source

    if sorted(sources) != list(range(len(subjects))):
        raise RuntimeError("wrong-subject pairing is not a permutation")
    if any(subjects[index] == subjects[source] for index, source in enumerate(sources)):
        raise RuntimeError("subject-aware context construction failed")
    return sources


def _dataset_subject_id(dataset, index):
    subject_id_at = getattr(dataset, "subject_id_at", None)
    if callable(subject_id_at):
        return str(subject_id_at(index))
    sample = dataset[index]
    if not isinstance(sample, Mapping) or "subject_id" not in sample:
        raise TypeError(
            "context-shuffle evaluation requires dataset rows with subject_id metadata"
        )
    return str(sample["subject_id"])


def _context_shuffle_plan(loader, seed):
    """Plan a full-loader cross-subject permutation without loading video tensors."""
    if not isinstance(loader, DataLoader):
        return {
            "status": "unavailable_non_dataloader",
            "source_index_batches": [],
            "source_positions": [],
            "subjects": [],
        }
    if loader.drop_last:
        raise ValueError("context-shuffle evaluation requires drop_last=False")
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError(
            "context-shuffle evaluation requires a deterministic sequential loader"
        )
    if getattr(loader, "in_order", True) is False:
        raise ValueError("context-shuffle evaluation requires in_order=True")
    if loader.batch_size is None:
        raise ValueError("context-shuffle evaluation requires a fixed batch_size")

    target_index_batches = []
    for batch in loader.batch_sampler:
        indices = [int(index) for index in batch]
        target_index_batches.append(indices)
    target_indices = [index for batch in target_index_batches for index in batch]
    if target_indices != list(range(len(loader.dataset))):
        raise ValueError(
            "context-shuffle evaluation requires every dataset row exactly once in order"
        )

    subjects = [_dataset_subject_id(loader.dataset, index) for index in target_indices]
    source_positions = balanced_wrong_subject_permutation(subjects, seed)
    if source_positions is None:
        return {
            "status": "infeasible_subject_distribution",
            "source_index_batches": [],
            "source_positions": [],
            "subjects": subjects,
        }
    source_indices = [target_indices[position] for position in source_positions]
    source_index_batches = []
    offset = 0
    for target_batch in target_index_batches:
        next_offset = offset + len(target_batch)
        source_index_batches.append(source_indices[offset:next_offset])
        offset = next_offset
    return {
        "status": "complete",
        "source_index_batches": source_index_batches,
        "source_positions": source_positions,
        "subjects": subjects,
    }


def _context_source_loader(loader, shuffle_plan, seed):
    """Clone validation-loading behavior for the planned source-index batches."""
    if shuffle_plan["status"] != "complete":
        return None
    generator = torch.Generator().manual_seed(int(seed))
    options = {
        "dataset": loader.dataset,
        "batch_sampler": shuffle_plan["source_index_batches"],
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": generator,
        "persistent_workers": loader.persistent_workers,
    }
    if loader.num_workers > 0:
        options["prefetch_factor"] = loader.prefetch_factor
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        options["pin_memory_device"] = pin_memory_device
    return DataLoader(**options)


def representation_health(metrics, cfg):
    issues = []
    if metrics["feature_std"] < float(cfg.get("min_feature_std", 1e-3)):
        issues.append("feature_std_below_threshold")
    if metrics["near_zero_variance_fraction"] > float(
        cfg.get("max_near_zero_variance_fraction", 0.5)
    ):
        issues.append("too_many_near_constant_dimensions")
    if metrics["effective_rank_ratio"] < float(
        cfg.get("min_effective_rank_ratio", 0.05)
    ):
        issues.append("effective_rank_below_threshold")
    context_gap = metrics.get(
        "subject_balanced_context_shuffle_loss_gap",
        metrics.get("context_shuffle_loss_gap", float("nan")),
    )
    context_status = metrics.get("context_shuffle_status")
    if context_status not in {None, "complete"}:
        issues.append("context_shuffle_unavailable")
    elif (
        metrics.get("context_shuffle_pairs", 0) <= 0
        or not math.isfinite(context_gap)
        or context_gap
        < float(cfg.get("min_context_shuffle_loss_gap", 0.0))
    ):
        issues.append("context_shuffle_gap_below_threshold")
    if metrics["min_feature_norm"] <= 1e-8:
        issues.append("near_zero_feature_norm")
    if metrics["max_feature_norm"] > float(cfg.get("max_feature_norm", 1e4)):
        issues.append("feature_norm_above_threshold")
    return {"representations_healthy": not issues, "health_issues": issues}


@torch.inference_mode()
def evaluate_jepa(
    context_encoder,
    target_encoder,
    predictor,
    loader,
    cfg,
    device,
    expected_split,
    mask_seed,
    mask_groups=DEFAULT_MASK_GROUPS,
    context_shuffle=True,
    context_seed=None,
):
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()
    mask_rng = random.Random(mask_seed)
    total_loss = total_cosine = 0.0
    examples = 0
    subject_loss = defaultdict(list)
    subject_cosine = defaultdict(list)
    pooled_features = []
    shuffle_gap_sum = 0.0
    shuffle_gap_examples = 0
    subject_shuffle_gaps = defaultdict(list)
    shuffle_plan = None
    if context_shuffle:
        if context_seed is None:
            raise ValueError("context_seed is required when context_shuffle=True")
        shuffle_plan = _context_shuffle_plan(loader, context_seed)
    source_loader = (
        _context_source_loader(loader, shuffle_plan, context_seed)
        if shuffle_plan is not None and shuffle_plan["status"] == "complete"
        else None
    )
    source_iterator = iter(source_loader) if source_loader is not None else None
    for batch_index, batch in enumerate(loader):
        video = video_from_batch(batch, device, cfg, expected_split)
        masks = multiblock_mask(
            cfg, video.size(0), mask_rng, device=device, mask_groups=mask_groups
        )
        with _autocast_context(cfg, device):
            online_tokens = context_encoder(video)
            pooled_features.append(online_tokens.float().mean(dim=1).cpu())
            targets, _ = encode_targets(target_encoder, video)
            batch_loss = torch.zeros(video.size(0), device=device)
            batch_cosine = torch.zeros(video.size(0), device=device)
            for mask_index, mask_group in enumerate(masks):
                loss, cosine = group_forward(
                    context_encoder,
                    predictor,
                    video,
                    targets,
                    mask_group,
                    mask_index,
                    float(cfg.get("loss_exp", 1.0)),
                )
                batch_loss += loss / len(masks)
                batch_cosine += cosine / len(masks)
            if shuffle_plan is not None and shuffle_plan["status"] == "complete":
                try:
                    source_batch = next(source_iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "context source loader ended before the target loader"
                    ) from error
                shuffled_video = video_from_batch(
                    source_batch, device, cfg, expected_split
                )
                target_subjects = [str(value).casefold() for value in batch["subject_id"]]
                source_subjects = [
                    str(value).casefold() for value in source_batch["subject_id"]
                ]
                if any(
                    target == source
                    for target, source in zip(target_subjects, source_subjects)
                ):
                    raise RuntimeError("context-shuffle plan paired the same subject")
                shuffled_loss = torch.zeros(video.size(0), device=device)
                for mask_index, mask_group in enumerate(masks):
                    loss, _ = group_forward(
                        context_encoder,
                        predictor,
                        shuffled_video,
                        targets,
                        mask_group,
                        mask_index,
                        float(cfg.get("loss_exp", 1.0)),
                    )
                    shuffled_loss += loss / len(masks)
                gaps = (shuffled_loss - batch_loss).float().cpu()
                shuffle_gap_sum += float(gaps.sum())
                shuffle_gap_examples += video.size(0)
                for subject, gap in zip(batch["subject_id"], gaps):
                    subject_shuffle_gaps[str(subject).casefold()].append(float(gap))
        batch_count = video.size(0)
        total_loss += float(batch_loss.sum())
        total_cosine += float(batch_cosine.sum())
        examples += batch_count
        for index, subject in enumerate(batch["subject_id"]):
            subject_loss[str(subject)].append(float(batch_loss[index]))
            subject_cosine[str(subject)].append(float(batch_cosine[index]))
    if source_iterator is not None:
        try:
            next(source_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("context source loader outlasted the target loader")
    if examples == 0:
        raise RuntimeError("evaluation loader produced no batches")
    diagnostics = representation_diagnostics(torch.cat(pooled_features, dim=0))
    diagnostics.update({
        "loss": total_loss / examples,
        "cosine": total_cosine / examples,
        "subject_balanced_loss": _subject_balanced_mean(subject_loss),
        "subject_balanced_cosine": _subject_balanced_mean(subject_cosine),
        "context_shuffle_loss_gap": (
            shuffle_gap_sum / shuffle_gap_examples
            if shuffle_gap_examples
            else float("nan")
        ),
        "subject_balanced_context_shuffle_loss_gap": _subject_balanced_mean(
            subject_shuffle_gaps
        ),
        "context_shuffle_pairs": shuffle_gap_examples,
        "context_shuffle_batches": (
            len(shuffle_plan["source_index_batches"])
            if shuffle_plan is not None and shuffle_plan["status"] == "complete"
            else 0
        ),
        "context_shuffle_subjects": len(subject_shuffle_gaps),
        "context_shuffle_unique_sources": (
            len(set(shuffle_plan["source_positions"]))
            if shuffle_plan is not None and shuffle_plan["status"] == "complete"
            else 0
        ),
    })
    if not all(
        math.isfinite(value)
        for key, value in diagnostics.items()
        if key not in {
            "context_shuffle_loss_gap",
            "subject_balanced_context_shuffle_loss_gap",
        }
    ):
        raise FloatingPointError(f"non-finite evaluation diagnostics: {diagnostics}")
    diagnostics.update({
        "representation_source": "context_encoder_final_norm_full_view_mean_pool",
        "context_shuffle_pairing": "global_seeded_cross_subject_permutation_v1",
        "context_shuffle_status": (
            shuffle_plan["status"]
            if shuffle_plan is not None
            else "disabled"
        ),
    })
    diagnostics.update(representation_health(diagnostics, cfg))
    return diagnostics


def _set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _scale_gradients(parameters, denominator):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_config(cfg):
    return {
        key: value
        for key, value in cfg.items()
        if isinstance(value, (str, int, float, bool, type(None), tuple, list, dict))
    }


def _checkpoint_payload(
    cfg,
    mask_groups,
    data_contract,
    context_encoder,
    target_encoder,
    predictor,
    optimizer,
    scaler,
    history,
    global_step,
    completed_epochs,
    mask_rng,
    loader_generator,
    best_val_loss,
    best_epoch,
    best_healthy_val_loss,
    best_healthy_epoch,
):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": MODEL_ARCHITECTURE,
        "config": _checkpoint_config(cfg),
        "mask_groups": [asdict(group) for group in mask_groups],
        "data_contract": data_contract,
        "context_encoder": context_encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "history": history,
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_healthy_val_loss": best_healthy_val_loss,
        "best_healthy_epoch": best_healthy_epoch,
        "mask_rng_state": mask_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": (
            loader_generator.get_state()
            if isinstance(loader_generator, torch.Generator)
            else None
        ),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }


def validate_resume_state(state, cfg, mask_groups, data_contract):
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is incompatible")
    if state.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError("checkpoint architecture is incompatible")
    if state.get("mask_groups") != [asdict(group) for group in mask_groups]:
        raise ValueError("checkpoint mask groups differ from this run")
    if state.get("data_contract") != data_contract:
        raise ValueError("checkpoint dataset/loader contract differs from this run")
    mutable = {
        "num_epochs",
        "eval_every_epochs",
        "train_eval_every_epochs",
        "checkpoint_every_epochs",
    }
    saved_cfg = state.get("config", {})
    current_cfg = _checkpoint_config(cfg)
    for key in set(saved_cfg) | set(current_cfg):
        if key not in mutable and saved_cfg.get(key) != current_cfg.get(key):
            raise ValueError(f"checkpoint config mismatch for {key!r}")


def _validate_compile_runtime(device):
    if device.type != "cuda":
        return
    try:
        from torch.utils._triton import has_triton

        triton_available = bool(has_triton())
    except Exception:
        triton_available = False
    if triton_available:
        return

    triton_spec = importlib.util.find_spec("triton")
    triton_version = None
    if triton_spec is not None:
        try:
            import triton

            triton_version = getattr(triton, "__version__", "unknown")
        except Exception as error:
            triton_version = f"import_failed:{error!r}"
    raise RuntimeError(
        "compile=True requested CUDA torch.compile, but TorchInductor cannot "
        "use Triton in this environment. Run this HAIC training job with "
        "CONFIG['compile'] = False, or repair the environment with "
        "`uv sync --frozen --reinstall-package torch --reinstall-package triton` "
        "and rerun a small torch.compile CUDA smoke test before training. "
        f"torch_version={torch.__version__}, "
        f"triton_module={triton_spec.origin if triton_spec else None}, "
        f"triton_version={triton_version}, python_executable={sys.executable}"
    )


def _maybe_compile(module, cfg, device):
    if not cfg.get("compile", False):
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile was requested but is unavailable")
    _validate_compile_runtime(device)
    return torch.compile(module, dynamic=True)


def train_jepa(
    cfg,
    train_loader,
    val_loader,
    data_contract,
    checkpoint_dir=None,
    resume_state=None,
    device=None,
    mask_groups=DEFAULT_MASK_GROUPS,
    train_eval_loader=None,
):
    updates_per_epoch = validate_training_config(cfg, train_loader)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    device = resolve_device(
        cfg.get("required_device", "auto") if device is None else device
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(cfg.get("tf32", False))
    context_encoder, target_encoder, predictor = build_models(cfg, device, mask_groups)
    param_groups = optimizer_param_groups(
        context_encoder, predictor, cfg["weight_decay"]
    )
    optimizer = optim.AdamW(param_groups, lr=float(cfg["lr"]))
    scaler = _make_scaler(cfg, device)
    trainable_parameters = [
        parameter
        for module in (context_encoder, predictor)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    loader_generator = getattr(train_loader, "generator", None)
    if isinstance(loader_generator, torch.Generator):
        loader_generator.manual_seed(seed)
    mask_rng = random.Random(seed)
    global_step = 0
    completed_epochs = 0
    history = []
    best_val_loss = math.inf
    best_epoch = None
    best_healthy_val_loss = math.inf
    best_healthy_epoch = None

    if resume_state is not None:
        validate_resume_state(resume_state, cfg, mask_groups, data_contract)
        context_encoder.load_state_dict(resume_state["context_encoder"], strict=True)
        target_encoder.load_state_dict(resume_state["target_encoder"], strict=True)
        predictor.load_state_dict(resume_state["predictor"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        if scaler.is_enabled() and resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        history = list(resume_state.get("history", []))
        global_step = int(resume_state["global_step"])
        completed_epochs = int(resume_state["completed_epochs"])
        best_val_loss = float(resume_state.get("best_val_loss", math.inf))
        best_epoch = resume_state.get("best_epoch")
        best_healthy_val_loss = float(
            resume_state.get("best_healthy_val_loss", math.inf)
        )
        best_healthy_epoch = resume_state.get("best_healthy_epoch")
        mask_rng.setstate(resume_state["mask_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if isinstance(loader_generator, torch.Generator) and resume_state.get("loader_rng_state") is not None:
            loader_generator.set_state(resume_state["loader_rng_state"])
        if device.type == "cuda" and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])

    context_runner = _maybe_compile(context_encoder, cfg, device)
    target_runner = _maybe_compile(target_encoder, cfg, device)
    predictor_runner = _maybe_compile(predictor, cfg, device)
    accumulation_steps = int(cfg["accumulation_steps"])
    max_steps = int(cfg["steps"])
    if int(cfg["num_epochs"]) * updates_per_epoch < max_steps:
        print(
            f"warning: num_epochs can produce only "
            f"{int(cfg['num_epochs']) * updates_per_epoch} of {max_steps} requested steps"
        )
    start_time = time.perf_counter()
    processed_examples = 0
    termination_reason = "num_epochs"

    for epoch in range(completed_epochs, int(cfg["num_epochs"])):
        if global_step >= max_steps:
            termination_reason = "max_steps_at_epoch_boundary"
            break
        dataset = getattr(train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        pending_examples = 0
        pending_microbatches = 0
        epoch_loss_sum = epoch_cosine_sum = 0.0
        epoch_examples = 0
        grad_norm_sum = 0.0
        grad_updates = 0
        last_lr = None
        last_tau = None
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch_index, batch in enumerate(train_loader):
            next_step = global_step + 1
            if pending_microbatches == 0:
                last_lr = learning_rate_for_step(cfg, next_step)
                _set_optimizer_lr(optimizer, last_lr)
            video = video_from_batch(batch, device, cfg, "train")
            masks = multiblock_mask(
                cfg, video.size(0), mask_rng, device=device, mask_groups=mask_groups
            )
            with _autocast_context(cfg, device):
                targets, _ = encode_targets(target_runner, video)
            batch_loss = batch_cosine = 0.0
            for mask_index, mask_group in enumerate(masks):
                with _autocast_context(cfg, device):
                    losses, cosines = group_forward(
                        context_runner,
                        predictor_runner,
                        video,
                        targets,
                        mask_group,
                        mask_index,
                        float(cfg.get("loss_exp", 1.0)),
                    )
                    group_loss = losses.mean()
                if not torch.isfinite(group_loss):
                    raise FloatingPointError("non-finite JEPA loss before backward")
                weighted = group_loss * video.size(0)
                scaler.scale(weighted).backward()
                batch_loss += float(losses.detach().sum()) / len(masks)
                batch_cosine += float(cosines.detach().sum()) / len(masks)
            pending_examples += video.size(0)
            pending_microbatches += 1
            epoch_examples += video.size(0)
            processed_examples += video.size(0)
            epoch_loss_sum += batch_loss
            epoch_cosine_sum += batch_cosine
            is_boundary = pending_microbatches == accumulation_steps or batch_index + 1 == len(train_loader)
            if is_boundary:
                scaler.unscale_(optimizer)
                _scale_gradients(
                    trainable_parameters, pending_examples * len(mask_groups)
                )
                max_grad_norm = float(cfg.get("grad_clip", 0.0))
                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_grad_norm if max_grad_norm > 0 else math.inf,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                last_tau = ema_tau_for_step(cfg, global_step)
                ema_update(target_encoder, context_encoder, last_tau)
                optimizer.zero_grad(set_to_none=True)
                pending_examples = pending_microbatches = 0
                grad_norm_sum += float(grad_norm)
                grad_updates += 1

        if epoch_examples == 0:
            raise RuntimeError("train_loader produced no batches")
        completed_epochs = epoch + 1
        should_eval = (
            completed_epochs % int(cfg.get("eval_every_epochs", 1)) == 0
            or global_step >= max_steps
            or completed_epochs == int(cfg["num_epochs"])
        )
        val_metrics = None
        if should_eval:
            val_metrics = evaluate_jepa(
                context_runner,
                target_runner,
                predictor_runner,
                val_loader,
                cfg,
                device,
                "val",
                mask_seed=seed + 1,
                mask_groups=mask_groups,
                context_shuffle=True,
                context_seed=seed + 2,
            )
        train_eval_metrics = None
        train_eval_every = int(cfg.get("train_eval_every_epochs", 0))
        if train_eval_loader is not None and train_eval_every > 0 and (
            completed_epochs % train_eval_every == 0 or global_step >= max_steps
        ):
            train_eval_metrics = evaluate_jepa(
                context_runner,
                target_runner,
                predictor_runner,
                train_eval_loader,
                cfg,
                device,
                "train",
                mask_seed=seed + 2,
                mask_groups=mask_groups,
                context_shuffle=False,
            )
        epoch_seconds = time.perf_counter() - epoch_started
        metrics = {
            "epoch": completed_epochs,
            "step": global_step,
            "lr": last_lr,
            "ema_tau": last_tau,
            "grad_norm": grad_norm_sum / max(1, grad_updates),
            "train_loss": epoch_loss_sum / epoch_examples,
            "train_cosine": epoch_cosine_sum / epoch_examples,
            "train_examples": epoch_examples,
            "epoch_seconds": epoch_seconds,
            "examples_per_second": epoch_examples / max(epoch_seconds, 1e-12),
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
            "val": val_metrics,
            "train_eval": train_eval_metrics,
        }
        history.append(metrics)
        selection_metric = str(cfg.get("selection_metric", "subject_balanced_loss"))
        if val_metrics is not None and selection_metric not in val_metrics:
            raise KeyError(f"unknown selection_metric {selection_metric!r}")
        if val_metrics is not None and val_metrics[selection_metric] < best_val_loss:
            best_val_loss = val_metrics[selection_metric]
            best_epoch = completed_epochs
        if (
            val_metrics is not None
            and val_metrics["representations_healthy"]
            and val_metrics[selection_metric] < best_healthy_val_loss
        ):
            best_healthy_val_loss = val_metrics[selection_metric]
            best_healthy_epoch = completed_epochs

        payload = _checkpoint_payload(
            cfg,
            mask_groups,
            data_contract,
            context_encoder,
            target_encoder,
            predictor,
            optimizer,
            scaler,
            history,
            global_step,
            completed_epochs,
            mask_rng,
            loader_generator,
            best_val_loss,
            best_epoch,
            best_healthy_val_loss,
            best_healthy_epoch,
        )
        if checkpoint_dir is not None and (
            completed_epochs % int(cfg.get("checkpoint_every_epochs", 1)) == 0
            or global_step >= max_steps
            or completed_epochs == int(cfg["num_epochs"])
        ):
            _atomic_torch_save(payload, Path(checkpoint_dir) / "latest.pt")
        if checkpoint_dir is not None and val_metrics is not None:
            if best_epoch == completed_epochs:
                _atomic_torch_save(payload, Path(checkpoint_dir) / "best_loss.pt")
            if best_healthy_epoch == completed_epochs:
                _atomic_torch_save(payload, Path(checkpoint_dir) / "best_healthy.pt")

        val_text = (
            f" | val_loss={val_metrics['loss']:.4f}, "
            f"effective_rank={val_metrics['effective_rank']:.1f}, "
            f"subject_balanced_context_shuffle_loss_gap="
            f"{val_metrics['subject_balanced_context_shuffle_loss_gap']:.4f}"
            if val_metrics is not None
            else ""
        )
        print(
            f"epoch={completed_epochs:03d} | step={global_step:05d} | "
            f"lr={last_lr:.2e} | train_loss={metrics['train_loss']:.4f}, "
            f"train_cosine={metrics['train_cosine']:.4f}{val_text}"
        )

    if global_step >= max_steps:
        termination_reason = "max_steps_at_epoch_boundary"
    elif completed_epochs >= int(cfg["num_epochs"]):
        termination_reason = "num_epochs_before_max_steps"
        print(
            f"warning: num_epochs ended at step {global_step}, below configured {max_steps}"
        )
    elapsed = time.perf_counter() - start_time
    return {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
        "optimizer": optimizer,
        "history": history,
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_healthy_val_loss": best_healthy_val_loss,
        "best_healthy_epoch": best_healthy_epoch,
        "termination_reason": termination_reason,
        "elapsed_seconds": elapsed,
        "examples_per_second": (
            processed_examples / elapsed
            if elapsed > 0
            else float("nan")
        ),
        "checkpoint_state": _checkpoint_payload(
            cfg,
            mask_groups,
            data_contract,
            context_encoder,
            target_encoder,
            predictor,
            optimizer,
            scaler,
            history,
            global_step,
            completed_epochs,
            mask_rng,
            loader_generator,
            best_val_loss,
            best_epoch,
            best_healthy_val_loss,
            best_healthy_epoch,
        ),
    }


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def healthy_checkpoint_path(checkpoint_dir, best_healthy_epoch):
    """Return the written healthy checkpoint path, or ``None`` when unselected."""
    if best_healthy_epoch is None:
        return None
    path = Path(checkpoint_dir) / "best_healthy.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"healthy epoch {best_healthy_epoch} was selected but {path} was not written"
        )
    return path


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
    "evaluate_jepa",
    "healthy_checkpoint_path",
    "learning_rate_for_step",
    "load_checkpoint",
    "multiblock_mask",
    "optimizer_param_groups",
    "representation_diagnostics",
    "resolve_device",
    "train_jepa",
    "validate_resume_state",
    "video_from_batch",
]
