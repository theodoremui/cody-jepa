"""Deterministic multi-block tube masking."""

from dataclasses import dataclass
import math

import torch


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
    # The 14x14 prototype grid needs more context than the large-scale preset.
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


def multiblock_mask(config, batch_size, rng, device=None, mask_groups=DEFAULT_MASK_GROUPS):
    """Sample disjoint context and target tubes for every mask group."""
    batch_size = int(batch_size)
    tubelet_size = int(config["tubelet_size"])
    patch_size = int(config["patch_size"])
    num_frames = int(config["num_frames"])
    image_size = int(config["img_size"])
    min_context_tokens = int(config.get("min_context_tokens", 1))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tubelet_size <= 0 or num_frames % tubelet_size:
        raise ValueError("tubelet_size must be positive and divide num_frames")
    if patch_size <= 0 or image_size % patch_size:
        raise ValueError("patch_size must be positive and divide img_size")
    temporal_grid = num_frames // tubelet_size
    spatial_grid = image_size // patch_size
    num_tokens = temporal_grid * spatial_grid * spatial_grid
    if int(config["num_tokens"]) != num_tokens:
        raise ValueError(
            f"num_tokens={config['num_tokens']} but geometry produces {num_tokens}"
        )
    min_context_cells = max(1, math.ceil(min_context_tokens / temporal_grid))

    output = []
    for group in mask_groups:
        target_sets, context_sets, block_shape = _sample_mask_sets(
            group, spatial_grid, min_context_cells, batch_size, rng
        )
        # Equalize ragged samples at whole-tube granularity so selected targets
        # still come from sampled blocks without time-major truncation bias.
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
        output.append(
            {
                "label": group.label,
                "block_shape": block_shape,
                "target_cells": target_keep,
                "context_cells": context_keep,
                "target_ratio": target_keep / (spatial_grid * spatial_grid),
                "context_ratio": context_keep / (spatial_grid * spatial_grid),
                "ctx": torch.tensor(contexts, dtype=torch.long, device=device),
                "pred": torch.tensor(targets, dtype=torch.long, device=device),
            }
        )
    return output


__all__ = ["DEFAULT_MASK_GROUPS", "MaskGroupConfig", "multiblock_mask"]
