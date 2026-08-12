"""Multi-block tube masking.

Each mask group samples a union of rectangular blocks on the spatial grid, then
extends them through time so context and target are disjoint tubes. Sampling is
driven by a caller-supplied `random.Random`, so a seeded run is reproducible.
"""

import math

import torch


DEFAULT_MASK_GROUPS = (
    {"label": "small_blocks", "num_blocks": 8, "spatial_scale": 0.15},
    {"label": "large_blocks", "num_blocks": 2, "spatial_scale": 0.55},
)
ASPECT_RATIO = (0.75, 1.5)


def _block_size(spatial_grid, scale, aspect_ratio):
    area = max(1, round(spatial_grid * spatial_grid * scale))
    height = min(spatial_grid, max(1, round(math.sqrt(area * aspect_ratio))))
    width = min(spatial_grid, max(1, round(math.sqrt(area / aspect_ratio))))
    return height, width


def _sample_block_union(num_blocks, block_shape, spatial_grid, rng):
    height, width = block_shape
    cells = set()
    for _ in range(num_blocks):
        top = rng.randrange(spatial_grid - height + 1)
        left = rng.randrange(spatial_grid - width + 1)
        cells.update(
            row * spatial_grid + column
            for row in range(top, top + height)
            for column in range(left, left + width)
        )
    return cells


def _expand_tubes(spatial_cells, temporal_grid, spatial_grid):
    cells_per_step = spatial_grid * spatial_grid
    return [
        step * cells_per_step + cell
        for step in range(temporal_grid)
        for cell in spatial_cells
    ]


def multiblock_mask(config, batch_size, rng, device=None, mask_groups=DEFAULT_MASK_GROUPS):
    """Sample disjoint context/target tube indices for each mask group."""
    temporal_grid = int(config["num_frames"]) // int(config["tubelet_size"])
    spatial_grid = int(config["img_size"]) // int(config["patch_size"])
    all_cells = set(range(spatial_grid * spatial_grid))
    min_context_cells = max(
        1, math.ceil(int(config.get("min_context_tokens", 1)) / temporal_grid)
    )

    output = []
    for group in mask_groups:
        aspect_ratio = rng.uniform(*ASPECT_RATIO)
        block_shape = _block_size(spatial_grid, group["spatial_scale"], aspect_ratio)
        target_sets, context_sets = [], []
        for _ in range(batch_size):
            for _attempt in range(256):
                target = _sample_block_union(
                    group["num_blocks"], block_shape, spatial_grid, rng
                )
                context = all_cells - target
                if target and len(context) >= min_context_cells:
                    target_sets.append(target)
                    context_sets.append(context)
                    break
            else:
                raise RuntimeError(
                    f"could not sample {group['label']!r} mask leaving "
                    f"{min_context_cells} context cells; lower spatial_scale/num_blocks"
                )
        # Blocks overlap by different amounts per sample, so trim to a common
        # count at whole-tube granularity to get a rectangular index tensor.
        target_keep = min(len(cells) for cells in target_sets)
        context_keep = min(len(cells) for cells in context_sets)
        targets = [
            _expand_tubes(sorted(rng.sample(sorted(cells), target_keep)), temporal_grid, spatial_grid)
            for cells in target_sets
        ]
        contexts = [
            _expand_tubes(sorted(rng.sample(sorted(cells), context_keep)), temporal_grid, spatial_grid)
            for cells in context_sets
        ]
        output.append(
            {
                "label": group["label"],
                "ctx": torch.tensor(contexts, dtype=torch.long, device=device),
                "pred": torch.tensor(targets, dtype=torch.long, device=device),
            }
        )
    return output
