"""Validation and normalization at the DataLoader-to-model boundary."""

from collections.abc import Mapping

import torch


def video_from_batch(batch, device, config, expected_split):
    if not isinstance(batch, Mapping) or "video" not in batch:
        raise TypeError("DataLoader batch must be a mapping containing 'video'")
    video = batch["video"]
    if not isinstance(video, torch.Tensor) or not video.is_floating_point():
        raise TypeError("batch['video'] must be a floating-point tensor")
    expected_shape = (
        int(config["num_frames"]),
        int(config["in_channels"]),
        int(config["img_size"]),
        int(config["img_size"]),
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
    mean = float(config.get("input_mean", 0.5))
    std = float(config.get("input_std", 0.5))
    if std <= 0:
        raise ValueError("input_std must be positive")
    return (video - mean) / std


__all__ = ["video_from_batch"]
