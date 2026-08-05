"""Frozen target-encoder feature export and table I/O."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from ..models.encoder import build_encoder
from ..training.batches import video_from_batch
from ..training.checkpoint import MODEL_ARCHITECTURE


FEATURE_SOURCE = "target_encoder_pre_norm_mean"
FEATURE_FORMULA = "target_encoder(video, return_pre_norm=True)[1].mean(dim=1)"
METADATA_COLUMNS = (
    "sequence_id",
    "recording_id",
    "source_video_id",
    "direction_clip_id",
    "split",
    "subject_id",
    "gait_system",
    "trial",
    "speed",
    "clothing",
    "direction",
    "window_start",
    "num_frames",
    "fps",
    "shortcut_log_frame_count",
    "shortcut_duration_seconds",
    "shortcut_horizontal_centroid_drift_signed",
    "shortcut_horizontal_centroid_drift_absolute",
    "shortcut_foreground_area_mean",
    "shortcut_foreground_area_std",
    "shortcut_foreground_area_q25",
    "shortcut_foreground_area_median",
    "shortcut_foreground_area_q75",
)
INTEGER_METADATA_COLUMNS = frozenset({"window_start", "num_frames"})
NUMERIC_METADATA_COLUMNS = frozenset(
    {"fps", *(name for name in METADATA_COLUMNS if name.startswith("shortcut_"))}
)


def _batch_values(batch, key, batch_size):
    if key not in batch:
        raise KeyError(f"feature export batch is missing metadata field {key!r}")
    values = batch[key]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    else:
        values = list(values)
    if len(values) != batch_size:
        raise ValueError(
            f"batch metadata {key!r} has {len(values)} values for batch size {batch_size}"
        )
    return values


@torch.inference_mode()
def export_frozen_features(target_encoder, loaders, cfg, device, show_progress=False):
    """Return one mean-pooled pre-final-LayerNorm target feature per clip."""
    if not isinstance(loaders, Mapping) or not loaders:
        raise TypeError("loaders must be a nonempty mapping from split to loader")
    device = torch.device(device)
    target_encoder.requires_grad_(False)
    target_encoder.eval()

    metadata_rows = []
    feature_batches = []
    feature_dim = None
    for expected_split, loader in loaders.items():
        batches = loader
        if show_progress:
            from tqdm.auto import tqdm

            batches = tqdm(loader, desc=f"export {expected_split}", unit="batch")
        for batch in batches:
            video = video_from_batch(batch, device, cfg, expected_split=str(expected_split))
            encoded = target_encoder(video, return_pre_norm=True)
            if not isinstance(encoded, tuple) or len(encoded) != 2:
                raise TypeError("target encoder did not return (normalized, pre_norm) tokens")
            pre_norm = encoded[1]
            if pre_norm.ndim != 3 or pre_norm.size(0) != video.size(0):
                raise ValueError("pre-normalization tokens must have shape [B, N, D]")
            features = pre_norm.mean(dim=1).float().cpu()
            if not torch.isfinite(features).all():
                raise FloatingPointError("target encoder produced non-finite features")
            if feature_dim is None:
                feature_dim = int(features.size(1))
            elif features.size(1) != feature_dim:
                raise ValueError("target encoder feature dimension changed between batches")
            feature_batches.append(features.numpy())

            values = {
                key: _batch_values(batch, key, video.size(0)) for key in METADATA_COLUMNS
            }
            for index in range(video.size(0)):
                metadata_rows.append({
                    key: (
                        int(values[key][index])
                        if key in INTEGER_METADATA_COLUMNS
                        else float(values[key][index])
                        if key in NUMERIC_METADATA_COLUMNS
                        else str(values[key][index])
                    )
                    for key in METADATA_COLUMNS
                })

    if not feature_batches:
        raise ValueError("feature export received no examples")
    metadata = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    features = np.concatenate(feature_batches, axis=0)
    feature_frame = pd.DataFrame(
        features,
        columns=[f"feature_{index}" for index in range(features.shape[1])],
    )
    table = pd.concat([metadata, feature_frame], axis=1)
    validate_feature_table(table)
    return table


def build_frozen_target_encoder(checkpoint, device):
    """Construct and strictly restore only the EMA target encoder."""
    if checkpoint.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError(
            f"checkpoint architecture must be {MODEL_ARCHITECTURE!r}; "
            f"got {checkpoint.get('architecture')!r}"
        )
    cfg = checkpoint.get("config")
    state_dict = checkpoint.get("target_encoder")
    if not isinstance(cfg, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint must contain config and target_encoder mappings")
    encoder = build_encoder(cfg, torch.device(device))
    encoder.load_state_dict(state_dict, strict=True)
    encoder.requires_grad_(False).eval()
    return encoder


def _feature_columns(table):
    columns = [column for column in table.columns if str(column).startswith("feature_")]
    try:
        ordered = sorted(columns, key=lambda column: int(str(column).removeprefix("feature_")))
    except ValueError as error:
        raise ValueError("feature columns must be named feature_0 ... feature_D") from error
    expected = [f"feature_{index}" for index in range(len(ordered))]
    if ordered != expected:
        raise ValueError("feature columns must be contiguous from feature_0")
    if not ordered:
        raise ValueError("feature table has no feature columns")
    return ordered


def validate_feature_table(table):
    if not isinstance(table, pd.DataFrame):
        raise TypeError("feature table must be a pandas DataFrame")
    missing = [column for column in METADATA_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {', '.join(missing)}")
    feature_columns = _feature_columns(table)
    if table.empty:
        raise ValueError("feature table is empty")
    text_columns = set(METADATA_COLUMNS) - INTEGER_METADATA_COLUMNS - NUMERIC_METADATA_COLUMNS
    for column in text_columns:
        if table[column].isna().any() or (table[column].astype(str).str.strip() == "").any():
            raise ValueError(f"feature table contains empty {column!r} values")
    if not set(table["split"].astype(str)).issubset({"train", "val"}):
        raise ValueError("feature table split values must be train or val")
    for column in INTEGER_METADATA_COLUMNS:
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"{column} values must be non-negative integers")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} values must be integers")
    for column in NUMERIC_METADATA_COLUMNS:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} values must be finite")
    if not set(table["speed"]).issubset({"UGS", "FGS"}):
        raise ValueError("feature table has invalid speed values")
    if not set(table["clothing"]).issubset({"WoJ", "WJ"}):
        raise ValueError("feature table has invalid clothing values")
    if not set(table["direction"]).issubset({"R2L", "L2R"}):
        raise ValueError("feature table has invalid direction values")
    num_frames = pd.to_numeric(table["num_frames"]).to_numpy(dtype=np.float64)
    fps = pd.to_numeric(table["fps"]).to_numpy(dtype=np.float64)
    if (num_frames <= 0).any() or (fps <= 0).any():
        raise ValueError("num_frames and fps must be positive")
    signed_drift = pd.to_numeric(
        table["shortcut_horizontal_centroid_drift_signed"]
    ).to_numpy(dtype=np.float64)
    absolute_drift = pd.to_numeric(
        table["shortcut_horizontal_centroid_drift_absolute"]
    ).to_numpy(dtype=np.float64)
    if (np.abs(signed_drift) > 1.0).any() or not np.allclose(
        absolute_drift, np.abs(signed_drift), rtol=1e-7, atol=1e-9
    ):
        raise ValueError("centroid-drift shortcut values are inconsistent")
    area_columns = [
        "shortcut_foreground_area_mean",
        "shortcut_foreground_area_q25",
        "shortcut_foreground_area_median",
        "shortcut_foreground_area_q75",
    ]
    areas = table[area_columns].apply(pd.to_numeric).to_numpy(dtype=np.float64)
    if ((areas < 0.0) | (areas > 1.0)).any():
        raise ValueError("foreground-area shortcut values must be in [0, 1]")
    if not np.all(areas[:, 1] <= areas[:, 2]) or not np.all(
        areas[:, 2] <= areas[:, 3]
    ):
        raise ValueError("foreground-area shortcut quantiles must be ordered")
    if (pd.to_numeric(table["shortcut_foreground_area_std"]) < 0.0).any():
        raise ValueError("foreground-area shortcut standard deviation is negative")
    if not np.allclose(
        pd.to_numeric(table["shortcut_log_frame_count"]),
        np.log(num_frames),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("shortcut_log_frame_count does not match num_frames")
    if not np.allclose(
        pd.to_numeric(table["shortcut_duration_seconds"]),
        num_frames / fps,
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("shortcut_duration_seconds does not match num_frames / fps")

    per_recording = (
        "direction_clip_id",
        "source_video_id",
        "subject_id",
        "split",
        "speed",
        "clothing",
        "direction",
    )
    if (table.groupby("recording_id", sort=False)[list(per_recording)].nunique() > 1).any().any():
        raise ValueError("recording_id maps to inconsistent hierarchy or factor metadata")
    source_descriptor = ("subject_id", "split", "speed", "clothing")
    inconsistent_sources = (
        table.groupby("source_video_id", sort=False)[list(source_descriptor)].nunique() > 1
    )
    if inconsistent_sources.any().any():
        raise ValueError("source_video_id maps to inconsistent scientific metadata")
    values = table[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("feature table contains non-finite features")
    duplicate_key = ["split", "sequence_id", "window_start"]
    if table.duplicated(duplicate_key).any():
        raise ValueError(
            "feature table contains duplicate clip windows by split/sequence_id/window_start"
        )
    return feature_columns


def _sidecar_path(path):
    path = Path(path)
    return path.with_suffix(path.suffix + ".metadata.json")


def _atomic_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(temporary)


def _write_json_atomic(payload, destination):
    temporary = _atomic_path(destination)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_feature_table(table, output_path, metadata=None):
    """Write a validated feature table as CSV or non-pickled compressed NPZ."""
    feature_columns = validate_feature_table(table)
    output_path = Path(output_path)
    suffix = output_path.suffix.casefold()
    if suffix not in {".csv", ".npz"}:
        raise ValueError("feature output must end in .csv or .npz")
    temporary = _atomic_path(output_path)
    try:
        if suffix == ".csv":
            table.to_csv(temporary, index=False, float_format="%.9g")
        else:
            arrays = {
                column: table[column].to_numpy(
                    dtype=(
                        np.int64
                        if column in INTEGER_METADATA_COLUMNS
                        else np.float64
                        if column in NUMERIC_METADATA_COLUMNS
                        else str
                    )
                )
                for column in METADATA_COLUMNS
            }
            arrays["features"] = table[feature_columns].to_numpy(dtype=np.float32)
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    description = {
        **(dict(metadata) if metadata is not None else {}),
        "feature_source": FEATURE_SOURCE,
        "feature_formula": FEATURE_FORMULA,
        "row_count": int(len(table)),
        "feature_dim": len(feature_columns),
    }
    _write_json_atomic(description, _sidecar_path(output_path))
    return {"features": output_path, "metadata": _sidecar_path(output_path)}


def read_feature_table(path, *, validate=True):
    """Read a CSV/NPZ feature table and its optional metadata sidecar.

    Callers that must isolate a split before semantic validation may set
    ``validate=False`` and validate the selected rows at their own boundary.
    Structural NPZ checks are always enforced.
    """
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        table = pd.read_csv(
            path,
            dtype={
                column: str
                for column in METADATA_COLUMNS
                if column not in INTEGER_METADATA_COLUMNS | NUMERIC_METADATA_COLUMNS
            },
        )
    elif path.suffix.casefold() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            missing = [column for column in METADATA_COLUMNS if column not in archive]
            if missing or "features" not in archive:
                raise ValueError(f"NPZ feature table is missing arrays: {missing or ['features']}")
            features = np.asarray(archive["features"], dtype=np.float32)
            if features.ndim != 2:
                raise ValueError("NPZ features array must have shape [examples, dimensions]")
            metadata_frame = pd.DataFrame({
                column: archive[column] for column in METADATA_COLUMNS
            })
            feature_frame = pd.DataFrame(
                features,
                columns=[f"feature_{index}" for index in range(features.shape[1])],
            )
            table = pd.concat([metadata_frame, feature_frame], axis=1)
    else:
        raise ValueError("feature input must end in .csv or .npz")
    if validate:
        validate_feature_table(table)
    sidecar_path = _sidecar_path(path)
    metadata = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    return table, metadata


__all__ = [
    "FEATURE_FORMULA",
    "FEATURE_SOURCE",
    "INTEGER_METADATA_COLUMNS",
    "METADATA_COLUMNS",
    "NUMERIC_METADATA_COLUMNS",
    "build_frozen_target_encoder",
    "export_frozen_features",
    "read_feature_table",
    "validate_feature_table",
    "write_feature_table",
]
