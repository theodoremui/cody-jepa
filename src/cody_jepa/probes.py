"""Frozen-representation export and deliberately simple linear probes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch

from .single_stream_jepa import (
    CHECKPOINT_SCHEMA,
    MODEL_ARCHITECTURE,
    build_encoder,
    video_from_batch,
)


FEATURE_TABLE_SCHEMA = 1
PROBE_RESULTS_SCHEMA = 1
FEATURE_SOURCE = "target_encoder_pre_norm_mean"
FEATURE_FORMULA = "target_encoder(video, return_pre_norm=True)[1].mean(dim=1)"
METADATA_COLUMNS = (
    "sequence_id",
    "split",
    "subject_id",
    "gait_system",
    "trial",
    "window_start",
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
    """Return one mean-pooled pre-final-LayerNorm target feature per clip.

    ``loaders`` is a mapping from split name to a deterministic, non-shuffled
    iterable. The encoder is forced into eval mode and permanently frozen.
    """
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
                        if key == "window_start"
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
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"checkpoint schema must be {CHECKPOINT_SCHEMA}; got {checkpoint.get('schema')!r}"
        )
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


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for column in METADATA_COLUMNS[:-1]:
        if table[column].isna().any() or (table[column].astype(str).str.strip() == "").any():
            raise ValueError(f"feature table contains empty {column!r} values")
    if not set(table["split"].astype(str)).issubset({"train", "val"}):
        raise ValueError("feature table split values must be train or val")
    window_starts = pd.to_numeric(table["window_start"], errors="coerce")
    if window_starts.isna().any() or (window_starts < 0).any():
        raise ValueError("window_start values must be non-negative integers")
    if not np.equal(window_starts, np.floor(window_starts)).all():
        raise ValueError("window_start values must be integers")
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
                    dtype=np.int64 if column == "window_start" else str
                )
                for column in METADATA_COLUMNS
            }
            arrays["features"] = table[feature_columns].to_numpy(dtype=np.float32)
            arrays["schema_version"] = np.asarray(FEATURE_TABLE_SCHEMA, dtype=np.int64)
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    sidecar = {
        **(dict(metadata) if metadata is not None else {}),
        "schema_version": FEATURE_TABLE_SCHEMA,
        "feature_source": FEATURE_SOURCE,
        "feature_formula": FEATURE_FORMULA,
        "row_count": int(len(table)),
        "feature_dim": len(feature_columns),
        "feature_table_sha256": checkpoint_sha256(output_path),
    }
    _write_json_atomic(sidecar, _sidecar_path(output_path))
    return {"features": output_path, "metadata": _sidecar_path(output_path)}


def validate_feature_metadata(table, feature_path, metadata):
    """Validate the provenance sidecar required by reproducible probe runs."""
    feature_path = Path(feature_path)
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError(f"feature metadata sidecar is required for {feature_path}")
    feature_columns = validate_feature_table(table)
    required = {
        "schema_version": FEATURE_TABLE_SCHEMA,
        "feature_source": FEATURE_SOURCE,
        "feature_formula": FEATURE_FORMULA,
        "row_count": int(len(table)),
        "feature_dim": len(feature_columns),
        "feature_table_sha256": checkpoint_sha256(feature_path),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"feature metadata {key!r} mismatch: "
                f"expected={expected!r}, actual={metadata.get(key)!r}"
            )
    checkpoint_hash = metadata.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_hash)
    ):
        raise ValueError("feature metadata checkpoint_sha256 must be a lowercase SHA-256")
    if not isinstance(metadata.get("checkpoint"), str) or not metadata["checkpoint"].strip():
        raise ValueError("feature metadata checkpoint must identify its source path")
    return feature_columns


def read_feature_table(path):
    """Read a CSV/NPZ feature table and its optional metadata sidecar."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        table = pd.read_csv(
            path,
            dtype={column: str for column in METADATA_COLUMNS if column != "window_start"},
        )
    elif path.suffix.casefold() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            missing = [column for column in METADATA_COLUMNS if column not in archive]
            if missing or "features" not in archive:
                raise ValueError(f"NPZ feature table is missing arrays: {missing or ['features']}")
            features = np.asarray(archive["features"], dtype=np.float32)
            if features.ndim != 2:
                raise ValueError("NPZ features array must have shape [examples, dimensions]")
            if "schema_version" not in archive:
                raise ValueError("NPZ feature table is missing schema_version")
            schema_version = int(np.asarray(archive["schema_version"]).item())
            if schema_version != FEATURE_TABLE_SCHEMA:
                raise ValueError(
                    f"NPZ feature schema must be {FEATURE_TABLE_SCHEMA}; "
                    f"got {schema_version}"
                )
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
    validate_feature_table(table)
    sidecar_path = _sidecar_path(path)
    metadata = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    return table, metadata


def _majority_baseline(labels):
    _, counts = np.unique(np.asarray(labels, dtype=str), return_counts=True)
    return float(counts.max() / counts.sum())


def _classification_metrics(
    task,
    protocol,
    feature_source,
    train_labels,
    true_labels,
    predicted_labels,
    extra=None,
):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )

    train_labels = np.asarray(train_labels, dtype=str)
    true_labels = np.asarray(true_labels, dtype=str)
    predicted_labels = np.asarray(predicted_labels, dtype=str)
    labels = sorted(set(train_labels) | set(true_labels) | set(predicted_labels))
    result = {
        "task": task,
        "protocol": protocol,
        "feature_source": feature_source,
        "train_examples": int(len(train_labels)),
        "val_examples": int(len(true_labels)),
        "num_classes": len(labels),
        "majority_baseline": _majority_baseline(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
        ),
        "class_labels": labels,
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=labels
        ).astype(int).tolist(),
    }
    if extra:
        result.update(extra)
    return result


def _linear_predictions(train_features, train_labels, val_features, max_iter, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(max_iter),
            class_weight="balanced",
            random_state=int(seed),
        ),
    )
    # LBFGS can briefly explore non-finite trial weights during its line search
    # even with finite standardized inputs, then converge normally. Suppress
    # only NumPy's transient matmul warning; convergence and final predictions
    # remain checked and recorded below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?:divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        model.fit(train_features, train_labels)
        predictions = model.predict(val_features)
    logistic = model.named_steps["logisticregression"]
    iterations = int(np.max(logistic.n_iter_))
    if iterations >= int(max_iter):
        raise RuntimeError("logistic-regression probe did not converge before max_iter")
    return predictions, iterations


def _closed_set_masks(table, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    subjects = table["subject_id"].astype(str).to_numpy()
    sequence_groups = (
        table["split"].astype(str) + "\0" + table["sequence_id"].astype(str)
    ).to_numpy()
    train_groups = set()
    val_groups = set()
    rng = np.random.default_rng(seed)
    for subject in sorted(set(subjects)):
        groups = np.asarray(sorted(set(sequence_groups[subjects == subject])), dtype=str)
        if len(groups) < 2:
            raise ValueError(
                f"identity_closed_set needs at least two sequences for subject {subject!r}"
            )
        rng.shuffle(groups)
        val_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
        val_groups.update(groups[:val_count])
        train_groups.update(groups[val_count:])
    train_mask = np.fromiter((group in train_groups for group in sequence_groups), dtype=bool)
    val_mask = np.fromiter((group in val_groups for group in sequence_groups), dtype=bool)
    if np.any(train_mask & val_mask) or not train_mask.any() or not val_mask.any():
        raise RuntimeError("failed to construct disjoint closed-set identity partitions")
    return train_mask, val_mask


def evaluate_identity_closed_set(
    table,
    feature_source=FEATURE_SOURCE,
    validation_fraction=0.25,
    max_iter=2000,
    seed=0,
    source_split="train",
):
    feature_columns = validate_feature_table(table)
    closed_set = table.loc[table["split"].astype(str) == str(source_split)].copy()
    if closed_set.empty:
        raise ValueError(
            f"identity_closed_set requires examples from source split {source_split!r}"
        )
    train_mask, val_mask = _closed_set_masks(closed_set, validation_fraction, seed)
    features = closed_set[feature_columns].to_numpy(dtype=np.float64)
    labels = closed_set["subject_id"].astype(str).to_numpy()
    predictions, iterations = _linear_predictions(
        features[train_mask], labels[train_mask], features[val_mask], max_iter, seed
    )
    return _classification_metrics(
        "identity_closed_set",
        "sequence_disjoint_stratified_logistic_regression",
        feature_source,
        labels[train_mask],
        labels[val_mask],
        predictions,
        {
            "max_iter": int(max_iter),
            "iterations": iterations,
            "validation_fraction": float(validation_fraction),
            "source_split": str(source_split),
            "subjects": int(closed_set["subject_id"].nunique()),
            "train_sequences": int(
                closed_set.loc[train_mask, ["split", "sequence_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "val_sequences": int(
                closed_set.loc[val_mask, ["split", "sequence_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
    )


def evaluate_identity_heldout_retrieval(
    table,
    feature_source=FEATURE_SOURCE,
    enrollment_sequences=1,
    seed=0,
):
    from sklearn.preprocessing import StandardScaler

    feature_columns = validate_feature_table(table)
    heldout = table.loc[table["split"].astype(str) == "val"].copy()
    if heldout.empty:
        raise ValueError("identity_heldout_retrieval requires validation examples")
    enrollment_sequences = int(enrollment_sequences)
    if enrollment_sequences <= 0:
        raise ValueError("enrollment_sequences must be positive")

    enrollment_indices = []
    query_indices = []
    rng = np.random.default_rng(seed)
    for subject in sorted(heldout["subject_id"].astype(str).unique()):
        subject_rows = heldout.loc[heldout["subject_id"].astype(str) == subject]
        sequences = np.asarray(sorted(subject_rows["sequence_id"].astype(str).unique()))
        if len(sequences) <= enrollment_sequences:
            raise ValueError(
                "identity_heldout_retrieval needs more sequences than enrollment_sequences "
                f"for subject {subject!r}"
            )
        rng.shuffle(sequences)
        selected = set(sequences[:enrollment_sequences])
        enrollment_indices.extend(
            subject_rows.index[subject_rows["sequence_id"].astype(str).isin(selected)]
        )
        query_indices.extend(
            subject_rows.index[~subject_rows["sequence_id"].astype(str).isin(selected)]
        )

    enrollment = heldout.loc[enrollment_indices]
    queries = heldout.loc[query_indices]
    scaler = StandardScaler().fit(enrollment[feature_columns].to_numpy(dtype=np.float64))
    enrollment_features = scaler.transform(
        enrollment[feature_columns].to_numpy(dtype=np.float64)
    )
    query_features = scaler.transform(queries[feature_columns].to_numpy(dtype=np.float64))
    enrollment_labels = enrollment["subject_id"].astype(str).to_numpy()
    labels = sorted(set(enrollment_labels))
    centroids = np.stack([
        enrollment_features[enrollment_labels == label].mean(axis=0) for label in labels
    ])
    squared_distances = ((query_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predictions = np.asarray(labels)[squared_distances.argmin(axis=1)]
    return _classification_metrics(
        "identity_heldout_retrieval",
        "heldout_subject_nearest_centroid",
        feature_source,
        enrollment_labels,
        queries["subject_id"].astype(str).to_numpy(),
        predictions,
        {
            "distance": "euclidean_after_enrollment_standard_scaling",
            "enrollment_sequences_per_subject": enrollment_sequences,
            "enrollment_sequences": int(enrollment["sequence_id"].nunique()),
            "query_sequences": int(queries["sequence_id"].nunique()),
        },
    )


def evaluate_gait_system(
    table,
    feature_source=FEATURE_SOURCE,
    max_iter=2000,
    seed=0,
):
    feature_columns = validate_feature_table(table)
    train = table.loc[table["split"].astype(str) == "train"]
    val = table.loc[table["split"].astype(str) == "val"]
    if train.empty or val.empty:
        raise ValueError("gait_system probe requires both train and validation examples")
    train_subjects = {value.casefold() for value in train["subject_id"].astype(str)}
    val_subjects = {value.casefold() for value in val["subject_id"].astype(str)}
    overlap = sorted(train_subjects & val_subjects)
    if overlap:
        raise ValueError(
            "gait_system subject-held-out protocol found train/val subject overlap: "
            + ", ".join(overlap[:10])
        )
    train_labels = train["gait_system"].astype(str).to_numpy()
    val_labels = val["gait_system"].astype(str).to_numpy()
    if set(train_labels) != set(val_labels):
        raise ValueError(
            "gait_system classes must match across train and val; "
            f"train={sorted(set(train_labels))}, val={sorted(set(val_labels))}"
        )
    predictions, iterations = _linear_predictions(
        train[feature_columns].to_numpy(dtype=np.float64),
        train_labels,
        val[feature_columns].to_numpy(dtype=np.float64),
        max_iter,
        seed,
    )
    return _classification_metrics(
        "gait_system",
        "subject_heldout_logistic_regression",
        feature_source,
        train_labels,
        val_labels,
        predictions,
        {"max_iter": int(max_iter), "iterations": iterations},
    )


def evaluate_all_probes(
    table,
    feature_source=FEATURE_SOURCE,
    validation_fraction=0.25,
    enrollment_sequences=1,
    max_iter=2000,
    seed=0,
):
    return [
        evaluate_identity_closed_set(
            table, feature_source, validation_fraction, max_iter, seed
        ),
        evaluate_identity_heldout_retrieval(
            table, feature_source, enrollment_sequences, seed
        ),
        evaluate_gait_system(table, feature_source, max_iter, seed),
    ]


def write_probe_results(results, output_dir, run_metadata=None):
    """Write full JSON plus a one-row-per-task CSV summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "probe_metrics.json"
    csv_path = output_dir / "probe_metrics.csv"
    payload = {
        "schema_version": PROBE_RESULTS_SCHEMA,
        **(dict(run_metadata) if run_metadata is not None else {}),
        "results": list(results),
    }
    _write_json_atomic(payload, json_path)
    csv_rows = []
    for result in payload["results"]:
        csv_rows.append({
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
            for key, value in result.items()
        })
    temporary = _atomic_path(csv_path)
    try:
        pd.DataFrame(csv_rows).to_csv(temporary, index=False)
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"json": json_path, "csv": csv_path}


__all__ = [
    "FEATURE_FORMULA",
    "FEATURE_SOURCE",
    "METADATA_COLUMNS",
    "build_frozen_target_encoder",
    "checkpoint_sha256",
    "evaluate_all_probes",
    "evaluate_gait_system",
    "evaluate_identity_closed_set",
    "evaluate_identity_heldout_retrieval",
    "export_frozen_features",
    "read_feature_table",
    "validate_feature_metadata",
    "validate_feature_table",
    "write_feature_table",
    "write_probe_results",
]
