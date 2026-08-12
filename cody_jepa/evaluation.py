"""Frozen-representation evaluation: feature export and linear probes.

Features come from the EMA target encoder, mean-pooled over tokens. Probes are
subject- or source-disjoint so a high score cannot come from matching the same
participant or the same source video on both sides of the split.
"""

import warnings

import numpy as np
import pandas as pd
import torch

from .engine import autocast, video_from_batch
from .models import build_encoder


METADATA_COLUMNS = (
    "subject_id",
    "split",
    "gait_system",
    "speed",
    "clothing",
    "direction",
    "recording_id",
    "source_video_id",
    "window_start",
)


def build_target_encoder(checkpoint, device):
    """Rebuild the frozen EMA target encoder from a saved checkpoint."""
    encoder = build_encoder(checkpoint["config"], device)
    encoder.load_state_dict(checkpoint["target_encoder"])
    return encoder.requires_grad_(False).eval()


@torch.inference_mode()
def export_features(encoder, loaders, config, device):
    """Return a DataFrame of pooled features plus clip metadata, one row per clip."""
    rows, features = [], []
    for loader in loaders:
        for batch in loader:
            video = video_from_batch(batch, device, config)
            with autocast(config, device):
                _, pre_norm = encoder(video, return_pre_norm=True)
            features.append(pre_norm.float().mean(dim=1).cpu().numpy())
            batch_size = video.size(0)
            for index in range(batch_size):
                rows.append(
                    {
                        column: _item(batch[column], index)
                        for column in METADATA_COLUMNS
                    }
                )
    features = np.concatenate(features).astype(np.float32)
    table = pd.DataFrame(rows)
    columns = pd.DataFrame(
        features, columns=[f"feature_{i}" for i in range(features.shape[1])]
    )
    return pd.concat([table, columns], axis=1)


def _item(values, index):
    value = values[index]
    return value.item() if torch.is_tensor(value) else value


def feature_columns(table):
    return [column for column in table.columns if column.startswith("feature_")]


def _metrics(task, train_labels, true_labels, predicted_labels):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    true_labels = np.asarray(true_labels, dtype=str)
    predicted_labels = np.asarray(predicted_labels, dtype=str)
    _, counts = np.unique(true_labels, return_counts=True)
    return {
        "task": task,
        "train_examples": int(len(train_labels)),
        "val_examples": int(len(true_labels)),
        "num_classes": int(len(set(true_labels))),
        "majority_baseline": float(counts.max() / counts.sum()),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
    }


def _logistic_predictions(train_features, train_labels, val_features, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    # The solver's line search can transiently overflow before converging to
    # finite weights; those warnings would otherwise bury the probe output.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?:divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        model.fit(train_features, train_labels)
        return model.predict(val_features)


def probe_gait_system(table, seed=0):
    """Classify walking speed (UGS/FGS) across held-out participants."""
    columns = feature_columns(table)
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    predictions = _logistic_predictions(
        train[columns].to_numpy(np.float64),
        train["gait_system"].astype(str).to_numpy(),
        val[columns].to_numpy(np.float64),
        seed,
    )
    return _metrics(
        "gait_system",
        train["gait_system"].astype(str).to_numpy(),
        val["gait_system"].astype(str).to_numpy(),
        predictions,
    )


def probe_identity(table, seed=0, split="train"):
    """Identify participants, holding out whole source videos from training."""
    columns = feature_columns(table)
    subset = table[table["split"] == split]
    rng = np.random.default_rng(seed)
    val_sources = set()
    for _, rows in subset.groupby(subset["subject_id"].astype(str)):
        sources = np.asarray(sorted(rows["source_video_id"].astype(str).unique()))
        if len(sources) < 2:
            raise ValueError("identity probe needs two source videos per participant")
        rng.shuffle(sources)
        val_sources.update(sources[: max(1, len(sources) // 4)])

    is_val = subset["source_video_id"].astype(str).isin(val_sources).to_numpy()
    features = subset[columns].to_numpy(np.float64)
    labels = subset["subject_id"].astype(str).to_numpy()
    predictions = _logistic_predictions(
        features[~is_val], labels[~is_val], features[is_val], seed
    )
    return _metrics("identity", labels[~is_val], labels[is_val], predictions)


def run_probes(table, seed=0):
    """Run every probe and return a one-row-per-task DataFrame."""
    return pd.DataFrame([probe_gait_system(table, seed), probe_identity(table, seed)])
