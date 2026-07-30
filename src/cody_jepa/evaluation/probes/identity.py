"""Closed-set and held-out identity evaluation protocols."""

import numpy as np

from ..features import FEATURE_SOURCE, validate_feature_table
from .common import classification_metrics, linear_predictions


def closed_set_masks(table, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    subjects = table["subject_id"].astype(str).to_numpy()
    source_groups = (
        table["split"].astype(str) + "\0" + table["source_video_id"].astype(str)
    ).to_numpy()
    train_groups = set()
    val_groups = set()
    rng = np.random.default_rng(seed)
    for subject in sorted(set(subjects)):
        groups = np.asarray(sorted(set(source_groups[subjects == subject])), dtype=str)
        if len(groups) < 2:
            raise ValueError(
                f"identity_closed_set needs at least two source videos for subject {subject!r}"
            )
        rng.shuffle(groups)
        val_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
        val_groups.update(groups[:val_count])
        train_groups.update(groups[val_count:])
    train_mask = np.fromiter((group in train_groups for group in source_groups), dtype=bool)
    val_mask = np.fromiter((group in val_groups for group in source_groups), dtype=bool)
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
    train_mask, val_mask = closed_set_masks(closed_set, validation_fraction, seed)
    features = closed_set[feature_columns].to_numpy(dtype=np.float64)
    labels = closed_set["subject_id"].astype(str).to_numpy()
    predictions, iterations = linear_predictions(
        features[train_mask], labels[train_mask], features[val_mask], max_iter, seed
    )
    return classification_metrics(
        "identity_closed_set",
        "source_video_disjoint_stratified_logistic_regression",
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
            "train_sources": int(
                closed_set.loc[train_mask, ["split", "source_video_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "val_sources": int(
                closed_set.loc[val_mask, ["split", "source_video_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
    )


def evaluate_identity_heldout_retrieval(
    table,
    feature_source=FEATURE_SOURCE,
    enrollment_sources=1,
    seed=0,
):
    from sklearn.preprocessing import StandardScaler

    feature_columns = validate_feature_table(table)
    heldout = table.loc[table["split"].astype(str) == "val"].copy()
    if heldout.empty:
        raise ValueError("identity_heldout_retrieval requires validation examples")
    enrollment_sources = int(enrollment_sources)
    if enrollment_sources <= 0:
        raise ValueError("enrollment_sources must be positive")

    enrollment_indices = []
    query_indices = []
    rng = np.random.default_rng(seed)
    for subject in sorted(heldout["subject_id"].astype(str).unique()):
        subject_rows = heldout.loc[heldout["subject_id"].astype(str) == subject]
        sources = np.asarray(sorted(subject_rows["source_video_id"].astype(str).unique()))
        if len(sources) <= enrollment_sources:
            raise ValueError(
                "identity_heldout_retrieval needs more source videos than enrollment_sources "
                f"for subject {subject!r}"
            )
        rng.shuffle(sources)
        selected = set(sources[:enrollment_sources])
        enrollment_indices.extend(
            subject_rows.index[subject_rows["source_video_id"].astype(str).isin(selected)]
        )
        query_indices.extend(
            subject_rows.index[~subject_rows["source_video_id"].astype(str).isin(selected)]
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
    return classification_metrics(
        "identity_heldout_retrieval",
        "heldout_subject_source_disjoint_nearest_centroid",
        feature_source,
        enrollment_labels,
        queries["subject_id"].astype(str).to_numpy(),
        predictions,
        {
            "distance": "euclidean_after_enrollment_standard_scaling",
            "enrollment_sources_per_subject": enrollment_sources,
            "enrollment_sources": int(enrollment["source_video_id"].nunique()),
            "query_sources": int(queries["source_video_id"].nunique()),
        },
    )


__all__ = [
    "closed_set_masks",
    "evaluate_identity_closed_set",
    "evaluate_identity_heldout_retrieval",
]
