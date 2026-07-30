"""Subject-held-out gait-system evaluation."""

import numpy as np

from ..features import FEATURE_SOURCE, validate_feature_table
from .common import classification_metrics, linear_predictions


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
    predictions, iterations = linear_predictions(
        train[feature_columns].to_numpy(dtype=np.float64),
        train_labels,
        val[feature_columns].to_numpy(dtype=np.float64),
        max_iter,
        seed,
    )
    return classification_metrics(
        "gait_system",
        "subject_heldout_logistic_regression",
        feature_source,
        train_labels,
        val_labels,
        predictions,
        {"max_iter": int(max_iter), "iterations": iterations},
    )


__all__ = ["evaluate_gait_system"]
