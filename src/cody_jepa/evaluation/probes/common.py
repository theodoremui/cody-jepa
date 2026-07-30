"""Shared classification helpers for representation probes."""

import warnings

import numpy as np


def majority_baseline(labels):
    _, counts = np.unique(np.asarray(labels, dtype=str), return_counts=True)
    return float(counts.max() / counts.sum())


def classification_metrics(
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
        "majority_baseline": majority_baseline(true_labels),
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


def linear_predictions(train_features, train_labels, val_features, max_iter, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(max_iter),
            class_weight="balanced",
            random_state=int(seed),
        ),
    )
    # The optimizer can emit transient NumPy matmul warnings during a line
    # search even when it later converges to finite weights.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.filterwarnings(
            "ignore",
            message=r"(?:divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        model.fit(train_features, train_labels)
        predictions = model.predict(val_features)
    iterations = int(np.max(model.named_steps["logisticregression"].n_iter_))
    if iterations >= int(max_iter):
        raise RuntimeError("logistic-regression probe did not converge before max_iter")
    return predictions, iterations


__all__ = ["classification_metrics", "linear_predictions", "majority_baseline"]
