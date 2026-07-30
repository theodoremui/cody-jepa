"""Representation health and cross-subject context diagnostics."""

from collections import defaultdict
from collections.abc import Mapping
import math
import random

import torch
from torch.utils.data import DataLoader, SequentialSampler


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


def subject_balanced_mean(values_by_subject):
    if not values_by_subject:
        return float("nan")
    return sum(
        sum(values) / len(values) for values in values_by_subject.values()
    ) / len(values_by_subject)


def balanced_wrong_subject_permutation(subject_ids, seed):
    """Return a seeded one-to-one cross-subject source permutation."""
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
    ordered_sources = ordered_targets[maximum_subject_rows:] + ordered_targets[:maximum_subject_rows]
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


def context_shuffle_plan(loader, seed):
    """Plan a full-loader cross-subject permutation without loading videos."""
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

    target_index_batches = [[int(index) for index in batch] for batch in loader.batch_sampler]
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


def context_source_loader(loader, shuffle_plan, seed):
    """Clone loader behavior for planned source-index batches."""
    if shuffle_plan["status"] != "complete":
        return None
    options = {
        "dataset": loader.dataset,
        "batch_sampler": shuffle_plan["source_index_batches"],
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": torch.Generator().manual_seed(int(seed)),
        "persistent_workers": loader.persistent_workers,
    }
    if loader.num_workers > 0:
        options["prefetch_factor"] = loader.prefetch_factor
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        options["pin_memory_device"] = pin_memory_device
    return DataLoader(**options)


def representation_health(metrics, config):
    issues = []
    if metrics["feature_std"] < float(config.get("min_feature_std", 1e-3)):
        issues.append("feature_std_below_threshold")
    if metrics["near_zero_variance_fraction"] > float(
        config.get("max_near_zero_variance_fraction", 0.5)
    ):
        issues.append("too_many_near_constant_dimensions")
    if metrics["effective_rank_ratio"] < float(
        config.get("min_effective_rank_ratio", 0.05)
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
        or context_gap < float(config.get("min_context_shuffle_loss_gap", 0.0))
    ):
        issues.append("context_shuffle_gap_below_threshold")
    if metrics["min_feature_norm"] <= 1e-8:
        issues.append("near_zero_feature_norm")
    if metrics["max_feature_norm"] > float(config.get("max_feature_norm", 1e4)):
        issues.append("feature_norm_above_threshold")
    return {"representations_healthy": not issues, "health_issues": issues}


__all__ = [
    "balanced_wrong_subject_permutation",
    "context_shuffle_plan",
    "context_source_loader",
    "representation_diagnostics",
    "representation_health",
    "subject_balanced_mean",
]
