"""Validation loop for JEPA prediction and representation health."""

from collections import defaultdict
import math
import random

import torch

from cody_jepa.masks import DEFAULT_MASK_GROUPS, multiblock_mask

from .batches import video_from_batch
from .diagnostics import (
    context_shuffle_plan,
    context_source_loader,
    representation_diagnostics,
    representation_health,
    subject_balanced_mean,
)
from .losses import encode_targets, group_forward
from .runtime import autocast_context


@torch.inference_mode()
def evaluate_jepa(
    context_encoder,
    target_encoder,
    predictor,
    loader,
    config,
    device,
    expected_split,
    mask_seed,
    mask_groups=DEFAULT_MASK_GROUPS,
    context_shuffle=True,
    context_seed=None,
):
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()
    mask_rng = random.Random(mask_seed)
    total_loss = total_cosine = 0.0
    examples = 0
    subject_loss = defaultdict(list)
    subject_cosine = defaultdict(list)
    pooled_features = []
    shuffle_gap_sum = 0.0
    shuffle_gap_examples = 0
    subject_shuffle_gaps = defaultdict(list)
    shuffle_plan = None
    if context_shuffle:
        if context_seed is None:
            raise ValueError("context_seed is required when context_shuffle=True")
        shuffle_plan = context_shuffle_plan(loader, context_seed)
    source_loader = (
        context_source_loader(loader, shuffle_plan, context_seed)
        if shuffle_plan is not None and shuffle_plan["status"] == "complete"
        else None
    )
    source_iterator = iter(source_loader) if source_loader is not None else None

    for batch in loader:
        video = video_from_batch(batch, device, config, expected_split)
        masks = multiblock_mask(
            config, video.size(0), mask_rng, device=device, mask_groups=mask_groups
        )
        with autocast_context(config, device):
            online_tokens = context_encoder(video)
            pooled_features.append(online_tokens.float().mean(dim=1).cpu())
            targets, _ = encode_targets(
                target_encoder,
                video,
                batch_standardize=bool(config.get("target_batch_standardize", False)),
            )
            batch_loss = torch.zeros(video.size(0), device=device)
            batch_cosine = torch.zeros(video.size(0), device=device)
            for mask_index, mask_group in enumerate(masks):
                loss, cosine = group_forward(
                    context_encoder,
                    predictor,
                    video,
                    targets,
                    mask_group,
                    mask_index,
                    float(config.get("loss_exp", 1.0)),
                )
                batch_loss += loss / len(masks)
                batch_cosine += cosine / len(masks)
            if shuffle_plan is not None and shuffle_plan["status"] == "complete":
                try:
                    source_batch = next(source_iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "context source loader ended before the target loader"
                    ) from error
                shuffled_video = video_from_batch(
                    source_batch, device, config, expected_split
                )
                target_subjects = [str(value).casefold() for value in batch["subject_id"]]
                source_subjects = [
                    str(value).casefold() for value in source_batch["subject_id"]
                ]
                if any(
                    target == source
                    for target, source in zip(target_subjects, source_subjects)
                ):
                    raise RuntimeError("context-shuffle plan paired the same subject")
                shuffled_loss = torch.zeros(video.size(0), device=device)
                for mask_index, mask_group in enumerate(masks):
                    loss, _ = group_forward(
                        context_encoder,
                        predictor,
                        shuffled_video,
                        targets,
                        mask_group,
                        mask_index,
                        float(config.get("loss_exp", 1.0)),
                    )
                    shuffled_loss += loss / len(masks)
                gaps = (shuffled_loss - batch_loss).float().cpu()
                shuffle_gap_sum += float(gaps.sum())
                shuffle_gap_examples += video.size(0)
                for subject, gap in zip(batch["subject_id"], gaps):
                    subject_shuffle_gaps[str(subject).casefold()].append(float(gap))
        total_loss += float(batch_loss.sum())
        total_cosine += float(batch_cosine.sum())
        examples += video.size(0)
        for index, subject in enumerate(batch["subject_id"]):
            subject_loss[str(subject)].append(float(batch_loss[index]))
            subject_cosine[str(subject)].append(float(batch_cosine[index]))

    if source_iterator is not None:
        try:
            next(source_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("context source loader outlasted the target loader")
    if examples == 0:
        raise RuntimeError("evaluation loader produced no batches")

    diagnostics = representation_diagnostics(torch.cat(pooled_features, dim=0))
    diagnostics.update(
        {
            "loss": total_loss / examples,
            "cosine": total_cosine / examples,
            "subject_balanced_loss": subject_balanced_mean(subject_loss),
            "subject_balanced_cosine": subject_balanced_mean(subject_cosine),
            "context_shuffle_loss_gap": (
                shuffle_gap_sum / shuffle_gap_examples
                if shuffle_gap_examples
                else float("nan")
            ),
            "subject_balanced_context_shuffle_loss_gap": subject_balanced_mean(
                subject_shuffle_gaps
            ),
            "context_shuffle_pairs": shuffle_gap_examples,
            "context_shuffle_batches": (
                len(shuffle_plan["source_index_batches"])
                if shuffle_plan is not None and shuffle_plan["status"] == "complete"
                else 0
            ),
            "context_shuffle_subjects": len(subject_shuffle_gaps),
            "context_shuffle_unique_sources": (
                len(set(shuffle_plan["source_positions"]))
                if shuffle_plan is not None and shuffle_plan["status"] == "complete"
                else 0
            ),
        }
    )
    if not all(
        math.isfinite(value)
        for key, value in diagnostics.items()
        if key
        not in {
            "context_shuffle_loss_gap",
            "subject_balanced_context_shuffle_loss_gap",
        }
    ):
        raise FloatingPointError(f"non-finite evaluation diagnostics: {diagnostics}")
    diagnostics.update(
        {
            "representation_source": "context_encoder_final_norm_full_view_mean_pool",
            "context_shuffle_pairing": "global_seeded_cross_subject_permutation_v1",
            "context_shuffle_status": (
                shuffle_plan["status"] if shuffle_plan is not None else "disabled"
            ),
        }
    )
    diagnostics.update(representation_health(diagnostics, config))
    return diagnostics


__all__ = ["evaluate_jepa"]
