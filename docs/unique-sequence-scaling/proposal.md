# Proposal: Unique-Sequence Diversity at Fixed Training Exposure

**Research direction:** existing unique-sequence scaling study

**Target:** ICLR 2027

**Abstract deadline:** September 18, 2026

**Paper deadline:** September 25, 2026

> **Status on August 8, 2026.** GFC-v2, GaitLU preparation, and fixed-exposure
> training are implemented and tested on constructed data. The private GaitLU corpus
> has not been prepared, the exposure tier has not been selected, and the 20 primary
> models have not been trained. The locked Health&Gait outcomes remain unopened.

This document covers the original unique-sequence scaling study. The alternative
hierarchical study is documented separately in
[hierarchical-diversity/proposal.md](../hierarchical-diversity/proposal.md).

## Research question

> With encoder architecture, JEPA training recipe, and a prespecified sampled-sequence
> exposure held fixed, does replacing repeated draws from an approximately
> 2,500-sequence, exact-content-deduplicated GaitLU pool with draws from the full
> eligible pool materially improve participant-averaged Health&Gait GFC-v2 top-1, and
> is that improvement distinguishable from the corresponding change in matched
> independent-factor completion?

In simpler terms, the study asks whether a video encoder supports better controlled
factor recombination
from seeing more unique walking sequences when sampled-example exposure stays constant. The
question is not whether a larger dataset helps when it also receives more optimization.
It is whether replacing repetition with unique data improves the target capability.

## Study design

GaitLU-1M supplies unlabelled silhouette sequences for encoder training. Health&Gait
supplies a separate, labelled evaluation with three controlled factors: walking speed,
clothing, and direction. No Health&Gait recording updates an encoder.

Five replicate seeds each define four nested GaitLU pools near 2,500, 25,000, 250,000,
and all eligible sequences. This produces 20 models. Every model uses the same
six-layer JEPA-style encoder, optimizer, masking policy, augmentations, effective batch
size, and sampled-sequence exposure. The primary exposure is 8,192,000 examples. A
prespecified throughput rule may reduce every run to 4,096,000 examples. The final-step
checkpoint is always used.

The primary comparison is the full-pool minus 2,500-pool change in GFC-v2 top-1 within
each replicate. Intermediate pools show the observed trajectory, but four data levels
do not justify a universal scaling law.

## Primary evaluation

Grounded Factorial Completion v2, or GFC-v2, tests whether information from two donor
recordings can identify a real target recording with the intended combination of
speed, clothing, and direction. Three development-fitted ridge heads align a frozen
recording representation with the three named factors. A query copies one factor block
from one donor and the other two blocks from a complementary donor. Neither donor may
come from the target's physical source walk. All eight factorial recordings remain in
the gallery, including both donors, so returning to a donor counts as a failure to
compose.

Each complete participant contributes 16 queries. The primary endpoint is
participant-averaged, full-gallery GFC-v2 top-1. The exact oracle spectrum is 12.5,
25, 50, and 100 percent when zero, one, two, or three binary factors are recovered.

## Controls and interpretation

The representation and nine non-learned acquisition cues receive the same three-head
ridge capacity. Independent-factor completion asks whether the three factor predictions
alone explain GFC-v2. Hard and soft completion are both computed, but their top-1
predictions coincide in the complete factorial gallery when temperature is positive.
They therefore form one top-1 control. Soft probabilities remain useful for calibration
and negative log-likelihood diagnostics.

The raw GFC-v2 scale contrast is primary. A paired contrast of GFC-v2 minus independent
completion is a required interpretation check. Gap equivalence supports an explanation
by independent completion at the frozen resolution. A materially nonzero gap is
necessary but not sufficient evidence for donor-based composition beyond independent
prediction. Any other gap result leaves that interpretation unresolved.

Context reliance, factor probes, effective rank, and cross-condition identity retrieval
are secondary capability measurements. They cannot replace the primary endpoint.

## Evidence and decision rules

The five trained-model contrasts are the primary inference units. Participants and
queries are repeated measurements within a trained model. The study reports a
Student's t interval over the five paired contrasts, every replicate trajectory, and
participant and crossed bootstraps as sensitivity analyses.

One of 16 queries corresponds to 6.25 percentage points. The implemented primary rule
classifies a result as meaningfully positive, positive but smaller than this resolution,
equivalent at this resolution, or inconclusive. A negative estimate is not automatically
promoted to a harm claim. Failure to reject zero is not evidence of equivalence.

## Contribution and limits

Fixed-compute data diversity, JEPA training, supervised factor alignment, and gait
representation learning already exist. The contribution is a controlled combination:
a replicated unique-data intervention, a source-disjoint real-target composition test,
an exact full-gallery oracle, and matched controls that limit the claim.

The strongest supported conclusion is specific to this encoder, training recipe,
GaitLU support manipulation, supervised alignment, and Health&Gait protocol. The study
cannot establish intrinsic compositionality, unsupervised factor discovery, clinical
utility, objective-general behavior, or transfer to RGB video.

The complete protocol appears in [method.md](method.md). Dataset preparation and role
boundaries appear in [data.md](data.md), and the evidence available before the primary
study appears in [results.md](results.md).
