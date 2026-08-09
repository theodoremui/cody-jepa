# Proposal: New Windows or New Sequences?

## Hierarchical data diversity in predictive video learning

**Research direction:** proposed hierarchical-diversity study

**Target:** ICLR 2027

**Abstract deadline:** September 18, 2026

**Paper deadline:** September 25, 2026

> **Status on August 9, 2026.** This would replace the
> [unique-sequence scaling study](../unique-sequence-scaling/proposal.md). No primary model
> is trained; Health&Gait outcomes remain unopened. The support audit and implementation
> remain pending.

## Research question

> **General question.** Suppose two video models train on the same number of clips. One
> sees clips from many sequences. The other revisits fewer sequences at more moments in
> time. Does this choice change what their representations support? In particular, can
> seeing more moments within fewer sequences match seeing one moment from many more
> sequences when a downstream task must combine information from different recordings,
> rather than simply recognize each factor separately?
>
> **Technical question.** A fixed clip budget can expose a video learner to more
> sequences or to more temporal windows from familiar sequences. We ask whether these
> sources of training support differ in the donor-based factor composition they enable
> after supervised alignment, beyond their effects on predicting each factor separately.
>
> Holding the encoder, latent-prediction objective, optimization, augmentations,
> nuisance draws, and total sampled clips fixed, does the effect of temporal-window
> resampling on factor-composition retrieval depend on whether training uses 2,500 or
> 250,000 exact-content-deduplicated sequences? Is this interaction distinguishable
> from the corresponding interaction in independent-factor prediction? At these
> allocations, is the small resampled pool equivalent to the 100×-larger frozen-window
> pool under a separately justified margin?

The design estimates effects at these allocations. It does not optimize a general data
allocation policy.

## Why this is novel and non-trivial

Different starts can reveal temporal phases, while new sequences may add motion, body
shape, clothing, framing, or acquisition variation. The two sources may interact.

Prior work compares video breadth with temporal information in continual-learning replay
([Just a Glimpse](https://arxiv.org/abs/2305.18418)) and video-language finetuning
([VideoWeave](https://arxiv.org/abs/2601.06309)). This study instead fully crosses
sequence-pool size with within-sequence window access during fixed-objective predictive
pretraining, then compares its effect on factor-composition retrieval with its effect on
separate factor prediction.

## Experiment

The primary design crosses two sequence pools with two temporal-window policies:

| | One frozen-random window per sequence | Resampled window on every draw |
|---|---:|---:|
| Approximately 2,500 sequences | Low breadth, low temporal support | Low breadth, high temporal support |
| Approximately 250,000 sequences | High breadth, low temporal support | High breadth, high temporal support |

Temporal anchors are eight frames apart, so adjacent 16-frame windows overlap by 50
percent and are not treated as independent examples. The inventory reports both this
anchor support and a conservative count based on non-overlapping windows.

A frozen anchor is sampled uniformly for each sequence and replicate, then reused. This
is an information-support control, not a recommended recipe. The resampled condition
draws from the same anchor set on every exposure. Spatial transformations and JEPA masks
are paired, so only the temporal anchor differs within each sequence-support pair.

Eight replicate blocks produce 32 models. Cells within a block share initialization and
optimization seeds. Policies share manifests, pools are nested, exposure is fixed, and
the final-step checkpoint is always used.

## Evidence for substitution

The primary endpoint remains full-gallery GFC-v2 top-1. The primary estimand is the
interaction between sequence support and window policy. A negative interaction means
that temporal resampling helps more when the sequence pool is small, which is consistent
with substitution.

Because top-1 is bounded, saturation can create an apparently negative raw interaction.
The main result therefore reports raw and prespecified clipped-logit interactions
together. A sign reversal blocks a substitution claim.

An interaction alone cannot establish substitution. The claim requires material benefits
from resampling at low sequence support and from sequence breadth under frozen anchors,
no material harm in the other two cells, a material negative interaction, and the same
sign in the ceiling sensitivity. Full performance replacement also requires equivalence
between the 2,500-sequence resampled condition and the 250,000-sequence frozen condition.
The separately justified 6.25-point margin is the largest average retrieval loss treated
as practically interchangeable at these allocations. It corresponds to one of the 16
GFC-v2 queries per participant and is frozen before outcomes are opened. If the small
resampled pool remains materially worse, the result supports partial replacement.

Before implementation, the private inventory audit reconstructs all nested pools and
reports anchor counts, non-overlapping counts, expected support, and overlap. Every pool
must have a median of at least four anchors and at least fourfold expected support under
resampling. Otherwise, the study will not launch. Anchors are support units, not
independent semantic examples.

## Representation claim

Independent-factor completion tests whether GFC-v2 is explained by separate speed,
clothing, and direction predictions. The raw GFC-v2 interaction is primary; the
GFC-v2-minus-completion interaction is the interpretation gate.

Gap equivalence supports an independent-completion explanation. A materially nonzero gap
is necessary but insufficient evidence beyond independent prediction. Other results are
unresolved. None establishes intrinsic compositionality or unsupervised factor discovery.

## Feasibility and decision gate

Thirty-two single-GPU models run in four waves if eight H100s are continuously reserved
and shared storage sustains the measured rate. Training takes about 3.7 to 6.3 elapsed
days and processes 262.144 million examples at full exposure.

The inventory gate must pass first. The switch then requires all four cells, an eight-job
storage probe, factorial analysis, and checkpoint provenance to pass by August 16.
Otherwise, the project retains the existing 20-model study. The studies will not be
combined after outcomes are observed.

The detailed implementation sequence and fallback rules appear in
[execution-plan.md](execution-plan.md).

## Scope

The hierarchy is windows nested within exact-content-deduplicated sequences, not verified
people, walks, cameras, or environments. One JEPA recipe, silhouette corpus, and
supervised evaluation cannot establish an objective-general law or RGB transfer.

The full protocol appears in [method.md](method.md).
