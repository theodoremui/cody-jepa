# Proposal: New Windows or New Sequences?

## Hierarchical data diversity in predictive video learning

**Research direction:** proposed hierarchical-diversity study

**Target:** ICLR 2027

**Abstract deadline:** September 18, 2026

**Paper deadline:** September 25, 2026

> **Status on August 8, 2026.** This is a proposed replacement for the
> [unique-sequence scaling study](../unique-sequence-scaling/proposal.md), not an additional experiment. The
> current code supports unrestricted temporal-window resampling and the 20-model scaling
> ladder. It does not yet support separated temporal anchors, the frozen-random policy,
> the 32-model factorial registry, or factorial inference. No GaitLU primary model has
> been trained and the locked Health&Gait outcomes remain unopened.

## Research question

> With encoder architecture, JEPA training recipe, and sampled-video exposure fixed,
> can temporal-window diversity within GaitLU sequences substitute for diversity across
> sequences in participant-averaged Health&Gait GFC-v2 top-1, and does any hierarchy
> effect remain after comparison with matched independent-factor completion?

In simpler terms, when sampled-example exposure is fixed, should a video model see new temporal
windows from familiar sequences or new sequences? Video data are nested. Frames form
windows, and windows belong to sequences. Counting every training clip as an unrelated
sample hides this structure.

## Why this is a non-trivial question

Repeated sampling from one sequence is not necessarily simple duplication. A new
16-frame start can reveal another temporal phase, while new masks and spatial crops
change the prediction problem. A new sequence can add different motion, body shape,
clothing, framing, or acquisition conditions. These sources of variation may be
substitutable, complementary, or useful for different representation properties.

Fixed-compute diversity studies such as [Hammoud et al.](https://openreview.net/forum?id=SLokff4aKI)
vary the number of unique samples in a flat pool. Hierarchical video methods such as
[HiCo](https://arxiv.org/abs/2204.03017) change the loss or positive-pair rule. This
study instead factorizes training support by hierarchy level while keeping the
objective, architecture, and total sampled exposure fixed. It asks whether the level
that supplies useful diversity changes the learned representation.

## Experiment

The primary design crosses two sequence pools with two temporal-window policies:

| | One frozen-random window per sequence | Resampled window on every draw |
|---|---:|---:|
| Approximately 2,500 sequences | Low breadth, low temporal support | Low breadth, high temporal support |
| Approximately 250,000 sequences | High breadth, low temporal support | High breadth, high temporal support |

Valid temporal anchors are spaced eight frames apart, so adjacent 16-frame windows
overlap by at most 50 percent. A frozen anchor is sampled uniformly once for each
sequence and replicate, then reused throughout training. The resampled condition draws
from the same anchor set on every exposure. Spatial transformations and JEPA masks are
paired across policies, so only the temporal anchor differs.

Eight replicate blocks produce 32 primary models. The four cells in a block share the
same initialization and optimization seed. The two window policies use the same
sequence manifest, and the 250,000-sequence pool contains the 2,500-sequence pool. Every
model processes the same prespecified number of sampled examples and uses its final-step
checkpoint.

The design intentionally replaces the four-level scaling curve. The earlier study asks
how performance changes along a data-size ladder. This study asks which level supplies
the useful diversity and allocates more model replication to that interaction.

## Evidence for substitution

The primary endpoint remains full-gallery GFC-v2 top-1. The primary estimand is the
interaction between sequence support and window policy. A negative interaction means
that temporal resampling helps more when the sequence pool is small, which is consistent
with substitution.

An interaction alone is not enough to claim substitution. The substitution-compatible
gate requires material benefits from temporal resampling at low sequence support and
from sequence breadth under frozen anchors, no material harm from either intervention
in the other policy, a material negative interaction, and the same interaction sign in
a ceiling sensitivity. Full performance replacement additionally requires equivalence
between the 2,500-sequence resampled condition and the 250,000-sequence frozen condition.
If the former remains materially worse, the result supports partial replacement.

Before training, one global eligibility rule will require at least two separated anchors
per sequence. The inventory will quantify anchor support and expected realized
sequence-anchor pairs at the lower possible exposure. If the median sequence has fewer
than four anchors or resampling provides less than fourfold expected support in either
pool, the hierarchy study will not launch. Anchors are support units, not independent
semantic examples.

## Representation claim

Independent-factor completion tests whether GFC-v2 is explained by separately predicting
speed, clothing, and direction. The raw GFC-v2 interaction is primary. The same
interaction in GFC-v2 minus completion is a required interpretation gate.

The completion-gap interaction has its own frozen margin. Gap equivalence supports an
independent-completion explanation at this resolution. A materially nonzero gap is
necessary but not sufficient evidence for composition beyond independent prediction
because ceiling and calibration can also affect the subtraction. Every other gap result
leaves the representation interpretation unresolved. None of these outcomes establishes
intrinsic compositionality or unsupervised factor discovery.

## Feasibility and decision gate

Thirty-two single-GPU models run in four waves if eight H100s are continuously reserved
and shared storage sustains the measured concurrent rate. At the frozen throughput
bound, training takes about 6.3 elapsed days. Historical rates imply roughly 3.7 to 5.8
days. The full-exposure study processes 262.144 million examples.

The switch occurs only if all four cells, an eight-job concurrent storage probe, the
factorial analysis, and checkpoint provenance pass end to end by August 16. Otherwise,
the project retains the existing 20-model study. The two studies will not be combined
after outcomes are observed.

The detailed implementation sequence and fallback rules appear in
[execution-plan.md](execution-plan.md).

## Scope

The defensible hierarchy is temporal windows nested within exact-content-deduplicated
GaitLU sequences. Anonymous paths and shard identifiers are not verified people,
physical walks, source videos, cameras, or environments. The study uses one JEPA recipe,
one silhouette corpus, and one supervised evaluation. It cannot support an
objective-general hierarchy law or a claim about RGB video.

The full protocol appears in [method.md](method.md).
