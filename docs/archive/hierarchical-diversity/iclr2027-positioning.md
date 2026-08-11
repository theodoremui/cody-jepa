# ICLR 2027 positioning: where video diversity lives

## Paper identity

The paper is a controlled representation-learning phenomenon study, supported by a reusable evaluation instrument. It is not a single-dataset ablation paper whose only claim is that one sampling choice improves gait accuracy.

One-sentence claim: **when video training examples have the same nominal sequence-origin catalog size and the same exposure, placing diversity across sequences versus across phase-separated views can change source-disjoint factor recombination differently from independent factor recovery.**

The title should make the controlled scope clear:

> **Where Does Diversity Live? An Iso-Catalog Study of Predictive Learning from Gait Silhouettes**

## Why one domain is defensible

Gait silhouettes are not offered as a representative sample of all video. They are an instrument. Their repeated, structured gait cycle makes it possible to define outcome-blind phase-separated origins, and Health&Gait provides a complete speed, clothing, and direction factorial gallery. That supports a source-disjoint retrieval test that ordinary video benchmarks do not make cleanly available.

The paper must say exactly this. It must not call the gallery an oracle for latent state, claim that gait is universal, or hide the silhouette modality. The narrow domain earns its place by making a causal data-allocation intervention and a compositional evaluation possible.

## Novelty wedge

Existing work shows that more data, more clips per video, or different training combinations can affect downstream quality. The surviving contribution is narrower: a naturalistic video SSL intervention that holds exposure and nominal sequence-origin cardinality fixed while changing the hierarchy level at which diversity is placed, evaluated against an independent-completion control on a matched continuous scale.

The work should cite temporal video pretraining allocation results, including Ghadiyaram et al. on video duration and breadth, MAE-ST on repeated sampling, and TCLR as adjacent temporal-contrastive work. It should also cite compositional-generalization work that separates factor recognition from novel combinations. Those papers make a broad claim of firstness untenable. They do not remove this specific hierarchy-allocation plus recombination-control question.

## Evidence story

1. Verify that semantic phase depth is genuinely different from nearby temporal jitter.
2. Follow the breadth, balanced, and phase-depth iso-catalog path with all paired blocks visible.
3. Use continuous GFC target margin as the primary score and require top-1 and MRR agreement.
4. Subtract identically scored independent completion to test whether the allocation changes recombination more than separate factor recovery.
5. Use locked factor-transport geometry as a supporting mechanism check.

The central figure should place all three primary allocation points on one x-axis, show each paired block, show GFC and independent-completion margins together, and place paired phase-depth versus jitter results beside the phase-depth point.

## Claims by outcome

- A phase-depth advantage over breadth plus jitter is a controlled demonstration that video diversity is not fully described by a flat clip count in this setting.
- A breadth advantage shows that equal nominal cardinality does not equal semantic breadth.
- A flat result is a bounded negative result: no allocation difference was resolved at the declared precision.
- A GFC residual that agrees with geometry supports a representation-level dissociation. Without that agreement, call the result supervised donor-based recombination only.

## Submission schedule

The ICLR 2027 abstract deadline is September 18, 2026 and the paper deadline is September 25. Results freeze on September 4. The period from September 5 through September 25 is paper-only work, with no optional experiments allowed on the critical path.

