# Where Does Video Diversity Live?

## A focused revision of the hierarchical-diversity study

Video data are hierarchical. A model can see a new walking sequence, or it can see a new part of a sequence it has already seen. These are both forms of diversity, but they are not the same intervention.

This study asks: **when the optimizer processes the same number of clips and has the same nominal catalog of sequence-origin atoms available, does it matter whether those atoms are spread across more sequences or concentrated in phase-separated views of fewer sequences?**

It is not asking whether more data are useful. It is not estimating a universal exchange rate between sequences and clips. It tests one controlled allocation path in gait-silhouette video self-supervised learning.

## The intervention

Every primary condition has nominal catalog size `U × k = 250,000`, where `U` is the number of eligible sequences and `k` is the number of deliberately selected phase origins per sequence.

| Allocation | Sequences `U` | Phase origins `k` | Nominal catalog |
| --- | ---: | ---: | ---: |
| Breadth | 250,000 | 1 | 250,000 |
| Balanced | 125,000 | 2 | 250,000 |
| Phase depth | 62,500 | 4 | 250,000 |
| Nearby-jitter diagnostic | 62,500 | 4 | 250,000 |

The first three rows form an iso-catalog allocation path. The fourth is a mechanism diagnostic, not a fourth path point. It replaces four phase-separated origins with four nearby origins around the same base phase. A phase-depth versus jitter difference is evidence that separated gait-cycle content matters beyond drawing different start indices.

`U × k` is a counting control, not a claim that atoms contain equal information. The paper will report phase coverage, window overlap, trajectory separation, and outcome-blind near-duplicate clusters.

## Comparable phase origins

For every eligible sequence, a frozen silhouette signal estimates stride period and confidence. All cells use the same `k = 4`-eligible corpus. A stable hash chooses a uniform base phase for each sequence and replicate block. The origin sets are nested: `k = 1` is the base phase, `k = 2` adds the antipodal phase, and `k = 4` adds the quarter-cycle phases. Nearby jitter uses symmetric small offsets around the same base phase.

The estimator, confidence threshold, clip construction, jitter offsets, and manual audit plan are frozen before outcomes are opened. If the audit does not establish reliable phase separation, the phase-allocation study does not launch.

## What stays fixed

Architecture, JEPA objective, sampled-clip exposure, optimizer, schedule, masks, spatial transforms, and checkpoint selection are fixed. An outcome-blind systems gate selects one exposure tier for every model: 8.192 million or 4.096 million clips. These imply about 32.77 or 16.38 planned draws per nominal atom.

Eight paired blocks contain breadth, balanced, and phase depth, for 24 primary models. Four prespecified blocks add nearby jitter, for four more. The study therefore trains 28 models, and the paired block is the model-level inference unit.

## The outcome and headline test

Health&Gait GFC v2 evaluates source-disjoint recombination of speed, clothing, and direction. It is a controlled representation instrument, not a clinical prediction task and not a test of unsupervised disentanglement.

The primary score is a continuous eight-gallery target margin:

$$
m(q)=d(q,\text{best non-target})-d(q,\text{true target}).
$$

Positive margin means the target wins. The same gallery construction is used for GFC recombination and independent factor completion. Top-1 and MRR are directional checks.

For each block and allocation, subtract independent-completion margin from GFC margin. The confirmatory contrast asks whether this residual differs between phase depth and breadth. This tests whether the allocation changes donor-based recombination more than separately predicted factors.

## Interpreting results

- Similar path points establish a useful null at the tested precision.
- Breadth winning shows equal nominal catalog size is not equal diversity.
- Phase depth winning and differing from nearby jitter shows value from temporally separated content, not only temporal randomness.
- A residual GFC effect that agrees with the locked factor-geometry diagnostic is evidence for a representation-level dissociation.

The strongest claim requires agreement among continuous margin, top-1, MRR, phase versus jitter, and geometry. The study remains a controlled case study in GaitLU silhouettes with one objective. It cannot establish a general video law, identify sequences as people, or make a clinical claim.

## ICLR 2027 schedule

The abstract deadline is September 18, 2026 and the paper deadline is September 25. Phase, catalog, metric, power, and systems gates must pass by August 17. Training and permitted systems reruns end by August 29. Results freeze by September 4, leaving September 5 through September 25 for paper writing only.
