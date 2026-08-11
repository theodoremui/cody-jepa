# Adversarial design review: iso-catalog phase allocation

## Verdict

The prior low-versus-high support by frozen-versus-resampled-anchor experiment should not be launched as the main ICLR study. It confounded extreme repetition, arbitrary start-index randomness, and unequal nominal support. The revised iso-catalog protocol is a credible controlled study only if its phase, metric, power, and systems gates pass before training.

The revised question is narrower and stronger: with fixed exposure and fixed nominal sequence-origin catalog size, does allocating that catalog across sequences or phase-separated origins change donor-based factor recombination beyond independent factor prediction?

## Main objections and repairs

| Objection | Required repair |
| --- | --- |
| A regular anchor grid counts starts, not gait content | Estimate stride period outcome-blind, use one common `k=4`-eligible corpus, and validate phase separation manually. |
| Fewer sequences can become an extreme repetition study | Fix `U × k = 250,000` and one shared exposure tier. Report planned recurrence. |
| `U × k` still does not make information equal | Call it nominal catalog cardinality and report overlap, trajectory separation, phase coverage, and near-duplicate clusters. |
| More temporal starts may only add randomness | Compare phase-separated `k=4` against four nearby origins centered on the same base phase with matched nuisance streams. |
| Three allocations cannot identify a substitution law | Call them an allocation path. Treat breadth versus phase depth as the confirmatory contrast and balanced as an intermediate diagnostic. |
| Top-1 has coarse shelves | Use continuous eight-gallery margin as the primary score, validate it synthetically, and require top-1 plus MRR directional concordance. |
| GFC can be explained by separate factor recovery | Use an identically scaled independent-completion control and test the residual GFC contrast. |
| Eight blocks may be underpowered | Run a legacy or development minimum-detectable-effect audit before launch. |
| Overlapping pools are not independent corpora | State inference conditionally on the fixed corpus and describe the repeated sources of variation honestly. |
| Silhouettes favor temporal information | Put the modality in the title, abstract, and limitations. Do not claim an RGB or universal video law. |
| Linear factor geometry can overstate composition | Validate factor transport on synthetic controls and use it only as supporting mechanism evidence. |

## Acceptance gates

The study proceeds only if all of the following are true before outcome access:

1. The phase estimator has adequate confidence and produces clearly separated semantic origins relative to jitter.
2. One common eligible catalog and common source-group rule can realize the required allocations.
3. The continuous score passes all synthetic positive and negative controls.
4. The primary eight-block contrast has useful prospective precision.
5. The registry, loader, provenance, evaluator, and figures work end to end.
6. A four-cell production pilot and storage probe pass at the intended checkpoint cadence.

## What would make the paper compelling

The highest-value outcome is not merely phase depth beating breadth. The compelling result is a coherent chain: phase depth differs from matched jitter, the continuous margin and rank metrics agree, the effect differs from independent completion, and factor-transport geometry supports the same direction. That chain would show that a flat count of clips misses a representation-relevant distinction in how video diversity enters training.

If this chain is incomplete, the paper should state the narrower result directly. A null or a raw allocation effect is useful when its controls are clean. It is not evidence for universal temporal semantics, intrinsic composition, or clinical utility.

## Explicitly rejected interpretations

- Phase origins are independent samples.
- Sequence count is a validated measure of people, walks, or environments.
- Three path points define a frontier, optimum, or scaling law.
- GFC is unsupervised disentanglement.
- The silhouette result transfers automatically to RGB, ambient sensing, or balance assessment.

