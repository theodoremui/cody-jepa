# 17. Iso-catalog phase allocation and paired inference

![The iso-catalog allocation experiment](../images/17_hierarchical_support_and_factorial_inference.svg)

## Why this capstone matters

This lesson connects sampling, representation evaluation, and small-sample inference into one scientific argument. The motivating question is: if two video models process the same number of clips, is it equivalent to obtain those clips from many walking sequences or from several deliberately different locations in fewer walking cycles?

The answer cannot come from counting files, clip starts, or training steps alone. A video has hierarchy. New sequences can add different recording conditions and trajectories. New temporal locations in one sequence can reveal different positions in a motion cycle. Those changes can be valuable in different ways. The experiment therefore controls a count of available sequence-origin atoms while measuring whether the placement of that count changes what a representation supports.

This is a capstone, not a universal law of video data. Its design is conditional on one silhouette corpus, one predictive learning objective, and one downstream factorial instrument. Its value is methodological: it makes the usually hidden choice of where diversity lives explicit, testable, and auditable.

## Prerequisites

Review [08. Group-aware sampling and shortcut learning](08_group_aware_sampling.md) for groups and temporal origins, [11. Factorial state spaces](11_factorial_state_spaces.md) through [13. Context interventions and identity geometry](13_context_interventions.md) for factorial retrieval, and [14. Paired inference](14_paired_inference.md) through [16. Reproducible scientific evaluators](16_reproducible_scientific_evaluators.md) for uncertainty and provenance.

## Learning goals

By the end of this lesson, you will be able to distinguish exposure from nominal catalog cardinality, make nested phase-origin sets, explain why nearby jitter is an important control, calculate the GFC-minus-independent-completion contrast, preserve paired blocks during uncertainty estimation, and state what this design can and cannot conclude.

## 1. Start with the allocation, not with an accuracy table

Let `U` be the number of eligible sequences and `k` the number of selected origins per sequence. The revised experiment uses the following allocations:

| Allocation | `U` | `k` | `U × k` |
| --- | ---: | ---: | ---: |
| Breadth | 250,000 | 1 | 250,000 |
| Balanced | 125,000 | 2 | 250,000 |
| Phase depth | 62,500 | 4 | 250,000 |
| Nearby jitter | 62,500 | 4 | 250,000 |

The first three rows are an allocation path. The fourth is a control for the phase-depth endpoint. It has the same number of sequences and starts as phase depth, but its four origins are close to the same base phase rather than separated around a gait cycle.

The equality `U × k = 250,000` matters because it removes one simple explanation: one allocation has a larger nominal catalog of sequence-origin choices. It does not make the allocations equally informative. A new sequence can contain more semantic variation than a new phase origin, and two origins can overlap substantially. For that reason, call this an *iso-catalog* study, not an equal-information study.

Every model also receives the same sampled-clip exposure, architecture, objective, optimizer, mask distribution, spatial transformations, and checkpoint rule. If exposure is `C`, the planned recurrence of each nominal atom is `C / 250,000`. At 8,192,000 clips this is about 32.77; at 4,096,000 clips it is about 16.38. The exposure tier is chosen once by an outcome-blind systems test. It is never tuned separately for a favorable cell.

## 2. A phase origin is a scientific object, not a frame index

Regularly spaced frame starts do not guarantee different gait information. A sequence may be short, irregular, poorly segmented, or sampled at a cadence unrelated to its stride. A phase allocation needs an outcome-blind construction that connects the word “phase” to a measured property of the signal.

For every sequence, estimate stride period and a confidence score using a frozen silhouette signal, such as area or width autocorrelation. Apply one eligibility rule to make a common corpus that supports four candidate origins. This common corpus is essential. If the four-origin cell receives only long, easy sequences while the one-origin cell receives all sequences, phase depth is confounded with data quality.

For sequence `i` and block `r`, use a stable hash to choose a base phase `b_ir` uniformly. The semantic origin sets are nested:

$$
O^{(1)}_{i,r}=\{b_{i,r}\},\qquad
O^{(2)}_{i,r}=\{b_{i,r},b_{i,r}+1/2\},
$$

$$
O^{(4)}_{i,r}=\{b_{i,r},b_{i,r}+1/4,b_{i,r}+1/2,b_{i,r}+3/4\}\pmod 1.
$$

Nestedness gives the phrase “more origins” a clear meaning. The one-origin set is contained in the two-origin set, which is contained in the four-origin set. Stable hashing prevents an accidental correlation between phase choice and worker assignment.

The jitter control chooses four distinct, symmetric small offsets around the same `b_ir`. It shares sequence draws, base phases, masks, spatial transforms, exposure, and optimization streams with phase depth. Its only intended change is local coverage rather than separated locations. A phase-depth advantage over jitter is therefore much more informative than a comparison between one fixed start and several randomly resampled starts.

Before training, audit a stratified sample. Report confidence, realized origin coverage, window overlap, pose-trajectory distance, and reasons sequences were excluded. Also report known source groups and outcome-blind near-duplicate clusters. These measurements do not turn origins into independent examples. They show whether the intended intervention exists.

![Temporal origins and matched transforms](../images/08_windows_and_transforms.svg)

## 3. Blocks are the trained-model sampling unit

Eight paired blocks contain breadth, balanced, and phase-depth models. Four blocks selected before outcome access also contain nearby jitter. Thus there are 24 primary models and four diagnostic models, or 28 in total.

Within a block, primary rows share declared optimization and replicate seeds. For phase depth and jitter, they additionally share sequence draws, base phases, masks, and spatial transformations. This pairing reduces irrelevant variation and makes a direct comparison possible. It also means the models are not 28 unrelated replicates.

After participant aggregation, a primary score can have shape `(8, 3)`: block by allocation. Before aggregation it may have shape `(8, 3, P)`, where `P` is the number of complete participants. Participants and queries make a model score more precise, but they do not create additional trained models. The main uncertainty is variation across the eight paired training blocks.

The pools can be nested, `62,500 ⊂ 125,000 ⊂ 250,000`, but overlapping pools do not make blocks independent samples of all possible gait corpora. The honest interpretation is variation over the declared construction, phase rotation, and optimization procedure, conditional on the fixed corpus.

## 4. Measure a capability on a continuous common scale

Health&Gait supplies a factorial gallery over speed, clothing, and direction. GFC creates a query using source-disjoint donors and asks which of eight gallery items is the intended combination. This is useful because a model can recognize each factor independently while still failing to recombine factors from different sources.

For query `q`, define the target margin

$$
m(q)=d(q,\text{best non-target})-d(q,\text{true target}).
$$

Positive margin means the intended target is closer than every competitor. Unlike top-1, the margin records whether a win is narrow or decisive. The competitor rule, feature normalization, gallery construction, participant aggregation, and tie policy must be frozen before outcomes. Top-1 and mean reciprocal rank remain useful checks, but they are not the primary continuous scale.

GFC is not unsupervised disentanglement. The factor map is supervised and the result is a source-disjoint donor-based recombination score. To test whether an allocation changes more than separate factor recovery, construct an independent completion control on the same eight-gallery margin. Before release, test the evaluator on synthetic perfect recovery, independent noisy recovery, a missing factor, donor attraction, an acquisition shortcut, representation collapse, and confidence rescaling. These tests validate measurement behavior. They do not validate a desired research result.

![Factorial gallery geometry](../images/12_blockwise_distances_and_ranking.svg)

## 5. The primary contrast is a residual difference across paired blocks

Let `G_ra` be the participant-averaged GFC margin for block `r` and allocation `a`, and let `C_ra` be its independent-completion margin. Define the residual

$$
D_{r,a}=G_{r,a}-C_{r,a}.
$$

The confirmatory contrast is phase-depth residual minus breadth residual:

$$
P_r=D_{r,\mathrm{phase\_depth}}-D_{r,\mathrm{breadth}}.
$$

This is not a claim that a significant GFC result with a nonsignificant completion result proves composition. Both scores are first placed on the same scale and then compared within each trained-model block. A positive or negative result says that the allocation changed donor-based recombination by a different amount than it changed the independent control.

The eight `P_r` values are summarized with their mean, standard deviation, all individual points, and the prospectively chosen small-sample interval. A minimum-detectable-effect audit on development or legacy encoders must show that the planned design can resolve an effect of scientific interest. Otherwise a null result would be too ambiguous to interpret.

The raw breadth-to-phase-depth GFC contrast is reported beside this residual contrast. Balanced is a middle path point. It can reveal a monotone, flat, or non-monotone pattern, but three correlated points cannot establish a data law, an optimum, or a Pareto frontier. The paired four-block phase-depth versus jitter result is a mechanism diagnostic with lower precision. Show its uncertainty, not a second confirmatory primary result.

## 6. Sensitivity analysis keeps the hierarchy intact

For a participant-only bootstrap, draw participants with replacement and apply the same draw to every block and allocation. For a crossed bootstrap, also draw blocks with replacement. Each selected block carries its complete breadth, balanced, and phase-depth triplet. Do not resample individual model rows. Doing so destroys the covariance generated by pairing and estimates a different experiment.

Do not let a large number of queries masquerade as a large number of trained models. Query and participant resampling describe evaluation uncertainty. The primary trained-model comparison still has eight blocks. This distinction matters when a narrow interval over queries conflicts with a wide interval over blocks.

## 7. Interpretation and failure modes

A breadth advantage means equal nominal cardinality did not provide equal useful diversity. A phase-depth advantage that also exceeds jitter means separated gait-cycle content mattered more than nearby start variation in this setting. A flat path means no allocation difference was resolved at the declared precision, not that all video hierarchies are interchangeable.

The strongest representation-level statement additionally needs a residual GFC result, agreement of margin, top-1, and MRR, and agreement with a locked factor-transport geometry diagnostic. If geometry conflicts, report the retrieval result as supervised donor-based recombination without claiming a particular representation mechanism.

Common failure modes are calling `U × k` equal information, calling sequence IDs people, altering jitter offsets after outcomes, selecting a metric after seeing results, treating the middle point as a fitted scaling law, treating four jitter blocks as eight primary blocks, or claiming that silhouettes validate RGB and clinical sensing. Every one of these changes the scientific question.

## Efficiency notes

Phase estimation and near-duplicate clustering run once before training and should be stored as a versioned catalog. Feature export should aggregate participant summaries early so later inference operates on small arrays. Use stable hashes rather than mutable worker random states. Validate all 28 registry rows and evaluator synthetic cases before expensive training. Save checkpoints at a planned cadence and include the phase-catalog digest in every resume and export check. These practices improve speed and protect the experiment from silently mixing protocols.

## Exercises

1. For a fixed `U × k`, calculate recurrence at both planned exposure tiers. Why does equal recurrence not imply equal information?
2. Construct a jitter set whose mean base phase matches semantic `k=4`. What other streams must remain paired for it to be a valid control?
3. Simulate eight blocks with a flat raw GFC path but a non-flat residual path. What would that say about independent completion?
4. Change phase-depth versus jitter from four to eight blocks. Which claim becomes more precise, and why must this choice be made before outcomes?
5. List one result that would support an ambient-sensing motivation and one result that would still be insufficient for a clinical balance conclusion.

## Summary

The revised study does not ask whether clips or sequences are universally better. It asks where a fixed nominal catalog lives in one hierarchical video domain. A valid answer needs semantic phase construction, a matched jitter control, fixed exposure, paired trained-model blocks, continuous matched GFC and independent-completion scores, and honest limits on the claim. This combination turns a sampling ablation into a controlled test of whether the hierarchy of video diversity matters for the representations that predictive learning produces.

- Notebook: [17. Iso-catalog phase allocation and paired inference](../implementations/17_hierarchical_support_and_factorial_inference.ipynb)
- Method: [Iso-catalog phase allocation](../../docs/hierarchical-diversity/method.md)
- Next: apply the same provenance discipline when reading or extending the research code.
