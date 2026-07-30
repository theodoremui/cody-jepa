# Method

## Research question

CoDy-JEPA asks whether a video representation contains factor structure that can be recombined to recover an observed target. Grounded Factorial Completion (GFC) tests this directly instead of treating prediction loss, representation breadth, or a linear probe as a complete measure of representation quality.

Health&Gait supplies three binary instructed factors for each participant:

- speed: usual (`UGS`) or fast (`FGS`);
- clothing: without a jacket (`WoJ`) or with a jacket (`WJ`);
- direction: right to left (`R2L`) or left to right (`L2R`).

A participant with one valid recording in every combination has eight cells. The factor meanings are experimental conditions; `UGS` and `FGS` are not substitutes for instrumented cadence or velocity.

## Recording features

Models produce window-level features. The Health&Gait configuration expects three deterministic windows per recording. Their elementwise arithmetic mean is computed in `float64` before adapters, normalization, query construction, or scoring. Windows are correlated views of one direction clip and never become independent gallery items.

The baseline disables horizontal flipping during self-supervised training because left-to-right versus right-to-left is an evaluated factor. Spatial crops remain clip-consistent. Any future flip augmentation must be treated as an explicit ablation rather than an unrecorded default.

Every window-level input row must identify the participant, recording, source video, direction clip, split, speed, clothing, and direction. Each recording must contribute three rows. Every row must also contain finite `feature_0` through `feature_D` values and the finite shortcut columns named in `configs/eval/gfc_healthgait.json`.

## Query and gallery construction

Every one of a participant's eight cells becomes a target. For target

$$
x_{s,p,c,d},
$$

the condition donor is the same participant, clothing, and direction at the opposite instructed speed:

$$
x^{\mathrm{cond}}=x_{s,\bar p,c,d}.
$$

The gait donors are the three cells at the target speed with a different clothing-direction pair:

$$
x^{\mathrm{gait}}_j=x_{s,p,c',d'},
\qquad (c',d')\ne(c,d).
$$

The condition block from the first donor and gait block from the second form the mixed query. The target supplies no query features. Eight targets times three gait donors gives 24 queries per complete participant.

The gallery begins with all eight cells and removes both donor recordings. It therefore contains the target and five distractors. Removing both donors is essential: each donor supplied half of the query and would otherwise receive a built-in similarity advantage.

The primary analysis is complete-case. A missing cell excludes that participant because the two donors and six gallery items exhaust all eight cells. Duplicate cell assignments are errors. The evaluator never shrinks the gallery or imputes a recording.

## Learned and shortcut paths

For a single-block encoder, two closed-form ridge adapters map the complete recording feature vector to:

- a four-dimensional condition block with targets `WoJ`, `WJ`, `R2L`, and `L2R`;
- a two-dimensional gait block with targets `UGS` and `FGS`.

Inputs are population-standardized using development-training participants only. Both fits use squared-error one-hot targets, an unpenalized intercept, coefficient-only L2 regularization, and `alpha = 1`. Model weights remain fixed during this evaluation.

The shortcut path uses declared acquisition cues as its condition and gait blocks directly; it does not pass them through the learned-feature ridge adapters. It otherwise uses the same participants, train-only normalization, queries, galleries, distance, and scoring rules:

- condition side: signed and absolute endpoint horizontal-centroid displacement plus foreground-area summaries;
- gait side: log frame count and duration.

The primary contrast is the participant-level learned top-1 score minus the participant-level shortcut top-1 score on identical queries.

## Normalization and sensitivity analyses

All normalization is fit on development-training participants and applied unchanged to evaluation participants. Condition and gait blocks are handled separately.

The primary analysis, `raw_retain_all`, population-standardizes every learned adapter output and direct shortcut coordinate, then L2-normalizes each block. It does not discard a coordinate.

Two sensitivity analyses test whether the result depends on that choice:

- `raw_effective_rank` retains the highest-variance raw coordinates, using the rounded entropy effective rank as the retained width;
- `pca_effective_rank` uses a training-only full SVD and retains the same effective-rank width in principal-component space.

For eigenvalues $\lambda_i$, define $p_i=\lambda_i/\sum_j\lambda_j$. Entropy effective rank is

$$
r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right).
$$

The retained width is at least one and uses nearest-integer half-up rounding. Population standard deviations and block norms have a `1e-12` floor. Nonfinite inputs are errors. A zero block remains zero and has cosine similarity zero with any block.

Sensitivity analyses are reported alongside the primary analysis; they are not used to select the more favorable result.

## Distance, ranking, and ties

Condition and gait blocks receive equal weight:

$$
d(q,g)=\tfrac12\left[1-\cos(q_{\mathrm{cond}},g_{\mathrm{cond}})\right]
+\tfrac12\left[1-\cos(q_{\mathrm{gait}},g_{\mathrm{gait}})\right].
$$

Aggregation, normalization, dot products, and score accumulation use `float64`. Distances differ only when their absolute difference exceeds `1e-12`; relative tolerance is zero.

If $a$ gallery items are strictly closer than the target and the target belongs to a tie of size $t$, its average occupied rank is

$$
r=a+\frac{t+1}{2}.
$$

The reciprocal-rank score is $1/r$. Top-1 credit is $1/t$ when $a=0$, otherwise zero. Donor attraction separately compares the target with the excluded gait donor and awards 1, 0, or one-half for target closer, donor closer, or a tie.

The six-item gallery has analytical top-1 chance $1/6$. Uniform tie-free ranks have expected MRR $49/120$. Constant blocks form a six-way tie and produce top-1 $1/6$, MRR $2/7$, and donor attraction $1/2$. Constant ties are not randomly broken.

## Aggregation and inference

Metrics are averaged over 24 queries within each participant and then across participants with equal participant weight. Windows and queries are not independent inference units.

Learned and shortcut results must contain identical participant, target-cell, gait-donor-cell, split, and gallery definitions. The participant-level differences are resampled with all paired queries kept together. The configured analysis uses 10,000 participant bootstrap draws, seed `20260728`, and a 95% percentile interval.

The prospective power calculation uses a meaningful effect of one additional successful query per participant, $1/24$, a two-sided `alpha = 0.05`, and a minimum target power of 0.80. It is a planning calculation, not a model result or a substitute for the participant bootstrap.

## Scientific safeguards

The focused tests cover the behavior that can change the scientific interpretation:

- eight cells generate 24 queries and six-item galleries;
- both donors are absent from every gallery;
- missing and duplicate cells follow the stated policy;
- exact-factor features retrieve every target;
- constant blocks reproduce the declared tied null;
- fractional top-1, MRR, and donor attraction handle ties correctly;
- windows are aggregated before fitted transformations;
- adapters and normalizers receive development-training rows only;
- adding held-out rows cannot alter a training fit;
- learned and shortcut paths use identical query keys;
- query scores are averaged within participant before inference;
- a fixed bootstrap seed gives repeatable numerical output.

## Interpretation

A positive learned-over-shortcut interval would show completion information beyond the declared shortcut set. It would not establish causal factors, eliminate every possible shortcut, prove identity invariance, or demonstrate clinical value. The present repository contains no empirical GFC estimate yet.

## References

- Eastwood, C., and Williams, C. K. I. (2018). [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ). ICLR.
- Gondal, M. W., et al. (2019). [On the Transfer of Inductive Bias from Simulation to the Real World: A New Disentanglement Dataset](https://proceedings.neurips.cc/paper/2019/hash/d97d404b6119214e4a7018391195240a-Abstract.html). NeurIPS.
- Gretton, A., Bousquet, O., Smola, A., and Schölkopf, B. (2005). [Measuring Statistical Dependence with Hilbert-Schmidt Norms](https://doi.org/10.1007/11564089_7). ALT.
- Hu, Q., Szabó, A., Portenier, T., Favaro, P., and Zwicker, M. (2018). [Disentangling Factors of Variation by Mixing Them](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Disentangling_Factors_of_CVPR_2018_paper.html). CVPR.
- Zafra-Palma, J., et al. (2025). [Health & Gait: A Dataset for Gait-Based Analysis](https://www.nature.com/articles/s41597-024-04327-4). Scientific Data.
