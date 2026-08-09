# 17. Hierarchical support interventions and blocked factorial inference

![The hierarchical support experiment from sequence pools to inference](../images/17_hierarchical_support_and_factorial_inference.svg)

## Why this capstone matters

Earlier lessons develop the pieces of a trustworthy video experiment. They explain temporal
windows, latent prediction, factor-aligned representations, retrieval, confidence intervals,
and reproducible software. This lesson connects those pieces into one complete scientific
argument.

The motivating question sounds simple. If two video models process the same number of clips,
should one model see more sequences, or should it revisit fewer sequences at more temporal
locations? The answer cannot come from counting files alone. It requires a controlled
intervention on two kinds of training support, repeated model training, a downstream outcome,
and an analysis that preserves the way the models were paired.

This is a capstone, not a replacement for the earlier foundations. Its purpose is to show
how a change in the experimental question changes the estimand, the probability model, the
array shapes, and the decision rules. The encoder and training objective can remain fixed
while the scientific mathematics changes substantially.

## Prerequisites

Review [08. Group-aware sampling and shortcut learning](08_group_aware_sampling.md) for
temporal anchors and random streams, [11. Factorial state spaces](11_factorial_state_spaces.md)
through [13. Context interventions and identity geometry](13_context_interventions.md) for
factor-composition retrieval, and [14. Paired inference](14_paired_inference.md) through
[16. Reproducible scientific evaluators](16_reproducible_scientific_evaluators.md) for
uncertainty and protocol contracts.

## Learning goals

By the end of this lesson, you will be able to:

1. distinguish sequence support, temporal support, exposure, and replication;
2. compute expected realized support under frozen and resampled anchor policies;
3. represent eight paired four-cell model blocks without breaking their pairing;
4. calculate an interaction, four simple effects, and a direct allocation contrast;
5. use a Student $t$ interval over model blocks rather than over participants or queries;
6. explain why an additive interaction can change sign after a logit transformation;
7. separate superiority, materiality, equivalence, and intersection decision gates;
8. calculate a completion-gap interaction and state its limited interpretation;
9. resample blocks and participants while preserving every four-cell comparison; and
10. validate the registry and provenance needed to make the analysis auditable.

## 1. Begin with the intervention, not the metric

The experiment crosses two factors. Sequence support has a low level $L$ and a high level
$H$. Temporal policy has a frozen level $F$ and a resampled level $R$. The four cells are

| | Frozen anchor $F$ | Resampled anchors $R$ |
|---|---:|---:|
| Low sequence support $L$ | low breadth, low temporal support | low breadth, high temporal support |
| High sequence support $H$ | high breadth, low temporal support | high breadth, high temporal support |

This is a factorial experimental design. It is different from a factorial outcome state
space, where factors such as speed, clothing, and direction enumerate the possible gallery
states. Both use Cartesian products, but they serve different roles. The experimental
factors define training interventions. The outcome factors define the retrieval task.

Every cell receives the same number $C$ of sampled clips. The architecture, objective,
optimizer, schedule, spatial transformations, masks, and checkpoint rule are also fixed.
Therefore a performance difference is not explained by one cell receiving more optimization
steps. It is associated with which sequence and temporal locations were available during
those steps.

Fixed exposure does not imply equal information. That distinction is the purpose of the
experiment. Exposure says how many draws the optimizer receives. Support says how many
distinct source units the draw process can reach.

## 2. Temporal anchors define available support

Consider sequence $i$ with $n_i$ frames and clip length $T=16$. The number of valid
contiguous starts is

$$
W_i=n_i-T+1.
$$

The experiment does not use every valid start. It uses starts separated by eight frames:

$$
\mathcal A_i=\{0,8,16,\ldots,8\lfloor(W_i-1)/8\rfloor\},
\qquad K_i=|\mathcal A_i|.
$$

Adjacent 16-frame windows overlap by 50 percent. They can reveal different temporal phases,
but they are not independent videos. A conservative audit also counts starts separated by
16 frames. The two counts answer different questions. $K_i$ describes treatment support.
The non-overlapping count describes how much of that support remains under a stricter notion
of temporal separation.

![Temporal starts and transform scopes](../images/08_windows_and_transforms.svg)

The frozen policy selects one member of $\mathcal A_i$ uniformly for each sequence and
replicate block, then reuses it for every draw in that run. The resampled policy selects a
member of the same set on every draw. Across repeated randomizations, both policies give
each anchor the same marginal probability. Within one frozen run, however, the conditional
distribution is concentrated on one anchor. That concentration is the intended reduction
in temporal support.

## 3. Expected support is an occupancy problem

Suppose a pool contains $U$ sequences and training samples sequences uniformly with
replacement for $C$ draws. Under the frozen policy, there are only $U$ reachable
sequence-anchor pairs. The expected number observed at least once is

$$
E_F(U)=U\left[1-\left(1-\frac{1}{U}\right)^C\right].
$$

This is an occupancy calculation. A particular sequence has probability
$(1-1/U)^C$ of never being drawn, so its probability of being seen is one minus that value.
Summing the identical probabilities over $U$ sequences gives the expectation.

Under resampling, sequence $i$ contributes $K_i$ reachable pairs. One particular pair from
that sequence has draw probability $1/(UK_i)$. Expected realized support becomes

$$
E_R(U)=\sum_{i=1}^{U}K_i
\left[1-\left(1-\frac{1}{UK_i}\right)^C\right].
$$

The ratio $E_R/E_F$ measures expected treatment separation at the sampled exposure. It does
not measure semantic independence. Large $K_i$ values also do not guarantee a large ratio
when $C$ is too small to visit many pairs. This is why the audit uses the actual pool,
anchor counts, and proposed exposure rather than reporting only theoretical capacity.

A prospective gate can require a minimum median $K_i$ and a minimum $E_R/E_F$ ratio in every
planned pool. The gate must be fixed before outcome results are available. Otherwise the
definition of an adequate intervention could move toward whichever experiment looks best.

## 4. Blocking turns four models into one comparison unit

One replicate block contains all four cells. Within the block, cells share the intended
initialization and optimization seed. The two temporal policies at a given sequence-support
level also share sequence draws, spatial transformations, and mask draws. Only the temporal
anchor policy changes within that pair.

With eight blocks, the outcome array after participant aggregation has shape `(8, 2, 2)`.
The axes are block, sequence support, and temporal policy. Before participant aggregation,
an array can have shape `(8, 2, 2, P)`, where $P$ is the number of complete outcome
participants.

This shape encodes the analysis. The 32 trained models are not 32 freely exchangeable
replicates. Four models belong to one matched block. Resampling or permuting individual
models across blocks would destroy the covariance created by shared seeds and paired
nuisance streams.

Participants are another repeated axis. More participants estimate each cell outcome more
precisely, but the primary interaction is repeated eight times, once per trained-model block.
Queries and windows are nested measurements. They do not turn the primary sample size into
thousands.

## 5. The interaction is a difference of differences

Let $Y_{r,u,w}$ be participant-averaged GFC-v2 top-1 for block $r$, sequence support $u$,
and temporal policy $w$. The primary interaction is

$$
I_r=(Y_{r,H,R}-Y_{r,L,R})-(Y_{r,H,F}-Y_{r,L,F}).
$$

The same expression can be rearranged as

$$
I_r=(Y_{r,H,R}-Y_{r,H,F})-(Y_{r,L,R}-Y_{r,L,F}).
$$

The first form compares the sequence-support effect under resampled and frozen windows.
The second compares the temporal-policy effect at high and low sequence support. Algebra
shows that they are the same interaction.

Under this sign convention, $I_r<0$ means resampling helps more when sequence support is
low. That pattern is consistent with substitution, but an interaction alone is not enough
to establish replacement.

The four simple effects are

$$
T_{L,r}=Y_{r,L,R}-Y_{r,L,F},\qquad
T_{H,r}=Y_{r,H,R}-Y_{r,H,F},
$$

$$
S_{F,r}=Y_{r,H,F}-Y_{r,L,F},\qquad
S_{R,r}=Y_{r,H,R}-Y_{r,L,R}.
$$

The direct allocation contrast is

$$
A_r=Y_{r,L,R}-Y_{r,H,F}.
$$

$A_r$ compares the low-sequence resampled allocation with the high-sequence frozen
allocation. It is not a comparison of equal information. It asks whether performance at
those two concrete allocations is practically interchangeable.

## 6. Primary uncertainty comes from eight block interactions

The primary estimate is the arithmetic mean

$$
\widehat I=\frac{1}{8}\sum_{r=1}^{8}I_r.
$$

Let $s_I$ be the sample standard deviation of the eight block interactions. A two-sided
Student $t$ interval is

$$
\widehat I\ \mathbin{\pm}\ t_{1-\alpha/2,7}\frac{s_I}{\sqrt{8}}.
$$

This is a small-sample procedure. Exact $t$ coverage requires independent normal block
interactions with nonzero finite variance. In practice, the eight blocks reuse an
overlapping finite corpus, so the interval is best understood as variation over the
declared pool ordering, anchor, and optimization procedure, conditional on that corpus.
Show all eight $I_r$ values. A prospective simulation should check whether eight blocks can
resolve the prespecified margins under plausible variation.

Participant-only and crossed bootstraps are useful sensitivities. They answer different
questions from the primary $t$ interval. They do not create additional trained models.

## 7. Bounded outcomes make interaction scale important

Top-1 lies between zero and one. Near a ceiling, a five-point raw improvement may reflect a
larger change in odds than the same raw improvement near the center. Therefore an additive
interaction on the percentage scale can be affected by saturation.

Define the clipped-logit cell value

$$
Z_{r,u,w}=\log\left(
\frac{\mathrm{clip}(Y_{r,u,w},\epsilon,1-\epsilon)}
{1-\mathrm{clip}(Y_{r,u,w},\epsilon,1-\epsilon)}
\right),
\qquad
\epsilon=\frac{1}{2\cdot308\cdot16}.
$$

Repeat the interaction calculation using $Z$. For example, a raw increase from 0.40 to
0.50 is 0.10, while an increase from 0.90 to 0.95 is only 0.05. On the logit scale, the
second change is larger because it represents a larger odds ratio. A raw interaction can
therefore be negative while the logit interaction is positive.

This does not make one scale universally correct. The raw percentage-scale interaction is
the confirmatory estimand. The clipped-logit interaction is a prespecified robustness
analysis. A sign reversal shows that the substitution interpretation depends on scale, so
the evidence is not stable enough for the substitution label.

## 8. Superiority, materiality, and equivalence answer different questions

Statistical superiority asks whether an interval excludes zero in the favorable direction.
Materiality adds a practical threshold. In this protocol, an effect is called materially
positive when its 95 percent interval lies above zero and its point estimate is at least the
relevant margin. This is not the stronger rule that the full interval must lie beyond the
margin. The exact definition must be taught and implemented consistently.

Equivalence asks whether the entire effect is estimated precisely enough to lie within a
practical band. At level 0.05, the usual TOST interval rule requires the 90 percent interval
to lie inside $[-\delta,+\delta]$.

![Superiority and equivalence occupy different decision regions](../images/14_decision_regions.svg)

The direct allocation margin should be justified for the replacement claim itself. It
should not be inherited silently from an interaction margin just because the values happen
to match.

A substitution-compatible label is an intersection decision. It requires beneficial
resampling at low sequence support, beneficial sequence breadth under frozen anchors, no
material harm in the other two simple effects, a materially negative interaction, and a
stable interaction sign under the ceiling sensitivity. Full replacement additionally
requires equivalence of $A$. Partial replacement instead means the low-sequence resampled
cell remains materially worse than the high-sequence frozen cell.

Every component must be shown. Passing one interaction test does not cause the other
requirements to become true.

## 9. Independent completion is an interpretation control

GFC-v2 builds a query from factor blocks supplied by complementary recordings. Independent
factor completion instead predicts speed, clothing, and direction separately and combines
the marginal decisions. Let $C_{r,u,w}$ be its top-1 outcome and define

$$
G_{r,u,w}=Y_{r,u,w}-C_{r,u,w}.
$$

The completion-gap interaction is

$$
J_r=(G_{r,H,R}-G_{r,L,R})-(G_{r,H,F}-G_{r,L,F}).
$$

Gap equivalence supports the explanation that the observed hierarchy effect is resolved by
independent factor prediction at the declared margin. The frozen gap margin is
$\delta_G=0.0625$. A resolved gap interaction requires a 95 percent interval that excludes
zero and $|\bar J|\ge\delta_G$. Gap equivalence requires the 90 percent interval to lie
entirely inside $[-\delta_G,+\delta_G]$. Every other result is unresolved. A resolved gap
is necessary but not sufficient evidence for donor-based composition beyond independent
prediction. Ceilings, probability calibration, and the subtraction of two bounded metrics
can also affect $J_r$.

This control narrows an interpretation. It does not prove intrinsic compositionality,
unsupervised factor discovery, or a particular neural mechanism.

## 10. Sensitivity resampling must preserve the four-cell block

For participant-only resampling, draw participants with replacement and use the same draw
in every block and cell. This preserves the repeated-cohort comparison. For crossed
resampling, also draw replicate blocks with replacement. Each selected block brings all
four cells with it.

The safe conceptual order is:

1. draw block indices of length eight;
2. draw participant indices of length $P$;
3. select all four cells for every drawn block and participant;
4. average selected participants within each block and cell;
5. calculate one interaction per selected block; and
6. average the eight selected interactions.

Do not draw the 32 models independently. Do not draw separate participants for each cell.
Either operation breaks pairing that the estimand relies on.

Bootstrap intervals describe the empirical resampling model. With only eight blocks, the
block distribution is coarse. Thousands of bootstrap iterations reduce Monte Carlo noise,
but they do not replace missing model runs.

## 11. The registry is part of the mathematics

A valid registry contains exactly one record for every combination of eight blocks, two
sequence-support levels, and two temporal policies. Each record should include the manifest
digest, actual sequence count, exposure, initialization and optimization seed, window-stream
version, temporal-stream version, spatial-stream version, mask-stream version, final
checkpoint identity, and protocol revision. Serialized rows use only the canonical values
`low`, `high`, `frozen_random`, and `resampled_anchor`. The recorded sequence count must
equal the count obtained from the verified manifest, rather than a separate expected literal.

Cross-record invariants are as important as individual field types. Low and high pools must
be nested according to the locked construction. The two temporal policies at a support level
must share the intended manifest and nuisance-stream definitions. Every block must contain
all four cells, and every low-pool sequence must retain the same frozen anchor in the high
pool. Evaluation should stop if a digest, seed version, stream version, exposure, anchor, or
checkpoint rule differs from the locked registry.

Public summaries should contain aggregate cell outcomes and diagnostics without participant
identifiers, embeddings, or nearest-neighbor examples. Reproducibility does not require
publishing private identity-capable material.

## 12. Interpreting common result patterns

Suppose resampling helps strongly at low sequence support, breadth helps strongly under
frozen anchors, the other two simple effects show no material harm, the interaction is
materially negative on the raw scale and remains negative on the logit scale, and $A$ is
equivalent. This complete pattern supports full performance replacement at the two tested
allocations. It does not imply that 2,500 resampled sequences will replace 250,000 frozen
sequences under a different encoder, objective, corpus, or exposure.

Suppose the raw interaction is negative but the logit interaction is positive. The result
is scale-dependent. Report both views and do not use the substitution label.

Suppose the interaction is negative, but resampling harms the high-support cell beyond the
margin. The interaction alone can look substitution-like because one policy fails in one
corner. The no-harm component prevents that pattern from being presented as successful
replacement.

Suppose GFC-v2 and independent completion have equivalent interactions. The hierarchy
effect may be explained by separate factor prediction at this resolution. The representation
can still be useful, but the experiment has not isolated an additional composition effect.

## 13. Efficiency and numerical notes

Keep participant scores in an array with explicit named axis documentation. Aggregate
participants with float64 before calculating block contrasts. Compute the four cell means
once, then derive all simple effects, interactions, and direct contrasts from that table.

Use `log1p` or a stable logit helper after clipping. Store both raw and transformed cell
values so a sign mismatch can be traced. Generate bootstrap replicates in chunks if a full
index tensor would be large. A loop over bootstrap replicates is acceptable when it keeps
block and participant selections visibly separate.

Validate constructed examples with algebraic identities. The interaction computed from
sequence-support differences must equal the interaction computed from temporal-policy
differences. Registry checks should verify exact discrete fields and use digests for durable
artifacts. Numeric output comparisons should use a declared tolerance only where floating
point operations justify it.

## 14. Failure modes

1. **Treating 32 models as independent:** this discards the eight-block pairing.
2. **Counting participants as model replicates:** participants refine cell outcomes but do
   not replace trained-model blocks.
3. **Calling anchors independent videos:** overlap and shared sequence content remain.
4. **Using capacity instead of realized support:** large anchor sets may remain mostly unseen
   at limited exposure.
5. **Changing one random stream for all nuisances:** temporal policy can accidentally alter
   crops or masks.
6. **Calling a negative interaction substitution:** the simple-effect, no-harm, scale, and
   allocation requirements still need evaluation.
7. **Calling nonsignificance equivalence:** equivalence requires its own interval and margin.
8. **Resampling models one by one:** a sensitivity bootstrap must carry all four cells in a
   selected block.
9. **Overinterpreting the completion gap:** subtraction of bounded metrics does not identify
   an intrinsic mechanism.
10. **Allowing registry mismatches:** an analysis over mixed protocols has no single estimand.

## Exercises

### Exercise 1

A sequence has 65 frames and $T=16$. List the valid anchors at spacing eight and calculate
$K_i$.

**Brief solution:** $W_i=50$, so anchors are 0, 8, 16, 24, 32, 40, and 48. Therefore
$K_i=7$.

### Exercise 2

Cell means are $Y_{L,F}=0.55$, $Y_{L,R}=0.68$, $Y_{H,F}=0.70$, and
$Y_{H,R}=0.75$. Calculate $T_L$, $T_H$, $S_F$, $S_R$, $I$, and $A$.

**Brief solution:** the effects are 0.13, 0.05, 0.15, 0.07, $-0.08$, and $-0.02$.

### Exercise 3

Why does a participant bootstrap use the same participant draw in every cell?

**Brief solution:** each participant is measured under every trained model. A common draw
preserves the repeated-cohort covariance across cells.

### Exercise 4

Why can an interaction be negative on the raw scale and positive on the logit scale?

**Brief solution:** the logit is nonlinear and expands changes near zero and one. Additive
differences are not invariant under nonlinear transformations.

### Exercise 5

A 90 percent interval for $A$ is `[-0.04, 0.03]` and the direct margin is 0.0625. What does
this component establish?

**Brief solution:** it establishes equivalence for the direct allocation contrast because
the full interval lies inside the declared band. The remaining substitution components
must still pass before full replacement can be claimed.

## Recap

The hierarchical-diversity study changes the scientific mathematics even though the encoder
can remain the same. Sequence support and temporal support form a controlled four-cell
experiment at fixed exposure. Temporal policies create different occupancy distributions
over sequence-anchor pairs. Eight matched model blocks provide the primary interaction
replicates. Simple effects, a direct equivalence contrast, a bounded-scale sensitivity, and
an independent-completion gap narrow the interpretation. Every analysis and bootstrap must
preserve the four-cell block. A complete registry connects those equations to the actual
models and prevents mixed protocols from producing an apparently precise but undefined
result.

## Continue

- Previous: [16. Reproducible scientific evaluators and numerical contracts](16_reproducible_scientific_evaluators.md)
- Notebook: [17. Hierarchical support interventions and blocked factorial inference](../implementations/17_hierarchical_support_and_factorial_inference.ipynb)
- Curriculum: [Tutorial README](../README.md)
