# Methods: Unique-Sequence Diversity at Fixed Training Exposure

This document specifies the original unique-sequence scaling study. It is written as
the methods section for the ICLR paper draft. The alternative hierarchical-diversity
study has a separate [proposal](../hierarchical-diversity/proposal.md) and
[methods section](../hierarchical-diversity/method.md). Dataset access, preparation, and
privacy rules are described in [data.md](data.md). Operator commands for this 20-model
design are in the [GaitLU runbook](../gaitlu_training.md).

## 1. Study design

The study tests whether unique unlabelled pretraining support changes a frozen video
representation when model architecture and sampled-sequence exposure remain fixed.
GaitLU-1M is used only to train encoders. Health&Gait is used only to fit evaluation
heads and score frozen representations. Constructed cases test the evaluator but do not
support empirical claims.

The research question is:

> With encoder architecture, JEPA training recipe, and a prespecified sampled-sequence
> exposure held fixed, does replacing repeated draws from approximately 2,500 unique
> GaitLU sequences with draws from the full eligible pool materially improve
> participant-averaged Health&Gait GFC-v2 top-1, and is that improvement distinguishable
> from the corresponding change in independent-factor completion?

The primary endpoint is learned full-gallery GFC-v2 top-1. The primary estimand is its
full-pool minus 2,500-pool change, averaged across five replicated data ladders.

## 2. Data and fixed roles

### 2.1 GaitLU pretraining data

Every GaitLU sequence is validated for decoding, frame count, silhouette range, and
empty-frame rate. Packed content and shape are hashed to remove exact duplicates. A
complete distributor-provided `sequence_id,source_group` map is preserved when
available. Otherwise, exact-content-deduplicated sequences are treated as singleton
groups, and the absence of verified source-level grouping is disclosed.

Approximately 10,000 group-disjoint sequences are reserved as a common holdout. Whole
groups are selected to reach the closest attainable sequence count. The remaining
eligible groups are placed in a reproducible random order for each of five replicate
seeds. Nested prefixes create pools near 2,500, 25,000, 250,000, and all eligible
sequences. Actual counts are reported, and a source group is never split to reach a
nominal target.

### 2.2 Health&Gait evaluation data

Health&Gait provides a controlled $2\times2\times2$ design for instructed speed
(usual or fast), clothing (without or with jacket), and walking direction
(right-to-left or left-to-right). One physical back-and-forth walk creates two
direction clips that share a `source_video_id`.

The existing 80-person group, with 76 complete participants, is the development cohort.
It fits factor heads, standardization, normalization, and soft-completion temperature.
The other 318 participants, with 308 complete participants, form the prospectively
locked outcome cohort. Earlier project work used these participants, so the outcome
cohort is prospectively locked but not historically untouched.

## 3. Encoder training

All 20 primary runs use the same six-layer, 384-dimensional, single-stream JEPA-style
Vision Transformer. Each input contains 16 one-channel silhouette frames at
$112\times112$ pixels. Masking, optimizer, learning-rate schedule, exponential moving
average schedule, spatial augmentation, effective batch size, and final pooling remain
fixed. Horizontal flipping is disabled because direction is an evaluated factor.

The primary sampled-sequence exposure is

$$
C=8{,}192{,}000
$$

examples per run, implemented as 128,000 updates at effective batch size 64. The
configuration uses microbatches of 16 with four-step gradient accumulation. Each of 125
virtual epochs contains 65,536 examples and 1,024 optimizer updates. The fixed-exposure
sampler draws sequences with replacement. Repeated draws receive new temporal windows,
spatial transformations, and masks.

Before primary training, an outcome-blind throughput gate is applied under the intended
storage and concurrency conditions:

- at least 60 examples per second per GPU selects $C=8{,}192{,}000$;
- 30 to 59 examples per second per GPU selects $C=4{,}096{,}000$ for every run;
- below 30 examples per second per GPU, or insufficient storage, cancels the scaling
  study.

Within each replicate, the same pool-ordering seed and optimization seed are used at
every data level.
The final-step checkpoint is primary. Training loss, throughput, and collapse checks may
identify systems failures, but no downstream result may select an epoch or replacement
seed.

Finalized manifests omit content hashes. The private registry stores a role-sensitive
digest over each ordered training and holdout manifest pair. Checkpoints record this
digest, loader settings, pool seed, optimization seed, unique-sequence count, exposure,
and the `splitmix64-v1` seed scheme. Resume and evaluation stop on a provenance mismatch.

## 4. Frozen recording representations

Each Health&Gait direction recording supplies three distinct, deterministic 16-frame
windows. The frozen encoder maps each window to one pooled vector. The three vectors are
averaged in float64 to produce one recording representation. Windows are repeated views
of one recording, not independent observations. A complete participant contributes
eight recording representations.

For every encoder, three separate ridge heads map recording vectors to two-dimensional
score blocks for speed, clothing, and direction. Development rows alone fit input
standardization, coefficients, and intercepts. The primary ridge penalty is
$\alpha=1$; $\alpha=0.1$ and $10$ are fixed sensitivities. The intercept is not
penalized. Development standardization uses population-statistic normalization. Each
factor block is normalized after mapping so that the three blocks enter the retrieval
distance on comparable scales.

This supervised alignment names known factors. It does not test whether the encoder
discovers those factors without labels.

## 5. Grounded Factorial Completion v2

### 5.1 Query construction

Let a target condition be $x=(s,c,d)\in\{0,1\}^3$. The focal factor $a$ is speed or
clothing. Direction is not focal because the opposite-direction clip at fixed speed and
clothing shares the target's physical source walk.

Two complementary donors are defined by

$$
u_a=x_a,\qquad u_j=1-x_j\quad(j\ne a),
$$

$$
v_a=1-x_a,\qquad v_j=x_j\quad(j\ne a).
$$

The query copies factor block $a$ from $u$ and the other two blocks from $v$. The target
contributes no feature values. Both donors must have `source_video_id` different from
the target. Eight targets and two focal factors produce 16 queries per complete
participant.

### 5.2 Retrieval and scoring

Every query is compared with all eight recordings from the same participant, including
both donors. For query $q$ and gallery item $g$, the retrieval distance is

$$
d(q,g)=\frac{1}{3}\sum_{k\in\{s,c,d\}}\left[1-\cos(q_k,g_k)\right].
$$

All arithmetic uses float64. Distances within the prespecified absolute tolerance are
tied. If $a$ items are strictly closer and $t$ items share the target's next rank
positions, the target receives average rank

$$
r=a+\frac{t+1}{2}.
$$

Reciprocal-rank credit is $1/r$. A first-place tie receives top-1 credit $1/t$;
otherwise top-1 credit is zero. Gallery order and random tie breaking are never used.

The exact full-gallery oracle spectrum is:

| Recovered binary factors | Tied cells | Top-1 |
|---:|---:|---:|
| 0 | 8 | 12.5% |
| 1 | 4 | 25% |
| 2 | 2 | 50% |
| 3 | 1 | 100% |

The primary participant score is mean fractional top-1 across 16 queries. MRR, focal
factor, donor attraction, and donor-excluded results are secondary.

## 6. Controls

### 6.1 Acquisition cues

The nine non-learned acquisition cues defined in
[data.md](data.md#35-extract-a-matched-acquisition-cue-baseline) receive the same three
ridge heads, development participants, labels, output widths, normalization, query
construction, and tie rules as learned representations. Cue top-1 is an absolute
shortcut diagnostic. Since cue inputs do not vary across GaitLU models, subtracting the
same cue score at both data levels does not create a distinct scaling estimand.

### 6.2 Independent-factor completion

Hard completion takes the highest-scoring label from each copied factor block and
retrieves the corresponding gallery cell. Soft completion fits one positive temperature
on development participants and scores each cell by the product of its three marginal
factor probabilities.

With a complete Cartesian gallery, hard and soft completion select the same top-1
factor tuple. Their equality must be verified rather than treated as two independent
pieces of evidence. Soft probabilities additionally provide negative log-likelihood
and calibration diagnostics.

Gap equivalence supports an explanation by independent completion at the frozen
resolution. A materially nonzero gap is necessary but not sufficient evidence for an
additional donor-composition effect. An imprecise gap leaves that interpretation
unresolved.

## 7. Secondary measurements

Factor-head balanced accuracy, effective rank, context reliance, and cross-condition
identity retrieval are secondary. Related families use Holm correction.

Context reliance is measured on the common 10,000-sequence GaitLU holdout. The target,
target positions, mask, and predictor inputs remain fixed while visible context is
replaced by a different sequence from the nearest decile under a fixed, non-learned
geometry descriptor. For sequence $i$,

$$
R_i^{\mathrm{near}}=
\frac{L_i^{\mathrm{near\ substitute}}-L_i^{\mathrm{true}}}
{\max(L_i^{\mathrm{true}},10^{-8})}.
$$

Both losses, their raw paired difference, the ratio distribution, and a sequence-level
paired bootstrap are reported. Temporal permutation, far substitutes, blank context,
and same-source alternatives when verified metadata permit them are sensitivities.

Identity enrollment averages the usual-speed, no-jacket recordings from both
directions. Other speed and clothing conditions are probes. Participant-weighted
rank-1 and MRR are reported only in aggregate.

## 8. Statistical analysis

Let $Y_{r,u}$ be participant-averaged GFC-v2 top-1 for replicate $r$ and data level
$u$. The primary paired contrast is

$$
d_r=Y_{r,\mathrm{full}}-Y_{r,\mathrm{2.5k}},
$$

and the primary estimate is $\hat\theta=\frac{1}{5}\sum_{r=1}^{5}d_r$. A Student's t
interval over the five $d_r$ values is primary. Every four-level trajectory is shown.
Participant and crossed bootstraps are sensitivity analyses and do not increase the
number of trained-model replicates.

The key construct-validity contrast is

$$
q_r=
\left(Y_{r,\mathrm{full}}-C_{r,\mathrm{full}}\right)
-\left(Y_{r,\mathrm{2.5k}}-C_{r,\mathrm{2.5k}}\right),
$$

where $C$ is independent-factor completion top-1. This contrast is a required
pre-freeze analysis addition. Its model-level interval and renderer must be implemented
and tested before it can gate interpretation.

The materiality margin is $\delta=6.25$ percentage points, equal to one of 16 queries
per participant. The implemented primary decision rule is:

- a 95% interval above zero with $\hat\theta\ge\delta$ is meaningfully positive;
- a 95% interval above zero with $0<\hat\theta<\delta$ is positive but small;
- a 90% interval contained in $[-\delta,+\delta]$ supports equivalence at this
  resolution;
- every other result is inconclusive.

The secondary completion-gap contrast uses a three-way interpretation. A 95% interval
excluding zero with $|\bar q|\ge\delta$ is a materially nonzero gap. A 90% interval
inside $[-\delta,+\delta]$ supports gap equivalence. Otherwise, the completion
interpretation is unresolved. Gap equivalence supports an explanation by independent
factor completion at this resolution. A materially nonzero gap is necessary but not
sufficient evidence for donor-based composition beyond independent prediction because
bounded scores and calibration can also change the difference.

Four data levels do not establish a functional scaling law. Participants, queries,
windows, and parity breakdowns are repeated measurements, not additional model
replicates.

## 9. Protocol freeze and quality checks

Before opening outcome aggregates, a timestamped commit freezes participant roles,
exclusions, manifests, exposure, all checkpoints, evaluator code, controls,
normalization, statistical decisions, and figure templates. An output counts as a
primary result only when its metadata record the GFC-v2 full-gallery protocol, 16
queries, three matched ridge heads, and the frozen cohort-role version.

Constructed tests must verify eight unique cells per complete participant, source-
disjoint donors, absence of target features from queries, retained donors, deterministic
ties, exact oracle values, matched learned and cue paths, development-only fitting, and
equal exposure across eligible models.

## 10. Scope and ethics

The strongest conclusion is limited to this JEPA recipe, encoder, fixed-exposure
GaitLU intervention, supervised factor alignment, and Health&Gait GFC-v2 protocol. The
study does not establish intrinsic compositionality, unsupervised disentanglement,
clinical utility, identity safety, or transfer to other objectives, populations,
modalities, or RGB video.

Health&Gait recordings, participant tables, embeddings, participant-level outputs,
nearest-neighbor examples, and identity-capable checkpoints remain private. Identity
results are reported only as aggregate capability measurements. No re-identification,
participant contact, deployment, or clinical claim is permitted.

## References

- Andreas, J. (2019). [Measuring Compositionality in Representation Learning](https://openreview.net/forum?id=HJz05o0qK7). ICLR.
- Fan, C., Hou, S., Huang, Y., and Yu, S. (2022). [Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark](https://arxiv.org/abs/2206.13964).
- Hammoud, H. A. K., et al. (2024). [On Pretraining Data Diversity for Self-Supervised Learning](https://openreview.net/forum?id=SLokff4aKI). ECCV.
- Hu, Q., et al. (2018). [Disentangling Factors of Variation by Mixing Them](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Disentangling_Factors_of_CVPR_2018_paper.html). CVPR.
- Zafra-Palma, J., et al. (2025). [Health & Gait: A Dataset for Gait-Based Analysis](https://www.nature.com/articles/s41597-024-04327-4). Scientific Data.
