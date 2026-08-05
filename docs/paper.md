# What Does Data Scale Buy a Predictive Video Representation?

## Factor recombination, context reliance, and identity in silhouette gait

> **Draft status (July 31, 2026).** This is the paper skeleton for the revised ICLR 2027
> study. Sections describing the prospective experiment use future tense because the 20
> primary GaitLU runs and locked Health&Gait outcome analysis have not been completed.
> Historical numbers are labelled *preliminary* or *legacy* and cannot be promoted into
> headline results.

## Abstract

Scaling self-supervised video pretraining commonly improves recognition accuracy, but
it is unclear whether more unique data changes how known factors can be extracted and
recombined. We propose a fixed-exposure study of one JEPA-style silhouette-video encoder
trained on four nested GaitLU-1M pools and five replicated ladders. Frozen encoders are
evaluated on Health&Gait using Grounded Factorial Completion v2 (GFC-v2), a supervised
linear-alignment protocol that composes speed, clothing, and direction blocks from two
session-independent donors and retrieves a real target from the full eight-cell
factorial gallery. The full gallery yields an exact partial-factor spectrum of
$1/8$, $1/4$, $1/2$, and $1$, avoiding the candidate-deletion bias found in our
preliminary protocol. Matched acquisition-cue and independent-factor controls bound the
interpretation. We jointly measure normalized context reliance, effective rank, factor
probes, and cross-condition identity retrieval. A prospectively locked outcome cohort,
five run-level contrasts, and prespecified superiority, equivalence, and inconclusive
rules prevent a null or unstable curve from being converted into a positive scaling
story. This draft states the protocol and preliminary motivation; the abstract will be
rewritten with genuine outcomes only after the frozen analysis is executed.

## 1. Introduction

Video representations are often selected by pretraining loss, effective rank, linear
probes, or identity retrieval. These measures answer different questions. Low loss does
not show that visible context is used; high rank does not show that named factors are
organized usefully; a factor probe does not show that separately observed factors can
be combined to identify a real target.

Health&Gait provides a controlled $2\times2\times2$ design for each complete participant:
usual or fast instructed speed, no jacket or jacket, and right-to-left or left-to-right
walking. This makes a grounded question possible: can factor blocks taken from two
non-target recordings retrieve the participant's observed recording with the requested
combination?

Our initial GFC implementation appeared encouraging, but adversarial audit changed its
interpretation. Deleting both donors distorted the chance structure; a non-learned
acquisition path nearly reached the relevant partial-factor oracle; one donor rule could
reuse the target's physical walk; and development participants influenced checkpoint
and method choices. These are not cosmetic limitations. They determine what the metric
measures.

The revised study asks a narrower scaling question:

> At fixed architecture and fixed training exposure, how does unique GaitLU pretraining
> data affect the supervised linear recoverability and recombination of speed, clothing,
> and direction?

We do not claim that fixed-compute scaling, JEPA training, latent mixing, gait identity
evaluation, or supervised factor alignment is new. The proposed contribution is their
careful combination with a session-safe real-target protocol, a full-gallery exact
oracle analysis, matched construct-validity controls, and a jointly replicated scaling
audit.

### Contributions

1. **Session-safe GFC-v2.** Two complementary donors construct a query without target
   features or target-source reuse; all gallery cells remain candidates.
2. **Protocol-compiled oracle analysis.** An enumerator calculates exact top-1 for every
   subset of recovered factors under a stated donor, gallery, and tie policy.
3. **Controlled scaling audit.** Five nested four-rung ladders measure recombination,
   context reliance, effective rank, probes, and identity at identical exposure.
4. **Calibrated interpretation.** Acquisition-cue and classifier-product controls,
   run-level inference, equivalence rules, and explicit dual-use limits constrain the
   conclusions.

## 2. Related work and novelty boundary

Disentanglement metrics already evaluate factor recoverability and compactness. Latent
mixing and analogy methods already compose representation parts, and Tree
Reconstruction Error provides a graded compositionality measure. Gait models have
separated identity from covariates and transferred covariate features. GaitLU/GaitSSB
already established large-scale unlabelled gait pretraining; Cosma et al. studied data,
model, and compute scaling for skeleton-based self-supervised gait recognition; BigGait
and BiggerGait use large vision representations for gait; and recent video work reports
data-scaling curves and compositional-reasoning failures.

Accordingly, this paper will not claim to be the first gait scaling study, the first
factor-mixing method, or a new JEPA objective. The most defensible methodological claim
is narrower: retaining every factorial candidate while enforcing source-session
independence gives a symmetric, exactly enumerable real-target test. Its use inside a
replicated multi-axis scale study is the empirical contribution.

Before submission, a closest-method matrix will compare real-target use, labelled
alignment, donor retention, exact subset-oracle enumeration, acquisition controls, and
session independence. If prior work contains the same mechanism, priority language will
be removed and the contribution narrowed.

## 3. Data and locked roles

### 3.1 GaitLU-1M

GaitLU-1M supplies encoder pretraining only. We validate decoding and silhouettes,
remove exact duplicates, preserve provided source groups, reserve a group-disjoint
10,000-sequence holdout, and report the actual eligible count $U_{\max}$. When source
metadata are unavailable, sequences become singletons after exact deduplication and the
absence of source-level independence is disclosed.

Five seeded group orders define nested pools near 2,500, 25,000, 250,000, and
$U_{\max}$. Actual counts and checksums are reported. The common holdout supplies
training-health and context-reliance measurements and never contributes encoder updates.

### 3.2 Health&Gait

Health&Gait supplies labelled frozen-feature evaluation only. Its hierarchy is
participant, speed/clothing source walk, direction clip, and frame. Opposite directions
from one back-and-forth walk share `source_video_id` and are not independent sessions.

![One participant's complete factorial grid](images/factorial-grid.svg)

The existing 80-person group, with 76 complete cases, is the development cohort for
factor adapters, normalizers, temperature calibration, and evaluator checks. The other
318 participants, with 308 complete cases, form the prospectively locked outcome cohort.
They are not an untouched external sample because their data informed earlier
Health&Gait experiments. New encoders, however, train only on GaitLU.

Outcome aggregates remain unopened until data roles, all runs, reference checkpoints,
evaluation code, thresholds, exclusions, and figure templates are frozen in a
timestamped commit.

## 4. Scaling experiment

### 4.1 Encoder and exposure

All runs use the same six-layer, 384-dimensional, single-stream JEPA-style ViT with 16
$112\times112$ silhouette frames. Architecture, mask policy, optimizer, augmentations,
batch size, learning-rate schedule, EMA schedule, and pooling are fixed. Horizontal
flipping is disabled because direction is evaluated.

The primary budget is 8,192,000 examples per run, or 128,000 updates at effective batch
64. A throughput gate may select 4,096,000 examples for every run before outcomes are
opened. The final-step checkpoint is primary.

### 4.2 Replication

Each of five replicate seeds defines one nested data ladder and one optimization seed,
giving 20 runs. Within replicate, larger rungs add sequences. Across replicates, the
design captures combined pool and optimization reproducibility. It does not estimate
separate variance components or support a universal power law.

## 5. Grounded Factorial Completion v2

### 5.1 Three supervised factor blocks

For each frozen recording representation, three separate ridge maps produce
two-dimensional score blocks for speed, clothing, and direction. Only development
participants fit standardization, coefficients, intercept, and normalization.
$\alpha=1$ is primary; $\alpha\in\lbrace0.1,10\rbrace$ are fixed sensitivities. The labelled
alignment is explicit, so the method does not claim unsupervised factor discovery.

The shortcut input contains log frame count, duration, signed and absolute endpoint
displacement, and five foreground-area summaries. It receives the same three-head ridge
capacity, labels, fitting participants, output dimensions, and normalization.

### 5.2 Complementary donors

For target $x=(s,c,d)$ and focal factor $a\in\lbrace s,c\rbrace$, define:

$$
u_a=x_a,\qquad u_j=1-x_j\quad(j\ne a),
$$

$$
v_a=1-x_a,\qquad v_j=x_j\quad(j\ne a).
$$

The query takes focal block $a$ from $u$ and the other two blocks from $v$. The target
contributes no features. Direction is not focal, and both donor `source_video_id` values
must differ from the target. Eight targets and two focal factors produce 16 queries per
complete participant.

![A session-safe full-gallery GFC-v2 query](images/gfc-query.svg)

### 5.3 Full gallery and scoring

All eight cells remain in the primary gallery, including both donors. The target is
retrieved by the equally weighted mean of three cosine block distances. Ties receive
fractional top-1 and average-rank MRR credit and are never broken randomly. Attraction
to each retained donor is reported.

The full-gallery partial-factor spectrum is exact:

| Factors recovered | Top-1 |
|---|---:|
| none | $1/8$ |
| any one | $1/4$ |
| any two | $1/2$ |
| all three | $1$ |

A secondary donor-excluded sensitivity reproduces the focal-dependent defect of
candidate deletion. It is shown for continuity, not substituted for the primary result.

### 5.4 Construct controls

Hard and soft factor-completion controls use the same three head outputs to retrieve the
cell implied by independent factor predictions. If they explain GFC-v2, the conclusion
is limited to joint linear factor recoverability. Evidence for additional compositional
geometry requires a reproducible gap beyond those controls.

## 6. Other fixed measurements

### 6.1 Context reliance

On the common GaitLU holdout, replace visible context with a different sequence from the
nearest decile under a fixed, non-learned geometry descriptor while retaining target
features, positions, mask, and predictor inputs. The primary ratio is:

$$
R_i^{\text{near}}=
\frac{L_i^{\text{near-substitute}}-L_i^{\text{true}}}
{\max\left(L_i^{\text{true}},10^{-8}\right)}.
$$

Both losses, the raw gap, distribution, and paired interval are reported. Same-source
segments, temporal shuffling, far substitutes, and blank context are secondary.

### 6.2 Probes, rank, and identity

Report balanced accuracy for each factor, pooled and token-level effective rank, and one
cross-condition identity protocol. Identity enrollment averages both directions at
usual speed without a jacket; remaining conditions are probes. Aggregate rank-1 and MRR
are reported with equal participant weighting.

GaitSSB and GaitBase/OpenGait are endpoint anchors when released checkpoints and
preprocessing are compatible. They are not points on the causal scaling curve. BigGait
and BiggerGait compatibility is optional and cannot block the primary study.

## 7. Statistical plan

The primary outcome is participant-level learned GFC-v2 top-1. For each ladder, compute
the full-minus-small contrast, then average the five replicate contrasts. Report a
conservative $t$ interval over replicate means, all five curves, and participant and
crossed bootstraps as sensitivities.

One of 16 queries is $\delta=6.25$ percentage points. A meaningful positive effect
requires a 95% interval above zero and an estimate at least $\delta$. A positive smaller
effect has an interval above zero but estimate below $\delta$. Equivalence requires the
90% interval inside $[-\delta,+\delta]$. All other cases are inconclusive. Failure to
reject zero is not called flat.

Factor probes, intermediate rungs, context, effective rank, and identity are secondary;
related families use Holm correction. Participant resampling cannot overcome only five
trained-model replicates.

## 8. Preliminary evidence and redesign trajectory

The historical Phase 0/1 experiments used Health&Gait encoder training and one seed.
Loss, effective rank, context gaps, speed probes, and identity selected different
checkpoints. Token features were broad (`381.58/384`) while pooled recording features
were narrow (`10.44/384`), and replacing context changed loss by only about `0.000156`.

The legacy GFC protocol fitted on 308 complete historical training participants and
evaluated 76 complete development participants. A00 learned top-1 was `69.79%`, the
shortcut was `65.46%`, and their difference was `+4.33` points with a participant
bootstrap interval of `[+0.27,+8.22]`. B01 did not separate from the shortcut, while B02
underperformed it.

![Legacy development GFC result; not a GFC-v2 outcome](../results/generated/legacy_gfc_comparison.png)

These numbers motivated the project but cannot serve as prospective results. The old
gallery deleted both donors, had a non-uniform oracle spectrum, permitted same-source
donors for some queries, used comparator-sensitive dimension reduction, and reused the
development split for model and method choices. The A00 effect was about one of 24
queries per participant, with estimated power around `0.52` at that resolution.

GFC-v2 responds by retaining donors, using 16 session-safe queries, matching shortcut
capacity, separating development and locked outcome roles, adding classification
controls, and moving encoder training entirely to GaitLU.

## 9. Prospective results section

This section remains outcome-neutral until the protocol is frozen and the outcome cohort
is opened once. The final paper will include:

1. throughput, actual rung sizes, exposure, exclusions, and optimization health;
2. all five four-rung learned GFC-v2 curves and the primary full-minus-small interval;
3. shortcut, hard-completion, and soft-completion controls;
4. per-factor probes, focal breakdowns, donor attraction, and adapter sensitivities;
5. normalized context reliance with raw components;
6. effective-rank and identity trajectories; and
7. superiority, equivalence, or inconclusive classification under the frozen rules.

No result will be described as causal disentanglement. If classifier controls explain
GFC-v2, the conclusion becomes joint linear factor recoverability. If replicates
disagree, the paper reports instability. If the protocol or full data rung fails, the
ICLR scaling claim is withdrawn rather than replaced by the legacy result.

## 10. Feasibility and reproducibility

Twenty primary runs process 163.84 million examples at the primary budget. Eight H100s
run eight independent single-device jobs concurrently. The conservative planning rate
is one million examples per hour across the machine; measured historical rates are
higher but full-pool streaming must be probed directly. Storage is budgeted at 250 GB
for packed silhouettes and temporary validation artifacts.

Compact results must record protocol version, source-group and role-map checksums, pool
and optimization seeds, actual rung size, exposure, checkpoint, evaluator commit,
gallery policy, query count, exclusions, and analysis-freeze commit. Tables and figures
must regenerate from those compact files without reading notebooks or prose.

## 11. Limitations

- One architecture and objective cannot establish a general scaling law.
- Four data rungs do not identify a universal functional form.
- Health&Gait provides one controlled dataset and no same-cell recording repeatability.
- The 318-person outcome cohort is prospectively locked but not historically untouched.
- Five model replicates may leave equivalence or superiority unresolved.
- Supervised factor heads prevent a claim of unsupervised disentanglement.
- Geometry matching does not isolate semantic context causally.
- Source-level independence in GaitLU may be unavailable when metadata are absent.
- Identity and clothing factors can be supported by body shape or acquisition cues.

## 12. Ethics

Health&Gait is human-participant data. Raw recordings, frames, tables, embeddings,
participant-level outputs, nearest-neighbor examples, and identity-capable checkpoints
remain private. Silhouettes and embeddings are not anonymous. No re-identification,
participant contact, clinical claim, or deployment claim is permitted.

Cross-condition identity retrieval is surveillance-relevant. It is included only to
test whether identity capability changes differently from factor recombination. The
paper reports aggregate capability and explicitly acknowledges that withholding
checkpoints reduces but does not eliminate dual-use risk.

## References

- Andreas, J. (2019). [Measuring Compositionality in Representation Learning](https://openreview.net/forum?id=HJz05o0qK7). ICLR.
- Assran, M., et al. (2023). [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html). CVPR.
- Bardes, A., et al. (2024). [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471).
- Cosma, A., Cătrună, A., and Rădoi, E. (2025). [On Model and Data Scaling for Skeleton-based Self-Supervised Gait Recognition](https://arxiv.org/abs/2504.07598).
- Eastwood, C., and Williams, C. K. I. (2018). [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ). ICLR.
- Fan, C., Hou, S., Huang, Y., and Yu, S. (2022). [Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark](https://arxiv.org/abs/2206.13964).
- Fan, C., et al. (2023). [OpenGait: Revisiting Gait Recognition Towards Better Practicality](https://openaccess.thecvf.com/content/CVPR2023/html/Fan_OpenGait_Revisiting_Gait_Recognition_Towards_Better_Practicality_CVPR_2023_paper.html). CVPR.
- Hu, Q., et al. (2018). [Disentangling Factors of Variation by Mixing Them](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Disentangling_Factors_of_CVPR_2018_paper.html). CVPR.
- Locatello, F., et al. (2019). [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://proceedings.mlr.press/v97/locatello19a.html). ICML.
- Ye, D., et al. (2024). [BigGait: Learning Gait Representation You Want by Large Vision Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_BigGait_Learning_Gait_Representation_You_Want_by_Large_Vision_Models_CVPR_2024_paper.pdf). CVPR.
- Ye, D., et al. (2025). [BiggerGait: Unlocking Gait Recognition with Layer-wise Representations from Large Vision Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6a5c23219f401f3efd322579002dbb80-Abstract-Conference.html). NeurIPS.
- Zafra-Palma, J., et al. (2025). [Health & Gait: A Dataset for Gait-Based Analysis](https://www.nature.com/articles/s41597-024-04327-4). Scientific Data.
