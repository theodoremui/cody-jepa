# Context-Sufficient World Models for Human Movement

**Integrated research plan for ICLR 2027, ambient intelligence, and biomechanics balance assessment**

**Prepared:** 2026-08-13

**ICLR 2027 deadlines:** abstract 2026-09-11, paper 2026-09-16, Anywhere on Earth

**Status:** decision document and execution guide, not a record of completed experiments

## Executive summary

The strongest research program is not simply "a JEPA for gait." Future motion forecasting, gait foundation models, clinical gait embeddings, and biomechanical generative models already exist. The sharper and more general question is:

> What is the minimum observed motion state needed to predict a meaningful future state, and how can we verify that a model actually uses that state rather than identity, camera, clothing, or target-location shortcuts?

This question connects three projects:

1. **ICLR 2027:** study a failure mode in joint-embedding predictive architectures. A representation can remain non-constant while its predictor barely uses the observed context. Measure this directly, then test whether informative targets and matched wrong-context training make context necessary.
2. **Stanford HAI ambient intelligence:** learn the least intrusive sensing representation that can detect meaningful within-person mobility change across homes, cameras, and clothing, while keeping raw video and personal baselines local.
3. **Scott Delp's biomechanics program:** learn a conditional response operator that predicts recovery after a specified perturbation from the smallest sufficient pre-perturbation state and a subject-specific baseline.

The immediate ICLR paper should be narrow. It should use the high-quality Health&Gait data already available, add a controlled motion testbed, and compare the current cody-jepa model with at least one independently trained public JEPA. It should not depend on GaitLU-1M, new clinical data collection, a full biomechanics-video bridge, or fall outcomes.

The recommended paper is:

> **When Non-Collapsed JEPAs Ignore Context: Measuring and Enforcing Target-Conditioned Dependence**

Its central claim should be empirical, not universal: marginal anti-collapse does not guarantee that a predictor uses its conditioning input. The paper must show when this occurs, how to measure it, and whether fixing it improves transfer under real paired changes in movement and appearance.

## Navigation

- [1. One research program, three tracks](#1-one-research-program-three-tracks)
- [2. What the current repository already tells us](#2-what-the-current-repository-already-tells-us)
- [3. What a high-quality dataset means](#3-what-a-high-quality-dataset-means)
- [4. The five ideas worth carrying forward](#4-the-five-ideas-worth-carrying-forward)
- [5. ICLR 2027 proposal](#5-iclr-2027-proposal)
- [6. Ambient intelligence extension](#6-ambient-intelligence-extension)
- [7. Biomechanics and balance extension](#7-biomechanics-and-balance-extension)
- [8. Shared dataset strategy](#8-shared-dataset-strategy)
- [9. What to borrow from g-jepa and gavd4-vicreg](#9-what-to-borrow-from-g-jepa-and-gavd4-vicreg)
- [10. Adversarial review and revisions](#10-adversarial-review-and-revisions)
- [11. Decision gates and failure plans](#11-decision-gates-and-failure-plans)
- [12. Execution plan](#12-execution-plan)
- [13. Claims that are and are not supportable](#13-claims-that-are-and-are-not-supportable)
- [14. Glossary](#14-glossary)
- [15. References](#15-references)

## 1. One research program, three tracks

All three tracks study the same scientific object: a **minimum sufficient state**. This is the smallest set of observations that supports a useful prediction.

| Track | Immediate question | Meaningful target | Primary test |
| --- | --- | --- | --- |
| ICLR representation learning | Does the predictor truly depend on the observed context? | Future or hidden motion representation | Correct context must beat matched wrong context and a position-only oracle |
| Ambient intelligence | What is the least intrusive sensing that preserves reliable within-person change? | Mobility trend with uncertainty | Cross-home, cross-camera, and cross-clothing generalization |
| Biomechanics balance | What pre-perturbation state predicts the response to a specified perturbation? | COM, CoP, WBAM, foot placement, recovery distribution | Held-out subjects and held-out perturbations |

The bridge between them is simple:

```text
observed state + target specification + optional action
                         |
                         v
              predicted future state
                         |
                         v
       context-use test + uncertainty + transfer
```

The ICLR project develops the representation-learning principle. The ambient project turns it into a sensing and workflow question. The biomechanics project gives it a physically meaningful action and target.

## 2. What the current repository already tells us

### 2.1 What cody-jepa is doing

The current repository trains a small video JEPA on cropped Health&Gait silhouettes. The online encoder sees visible tokens. A predictor receives those tokens plus the locations of masked targets. An exponential-moving-average encoder supplies target embeddings.

The implementation is compact and useful as a research instrument, but the current task does not require temporal understanding:

- [`cody_jepa/masks.py:41`](cody_jepa/masks.py#L41) repeats each spatial mask through every temporal step. This is spatial tube completion, not future prediction.
- [`cody_jepa/models.py:183`](cody_jepa/models.py#L183) gives the predictor the target positions. A predictor can therefore exploit where a target is located even if the observed video contributes little.
- [`configs/healthgait.json:25`](configs/healthgait.json#L25) uses 16 grayscale frames at 112 by 112 pixels. This is a short temporal window and discards Health&Gait's color-coded DensePose and optical-flow semantics.
- [`cody_jepa/data.py:165`](cody_jepa/data.py#L165) converts every loaded frame to grayscale. A multimodal target experiment needs a loader that preserves each modality's meaning.
- [`cody_jepa/engine.py:161`](cody_jepa/engine.py#L161) constructs a diagnostic "blank" after normalization by setting the tensor to zero. With the default normalization, zero is raw mid-gray, not the black background of a silhouette. The diagnostic must construct the blank in raw pixel space and then normalize it.

### 2.2 The important empirical signal

The archived, pre-fix diagnostic is unusually informative:

- The archived checkpoints have pooled effective rank near 10 to 12 out of 384 dimensions, depending on the exact representation and export.
- Replacing the context with another context changes prediction loss by only about 0.000154.
- Only 9.62 percent of masked target tokens contain foreground.
- The wrong-context gap is larger on background targets than on foreground targets.
- The token representation can have near-full effective rank while the pooled clip representation uses only a small fraction of its dimensions.

These observations support a hypothesis, not a final conclusion:

> The predictor may solve much of the task from target position and population regularity, while the anti-collapse regularizer keeps marginal token statistics diverse.

This is more precise than saying that "the model collapsed." There are at least two distinct failures:

1. **Marginal collapse:** embeddings are constant or low-rank across samples.
2. **Conditional neglect:** embeddings may be diverse, but the predictor does not need the conditioning input.

VICReg, SIGReg, and related regularizers primarily address the first issue. The proposed paper asks how to measure and address the second.

### 2.3 What must be reproduced

The strongest diagnostics came from archived artifacts. They should not be treated as publication evidence until they are reproduced under the current input pipeline and current checkpoints.

The reproducibility repair must include:

- correct context, within-subject wrong context, cross-subject wrong context, temporal shuffle, raw-black blank, and position-only conditions;
- foreground and background target decomposition;
- token-level and clip-level effective rank on the exact representation used by downstream evaluation;
- recording-level and subject-level statistics;
- random initialization, handcrafted motion, and direct supervised controls;
- at least three training seeds for any result used in the main claim.

## 3. What a high-quality dataset means

High resolution is helpful, but it is only one part of data quality. A useful scientific dataset needs five forms of fidelity.

| Fidelity | Question | Health&Gait | GaitLU-1M |
| --- | --- | --- | --- |
| Pixel fidelity | Are there enough person pixels to recover motion? | Strong source video and 960 by 540 derived modalities | Weak, 64 by 44 binary silhouettes |
| Measurement fidelity | Are targets measured reliably? | Mixed, with measured and vision-derived gait variables requiring QC | No clinical or biomechanical targets |
| Design fidelity | Can nuisance and movement changes be separated? | Useful paired speed and jacket conditions within person | No comparable paired condition design |
| Population fidelity | Does the cohort represent the intended use? | Healthy adults aged 18 to 64 in one controlled site | Large but unlabeled and biometric-oriented |
| Deployment fidelity | Does capture resemble a home or clinic? | Limited, controlled corridor and camera | Limited and very low resolution |

This leads to a concrete decision:

> Use Health&Gait as the main real-video testbed. Use GaitLU only as a low-resolution stress test, never as the main source of scientific conclusions.

Health&Gait contains 1,564 videos from 398 participants. Original video was recorded at 1920 by 1080 and 30 Hz. Released modalities include 960 by 540 silhouettes and semantic segmentation, 480 by 270 optical flow, 2D pose, anthropometrics, and gait parameters. Participants walked at usual and fast instructed speeds, with and without their own jacket when possible.

Three boundaries matter:

1. The jacket was not described as weighted. It changes appearance and may also change movement slightly.
2. The two walking directions are two segments split from the same source video. They are not an independently assigned experimental factor.
3. The cohort does not contain balance impairments, falls, disease progression, or longitudinal home monitoring.

Therefore Health&Gait can support conclusions about representation behavior under paired movement and appearance conditions. It cannot validate a fall-risk biomarker or a clinical balance assessment.

## 4. The five ideas worth carrying forward

Scores use a 1 to 5 scale. Novelty asks whether the central contribution is crowded. Usefulness asks whether success would change research or practice. Feasibility assumes the ICLR paper deadline is 2026-09-16 and that no new clinical cohort can be collected.

### Idea 1: Necessary-context JEPA

**Question:** Can a non-collapsed JEPA still ignore its context, and can training make the correct context necessary for accurate prediction?

**Core method:** compare correct context with carefully matched wrong contexts. Train with a margin that requires correct context to predict the same target better than wrong context.

| Audience | Novelty | Usefulness | Feasibility | Assessment |
| --- | ---: | ---: | ---: | --- |
| ICLR | 5 | 5 | 4 | Best flagship if the failure generalizes beyond one local checkpoint |
| HAI | 3 | 4 | 4 | Important safeguard against room, clothing, and identity shortcuts |
| Delp lab | 4 | 5 | 3 | Strong when context is pre-perturbation state and the action is controlled |

**Why it matters:** a diverse embedding distribution does not prove conditional prediction. This distinction is relevant across video, robotics, healthcare, and scientific modeling.

**Main risk:** the local failure may be caused only by sparse silhouettes and a small model. The paper must test at least one public JEPA and one controlled second domain.

### Idea 2: Target-information and sensing-fidelity ladder

**Question:** Which target and input representations make the prediction meaningful, and what information is lost at each sensing rung?

**Target rungs:** generic latent tokens, foreground-only latent tokens, anatomical parts, pose, DensePose, flow, measured gait variables, and later biomechanics.

**Input rungs:** high-quality person crop, 112-pixel crop, 64 by 44 enlarged silhouette, silhouette only, and later privacy-reduced on-device features.

| Audience | Novelty | Usefulness | Feasibility | Assessment |
| --- | ---: | ---: | ---: | --- |
| ICLR | 2 | 4 | 5 | Essential evaluation spine, but weak as a standalone method paper |
| HAI | 4 | 5 | 4 | Directly answers privacy, bandwidth, compute, and sensor-cost questions |
| Delp lab | 3 | 4 | 3 | Useful for deciding whether video can preserve biomechanical endpoints |

**Why it matters:** "high resolution" is not a binary property. The ladder measures the lowest sensing cost that preserves a specified utility.

**Main risk:** DensePose and flow are computed from the same RGB video. They are useful derived views, not independent sensor ground truth.

### Idea 3: Paired-condition transport and perturbation-response operators

**Question:** Does a learned representation encode a change in movement as a reusable direction or operator, and can that idea extend to a controlled physical perturbation?

For Health&Gait, estimate within-person changes such as usual to fast walking and no-jacket to jacket. For biomechanics, condition on a perturbation token containing direction, magnitude, and gait phase, then predict the distribution of the response.

| Audience | Novelty | Usefulness | Feasibility | Assessment |
| --- | ---: | ---: | ---: | --- |
| ICLR | 4 | 4 | 4 | Strong real-world transfer test for Idea 1 |
| HAI | 3 | 4 | 3 | Supports personal trend interpretation, but staged conditions are not homes |
| Delp lab | 5 | 5 | 3 | Best long-term biomechanics contribution |

**Why it matters:** average accuracy can hide whether a representation tracks movement or appearance. Within-person transport asks whether the same change behaves consistently across people and nuisances.

**Main risk:** the Health&Gait conditions do not identify a pure causal operator. Fast walking changes the full dynamical system, and a jacket may change both appearance and mechanics. Use the term **paired-condition transport**, not causal intervention, for this dataset.

### Idea 4: Privileged biomechanical targets with cheap inference

**Question:** Can expensive measurements available during training teach a model that later runs from a cheap, unobtrusive sensor?

Examples include training with pose, flow, 3D kinematics, ground reaction forces, COM, CoP, or WBAM while deploying from a single camera or an on-device silhouette stream.

| Audience | Novelty | Usefulness | Feasibility | Assessment |
| --- | ---: | ---: | ---: | --- |
| ICLR | 3 | 5 | 4 for RGB-derived targets | Useful component, but privileged distillation is established |
| HAI | 4 | 5 | 3 | Strong deployment path if compute and privacy are measured end to end |
| Delp lab | 3 | 5 | 2 | No public perturbation dataset currently supplies synchronized RGB and full kinetics |

**Why it matters:** a state relevant to biomechanics or care may be easier to define with laboratory instruments than with raw home video.

**Main risk:** training data must actually align the cheap and privileged modalities. AddBiomechanics and the public Delp perturbation data are not synchronized video datasets.

### Idea 5: Personalized, identity-contained change state

**Question:** Can a system preserve a personal mobility baseline and meaningful deviation while reducing unnecessary identity leakage?

A useful conceptual factorization is:

\[
z_{s,t} = b_s + \phi_{s,t} + d_{s,t},
\]

where \(b_s\) is a local personal baseline, \(\phi_{s,t}\) is gait phase or ordinary short-term variation, and \(d_{s,t}\) is the change that may merit review.

| Audience | Novelty | Usefulness | Feasibility | Assessment |
| --- | ---: | ---: | ---: | --- |
| ICLR | 4 | 4 | 3 | Interesting if tied to context sufficiency and adaptive leakage tests |
| HAI | 4 | 5 | 3 | Most directly aligned with dignified longitudinal monitoring |
| Delp lab | 4 | 5 | 4 | Supported by the value of subject-specific balance baselines |

**Why it matters:** removing every stable personal signal can destroy the longitudinal baseline needed for care. A better design keeps the baseline locally and shares only change, uncertainty, and provenance.

**Main risk:** low identity-probe accuracy is not anonymity. Gait is biometric. Evaluation needs adaptive attackers, cross-session linkability, attribute leakage, and membership tests.

### Recommended combination

Do not present these as five separate papers.

- **ICLR flagship:** Idea 1, supported by Idea 2 and evaluated with Idea 3.
- **Ambient extension:** Ideas 2, 4, and 5, with Idea 1 as a shortcut safeguard.
- **Biomechanics extension:** Idea 3, strengthened by Ideas 1 and 5. Idea 4 becomes possible after synchronized video and laboratory sensing exist.

Generic phase-residual forecasting should not be the headline. GaitForeMer already used future motion forecasting for few-shot Parkinson's gait severity, and Perturbation Recovery Time already formalized return to a subject-specific steady-state neighborhood after perturbation.

### Audience-specific ranking

These rankings are risk-adjusted, not statements that a lower-ranked idea has less long-term scientific value.

| Rank | ICLR by September 2026 | HAI ambient program | Delp biomechanics program |
| ---: | --- | --- | --- |
| 1 | Necessary-context JEPA | Personalized, identity-contained change state | Perturbation-response operator |
| 2 | Paired-condition transport as the utility test | Privileged targets with cheap inference | Necessary-context analysis of recovery |
| 3 | Target-information ladder | Sensing-fidelity ladder | Personalized baseline and change state |
| 4 | Personalized change state | Necessary-context shortcut test | Privileged video student |
| 5 | Privileged biomechanics targets | Paired-condition transport | Sensing-fidelity ladder |

The practical ICLR package combines ranks 1 through 3. The HAI and Delp rankings describe later programs, not work that should be forced into the September paper.

## 5. ICLR 2027 proposal

### 5.1 Proposed title

**When Non-Collapsed JEPAs Ignore Context: Measuring and Enforcing Target-Conditioned Dependence**

An alternative title with less emphasis on failure is:

**Necessary Context for Joint-Embedding Prediction**

### 5.2 Research questions

**RQ1. Diagnostic validity:** Can a JEPA satisfy common marginal anti-collapse metrics while its predictor is nearly insensitive to the observed context?

**RQ2. Target design:** Which targets create genuine conditional prediction instead of a task solvable from position, background frequency, or population averages?

**RQ3. Training:** Does a matched wrong-context objective increase dependence on the correct context without harming prediction accuracy or increasing shortcut leakage?

**RQ4. Utility:** Does stronger correct-context dependence improve held-out-subject transfer, paired-condition transport, and robustness to appearance changes?

**RQ5. Sufficiency:** How much temporal history and which body regions are actually needed for the target?

### 5.3 Hypotheses

**H1:** marginal rank and variance can remain healthy while the correct-versus-wrong-context prediction gap is near zero.

**H2:** foreground or anatomy-constrained targets produce a larger correct-context advantage than uniformly sampled targets when target count and difficulty are matched.

**H3:** matched wrong-context training improves the correct-context gap more reliably than anti-collapse regularization alone.

**H4:** a larger correct-context gap is scientifically useful only when it predicts better transfer under unseen people, speeds, and clothing.

**H5:** gains will be concentrated in targets whose uncertainty can actually be reduced by the observed context. A target that is nearly deterministic from location does not provide a valid test.

### 5.4 Formal problem

Let \(C^+\) be the correct observed context, \(C^-\) a matched but incorrect context, \(M\) the target specification, and \(T\) the target embedding. The predictor is \(g(C,M)\) and the prediction loss is \(\ell\).

Define the context advantage:

\[
\Delta_{ctx} = \mathbb{E}\left[\ell(g(C^-,M),T) - \ell(g(C^+,M),T)\right].
\]

A positive value means the correct context helps. It does not by itself prove that useful motion was learned. The negative context must be matched closely enough that identity, camera, or background cannot solve the comparison.

The training objective is:

\[
\mathcal{L} = \mathcal{L}_{JEPA}
+ \lambda \max\left(0, m - \left[\ell(g(C^-,M),T)-\ell(g(C^+,M),T)\right]\right)
+ \beta \mathcal{R}_{marginal}.
\]

Here \(m\) is a required context margin and \(\mathcal{R}_{marginal}\) is VICReg, SIGReg, or another independently controlled anti-collapse term.

The variance identity explains why target choice matters:

\[
\operatorname{Var}(T\mid M) =
\operatorname{Var}(\mathbb{E}[T\mid C,M]\mid M)
+ \mathbb{E}[\operatorname{Var}(T\mid C,M)\mid M].
\]

The first term is context-explainable variation. If it is small, a predictor has little reason to use context. The second term is residual ambiguity. If it is too large, the target is not predictable even from good context. Useful targets need enough context-explainable variation and manageable residual ambiguity.

### 5.5 Required data

#### Required dataset A: Health&Gait

Use Health&Gait as the high-quality real-video testbed because it already provides:

- 398 people and 1,564 video sequences;
- paired usual and fast walking;
- paired with-jacket and without-jacket conditions for many participants;
- silhouette, DensePose-like semantic segmentation, pose, and optical flow derived from the same event;
- participant-disjoint evaluation;
- measured and vision-estimated gait variables for secondary analysis.

Primary labels should be the assigned walking instruction and paired condition identity. Absolute gait variables should be secondary and pass a reliability audit before use.

#### Required dataset B: a controlled motion testbed

The paper needs a second domain where the information needed for prediction is known. The preferred option is a small rendered subset of AMASS with controlled changes in motion, body shape, viewpoint, texture, and phase.

The controlled testbed should answer questions that Health&Gait cannot:

- whether target position alone is sufficient;
- how the gap changes as context is removed;
- whether shape and motion factors can be crossed independently;
- whether the method recovers a known minimum sufficient context.

If AMASS access or rendering is not ready by the first project gate, use a small procedural articulated-motion benchmark. It is better to have a transparent controlled mechanism than a rushed, opaque data pipeline.

#### Required external model: a public JEPA

Run the diagnostic on at least one independently trained public model such as V-JEPA or V-JEPA 2, provided the released artifact includes the predictor and its masking contract. An encoder-only release cannot answer whether the predictor uses context. If a compatible full checkpoint is unavailable, train a documented public JEPA recipe on the controlled domain. This is not a new clinical dataset requirement. It is a generality test.

If the public model does not exhibit context neglect, that is still informative. The claim must then become conditional: small or domain-specific JEPAs fail under sparse, low-entropy, or poorly chosen targets. The paper must not claim that all JEPAs share the local failure.

#### Optional, not on the critical path

- **AddBiomechanics:** useful for learning physically grounded target encoders or future extensions. Most sequences are not paired with synchronized ambient video.
- **GaitLU-1M:** useful only for a low-resolution stress test or an existing public checkpoint.
- **GAVD:** useful as a pathological-gait transfer study after its target, regularizer, and probe definitions are decoupled.
- **Delp and Georgia Tech perturbation datasets:** valuable for the biomechanics extension, not necessary for the September ICLR submission.

### 5.6 Required method changes

#### A. True temporal prediction

Add masks in which the context occurs before the target. Compare:

1. current spatial tubes across the full clip;
2. space-time blocks;
3. past-to-future masks;
4. phase-aware past-to-future masks, if phase estimation is reliable.

Use 32 to 64 frames when compute permits. Keep a 16-frame baseline to separate a longer horizon effect from the new objective.

#### B. Informative target families

Construct a factorial target experiment:

| Factor | Level 1 | Level 2 |
| --- | --- | --- |
| Target eligibility | Uniform spatial target | Foreground or anatomy-constrained target |
| Necessary-context loss | Off | On |

Keep the following fixed across cells:

- number of target tokens;
- spatial and temporal support;
- empirical baseline difficulty as closely as possible;
- encoder, predictor, optimizer, compute budget, and seeds;
- anti-collapse regularizer \(R_{regularizer}\);
- downstream probe pooling \(P_{probe}\).

This separation is essential. Otherwise changing a body-joint list changes several parts of the system at once and the result is uninterpretable.

#### C. Matched wrong contexts

Use a hierarchy from easiest to hardest:

1. cross-subject context;
2. same-subject, different-condition context;
3. same-subject, same-appearance, temporally mismatched context;
4. within-recording phase-mismatched context;
5. context-free position-only oracle.

The main training negative should preserve as much static appearance as possible. A negative from the same recording at the wrong time is preferable to a negative from a different person because it reduces identity and background shortcuts. Periodic gait creates false negatives, since a different time can contain the same phase and state. Estimate phase, exclude phase-equivalent windows, and report results both with and without phase-aware filtering.

#### D. Context-sufficiency curves

Vary the amount of available state:

- number of past frames;
- temporal distance to target;
- body regions present;
- resolution and modality;
- phase information present or absent.

Plot prediction quality and context advantage against context budget. The elbow is an empirical estimate of the minimum sufficient state for that target.

#### E. Multimodal targets without false ground truth

Health&Gait's pose, segmentation, and flow can define target families or privileged teachers. They should be described as model-derived views of RGB, not independent ground truth.

Semantic target selection can itself leak target content. If the target-frame segmentation chooses the indices, the predictor learns from the fact that every selected location contains a person. The primary experiment should choose candidate regions from the observed context track or a fixed anatomy-balanced schedule. Target-frame semantic selection should be a labeled upper bound, and the position-only oracle must receive the same indices. Keep the prediction target in the same latent space at first. A cross-modal target encoder is an extension after the core 2 by 2 result works.

### 5.7 Baselines

The baseline table should include:

- current cody-jepa with spatial tube masks;
- current cody-jepa with true temporal masks;
- target position and mask only, with no video context;
- raw-black blank context;
- temporally shuffled context;
- same-subject and cross-subject wrong contexts;
- random initialization with identical architecture;
- direct supervised classification or regression;
- handcrafted silhouette area, bounding box, centroid, cadence, and optical-flow summaries;
- a contrastive method such as InfoNCE;
- VICReg or LeJEPA-style marginal regularization without the context margin;
- at least one frozen public video encoder for the paired-condition evaluation.

The position-only oracle is critical. If it predicts the target almost as well as the full model, the target design is still vacuous.

### 5.8 Evaluation

#### Representation behavior

- correct-context advantage and confidence interval;
- normalized context advantage, divided by correct-context loss;
- foreground and background context advantage;
- token-level and clip-level effective rank;
- sensitivity to temporal order;
- context-sufficiency curves;
- variance explained by target position alone.

#### Useful transfer

- usual versus fast walking on held-out participants;
- paired-condition transport consistency across participants;
- transfer of the usual-to-fast direction across jacket conditions;
- transfer across direction, treated as correlated segments rather than independent trials;
- robustness to reduced resolution;
- secondary regression to quality-controlled gait variables.

#### Leakage and nuisance

- closed-set identity as a diagnostic only;
- held-out-subject and cross-session retrieval when possible;
- clothing and direction prediction;
- source-video, camera, and background probes;
- adaptive nonlinear attackers for any privacy-related result.

#### Statistics

- split by person, never by clip window;
- aggregate windows to recording and subject before confidence intervals;
- three or more seeds for trained comparisons;
- subject-level bootstrap intervals;
- a fixed tuning split and untouched test split;
- report effect sizes, not only significance;
- publish the complete exposure ledger for every probe.

### 5.9 What would make this an ICLR paper

The paper needs all four contributions:

1. **A distinction:** marginal non-collapse is not conditional context use.
2. **A measurement:** matched context-substitution and sufficiency curves.
3. **A mechanism:** target information and mask design determine whether context is necessary.
4. **A consequence:** improving context use improves transfer under real movement and appearance changes.

A diagnostic on one failed gait model is not enough. A gait benchmark with frozen encoders is useful but may not be enough for a main-track representation-learning paper. The combination is substantially stronger.

The proposal should be judged against four reviewer questions:

| Review dimension | Evidence required |
| --- | --- |
| Originality | Conditional context use is distinguished from marginal non-collapse, with clear separation from standard contrastive ranking |
| Technical quality | Matched negatives, target controls, multiple seeds, subject-level statistics, and complete shortcut baselines |
| Significance | The result appears beyond one local checkpoint and changes useful transfer behavior |
| Clarity and reproducibility | Fixed split contract, exposure ledger, artifact hashes, executable diagnostics, and claim boundaries |

### 5.10 ICLR kill criteria

Stop or narrow the claim if any of the following occurs:

- context neglect appears only in one archived cody-jepa checkpoint;
- the margin loss increases the diagnostic gap but not held-out transfer;
- gains disappear after matching target count, support, and difficulty;
- the position-only oracle matches the proposed model;
- the method increases identity, clothing, or camera leakage enough to explain the gain;
- the result requires unreliable absolute gait labels;
- one seed drives the result;
- the controlled testbed works but Health&Gait does not.

## 6. Ambient intelligence extension

### 6.1 Goal

The ambient project is not "put the gait model in a home." Its goal is:

> Detect meaningful within-person mobility change with the least burdensome sensing, while preserving dignity, uncertainty, and a clear human decision path.

Stanford HAI describes ambient intelligence as AI-enabled physical spaces that anticipate and respond to needs through embedded sensing while improving care and respecting privacy and human dignity. Stanford's current senior-apartment work emphasizes consented passive sensing, personalized longitudinal dashboards, and eventual clinical comparison.

### 6.2 Research questions

**HAI-RQ1:** What is the lowest sensing fidelity that preserves sensitivity to meaningful mobility change?

**HAI-RQ2:** Can expensive training-time supervision produce a cheap on-device representation at deployment?

**HAI-RQ3:** Can a local personal baseline improve change detection without transmitting an identity-rich embedding?

**HAI-RQ4:** Does the representation remain calibrated across rooms, cameras, clothing, lighting, mobility aids, and demographic groups?

**HAI-RQ5:** What action follows an alert, who receives it, and how is uncertainty shown?

**HAI-RQ6:** What identity, attribute, membership, and cross-session information can an adaptive attacker recover?

### 6.3 Proposed system contract

```text
home sensor
    |
    v
on-device person and motion extraction
    |
    +--> raw video deleted under a declared retention policy
    |
    v
local personal baseline + current deviation + uncertainty
    |
    v
trend and out-of-distribution checks
    |
    v
clinician or caregiver dashboard
    |
    v
human review, confirmatory test, or no action
```

The model output has no value without a decision protocol and someone with the capacity to act. A research prototype should specify:

- who sees the alert;
- what threshold triggers review;
- what confirmatory assessment follows;
- how often false alerts can be tolerated;
- how the resident pauses or opts out;
- what data are retained and where;
- what happens when the model is uncertain or out of distribution.

### 6.4 Methods

#### Sensing-fidelity ladder

Compare high-quality RGB or derived video with progressively cheaper or more private inputs:

1. person crop from native video;
2. 224 or 112 pixel crop;
3. low-resolution crop;
4. silhouette or pose only;
5. compact on-device motion state;
6. non-camera sensor when available.

At each rung report:

- mobility-change utility;
- calibration and out-of-distribution detection;
- identity and attribute leakage;
- latency, energy, memory, and bandwidth;
- failure by subgroup and environment.

Low resolution and silhouettes are not automatically private. Gait itself can identify people.

#### Local personalization

Keep the stable baseline \(b_s\) on the resident's device. Transmit a change vector, uncertainty, timestamp, and sensor-health metadata. Compare this with both full identity erasure and a cloud-hosted personal embedding.

The expected result is not necessarily that identity removal is best. Longitudinal change often requires stable within-person information. The goal is data minimization and contained personalization, not pretending the gait signal is anonymous.

#### Privileged training

Use better measurements in the laboratory to define the desired state, then distill into a deployment model. This can include pose, 3D kinematics, force, COM, CoP, and WBAM once aligned data exist.

### 6.5 Data roadmap

Health&Gait is useful for prototyping clothing and resolution robustness. It is not an ambient dataset.

A genuine ambient study later requires:

- repeated measurements over weeks or months;
- real rooms and camera changes;
- older adults and clinically relevant mobility variation;
- informed consent, understandable controls, and opt-out;
- practitioner-designed trend displays;
- prespecified clinical comparison or workflow endpoint;
- governance for raw data, derived features, and access logs.

### 6.6 Claims boundary

Do not call a staged laboratory video model ambient intelligence. Do not call an embedding a digital biomarker without a defined measure, context of use, clinical validation, and action. Do not describe low attacker accuracy as a privacy guarantee.

The defensible near-term statement is:

> The representation is designed and stress-tested for low-burden longitudinal sensing, but ambient and clinical effectiveness require prospective in-home validation.

## 7. Biomechanics and balance extension

### 7.1 Goal

The strongest Delp-lab question is not generic gait forecasting or a generic biomechanics latent. GaitDynamics already models flexible gait kinematic and force variables, and GaitEncoder already targets compact clinical gait representations.

The open question is more specific:

> Given a person's pre-perturbation state and a specified perturbation, what distribution of recovery responses should we expect, and what is the smallest state needed to predict it?

### 7.2 Research questions

**BIO-RQ1:** Which pre-perturbation variables are necessary to predict recovery after controlling for perturbation direction, magnitude, and phase?

**BIO-RQ2:** Does a subject-specific baseline outperform population normalization for detecting impaired or unusual recovery?

**BIO-RQ3:** Can a learned response operator generalize to held-out subjects, perturbation directions, magnitudes, phases, and impairment types?

**BIO-RQ4:** Which body regions and history lengths carry unique predictive information?

**BIO-RQ5:** Can laboratory kinetics supervise a later video model without losing calibration or hiding sensor error?

### 7.3 Existing evidence and novelty boundary

Wu and colleagues showed that steady-state step-width variability, step-time variability, and foot-placement predictability can detect artificial impairments well when subject-specific baselines are used. Their public dataset contains only 10 healthy participants but rich measurements, including motion capture, forces, EMG, wearable motion, and OpenSim-derived states.

Perturbation Recovery Time already defines recovery as return to a subject's steady-state neighborhood and identifies useful COM-CoP and WBAM features. Therefore the extension must go beyond estimating one recovery time.

The new object is a conditional response distribution:

\[
p\left(z_{t+1:t+H}\mid z_{t-k:t}, a, b_s\right),
\]

where:

- \(z_{t-k:t}\) is recent biomechanical state;
- \(a\) contains perturbation direction, magnitude, and onset phase;
- \(b_s\) is the subject-specific baseline;
- the output is a distribution over post-perturbation trajectories and recovery events.

### 7.4 Targets

Predict physically meaningful endpoints rather than a generic latent alone:

- COM-CoP relationship;
- frontal and sagittal whole-body angular momentum;
- center-of-mass height, velocity, and acceleration;
- first recovery-step timing, length, and width;
- number of recovery steps;
- margin of stability, when measurement validity supports it;
- harness load, assistance, or unsuccessful recovery;
- calibrated prediction intervals.

### 7.5 Data

#### Delp and Collins public perturbation data

Strengths:

- synchronized laboratory kinematics and kinetics;
- four artificial balance conditions;
- 640 perturbation trials;
- subject-specific steady-state baselines;
- perturbation direction and magnitude.

Limitations:

- only 10 healthy participants;
- trials are correlated and must be split by subject;
- treadmill speed and perturbation phase are fixed;
- artificial braces, blocked vision, and foot jets are not diseases;
- apparatus can create visible or sensor shortcuts;
- no synchronized RGB video for a camera student.

#### Georgia Tech perturbation data

Use as an external mechanism test because it contains multiple directions, magnitudes, and gait phases. Confirm exact sensor compatibility and outcome definitions before combining datasets.

#### Later synchronized video collection

A real video-to-biomechanics study requires synchronized camera, motion capture, force, and perturbation measurements. OpenCap can help bridge ordinary movements, but validation on walking, squats, and sit-to-stand does not establish accuracy for small sway or fast reactive balance.

### 7.6 Methods

1. Encode pre-perturbation biomechanical state.
2. Represent the perturbation explicitly as an action token.
3. Predict a calibrated distribution over the post-perturbation trajectory.
4. Apply the necessary-context diagnostic while holding the action fixed.
5. Remove history, phase, or body regions to find the minimum sufficient state.
6. Compare global normalization with subject-specific baselines.
7. Test held-out perturbation combinations, not only held-out trials.

### 7.7 Baselines

- Perturbation Recovery Time;
- the best steady-state balance metrics from Wu and colleagues;
- linear and nonlinear state-space models;
- raw OpenSim features with a simple predictor;
- GaitDynamics where input and output contracts are compatible;
- population-average response;
- subject-specific nearest-neighbor response;
- perturbation-only and history-only oracles.

The perturbation-only oracle is the biomechanics version of the position-only JEPA oracle. If direction, magnitude, and phase already explain the outcome, the person-state encoder has not added value.

### 7.8 Claims boundary

Recovery metrics are not falls. Artificial impairments are not clinical disease. A model trained on healthy participants under laboratory perturbations cannot claim prospective fall-risk prediction.

Fall-risk claims require older or clinical populations, validated assessments, relevant confounders, and prospective fall outcomes. Health&Gait cannot supply these endpoints.

## 8. Shared dataset strategy

### 8.1 Dataset ladder

| Stage | Dataset | Purpose | What it cannot prove |
| --- | --- | --- | --- |
| Mechanism | Rendered AMASS or procedural articulated motion | Known factor control and minimum-context tests | Real-world or clinical usefulness |
| Real video | Health&Gait | High-quality person pixels, paired speed and jacket conditions | Balance, disease, falls, or longitudinal change |
| Physics | AddBiomechanics | Motion and force representation learning | Camera deployment without aligned video |
| Reactive balance | Delp/Collins and Georgia Tech perturbation data | Controlled action-response modeling | General clinical fall risk |
| Ambient validation | Future longitudinal home cohort | Workflow, calibration, acceptability, and trend validity | Broad clinical effectiveness without prospective outcomes |

### 8.2 Required versus optional for ICLR

**Required now:**

- Health&Gait;
- one controlled second domain;
- current cody-jepa checkpoints and reproducible training;
- at least one independent public JEPA diagnostic;
- person-disjoint evaluation and three seeds.

**Optional now:**

- AddBiomechanics;
- GAVD transfer;
- GaitLU stress testing;
- cross-modal privileged targets beyond segmentation-guided target selection;
- balance perturbation experiments.

This boundary protects the September deadline. A smaller complete mechanism story is stronger than an incomplete multimodal system.

## 9. What to borrow from g-jepa and gavd4-vicreg

### 9.1 From g-jepa

The `g-jepa` folder is a conceptual literature and proposal collection, not an executable research repository. It provides useful framing, but its generic future-gait forecasting proposal is not novel enough by itself. GaitForeMer already combined future human-motion forecasting with few-shot Parkinson's gait severity estimation.

Borrow:

- the emphasis on representation-space prediction;
- the idea of privileged biomechanical targets;
- the separation between strong evidence and speculative long-term claims;
- the view that useful world models should support prediction under change.

Do not borrow as a headline:

- "V-JEPA for gait";
- generic low-label gait severity transfer;
- fall anticipation without fall or near-fall data;
- long-horizon hierarchical health claims without longitudinal observations.

### 9.2 From gavd4-vicreg

The strongest idea is to define a semantically meaningful target vocabulary, such as body regions or joints, while allowing broader context.

The current notebooks contain a serious confound. The same 12-joint list influences:

1. target eligibility;
2. VICReg pooling;
3. downstream probe features.

This makes it impossible to tell whether a result comes from the prediction task, anti-collapse regularization, or readout definition.

Borrow:

- anatomy-aware target eligibility;
- exposure ledgers for every readout;
- multiple reference lines, including shortcut floors and handcrafted ceilings;
- artifact fingerprints and result histories.

Revise before borrowing:

- define \(T_{target}\), \(R_{regularizer}\), and \(P_{probe}\) separately;
- keep \(R\) and \(P\) fixed while changing \(T\), or explicitly cross all factors;
- remove label-aware group loss from any self-supervised claim;
- match target count and difficulty;
- avoid calling the 12-joint whitelist a novel method by itself.

Anatomical and semantic masking are already crowded. The novel combination is informative target selection plus a direct test that the correct temporal context is necessary.

## 10. Adversarial review and revisions

This section records the most damaging reviewer objections and the changes they force.

| Initial assumption | Adversarial objection | Revision adopted |
| --- | --- | --- |
| A healthy embedding rank means the JEPA learned prediction | Marginal diversity does not prove conditional dependence | Make correct-versus-matched-wrong context the primary diagnostic |
| A larger context gap means a better representation | A model can use identity or background to reject wrong contexts | Use same-recording or same-person matched negatives and require downstream transfer |
| Foreground targets solve the problem | Foreground can still be predictable from position, and anatomy masking is crowded | Match target difficulty and combine target selection with necessary-context training |
| Semantic target selection is harmless | Selecting indices from the hidden target reveals which locations contain a person | Select from observed context or a fixed schedule; give identical indices to the position-only oracle |
| Every wrong-time context is a valid negative | Periodic gait can return to the same phase and make the alleged negative correct | Exclude phase-equivalent windows and report sensitivity to the negative sampler |
| The context-margin loss is novel by itself | Its mathematical form is a standard ranking or contrastive loss | Base novelty on the conditional diagnostic, target-information analysis, matched sampling, and transfer evidence |
| High resolution makes a dataset scientifically good | Pixel quality cannot repair weak labels, narrow populations, or poor experimental design | Use a five-part data-quality audit and a dataset ladder |
| Health&Gait provides three causal interventions | Jacket use was optional and direction is split from one recording | Use "paired conditions" and treat direction as correlated segments |
| Health&Gait is a complete randomized factorial | Jacket availability can create missingness, and usual walking was recorded before fast walking in the stated protocol | Publish the condition inventory, model missingness, and avoid randomization claims |
| The jacket is weighted | The source paper says participants used their own jacket | Correct the description everywhere |
| DensePose and flow are additional ground-truth sensors | Both are model-derived from the same RGB video | Call them derived views or privileged targets, not independent modalities |
| Absolute Health&Gait gait variables are reliable targets | Local audits and the source paper show uneven agreement | Use instructed usual versus fast as primary; audit each continuous label |
| AMASS success proves real-world usefulness | Synthetic or rendered data can reveal a mechanism but not deployment validity | Require Health&Gait transfer and clearly separate claims |
| AddBiomechanics enables a video model directly | Most data do not contain aligned ambient video | Use it for biomechanics representation learning, not a direct RGB pairing claim |
| Low identity accuracy proves privacy | Gait is biometric and attackers adapt | Test linkability, attributes, membership, and keep personalization local |
| Removing identity is always desirable | A personal baseline is often necessary for longitudinal monitoring | Contain the baseline locally and transmit minimal change state |
| Healthy gait forecasting supports balance or fall claims | Forecasting periodic gait is not predicting recovery or prospective falls | Keep balance claims tied to controlled perturbation endpoints |
| Generic recovery-time modeling is novel | Perturbation Recovery Time already occupies that space | Predict a conditional response operator and held-out perturbations |
| A generic biomechanics latent is novel | GaitDynamics and GaitEncoder occupy much of this space | Focus on minimum necessary pre-perturbation state and personalized response |
| The local archived failure generalizes to all JEPAs | Large public models may use context well | Test public weights and narrow the claim if needed |
| A zero normalized tensor is a black blank input | Under current normalization it is raw mid-gray | Construct blank inputs before normalization and verify pixels |
| More datasets automatically strengthen the paper | Every bridge adds alignment, licensing, and evaluation risk | Keep only Health&Gait, one controlled domain, and one public JEPA on the critical path |

### 10.1 Strongest hostile review of the ICLR proposal

> "You found that a small model trained on sparse binary silhouettes predicts mostly background. This is an implementation postmortem, not a representation-learning contribution. Your loss simply turns wrong contexts into negatives, and any gain could come from subject identity."

The paper survives only if it answers every part:

- reproduce the failure under the corrected pipeline;
- show the distinction between marginal non-collapse and conditional neglect on more than one model or domain;
- use matched negatives that preserve identity and appearance;
- hold target cardinality, support, difficulty, regularization, and probe features fixed;
- show improved prediction and held-out transfer, not only a larger training margin;
- include position-only, random-init, handcrafted, contrastive, and direct supervised controls.

### 10.2 Strongest hostile review of the ambient extension

> "This is a staged gait-video model with privacy language added after the fact. There is no home deployment, resident control, decision protocol, longitudinal endpoint, or privacy guarantee."

The extension survives only after a prospective ambient study adds those elements. Until then, call it an ambient-ready sensing and representation study, not validated ambient intelligence.

### 10.3 Strongest hostile review of the biomechanics extension

> "Ten healthy subjects with artificial impairments and hundreds of correlated trials cannot support a clinical balance claim. Recovery time and gait foundation models already exist, and there is no synchronized RGB."

The extension survives by splitting subjects, modeling explicit perturbation conditions, testing held-out perturbations, comparing with existing recovery metrics and GaitDynamics, and avoiding camera or fall-risk claims until aligned prospective data exist.

## 11. Decision gates and failure plans

### Gate 1: Reproducible local signal

**Deadline:** 2026-08-16

Pass if the corrected diagnostic reproduces weak context use and the result is stable across current checkpoints. Fix the raw-black blank, recording aggregation, and exposure ledger first.

If it fails, stop treating the archived result as a mechanism. Continue with the paired-condition representation benchmark as the lower-risk paper.

### Gate 2: Externality

**Deadline:** 2026-08-20

Pass if context neglect, target-position sufficiency, or a meaningful boundary appears in a public JEPA or controlled second domain.

If it fails, narrow the paper to domain-specific conditions that induce context neglect. Do not use a universal title.

### Gate 3: Method signal

**Deadline:** 2026-08-24

Pass if the 2 by 2 target and loss factorial shows that at least one design choice improves correct prediction and context advantage after matching difficulty.

If only the diagnostic gap improves, drop the method claim.

### Gate 4: Useful transfer

**Deadline:** 2026-08-31

Pass if the method improves held-out-subject paired-condition transfer or another prespecified utility metric without a leakage increase that explains the gain.

If it fails, the method is not the paper. Submit the measurement and boundary result only if it generalizes strongly, or fall back to the paired-condition benchmark.

### Gate 5: Submission quality

**Deadline:** 2026-09-05

Pass if all main comparisons have three seeds, subject-level intervals, frozen splits, complete baselines, artifact provenance, and a one-sentence claim that matches the evidence.

No new major method enters after this gate.

## 12. Execution plan

### 12.1 Immediate repository work

1. Restore or rebuild the complete context-substitution diagnostic in the current code.
2. Correct the normalization-aware blank input.
3. Aggregate windows to recording and subject for evaluation.
4. Restore open-set identity and nuisance retrieval tests.
5. Add deterministic seed control and three-seed orchestration.
6. Add temporal and space-time mask generators without removing the current baseline.
7. Add a semantic target-eligibility interface.
8. Separate target selection, regularization pooling, and probe pooling in configuration.
9. Add a context-free position-only predictor path.
10. Add matched wrong-context sampling and the margin loss.
11. Add a compact exposure ledger and artifact fingerprint to every result.

### 12.2 Experiment order

Run cheap experiments that can kill the idea before expensive training.

| Order | Experiment | Decision it supports |
| ---: | --- | --- |
| 1 | Corrected diagnostics on existing checkpoints | Is the local signal real? |
| 2 | Position-only and handcrafted controls | Is the target task vacuous? |
| 3 | Frozen public JEPA diagnostic | Is the issue broader than cody-jepa? |
| 4 | Small target and mask pilot | Does informative target design help? |
| 5 | Necessary-context loss pilot | Can correct context be made useful without gaming? |
| 6 | Full 2 by 2 factorial, three seeds | Is the mechanism identifiable? |
| 7 | Health&Gait paired-condition transfer | Does it improve useful representation behavior? |
| 8 | Controlled second-domain result | Does the explanation match a known mechanism? |
| 9 | Resolution and modality ladder | What information is necessary? |
| 10 | Final leakage, calibration, and robustness suite | Are gains trustworthy? |

### 12.3 Calendar

| Dates | Output |
| --- | --- |
| Aug 13 to Aug 16 | Reproducible diagnostic, fixed blank, exposure ledger, baseline controls |
| Aug 17 to Aug 20 | Public JEPA audit and controlled-domain setup |
| Aug 21 to Aug 24 | Pilot 2 by 2 target and context-loss experiment |
| Aug 25 to Aug 31 | Full runs, three seeds, paired-condition transfer |
| Sep 1 to Sep 5 | Ablations, statistics, leakage, figures, claim freeze |
| Sep 6 to Sep 10 | Paper writing, internal hostile review, genuine abstract |
| Sep 11 | ICLR abstract deadline |
| Sep 12 to Sep 16 | Final paper, reproducibility package, submission |

This schedule is aggressive. New data collection, a full OpenSim benchmark, a longitudinal ambient study, and synchronized perturbation video are outside the ICLR critical path.

### 12.4 Minimum viable paper package

If time becomes limiting, keep:

- the diagnostic distinction;
- Health&Gait;
- one controlled testbed;
- one public JEPA comparison;
- the 2 by 2 target and context-loss experiment;
- position-only and matched-wrong-context controls;
- paired-condition transfer;
- three seeds and subject-level statistics.

Drop first:

- AddBiomechanics integration;
- GAVD transfer;
- GaitLU retraining;
- cross-modal teacher-student training;
- phase-residual forecasting;
- balance perturbation modeling.

## 13. Claims that are and are not supportable

### Supportable after the proposed ICLR experiments

- Some marginally non-collapsed JEPAs can neglect context under identifiable target and mask conditions.
- Matched context substitution measures a behavior that rank alone does not.
- Informative target selection and necessary-context training can be evaluated independently.
- A method improves paired movement-versus-appearance transfer on held-out Health&Gait participants, if the result is observed.
- High-quality source data can reveal information lost by aggressive low-resolution preprocessing.

### Not supportable from the current data

- all JEPAs ignore context;
- a universal conditional-collapse law;
- causal effects of speed, jacket, or direction on the representation;
- privacy or anonymity from silhouettes or low-dimensional embeddings;
- clinical balance assessment from Health&Gait;
- fall-risk prediction;
- ambient intelligence effectiveness;
- disease progression or early warning;
- video prediction of COM, CoP, WBAM, or force without aligned validation;
- superiority to GaitDynamics or GaitEncoder without direct compatible benchmarks.

### Preferred wording

| Avoid | Use instead |
| --- | --- |
| causal intervention in Health&Gait | paired condition or assigned walking instruction |
| ground-truth DensePose or flow modality | RGB-derived semantic or flow view |
| anonymous gait representation | reduced leakage under specified attacks |
| digital biomarker | candidate measurement requiring clinical validation |
| fall prediction | perturbation-response or mobility-change prediction |
| ambient system | ambient-ready representation or prospective ambient prototype |
| clinically meaningful | mechanically meaningful, unless clinical validity is established |
| world model because it predicts embeddings | conditional predictive model with explicit state and target contracts |

## 14. Glossary

**Context:** the observed portion of a video or motion sequence supplied to the predictor.

**Target:** the hidden or future state the model must predict. In a JEPA, this is usually an embedding rather than raw pixels.

**JEPA:** a joint-embedding predictive architecture. It learns by predicting a target representation from a context representation.

**Marginal non-collapse:** embeddings vary across samples and dimensions. This does not guarantee that a predictor uses its input context.

**Conditional neglect:** the predictor's output changes little when the correct context is replaced with an incorrect one.

**Minimum sufficient state:** the smallest observation that retains the information needed for a specified prediction.

**Paired-condition transport:** whether a within-person change learned in one setting behaves consistently in another setting or another person.

**Privileged target:** an expensive or information-rich signal available during training but not required at deployment.

**COM:** center of mass.

**CoP:** center of pressure.

**WBAM:** whole-body angular momentum.

**Exposure ledger:** a record of which labels, groups, subjects, splits, and representation components were available to every stage of an experiment.

## 15. References

### Joint-embedding prediction and representation learning

- Assran et al. [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), CVPR 2023.
- Bardes et al. [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv), ICLR 2024.
- Bardes et al. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/), 2025.
- Balestriero and LeCun. [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544), 2025.
- Nam et al. [Causal-JEPA: Learning World Models through Object-Level Latent Interventions](https://arxiv.org/abs/2602.11389), 2026.
- Bardes et al. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906), ICLR 2022.

### Gait and motion data

- Gutierrez-Martinez et al. [Health & Gait: a dataset for gait-based analysis](https://www.nature.com/articles/s41597-024-04327-4), Scientific Data 2024.
- Mahmood et al. [AMASS: Archive of Motion Capture as Surface Shapes](https://amass.is.tue.mpg.de/), ICCV 2019.
- Werling et al. [AddBiomechanics Dataset: Capturing the Physics of Human Motion at Scale](https://arxiv.org/abs/2406.18537), 2024.
- Endo et al. [GaitForeMer: Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for Few-Shot Gait Impairment Severity Estimation](https://arxiv.org/abs/2207.00106), MICCAI 2022.
- Fan et al. [GaitSSB: A Large-Scale Gait Dataset and Benchmark](https://arxiv.org/abs/2206.13964), 2022.
- [GaitEncoder: A Foundation Model of Gait Kinematics for Diverse Clinical Applications and Pathologies](https://www.medrxiv.org/content/10.64898/2026.07.07.26357479v1.full), medRxiv 2026.

### Biomechanics and balance

- Wu et al. [Detecting artificially impaired balance in human locomotion: metrics, perturbation effects and detection thresholds](https://pmc.ncbi.nlm.nih.gov/articles/PMC12148027/), Journal of Experimental Biology 2025.
- [Public Dryad data for the balance-impairment and perturbation study](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqch3).
- Wu et al. [Perturbation Recovery Time Identifies Subtle Human Balance Impairments and Features](https://sciety.org/articles/activity/10.1101/2025.06.26.661833), IEEE Transactions on Biomedical Engineering 2026.
- Tan et al. [GaitDynamics: a generative foundation model for analyzing human walking and running](https://www.nature.com/articles/s41551-025-01565-8), Nature Biomedical Engineering 2026.
- [Georgia Tech public multidirectional gait-perturbation dataset](https://repository.gatech.edu/entities/publication/73a7c133-6535-4a88-b81e-5c39df5efb3e).
- Uhlrich et al. [OpenCap: Human movement dynamics from smartphone videos](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462), PLOS Computational Biology 2023.

### Ambient intelligence and clinical translation

- Stanford HAI. [Ambient Technology/Intelligence mission](https://hai.stanford.edu/centers-labs).
- Stanford HAI. [Ambient Intelligence, Human Impact](https://hai.stanford.edu/news/ambient-intelligence-human-impact), 2025.
- Haque et al. [Illuminating the dark spaces of healthcare with ambient intelligence](https://www.nature.com/articles/s41586-020-2669-y), Nature 2020.
- Callahan et al. [Standing on FURM ground: A framework for evaluating Fair, Useful, and Reliable AI Models in healthcare systems](https://arxiv.org/abs/2403.07911), 2024.
- Duan et al. [GaitProtector: Impersonation-Driven Gait De-Identification via Training-Free Diffusion Latent Optimization](https://arxiv.org/abs/2605.12431), 2026.
- [A framework of digital biomarkers for neurodegenerative diseases](https://pmc.ncbi.nlm.nih.gov/articles/PMC13229411/), 2026.

### Submission information

- [ICLR 2027 author guidelines and official deadlines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

### Repository evidence

- [`README.md`](README.md)
- [`REVIEW-2026-08-12.md`](REVIEW-2026-08-12.md)
- [`outputs/phase0/baseline/context_use.metadata.json`](outputs/phase0/baseline/context_use.metadata.json)
- [`outputs/phase0/regen-r4/report.md`](outputs/phase0/regen-r4/report.md)
- [`configs/healthgait.json`](configs/healthgait.json)
- [`cody_jepa/masks.py`](cody_jepa/masks.py)
- [`cody_jepa/models.py`](cody_jepa/models.py)
- [`cody_jepa/data.py`](cody_jepa/data.py)
- [`cody_jepa/engine.py`](cody_jepa/engine.py)
- `/Users/theodoremui/dev/g-jepa/`
- `/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd4-vicreg/`
