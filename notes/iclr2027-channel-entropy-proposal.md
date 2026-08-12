# ICLR 2027 proposal — channel entropy as a precondition for joint-embedding prediction

Date: 2026-08-12. Deadlines: abstract Sep 18, paper Sep 25. Experiments stop Sep 4,
leaving **18 working days**.

This supersedes the flagship recommendation in `iclr2027-direction.md` §3. That document's
verified findings (§1.1–§1.8) all still hold and are load-bearing here; only the choice of
paper changes. The counterfactual-probing flagship is retained as fallback B.

---

## 1. The claim

> Joint-embedding prediction has an unstated dependency on its **input channel**, not just
> on data quantity. EMA targets, stop-gradient and masking prevent latent collapse only
> while the target encoder's outputs stay diverse — and that diversity is supplied largely
> by *appearance* variation, precisely the nuisance the representation is meant to discard.
> Strip appearance and the mechanism fails silently: the loss still descends, the predictor
> stops conditioning on context, and rank collapses. Data scale does not rescue it.

The quotable form: **latent prediction is powered by the nuisance variability it is
supposed to remove.** This reverses InfoMin's prescription (minimise nuisance mutual
information between views) specifically for predictive objectives, where it is known to
help contrastive ones.

Scope of the claim is deliberately broad and testable outside gait: any domain whose input
is already a clean, low-entropy, task-focused channel — medical segmentation masks, depth,
radar, event cameras, LiDAR occupancy, simulator state — should exhibit the same silent
collapse under JEPA-style objectives.

## 2. Why this rather than the other candidates

- It does **not** touch the OptoGait parameters, so §1.1 (label reliability ~0.14,
  ceiling r ~ 0.372) is irrelevant to every number in the paper.
- It does **not** require phase, so §1.4 (0.52 gait cycles per clip) is irrelevant.
- It does **not** require a working representation. The repo's eleven collapsed checkpoints
  (effective rank 5–11 of 384, `wrong_context_gap ~ 1e-4`, held-out retrieval 4.8%) stop
  being an embarrassment and become the first data point.
- It makes GaitLU-1M and the in-house JEPA **load-bearing** rather than optional, which was
  the explicit requirement.
- The word "gait" need not appear in the title. (Inherited non-negotiable #7.)

## 3. Assets, verified state

| Asset | State | Verified |
| --- | --- | --- |
| GaitLU-1M shards | On HAIC. Phase-aware loader + production launcher **unfinished**. | User, 2026-08-12 |
| GaitLU-1M local copy | `data/gaitlu-1m/`, 52 GB unopened multi-part ZIPs. Not usable. | This session |
| Health&Gait | 1,564 videos, 398 participants, on disk (57 GB) | This session |
| Health&Gait modalities | silhouette (YOLOv8, JPEG), semantic segmentation (DensePose, PNG), optical flow (**TVL1 and GMFlow**, PNG), pose (AlphaPose, JSON) | This session |
| Existing checkpoints | 11 runs, `results/phase1_summary.csv` | Repo |
| CASIA-B / OU-MVLP | Neither held. Registration required. | User |
| Compute | ~708 GPU-hr class budget available to Sep 4 | User |

**Open data check:** 398 participants x 4 (speed x jacket) = 1,592, but the README states
1,564 videos. Reconcile before building manifests — likely missing cells, and the
per-participant cell count must be audited, not assumed.

### 3.1 Why the Health&Gait modalities are the right instrument

Prior work on what SSL needs from its data (e.g. *On Pretraining Data Diversity for SSL*,
ECCV 2024) varies the **dataset**, confounding channel with content. Health&Gait ships four
renderings of the **identical** physical events — same frames, same participants, same
camera, same randomised cells. Content, motion and camera are held exactly fixed; only the
channel varies. That isolation is the contribution's foundation.

Better still, the four channels **dissociate two properties that are usually conflated**:

| Channel | Pixel entropy | Task-relevant information |
| --- | --- | --- |
| Silhouette | lowest (1 bit) | moderate |
| Semantic segmentation (DensePose IUV) | high | high |
| Optical flow (TVL1, GMFlow) | high | high |
| Pose (17 keypoints) | near-zero | near-sufficient |

Pose is the pivot. If collapse tracks **pixel entropy** and ignores **task-relevant
information** — collapsing on silhouette *and* pose while surviving on flow and
segmentation — the claim is sharp and slightly counter-intuitive. If instead collapse
tracks task-relevant information, the thesis is wrong in an interesting way and the paper
becomes a different, still-reportable finding.

TVL1 vs GMFlow gives a free same-channel / different-extractor control.

### 3.2 The corroborating literature observation

Every published silhouette-gait SSL model is contrastive or generative — GaitSSB
(contrastive, on this exact corpus), FoundationGait, the diffusion-pretraining line. No
published masked-latent-prediction gait model exists. The field has been routing around
this without naming it. This belongs in the introduction as motivation, not as evidence.

**Do not** use GaitSSB's published numbers as the contrastive arm. It uses a different
(CNN gait-recognition) backbone and augmentation stack, so that comparison confounds
objective with architecture. The arm must be a matched-encoder contrastive run trained
in-house on the same shards. Verify GaitSSB's backbone before writing the related-work
paragraph.

## 4. Experimental design

Two axes crossing at one corner, plus a fix.

**Axis A — scale (GaitLU, silhouette only).** Pretrain at 10^3 / 10^4 / 10^5 / 10^6
sequences. Objectives: JEPA, matched-encoder contrastive, pixel-MAE. Outcome: effective
rank, `wrong_context_gap`, target-embedding pairwise distance, downstream transfer.
*This is a scaling curve, not a point.* It also retires I6: the 28-model allocation study
is replaced by a curve with real dynamic range instead of a contrast designed to be null.

**Axis B — channel (Health&Gait, matched small scale).** Four renderings x three
objectives, identical encoder and budget.

**Axis C — early-warning diagnostic.** Instrument every run: does target-embedding pairwise
distance over the first ~500 steps predict final effective rank? If yes, the paper ships a
cheap pre-flight check that tells a practitioner whether their channel will support a JEPA
before they spend the compute. This is the most directly useful artefact in the paper.

**The fix (turns a study into a method).** Cross-channel JEPA: context in the low-entropy
channel, targets in a higher-entropy one (silhouette -> flow). Exact frame alignment makes
this free to construct. Compare against LeJEPA-style explicit variance regularisation as
the competing remedy. If both work, the mechanism is confirmed twice over.

**Replication.** Derive the same channel ladder from a no-registration public RGB corpus
using off-the-shelf extractors. This replicates the *claim* rather than the dataset, which
is stronger for this thesis than another gait benchmark, and satisfies non-negotiable #6.

**Transfer evaluation.** Health&Gait identity retrieval across factor changes (398
identities, held-out participants). CASIA-B added only if the request lands in time.

## 5. Predictions, and what falsifies them

| Prediction | Falsified by |
| --- | --- |
| JEPA effective rank degrades monotonically as channel entropy drops | flat or non-monotone ladder |
| Contrastive stays approximately flat across the same ladder | contrastive collapsing too — implies data/encoder cause, not objective |
| GaitLU scale curve saturates early and low | rank/transfer continuing to climb with corpus size |
| Collapse tracks pixel entropy, not task-relevant information | pose behaving like flow rather than like silhouette |
| Early target-distance predicts final rank | no correlation across the run grid |

The **second row is load-bearing**. It is what rules out both "silhouettes are just hard"
and "you have a bug," because data, encoder, budget and evaluation are identical across
objectives.

## 6. Decision gate — Aug 14

Four small runs on Health&Gait: {silhouette, flow} x {JEPA, VICReg}. Requires no new code
and no GaitLU.

- **Contrastive flat where JEPA collapses** -> thesis alive, commit to the full grid.
- **Both collapse** -> the cause is the setup, not the channel. Pivot to diagnosing it,
  which yields working checkpoints and fallback B.

## 7. Schedule

| Dates | Work |
| --- | --- |
| Aug 12–14 | **Track A:** finish phase-aware loader + launcher on HAIC, validate shards, training smoke test. **Track B:** the Aug 14 decision gate (4 runs, no new code). Also: file CASIA-B request; audit the 1,564 vs 1,592 discrepancy; build channel-matched normalisation controls. |
| Aug 17–21 | Axis A scale ladder (4 scales x 3 objectives, GaitLU). Axis B full channel ladder (4 channels x 3 objectives). Axis C instrumentation in every run. |
| Aug 24–28 | Cross-channel JEPA fix. LeJEPA-style variance-regularisation comparison. Replication ladder on public RGB corpus. Transfer eval. |
| Aug 31–Sep 4 | Controls (random-init, classical baselines, normalisation-matched channels), robustness, freeze analysis and figures. Stop experiments. |
| Sep 5–17 | Writing. |
| Sep 18 | Abstract. Sep 25 submission. |

Track A and Track B are independent for the first three days, which is why the unfinished
launcher costs no science time.

## 8. Reviewer objections, prepared answers

**"You have an implementation bug."** The masking matches V-JEPA's design (spatial blocks
replicated across the full temporal extent — confirm against the paper and quote it). More
decisively, the objective crossover holds data, encoder, budget and evaluation fixed.

**"It is known that JEPAs need anti-collapse heuristics."** Known that heuristics are
needed; not known that their efficacy is a function of input-channel entropy, nor that it
is measurable and predictable within the first few hundred steps. LeJEPA is the natural
related work *and* a confirming remedy, not a scoop.

**"You measured normalisation, not information."** The most serious objection. Flow and
DensePose PNGs differ from binary silhouettes in dynamic range and sparsity. Requires
channel-matched controls: quantise flow to 1 bit, dilate silhouettes to match foreground
density, match per-channel mean/variance. **Build these in week one, not week four.**

**"Scale and channel cross at one corner."** True and must be stated. GaitLU is
silhouette-only and its RGB source is gone, so the scale axis cannot be run at higher
entropy. Mitigation is the RGB replication corpus giving a second scale curve at a second
channel.

**"One dataset, one architecture."** Replication corpus plus at least one public-weight
encoder in the evaluation tables.

## 9. Inherited non-negotiables

1. Participant-disjoint splits, stated in the main text.
2. A random-init encoder row in every table.
3. A non-learned classical baseline in every table.
4. Height / BMI / sex / age partialled out of any anthropometric claim.
5. At least two public-weight encoders, not only the in-house 6-layer ViT.
6. At least one replication dataset.
7. The word "gait" absent from the title.

## Sources

- [V-JEPA (OpenReview)](https://openreview.net/forum?id=WFYbBOEOtv) · [V-JEPA 2](https://arxiv.org/html/2506.09985v1) · [I-JEPA](https://arxiv.org/pdf/2301.08243)
- [GaitSSB / GaitLU-1M (TPAMI 2023)](https://arxiv.org/pdf/2206.13964) · [FoundationGait](https://www.emergentmind.com/papers/2512.00691)
- [UR-JEPA](https://arxiv.org/pdf/2606.01443) · [From Alignment to Prediction](https://arxiv.org/html/2604.13518v1)
- [On Pretraining Data Diversity for SSL (ECCV 2024)](https://arxiv.org/abs/2403.13808)
- [Health&Gait (Sci Data 2025)](https://www.nature.com/articles/s41597-024-04327-4)
