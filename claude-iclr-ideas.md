# cody-jepa → ICLR 2027, v2: cross-repo synthesis, adversarial review, and the top five

**Prepared** 2026-08-12 · **Repo state** `03f95ae` (main), untracked `ICLR2027-ANALYSIS-2026-08-12.md`
**Supersedes** the *recommendation* in `ICLR2027-ANALYSIS-2026-08-12.md`. It does **not** supersede that document's
audit of the code or its verified empirical findings, which I re-checked and endorse.

**What is new here:** a systematic read of `~/dev/g-jepa` (conceptual corpus) and
`~/dev/alexpose/experiments/sjepa/gavd4-vicreg` (working S-JEPA on pathological gait), an adversarial
debate over eleven candidate directions, a ranked top five, a resolved answer to the GaitLU-1M data-quality
problem, and a three-audience evaluation (ICLR / Stanford HAI ambient intelligence / Delp balance assessment).

**Provenance discipline.** Every number below is tagged `[disk]` (recomputed from files in these repos today),
`[web]` (verified against a source I fetched today), or `[inherited]` (taken from the prior analysis without
re-derivation). Nothing is asserted from memory alone.

---

## Part 0 — The one-paragraph answer

Stop treating Health&Gait as a *training corpus* that your JEPA is failing to learn from, and start treating it
as a **measuring instrument** that almost nobody else has: 398 human bodies, each recorded under a *randomly
assigned physical intervention* (walk fast), a *randomly assigned appearance nuisance* (put a jacket on), and a
*viewpoint nuisance* (walk back the other way) — with the same physical event simultaneously rendered four
different ways at 960×540. That is a real-world, human-scale, causally-controlled test bed for the single
question the joint-embedding community argues about most and measures least: **does the representation encode
how the body moves, or what it looks like?** You can answer it with frozen public checkpoints, on a laptop's
worth of GPU, with zero pretraining risk, in the time you have. Retire GaitLU-1M — it is 64×44 binary
silhouettes `[disk]` and it cannot support any causal or clinical claim you would want to make.

---

## Part 1 — What is actually on disk, verified today

### 1.1 The data-quality question, settled

The user's instinct about GaitLU is correct, and the situation is starker than "low resolution."

| | Health&Gait | GaitLU-1M | GAVD (in `alexpose`) |
|---|---|---|---|
| Frame size | **960 × 540** `[disk]` | **64 × 44** `[disk]` | 1280 × 720 source `[disk]` |
| Pixel depth | 8-bit L / RGB per modality | **binary** `[disk]` | RGB |
| Subjects | 398, with IDs | none — unlabelled by design | 400+ |
| Modalities of the *same event* | **4 co-registered** | 1 | 1 |
| Assigned intervention | **yes — speed, jacket, direction** | no | no |
| Anthropometrics | 11 fields, n=397 | no | no |
| Clinical labels | OptoGait / MuscleLAB (unreliable, see 1.3) | no | **8+ pathology classes** |
| On-disk size | 56 GB, unpacked | 51 GB, **still in multipart zips** | ~1.3 GB extracted subset |

`[disk]` verification of the two headline numbers:

- Health&Gait silhouette `PA332/FGS/WoJ_2_YOLOV8/015.jpg` → `(960, 540)`, mode `L`.
  DensePose `PA332/FGS/WoJ_2_DensePose/016.png` → `(960, 540)`, mode `RGB`.
- GaitLU shard `gaitlu-000.tar.gz`, sequence `000/030/001/001.pkl` → 66 frames, **64 × 44 pixels**
  (read out of `gaitlu-000.html`, the notebook you ran on 2026-08-12).

**The perceived "low resolution problem" is a GaitLU problem, not a Health&Gait problem.** The current
pipeline downsamples Health&Gait to 112×112 after a foreground crop (`configs/healthgait.json`) and uses
exactly one of its four modalities. You are discarding roughly 97% of the pixels and 75% of the channels of
the *good* dataset while contemplating unpacking 51 GB of the bad one.

### 1.2 The asset nobody has costed correctly

Health&Gait's directory grammar is `<modality>/PA###/<UGS|FGS>/<WoJ|WJ>_<1|2>` `[disk]`. Decoded:

- **UGS / FGS** — usual vs. fast gait *instruction*. This is an **assigned within-subject intervention** with
  near-perfect compliance: 98.2% of participants increase velocity by the OptoGait labels, 99.7% by the video
  estimator, **100.0% increase cadence** by the video estimator `[inherited, §2.5 of v1]`.
- **WoJ / WJ** — without / with a weighted jacket. Changes the silhouette's *shape and appearance*; the
  authors' design intent is a load manipulation, but for representation-learning purposes the important thing
  is that it is an **assigned appearance perturbation with the same body and the same instruction**.
- **_1 / _2** — the two directional segments of one walk `[disk, build_manifest.py:53]`. **Viewpoint only.**

So for each of 398 subjects you have, in one session, on the same camera:

```
                    do(speed)              do(jacket)             do(view)
semantics:      dynamics change       appearance change      viewpoint change
                appearance fixed      dynamics fixed         everything fixed
```

**Three interventions whose true causal semantics are known a priori, assigned rather than observed,
within-subject, at n=398, on real human bodies.** I went looking for anything comparable and did not find it.
MPI3D is a robot arm; CausalVerse `[web, arXiv:2510.14049]` is high-fidelity *simulation*; CEBaB is
human-authored *text* edits; the entire disentanglement benchmark corpus (dSprites, Shapes3D, Causal3DIdent)
is synthetic. **This is, as far as I can establish, the only real-world human-motion dataset with orthogonal
assigned physical factors at three-figure subject scale.**

That is the paper. Everything else in this document is either how to get there or what to do next.

### 1.3 What is broken, and what that is worth

Re-verified `[disk]`, from `outputs/phase0/baseline/` and `outputs/phase1/`:

| Measurement | Value | Chance / reference |
|---|---:|---:|
| Blank-context loss gap | 9.531e-5 (0.025% of loss 0.3868) | — |
| Cross-subject − same-subject gap | 3.3e-6 | — |
| Temporal-shuffle gap | 4.716e-5 — **half** the blank gap | — |
| Foreground-token gap ÷ all-token gap | **0.299** | 1.0 if uniform |
| Context-token effective rank | 381.6 / 384 | — |
| **Evaluated (EMA, clip-pooled) effective rank** | **11.5 / 384** | — |
| identity, held-out-subject retrieval, 80 classes | **0.0245** | 0.0129 |
| identity, closed-set, 318 classes | 0.0925 | 0.0032 |
| `gait_system` (2-class), held-out subjects | 0.926 | 0.500 |

The diagnostic harness that produced these is genuinely good — subject-level bootstrap over 80 held-out
subjects, 10,000 iterations, sha256 provenance on checkpoint / CSV / manifest / dataset `[disk,
context_use.metadata.json]`. Effect size `d_z = 6.06` on the blank condition with
`positive_subject_fraction = 1.0`. This is publication-grade instrumentation attached to a model that does not
work. **Keep the instrument, change what you point it at.**

Two structural causes, both confirmed in code:

1. **Masking is spatial-only.** `_expand_tubes` (`masks.py:41`) copies each selected spatial cell to every
   temporal index. Context and target both span the whole clip; Δt is never an input and never supervised.
   This is why temporal shuffling hurts *less* than blanking — the co-temporal neighbours the predictor leans
   on survive a frame permutation.
2. **Targets are batch-standardized per dimension** (`losses.py:31`, `target_batch_standardize: true`). The
   target distribution looks diverse regardless of what the encoder does.

---

## Part 2 — What to borrow from the two sibling repos

### 2.1 From `gavd4-vicreg` — take the epistemics and one idea

**Take (high value, port directly):**

| Asset | Why it matters here | Port cost |
|---|---|---|
| **Exposure ledger** — every score printed beside its row/group/representation/label exposure | cody-jepa currently reports `identity` on **training** participants by default (`evaluation.py:173`, `split` defaults to `"train"`). The ledger discipline catches exactly this. | 1 day |
| **Three reference lines on every readout** — majority baseline / shortcut floor / handcrafted ceiling | cody-jepa has the random-init control but no *shortcut floor* and no *classical ceiling*. Reviewers ask for both. | 1 day |
| **Missingness-only / shortcut control** | The pixel analogue is a silhouette-area-only and bbox-only probe. Given foreground fraction is 9.62% of masked targets, a shape-free baseline is essential. | 1 day |
| **Checkpoint fingerprint bound to data payload** | cody-jepa already does sha256; gavd4 binds the *classifier contract* to the fingerprint so downstream results cannot drift off their encoder. | 2 days |
| **`result_history.csv` distinguishing "model changed" from "evaluation changed"** | You have eleven phase-1 variants inside ±2 SE of each other `[inherited]`. This file is how you avoid a twelfth. | half day |

**Take (the one genuinely novel idea in that repo):**

> **Target vocabulary** — the prediction-target set is restricted to a fixed *anatomical* subset (12 of 33
> landmarks) while the context encoder still sees everything.

Their own analysis correctly identifies this as unstudied *as a general mechanism*, and correctly notes that
every JEPA/MAE variant masks uniformly, by learned salience, or by geometric blocks — never by a fixed
semantic vocabulary with the asymmetry *measured*. **And cody-jepa can do this better than they can**, because
Health&Gait ships DensePose semantic segmentation co-registered with the silhouettes `[disk]`: you get 24 body-part
labels *per pixel*, which means you can define anatomically-specified target token sets **in a video JEPA**,
where it has not been done. gavd4 can only do it in skeleton space. See idea **#4**.

**Do not take:**

- The **condition curriculum** (normal → PD → stroke → myopathic → CP). Their own audit shows the realized
  mask fraction drifts 0.551 → 0.423 across stages, so difficulty and stage are confounded; and stages 1–4 use
  a label-aware group loss, so the pipeline is not self-supervised after stage 0.
- The **DINO-style cross-entropy taken over the 96 embedding dimensions**. That is a deviation from both DINO
  and S-JEPA and has no clear geometric meaning. cody-jepa's L1-in-embedding-space is the better choice.
- The **five-condition classification framing** on 159 sequences from 35 videos. Underpowered by construction.

### 2.2 From `g-jepa` — take one proposal and one discipline

The `g-jepa` wiki is a conceptual corpus (56 pages, ~5,650 lines `[disk]`), not code. Its value is that it has
already done the literature triage against the Fei-Fei Li gait corpus and the LeCun world-models program, and
it has already sorted its six proposals into firm and soft ground.

| g-jepa proposal | Verdict for this repo, this cycle |
|---|---|
| P1 G-JEPA self-supervised gait world model | **This is what cody-jepa already is, and it does not work.** Do not re-attempt as the headline. |
| P2 Fall anticipation as violation-of-expectation | **Blocked on data.** Health&Gait has zero falls and zero near-falls; GAVD has none either. Also crowded (T-SAR-JEPA, MTS-JEPA). Out. |
| P3 Multimodal privacy distillation | **Half-blocked.** All four Health&Gait modalities are vision-derived from one camera; there is no IMU or radar. But the *privacy* half survives and is strong — see idea **#3**. |
| P4 Objective-driven fall prevention | Out. No actuator, no simulator, no cohort. Multi-year. |
| P5 Hierarchical multi-timescale H-JEPA | Out. No longitudinal data. LeCun himself calls abstraction discovery unsolved. |
| **P6 Gait-Physics-IQ** | **Take it.** See idea **#2**. This is the best idea in the g-jepa corpus and it is the one that is currently unoccupied in the literature. |

**Take the discipline too:** g-jepa's `critical-appraisal.md` splits LeCun's claims into best-supported
(predict in representation space; violation-of-expectation) and contested (intrinsic safety; hierarchical
abstraction). Papers built on the first ship; papers built on the second do not. All five ideas below sit on
the firm half.

---

## Part 3 — Adversarial review

I ran eleven candidates through proposal → red team → verdict. Compressed to the decisions that changed
something.

### Round 1 — against the v1 document's primary recommendation

**Proposition (v1 Candidate C):** *Silent failure in latent-prediction SSL — an axis-resolved diagnostic, a
mechanism, and a fix.* The 33× rank discrepancy and the 0.025% blank-context gap are Table 1.

**Red team.** The paper's survival depends entirely on its own §6 non-negotiable: *"≥2 public-weight encoders
running the same diagnostic battery… this is what answers 'does this matter outside your broken 6-layer
model?'"* The v1 document names this as the kill criterion and then schedules it for week 2. **I estimate that
test fails.** V-JEPA 2 is trained on 22M videos with volumetric tube masking on natural RGB; its clip-pooled
rank will not be 11/384. VideoMAE-v2 likewise. The honest result you will get is *"a 6-layer JEPA trained on
binary silhouettes with spatial-only masking collapses on the clip axis"* — which is a specific instance of
dimensional collapse (Jing et al., ICLR 2022) plus a known low-input-entropy pathology. That caps the paper at
borderline.

**Counter (partial).** The *axis-mismatch* framing survives even when the failure does not generalize, because
the claim can be restated as a claim about **metrics** rather than about **models**: token-axis VICReg and
token-axis rank are routinely reported as health signals and they are uninformative about the axis probes
actually consume. That is adoptable practice, and it holds whether or not big checkpoints are sick.

**Verdict.** Demote from primary to (a) a section of the winning paper and (b) a standalone fallback.
Retained as idea **#5**.

### Round 2 — against the channel-entropy proposal

**Proposition:** latent prediction is powered by the nuisance variability it is supposed to discard; strip
appearance and the mechanism fails silently.

**Red team.** Three problems. (i) The four Health&Gait channels differ in resolution, codec, extractor and
preprocessing artifacts, not only entropy — v1 spots this and proposes a within-channel ladder, which is the
right fix. (ii) Even fixed, the claim's nearest neighbours are close: InfoMin, dimensional collapse, and the
SSL-data-diversity line. (iii) Most damaging: a reviewer reads it as *"low-information inputs yield
low-information representations,"* which is not surprising.

**But the red team also handed over the answer.** The reason the four-channel comparison is confounded as a
*causal* claim is exactly the reason it is excellent as a *robustness* axis. Four renderings of an identical
physical event is a gift for any measurement you want to show is not an artifact of one input format.

**Verdict.** Killed as a standalone paper. Absorbed twice: as the robustness axis in **#1**, and as the
privacy ladder in **#3**.

### Round 3 — the decisive round: counterfactual probing, escalated

**Proposition (v1 Candidate A):** fit `w_obs` observationally; compute `Δ_int` within-subject; measure
`cos(w_obs, Δ_int)`; test transport and universality.

**Red team.** CEBaB (NeurIPS 2022) and Causal Proxy Models (ICML 2023) pre-claim "counterfactual concept
effects." Novelty available is narrow: *physical* rather than text interventions. Audit papers cap at
borderline. Only two clean assigned factors. And v1's own §2.5 finding (Δ magnitude correlates with sex
r = 0.264 and height r = 0.247) predicts the universality experiment returns the messy answer.

**Escalation — this is where the debate turned.** Every objection above is an objection to treating `Δ_int` as
a *correlation to be measured*. None of them survives treating it as an **operator to be learned, transported,
and composed**:

- `cos(w_obs, Δ_int)` is a probing statistic. **`T_jacket ∘ T_speed` vs. the true joint displacement is an
  experiment**, and it is one that no dataset in the disentanglement literature can run on real bodies.
- Two factors is thin for a correlation study. For a *composition* study, two orthogonal factors plus a
  viewpoint nuisance is exactly the minimal sufficient design: you need 2 to compose and a 3rd to show the
  composition is selective.
- "Audit papers cap at borderline" applies to audits. Adding an **intervention-equivariant training
  objective** whose effect you measure on the same instrument makes it a method paper.
- The messy universality answer stops being a risk and becomes a *result*: a rank-1 spectrum validates the
  linear representation hypothesis on real physical interventions; a high-rank spectrum falsifies it in a
  setting people care about. Both publish.

**Literature check, run today.** The nearest neighbours and why each leaves the ground open:

| Prior work | What it does | Why the gap survives |
|---|---|---|
| CausalVerse `[web, arXiv:2510.14049]` | Benchmarks causal representation learning with configurable high-fidelity **simulation** | Simulated. The entire point of the proposed benchmark is that the intervention was applied to a real body by a real instruction. |
| Delta Embeddings `[web, arXiv:2508.04492]` | Learns robust intervention representations from **computational** perturbations of inputs | Interventions are algorithmic edits, not physical assignment. Nearest neighbour; must be cited in paragraph 1. |
| SIE — Split Invariant-Equivariant SSL `[web, arXiv:2302.10283, ICML 2023]` | Splits representation into invariant and equivariant parts w.r.t. **image augmentations** | Augmentations, not physical interventions. Also single-factor: no composition of two different interventions. |
| von Kügelgen et al., NeurIPS 2021 | *Proves* SSL with data augmentations isolates content from style | Proof holds under assumed augmentation-as-style. **Nobody has tested it where style is a real jacket and content is a real change of gait.** |
| CEBaB / Causal Proxy Models | Real-world counterfactual concept effects | Text; human-authored edits; no composition; no vision. |

**Verdict.** Promoted to **#1**. This is the paper.

### Round 4 — Gait-Physics-IQ, and the objection that made it better

**Proposition (g-jepa P6):** matched physically-plausible / implausible gait pairs; a good world model should
show high prediction energy on the impossible member.

**Red team.** The g-jepa wiki names the killer objection itself: *"a stroke gait is abnormal but entirely
physical."* If your benchmark rewards surprise at abnormal gait, you have built an abnormality detector and
called it a physics test. Second objection: sim-to-real gap in the oracle. Third: it is 6–10 weeks of pipeline
work you do not have before September.

**The objection contains the contribution.** Every existing intuitive-physics benchmark — IntPhys 2
`[web, arXiv:2506.09849]`, Physion, X-VoE `[web, arXiv:2308.10441]`, GRASP, Physics-IQ — tests *objects*:
permanence, solidity, spatio-temporal continuity, immutability. Objects have no category of "unusual but
legal." **Human bodies do.** So the discriminating test is not impossible-vs-normal, it is:

```
   impossible          vs.    pathological-but-possible      vs.    typical
   (CoM outside BoS,          (hemiparetic, Parkinsonian,           (healthy
    ground penetration,        antalgic — from GAVD)                 gait)
    non-ballistic flight)
```

A model that scores impossible > pathological > typical understands balance physics. A model that scores
pathological ≈ impossible is an outlier detector. **That three-way structure is new, it is only possible
because human movement has a legal-abnormal class, and it is the thing that makes this more than
"IntPhys with people."**

I confirmed today that no human-balance intuitive-physics benchmark exists `[web]`; the search space returns
only object-physics benchmarks and unrelated OpenSim material.

**Verdict.** **#2** on merit, but it does not fit before September. Next cycle, or a parallel track.

### Round 5 — the ideas that lost

| Candidate | Killed because |
|---|---|
| Fall anticipation via violation-of-expectation | No falls in any dataset on disk. Field already crowded. |
| Hierarchical multi-timescale H-JEPA | No longitudinal data. Abstraction discovery unsolved. |
| Objective-driven fall prevention | Multi-year, regulated, no actuator. |
| Fold-local leakage gap (gavd4 P2) | "Everyone knows" objection needs a survey of N papers to answer; and it is a section of every idea below rather than a paper. **Adopt the protocol, drop the paper.** |
| Data fidelity vs. scale (GaitLU vs Health&Gait) | Confounded on five axes at once; and GaitSSB `[web, arXiv:2206.13964]` already published the "scale helps gait recognition" result using this exact corpus. Derivative by construction. |
| Multimodal privacy **distillation** | No non-vision modality on disk. The distillation half dies; the privacy half becomes **#3**. |

---

## Part 4 — The top five, ranked

Scoring: **Novelty** (is the claim unoccupied?), **Significance** (would the community change practice?),
**Feasibility** (can it be done, by this repo, on this hardware, by 2026-09-04?), **Branch-robustness** (do
*all* possible outcomes yield a paper?), **ICLR fit** (main-track representation-learning shaped?).
Weights: 25 / 25 / 25 / 15 / 10.

| # | Idea | Nov | Sig | Feas | Branch | Fit | **Score** |
|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **INTERVENE** — physically-assigned interventions as a probe of content–style separation | 8.5 | 9.0 | 9.0 | 9.5 | 9.0 | **8.9** |
| **2** | **BALANCE-IQ** — intuitive physics of human balance, with a pathological-but-possible control | 8.5 | 9.0 | 5.5 | 7.0 | 8.5 | **7.7** |
| **3** | **PRIVACY FRONTIER** — what identity-removal costs clinical signal, measured on co-registered renderings | 7.0 | 8.0 | 8.5 | 8.5 | 7.0 | **7.7** |
| **4** | **TARGET VOCABULARY** — anatomically-specified prediction targets as a controllable prior | 7.0 | 7.0 | 8.0 | 8.0 | 7.5 | **7.4** |
| **5** | **AXIS-RESOLVED DIAGNOSTICS** — silent failure in latent-prediction SSL | 6.5 | 7.5 | 9.5 | 8.5 | 7.0 | **7.7** |

Three ideas tie at 7.7 on different profiles. The tiebreak is **what each one needs from you**: #5 is nearly
written, #3 needs three loaders, #2 needs a simulator. So the *execution* order is 1 → 5 → 3 → 4 → 2, and the
*merit* order is 1 → 2 → {3,5} → 4.

---

### #1 · INTERVENE — *Do video representations encode how a body moves, or what it looks like?*

**Claim.** Given a representation `f` and three physically assigned interventions with known causal semantics,
one can measure — without any labels and without training `f` — whether `f` separates dynamics from
appearance, whether the intervention acts as a transportable operator across bodies, and whether two
interventions compose. On 398 real human bodies, current video encoders fail the first test in a specific and
memorable way.

**The predicted headline** (state it as a pre-registered prediction, then test it): *state-of-the-art video
encoders move further in representation space when a person puts on a jacket than when the same person
increases their walking speed by 40%.* If true, it is a clean, damaging, one-sentence result. If false, it is
the first empirical confirmation of content–style identifiability theory (von Kügelgen et al., NeurIPS 2021) on
real physical interventions — which is also a paper.

**Five measurements, all on frozen features, all linear algebra.**

1. **Selectivity matrix** `S[a,b]` = fraction of `Δ_a`'s energy lying in the subspace spanned by `{Δ_b}`.
   Diagonal = separation. Off-diagonal = entanglement. 3×3, one per encoder per rendering.
2. **Transport.** Estimate `Δ_speed` on donor subjects; apply to a *held-out* subject's UGS embedding; measure
   retrieval rank of their true FGS embedding among all held-out cells. Subject-disjoint by construction.
3. **Composition.** Is `Δ_speed + Δ_jacket` ≈ the true `(UGS,WoJ) → (FGS,WJ)` displacement? Do the operators
   commute? Is a learned `T_a` better approximated as identity-plus-low-rank, affine, or genuinely nonlinear?
   **This is the experiment nobody else can run.**
4. **Universality spectrum.** SVD of the 398 × D matrix of per-subject `Δ_speed`. v1 §2.5 predicts non-rank-1
   (Δ magnitude correlates with sex r = 0.264, height r = 0.247). Pre-register the prediction before running.
5. **Nuisance dominance.** `‖Δ_jacket‖ / ‖Δ_speed‖` and `‖Δ_view‖ / ‖Δ_speed‖`, per encoder. The headline number.

**Models** (rows of the main table): V-JEPA 2 and V-JEPA 2.1 `[web, arXiv:2603.14482]`, VideoMAE-v2,
InternVideo2, DINOv3 frame-pooled, GaitBase / DeepGaitV2, **GaitSSB** (pretrained on GaitLU-1M — this is where
GaitLU earns its keep, as a *checkpoint*, not as training data), cody-jepa's own best checkpoint, and a
random-init control (already implemented, `evaluation.py:38`).

**The input-domain problem, and its fix.** Frozen RGB video encoders on binary silhouettes is a large domain
gap and V-JEPA 2 may produce noise. **Do not fight this — use the four renderings.** DensePose IUV is a
smooth, coloured, textured RGB image `[disk]` and is by far the best input for a pretrained RGB encoder;
GMFlow and TVL1 flow are also RGB PNGs. So the design is
`{V-JEPA2, V-JEPA2.1, VideoMAE2, InternVideo2, DINOv3} × {DensePose, GMFlow, TVL1}` plus
`{GaitSSB, GaitBase, cody-jepa, random-init} × {silhouette}`. Report a domain-gap gate for every cell:
does the frozen encoder beat random-init on a trivial probe? If not, the cell is excluded and *that is
reported*.

**The method half** (so it is not only a benchmark): **intervention-equivariant fine-tuning.** Train a
lightweight operator head `T_a` on training subjects with an equivariance loss for `do(speed)` and an
invariance loss for `do(jacket)` / `do(view)`; show it drives the selectivity matrix toward diagonal and
improves clinical decodability on held-out subjects. This is SIE's mechanism pointed at physical rather than
synthetic transformations — a distinction the paper must own explicitly in related work.

**Compute.** ~3,130 recordings × ~3 windows × 3 renderings × ~9 encoders ≈ 250k clip forwards ≈ **6–10
GPU-hours total**. All downstream analysis is CPU linear algebra. **There is no pretraining on the critical
path.** For a five-week deadline against a repo whose training pipeline is the broken part, this is decisive.

**Weaknesses, stated plainly.** Only three factors and four cells — thin compositional space; mitigate with
graded speed (the video estimator gives continuous Δv, so `T_speed` can be a flow rather than a discrete step)
and an external replication on OU-ISIR Treadmill A (speed-varied gait). Health&Gait is 398 healthy adults
aged 19–64, so nothing here is a clinical claim. Frame rate is undocumented, so avoid absolute-time
statements or measure fps empirically and say how.

---

### #2 · BALANCE-IQ — *Does a video world model know balance physics?*

**Claim.** Intuitive-physics evaluation of world models has tested only object physics. Human balance is a
richer test because it admits a third category — abnormal but physically legal — that objects do not, and that
third category is exactly where a naive surprise-based model fails.

**Construction.** AMASS/SMPL mocap → matched triples, rendered identically (same body, same camera, same
texture; render to silhouette so appearance is exactly controlled and the inputs match Health&Gait):

- **Impossible**: centre-of-mass projection leaves the base of support with no compensatory step; foot
  penetrates the ground plane; flight-phase COM trajectory violates projectile motion; limb length changes
  mid-stride; angular momentum non-conservation during swing.
- **Pathological-but-possible**: real hemiparetic / Parkinsonian / antalgic gait from GAVD, or simulated from
  OpenSim with altered muscle parameters.
- **Typical**: healthy gait.

**Metrics.** Energy gap `E(impossible) − E(typical)`; **the discriminating gap** `E(impossible) −
E(pathological)`; and the correlation between the discriminating gap and downstream clinical decodability. A
model with a large first gap and a near-zero second gap is an outlier detector wearing a physics costume, and
naming that failure is worth the paper on its own.

**Why this is the best long-term bet.** It is the only idea here that is simultaneously (a) unoccupied in the
literature `[web, verified today]`, (b) natively in Scott Delp's toolchain — OpenSim *is* the oracle, and the
g-jepa corpus already identifies this as the concrete bridge to Mobilize — and (c) a genuine test of the LeCun
world-models program rather than an application of it.

**Why not now.** Building the simulator + renderer + violation library is 6–10 weeks and is a *new* pipeline,
not a modification of an existing one. Start it in parallel; target the cycle after.

---

### #3 · PRIVACY FRONTIER — *What does anonymization cost clinical signal?*

**Claim.** Health&Gait's four renderings are a ladder of privacy-preserving transforms of an *identical*
physical event `[disk]`. Measure, at every rung, both axes that matter for ambient healthcare:
**re-identification risk** (open-set subject retrieval — already implemented, currently 0.0245 vs 0.0129
chance `[disk]`) and **clinical utility** (decodability of the assigned speed condition, anthropometrics,
gait parameters). Produce the Pareto frontier.

**Grounding, verified today.** The privacy-video field is active — ReGenHuman `[web, arXiv:2606.14972]`,
Privacy Beyond Pixels `[web, arXiv:2511.08666]`, temporally-consistent token pruning `[web, arXiv:2603.26336]` —
but two findings define the gap:

1. *"Anonymization removes appearance but leaves gait identity largely intact, indicating that pose-driven
   anonymization is insufficient for privacy protection"* `[web, PMC12788357]`. The thing you are trying to
   preserve is the thing that identifies people. That tension has no measured frontier.
2. *"Existing anonymization methods, which typically use action recognition as the sole proxy utility task,
   significantly degrade performance on alternate downstream tasks"* `[web]`. **Nobody has used clinical gait
   function as the utility axis.**

Plus the structural advantage: existing work holds content only *approximately* fixed across anonymization
methods. Health&Gait's renderings are exactly co-registered — same frames, same camera, same instant.

**The method half.** An objective that is *equivariant to the intervention and invariant to identity* —
i.e. #1's operator machinery pointed at deployment. #1 and #3 share their entire infrastructure, which is why
#3 is cheap once #1 exists.

**Weakness.** 398 healthy adults is not a clinical utility axis. Add GAVD for pathology, and be explicit that
the frontier is measured on a healthy cohort.

---

### #4 · TARGET VOCABULARY — *What a JEPA is asked to predict determines what it encodes*

**Claim** (ported from gavd4's P1, upgraded to video). The target set is a first-class design variable, not a
sampling detail, and it steers representation content measurably — via an **encoding asymmetry**: variables
depending on target tokens become more linearly decodable than variables depending on context-only tokens,
though the encoder sees both.

**Why cody-jepa can do this better than gavd4 can.** Health&Gait's DensePose gives 24 per-pixel body-part
labels co-registered with the silhouettes `[disk]`. So the target arms are definable in *pixel* space:
`T-random` (current multiblock), `T-foreground`, `T-distal` (shank/foot), `T-proximal` (torso/pelvis),
`T-unilateral`, `T-swing-leg` (phase-dependent — the most interesting arm), each at **matched absolute token
count** so set choice is decoupled from difficulty. gavd4's own audit shows their mask ratio drifted with
curriculum stage; do not repeat that.

**This is also the fix for the verified failure.** 90.4% of masked target tokens are background and the
context-substitution gap is 3.6× *larger* on background than foreground `[disk]`. Anatomically-grounded
targets attack that directly.

**Red-team caveats to answer in the paper.** Semantic masking as a *method* is occupied — SemMAE
(NeurIPS 2022), AttMask, MaskSem, MAMP. The novelty must be the **asymmetry measurement**, not the masking.
And V-JEPA 2.1's dense predictive loss, where all tokens contribute `[web, arXiv:2603.14482]`, partially
addresses target vacuity by a different route — it must be an ablation arm, not an uncited neighbour.

---

### #5 · AXIS-RESOLVED DIAGNOSTICS — *Silent failure in latent-prediction SSL*

**Claim, narrowed to what survives red-teaming.** Not *"JEPAs fail"* but *"the health metrics JEPA papers
report are measured on an axis nobody evaluates."* Token-axis effective rank 381.6/384 while the clip-pooled
EMA representation the probes actually consume sits at 11.5/384 — a 33× discrepancy on the same forward pass
`[disk]`. Batch-standardized targets hide it further.

**The battery** (all already implemented or trivially restorable): axis-resolved effective rank; the
context-substitution ladder (self / same-subject / cross-subject / temporal-shuffle / blank) with
subject-level bootstrap; the foreground/background gap decomposition.

**The one novel fix.** `−λ·(blank_context_loss − loss)` as an auxiliary term: explicitly train the predictor
to be *worse* without its context. This converts the diagnostic into a regularizer, costs one extra forward
pass with `torch.zeros_like`, and I have not seen it proposed. Worth a section wherever it lands.

**Role.** Section 5 of #1 (it is how you show the benchmark discriminates: an encoder that fails the
diagnostics should also fail the selectivity matrix, and cody-jepa is the existence proof). Standalone
fallback if #1 stalls. Not the flagship — its generality is a single point of failure and I estimate it fails.

---

## Part 5 — The data strategy

### 5.1 GaitLU-1M: retire it

**Verdict: do not unpack the 51 GB.** Grounds, in order of force:

1. **64 × 44 binary** `[disk]`. At any reasonable patch size the person occupies a handful of tokens and the
   per-frame information content is roughly an outline. It cannot support a resolution-sensitive claim.
2. **No subject IDs, no clinical labels, no interventions, no anthropometrics.** Unlabelled is the point of the
   corpus. Every idea in Part 4 requires at least one of these.
3. **The one claim it can support is already published.** GaitSSB / *Learning Gait Representation from Massive
   Unlabelled Walking Videos* `[web, arXiv:2206.13964]` **is** the GaitLU-1M paper. Re-deriving "pretraining
   scale helps gait recognition" on the same corpus is derivative by construction.
4. **Cost.** 51 GB multipart zip plus an unwritten shard pipeline ≈ 4–8 elapsed days `[inherited]` for an axis
   that is not load-bearing under any of the five ideas.

**The one legitimate use:** the public **GaitSSB checkpoint**, pretrained on GaitLU-1M, is a strong and
appropriate row in #1's model table — it represents "trained at scale on exactly this input format." That
gives you GaitLU's value for the cost of a `wget`.

### 5.2 The replacement ladder

**Tier 1 — on disk, use now.** Health&Gait at native 960×540, **all four modalities**. The immediate repo work
is generalizing `build_manifest.py` past `--modality silhouette`; the tree grammar already supports the other
three `[disk]`, and pose needs a renderer to enter the same Conv3d patchifier.

**Tier 2 — acquire, ranked by value ÷ effort.**

| Dataset | What it buys | Which ideas | Effort |
|---|---|---|---|
| **OpenCap** (Uhlrich et al., PLOS Comp Biol 2023 — **Delp lab**) | ~100 subjects, two-smartphone video **with validated OpenSim 3D kinematics and kinetics**. Ground-truth biomechanics paired with video. | #1 external validity, #2 oracle, Delp bridge | 1–2 weeks |
| **AMASS** (arXiv:1904.03278) | 40+ h mocap in SMPL → unlimited controlled renders, exact counterfactuals, physics-violation pairs | #2 (required), #1 synthetic tier | 1–2 weeks |
| **GAVD** (arXiv:2407.04190) — already partially extracted in `alexpose` | 1,874 sequences, 400+ subjects, 8+ pathology classes at 1280×720 `[disk]` | #2's pathological-but-possible arm, #3's clinical utility axis | days (link rot expected — report yield) |

**Tier 3 — optional external validity.** OU-ISIR Treadmill A (speed-varied gait; a direct external replication
of `do(speed)`), CASIA-B / OU-MVLP (gait-recognition transfer baselines), MM-Fi (mmWave + video, if the
privacy story ever needs a non-vision modality).

**The principle behind the ladder:** each tier answers the standing objection to the tier below. Synthetic
gives exact ground truth but no external validity; Health&Gait gives real bodies with assigned factors but no
pathology; GAVD gives pathology but no control. Say this in the paper and the reviewer's objection is
pre-answered.

---

## Part 6 — Three-audience evaluation

### 6.1 ICLR 2027

| Idea | Novelty | Significance | Soundness risk | Verdict |
|---|---|---|---|---|
| #1 INTERVENE | **8.5** — the *real physical intervention* boundary is defensible against CausalVerse (sim), Delta Embeddings (computational), SIE (augmentations), CEBaB (text) | **9** — gives the community a measuring instrument and a memorable number | Low. Frozen features; no training on the critical path | **Submit this.** Main track, representation learning |
| #2 BALANCE-IQ | **8.5** — no human-balance intuitive-physics benchmark exists `[web]`; the pathological-but-possible arm is genuinely new | **9** | Medium-high: oracle fidelity, "impossible" boundary | Next cycle. D&B or main track |
| #3 PRIVACY | 7 — crowded field, but clinical-function-as-utility is unoccupied | 8 | Low | Main track or D&B |
| #4 TARGET VOCAB | 7 — asymmetry measurement is new; semantic masking is not | 7 | Medium: must ablate against V-JEPA 2.1 dense loss | Section of #1, or its own paper next cycle |
| #5 DIAGNOSTICS | 6.5 — RankMe / dimensional collapse are close | 7.5 | **Generality is a single point of failure** | Section 5 of #1; fallback paper |

**Title discipline for #1.** Lead with representation learning. "Gait" should not appear in the title, the
first two sentences of the abstract, or Figure 1 `[inherited, and I endorse]`. It is the testbed, not the
subject. Save the health framing for #3.

### 6.2 Stanford HAI — ambient intelligence

The reference frame is Haque, Milstein & Fei-Fei, *Illuminating the dark spaces of healthcare with ambient
intelligence*, Nature 585:193–202 (2020), which sets the contactless-sensing agenda **and** names the privacy
tension as the central obstacle.

| Idea | Fit | Why |
|---|---|---|
| **#3 PRIVACY** | **Highest** | This is a direct, quantitative answer to the exact tension the Nature review raises, on real co-registered data, with re-identification risk as an explicit measured axis. It is the HAI-facing paper. |
| **#1 INTERVENE** | **High** | Ambient monitoring is worthless if the representation tracks clothing rather than movement. `‖Δ_jacket‖ / ‖Δ_speed‖` is precisely the deployment-robustness question, and no in-home system has ever reported it. |
| #2 BALANCE-IQ | Medium-high | Trustworthiness of an anticipation model in a home is exactly the "surprised by the right things" question. |
| #4 TARGET VOCAB | Medium | Label-efficiency argument (their Gap 1). |
| #5 DIAGNOSTICS | Medium | Silent failure in a deployed health model is the nightmare scenario; a cheap battery is directly useful. |

The `g-jepa` corpus already did the gap analysis against the Li corpus — Gap 1 label hunger, Gap 2 pose
bottleneck, Gap 6 modality silos and privacy. **#1 and #3 together address Gaps 2 and 6.** That framing is
ready to use verbatim in an HAI proposal.

### 6.3 Scott Delp — balance assessment

This is the audience where the ranking changes most, and where one literature finding reframes everything.

**The finding, verified today.** *"Fast gait speed showed stronger associations and better predictive
capabilities compared with usual gait speed with physical performance measures and balance confidence in
community-dwelling older adults"* `[web, PubMed 37820362]`. And: *"step width variability was higher in the
fall-risk group under both speed conditions and showed **greater discriminative power at increased walking
speed**"* `[web, Sci Rep 2025 / PMC12081882]`.

**Consequence.** Health&Gait's `do(speed)` intervention is not merely a statistically convenient assigned
factor. **It is the clinical stress test that the balance literature says is more discriminative than usual
gait** — the capacity-under-load manipulation, sometimes called gait speed reserve. That means:

> the per-subject residual of the learned intervention operator — how *this* body reorganizes when asked to
> walk fast, beyond the population-average `Δ_speed` — is a principled candidate digital biomarker of motor
> reserve.

And #1's universality spectrum (measurement 4) is exactly the experiment that tells you whether such a
biomarker can exist: a rank-1 spectrum means everyone reorganizes the same way and there is no per-subject
signal; a high-rank spectrum means there is. v1 §2.5's sex and height correlations predict high rank. **Either
answer is a result Delp's group would want.**

| Idea | Fit | Why |
|---|---|---|
| **#2 BALANCE-IQ** | **Highest** | OpenSim *is* the oracle. Musculoskeletal simulation as ground truth for a world-model benchmark is his toolchain used for a new purpose. The strongest possible collaboration hook. |
| **#1 INTERVENE** | **High** | The gait-speed-reserve framing above; plus OpenCap gives validated 3D kinematics to check that the learned `Δ_speed` operator corresponds to real biomechanical change and not a rendering artifact. |
| #3 PRIVACY | Medium | Deployment-relevant, less biomechanically deep. |
| #4 TARGET VOCAB | Medium | Distal vs. proximal target arms map onto a real biomechanical hypothesis (distal control degrades first in ageing). |
| #5 DIAGNOSTICS | Low-medium | Methodological hygiene; not his question. |

**The honest limitation to lead with, not bury:** Health&Gait is 398 healthy adults aged 19–64. There is no
balance-impaired cohort, no falls, no elderly. Nothing in #1 is a clinical claim, and pretending otherwise
with this audience is the fastest way to lose them. The right framing is: *here is a validated measuring
instrument built on a healthy cohort; the clinical cohort is what a collaboration would supply.*

---

## Part 7 — What to do, in order

**Now → Aug 21 (the critical path for #1).**

1. Generalize `build_manifest.py` past `--modality silhouette` — DensePose, GMFlow, TVL1. The tree grammar
   already supports them `[disk]`. This is the single blocking task.
2. Build the **cell table**: subject × {UGS,FGS} × {WoJ,WJ} × {dir 1,2}. Audit the 27 missing cells
   (1,565 vs 398×4 = 1,592) `[inherited]` and report the completeness rate. A reviewer will check this.
3. Frozen-feature export for V-JEPA 2 / 2.1, VideoMAE-v2, InternVideo2, DINOv3 on the three RGB renderings.
   Domain-gap gate on every cell (beat random-init or be excluded and reported).
4. Compute the selectivity matrix, the nuisance-dominance ratio, and the Δ spectrum. **This is the go/no-go.**
   If `‖Δ_jacket‖/‖Δ_speed‖ > 1` for the major encoders, you have the headline.

**In parallel, cheap, and independently valuable.**

5. `evaluation.py:173` — pass `split="test"` to `probe_identity`. Identity is currently measured on training
   participants `[disk]`. There is no open-set biometric number in the repo until this is fixed.
6. Recording-level (not window-level) aggregation before scoring `[disk, evaluation.py:84]`; `--seed` on
   `train.py`; classical baselines (silhouette-area FFT, bbox-aspect autocorrelation, temporal
   self-similarity).
7. Port gavd4's exposure ledger and three-reference-line reporting.

**Deferred deliberately.** GaitLU unpacking (never); the phase-1 grid rerun (eleven variants inside ±2 SE
`[inherited]`); any Koopman/operator-dynamics paper (occupied, and the enabling repo work does not fit);
BALANCE-IQ's simulator (start it, but off the critical path).

**Pre-register before running #1.** Three predictions, written down first: (a) `‖Δ_jacket‖ > ‖Δ_speed‖` for
the majority of frozen encoders; (b) the per-subject `Δ_speed` spectrum is **not** rank-1; (c) the
selectivity matrix is closer to diagonal for encoders pretrained on natural video than for those pretrained
on silhouettes. Pre-registration is cheap here and it converts a fishing expedition into an experiment.

---

## Sources

**Verified today (web).**
CausalVerse — [arXiv:2510.14049](https://arxiv.org/pdf/2510.14049) ·
Learning Robust Intervention Representations with Delta Embeddings — [arXiv:2508.04492](https://arxiv.org/pdf/2508.04492) ·
SIE, Split Invariant-Equivariant SSL — [arXiv:2302.10283](https://arxiv.org/pdf/2302.10283) ·
IntPhys 2 — [arXiv:2506.09849](https://arxiv.org/abs/2506.09849) ·
X-VoE — [arXiv:2308.10441](https://arxiv.org/pdf/2308.10441) ·
V-JEPA 2.1, dense features — [arXiv:2603.14482](https://arxiv.org/html/2603.14482v2) ·
GaitSSB / GaitLU-1M — [arXiv:2206.13964](https://arxiv.org/pdf/2206.13964) ·
Privacy Beyond the Face (gait identity survives anonymization) — [PMC12788357](https://pmc.ncbi.nlm.nih.gov/articles/PMC12788357/) ·
Privacy Beyond Pixels — [arXiv:2511.08666](https://arxiv.org/pdf/2511.08666) ·
ReGenHuman — [arXiv:2606.14972](https://arxiv.org/html/2606.14972v1) ·
Token-pruning video anonymization — [arXiv:2603.26336](https://arxiv.org/pdf/2603.26336) ·
Fast vs. usual gait speed and balance confidence — [PubMed 37820362](https://pubmed.ncbi.nlm.nih.gov/37820362/) ·
Step-width variability at increased gait speed predicts falls — [Sci Rep 2025](https://www.nature.com/articles/s41598-025-02128-2) ·
Health&Gait — [Sci Data 2025](https://www.nature.com/articles/s41597-024-04327-4), [code](https://github.com/AVAuco/healthgait), [Zenodo](https://zenodo.org/record/14039922) ·
GAVD — [arXiv:2407.04190](https://arxiv.org/html/2407.04190v1) ·
Self-Supervised Learning of Gait-Based Biomarkers — [arXiv:2307.16321](https://arxiv.org/pdf/2307.16321)

**Cited from established literature (not re-fetched today):** von Kügelgen et al., *Self-Supervised Learning
with Data Augmentations Provably Isolates Content and Style* (NeurIPS 2021, arXiv:2106.04619) · I-JEPA
(CVPR 2023, arXiv:2301.08243) · V-JEPA 2 (arXiv:2506.09985) · VICReg (ICLR 2022, arXiv:2105.04906) · LeJEPA /
SIGReg (arXiv:2511.08544) · RankMe (ICML 2023, arXiv:2210.02885) · Jing et al., dimensional collapse
(ICLR 2022, arXiv:2110.09348) · CEBaB (NeurIPS 2022, arXiv:2205.14140) · Causal Proxy Models (ICML 2023) ·
SemMAE (NeurIPS 2022, arXiv:2206.10207) · MPI3D (NeurIPS 2019, arXiv:1906.03292) · CITRIS (ICML 2022,
arXiv:2202.03169) · AMASS (ICCV 2019, arXiv:1904.03278) · OpenCap (Uhlrich et al., PLOS Comp Biol 2023) ·
Garrido et al., intuitive physics from V-JEPA (arXiv:2502.11831) · Physics-IQ (arXiv:2501.09038) ·
Haque, Milstein & Fei-Fei, Nature 585:193–202 (2020) · Moscovich & Rosset (JRSS-B 2022)

**Repo-internal (recomputed today).** `configs/healthgait.json` · `cody_jepa/{masks,losses,models,evaluation}.py` ·
`outputs/phase0/baseline/{probes-best_loss.csv, context_use.metadata.json}` ·
`data/healthgait/raw/Health_Gait/**` (image geometry) · `gaitlu-000.html` (GaitLU frame geometry) ·
`~/dev/g-jepa/wiki/**` · `~/dev/alexpose/experiments/sjepa/gavd4-vicreg/{README.md, notes/11_iclr_paper_analysis.md}` ·
`~/dev/alexpose/data/GAVD_Clinical_Annotations_1*.csv`
