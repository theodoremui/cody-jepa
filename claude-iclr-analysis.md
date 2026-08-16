# cody-jepa → ICLR 2027: repo analysis and paper strategy

**Prepared:** 2026-08-12 · **Repo state:** `03f95ae` (main) plus `archive/pre-refactor` and the git-ignored `outputs/` tree
**Deadlines (verified):** abstract **Sep 18, 2026**, paper **Sep 25, 2026** (AoE) → experiments must freeze ~**Sep 4** → **16 working days**

Everything numeric below I re-derived from the repo's own files rather than quoting the existing
`REVIEW-2026-08-12.md` or the archived notes. Where I contradict or extend them, I say so.

---

## Part 1 — What the repo actually is

### 1.1 The method

A single-stream masked video JEPA, ~1,100 lines, no pixel reconstruction.

| Component | Implementation | Notes |
|---|---|---|
| Encoder | 6-layer ViT, `embed_dim=384`, 6 heads, Conv3d patchify | `models.py:85` |
| Token grid | `16 frames / tubelet 2` × `112 / patch 8` × `112 / patch 8` = **8 × 14 × 14 = 1,568 tokens** | `models.py:110` |
| Position | fixed 3D sin-cos, **absolute only, no relative/Δ encoding** | `models.py:51` |
| Predictor | 6-layer, `pred_dim=192`, per-mask-group learned mask token | `models.py:152` |
| Target | EMA copy, `τ: 0.99 → 1.0` linear, LayerNorm + **per-batch per-dim standardization** | `engine.py:51`, `losses.py:31` |
| Loss | L1 in embedding space + VICReg | `losses.py:7` |
| Masking | multi-block union on the **14×14 spatial grid**, replicated across all 8 temporal indices | `masks.py:41` |

Two facts about this design matter more than anything else downstream:

1. **Masking is spatial-only.** `_expand_tubes` takes each selected spatial cell and copies it to every
   temporal index. Context and target both span the full clip. The predictor is a spatial inpainter with
   bilateral temporal context; Δt is never an input and never a supervision signal. Maximum horizon is
   structurally 0.
2. **Targets are batch-standardized** (`target_batch_standardize: true`). Per-dimension whitening across the
   batch makes the *target* distribution look diverse regardless of what the encoder is doing. This is a
   second, independent reason the training-time health signals read green — I have not seen this pointed out
   anywhere in the repo's own notes.

### 1.2 The data

Health&Gait, on disk (~57 GB), and it is a better asset than the repo currently uses.

- **398 participants**, complete anthropometrics for 397, sex balanced exactly 199/199, age 19–64 (mean 42.0).
- Directory grammar is `<modality>/PA###/<UGS|FGS>/<WoJ|WJ>_<1|2>` → **2 assigned factors** (speed instruction,
  jacket) × **2 directions**. Note `source_video_id` in `build_manifest.py:53` collapses direction: the two
  directions are two segments of **one** walk. So the design is **4 assigned cells × 2 directional segments**,
  not a 2×2×2 of independently assigned factors. The archived note calls it "2×2×2 randomised"; that is
  half right and a reviewer will check it.
- Verified counts from `data/healthgait/diagnostics/healthgait_manifest_summary.csv`: **3,130 recordings,
  1,565 source videos, 398 participants**, 0 corrupt frames, frame counts 47–199 (mean 103).
  1,565 vs 398×4 = 1,592 → **27 cells are missing**; audit before claiming a complete factorial.
- **Four co-registered renderings of the identical physical events**: silhouette (YOLOv8 JPEG), semantic
  segmentation (DensePose PNG), optical flow (TVL1 **and** GMFlow PNG), pose (AlphaPose JSON). This is rare
  and it is the single most under-exploited asset in the repo.
- **Frame rate is unknown.** The Health&Gait README never states it — `-f fps` is an argument you supply to
  *their* estimation script. The current `build_manifest.py` has no fps flag at all. Any claim in absolute
  seconds (cadence, horizon, "0.53 s per clip") rests on an assumption, not a documented fact.

### 1.3 Current code state vs. `REVIEW-2026-08-12.md`

The review is a few hours stale. Most P0/P1 items are **already fixed** in the working tree:

| Review item | Status now |
|---|---|
| No person crop, 1.78× squash | **Fixed** — `data.py:126` foreground bbox over the window, `ImageOps.contain` + centered letterbox |
| VICReg on the wrong axis | **Fixed** — `engine.py:288` splits `clip_var_coef`/`clip_cov_coef` (default on) from `token_*` (default off) |
| No random-init control | **Fixed** — `evaluation.py:38` `build_random_target_encoder`, wired through `export_features.py` |
| Monitored ≠ evaluated representation | **Fixed** — `evaluate()` pools `target_encoder(..., return_pre_norm=True)`, same as probes; logs clip *and* token rank |
| No blank-context diagnostic | **Partially** — `blank_context_loss_delta` is in `evaluate()`; same-subject / cross-subject / temporal-shuffle are **still gone** |
| `--resume` restarts schedules | **Fixed** — `global_step`, `best_loss`, RNG states all checkpointed and restored |
| `torch.load(weights_only=False)` | **Fixed** — `safe_globals([TorchVersion])` |
| Trailing accumulation group mis-scaled | **Fixed** — `engine.py:368` now divides by `current_accumulation` |

**Still live, and each one will be caught in review:**

- `evaluation.py:173` — `run_probes` calls `probe_identity(table, seed, model=model)`; `split` defaults to
  `"train"`. **Identity is measured on training participants.** The held-out-subject retrieval probe is still
  deleted. The repo currently has **no open-set biometric measurement**.
- `evaluation.py:84` — `_metrics` counts rows, and `export_features.py` defaults to 3 windows per recording.
  Reported `val_examples` is ~3× the number of independent units. Any CI from it is ~√3 too narrow.
- `masks.py:84` — `rng.sample` trims to the batch minimum, so effective context size is batch-composition
  dependent. Adds variance; harmless for correctness.
- No `--seed` flag on `train.py`. Every number in the repo is single-seed.
- `configs/healthgait.json` declares `amp_dtype: bfloat16`, but `engine.py:33` silently returns
  `nullcontext` on MPS. The archived runs were on MPS, i.e. fp32, i.e. **the config does not describe them.**

---

## Part 2 — The verified empirical findings

I recomputed these from `outputs/phase0/baseline/context_use.csv` (18,720 rows) and the raw dataset CSVs.
They are the strongest thing in this repository and they all point the same way.

### 2.1 The predictor does not use its context

Mean prediction loss is **0.3868**. Substituting the context:

| Substitution | Δloss | as % of loss |
|---|---:|---:|
| Same clip (control) | 0.0 | 0% |
| Different trial, **same subject** | 1.596e-4 | 0.041% |
| **Different subject** | 1.563e-4 | 0.040% |
| Temporally shuffled frames | 4.716e-5 | 0.012% |
| **All-zero blank video** | 9.531e-5 | 0.025% |

Three readings, in ascending order of damage:

- Blanking the entire input changes the objective by **0.025%**. The predictor is, to three decimal places,
  a function of position alone.
- same-subject − cross-subject = **3.3e-6**. The model cannot distinguish whose gait it is looking at,
  measured across 80 held-out subjects.
- **Temporal shuffling hurts *half* as much as blanking.** Destroying all temporal order is *less* damaging
  than removing the input entirely. This is the direct empirical signature of §1.1's spatial-only masking:
  the co-temporal neighbours the predictor actually leans on survive a frame permutation.

### 2.2 The context is used to predict emptiness

From `context_use.metadata.json` `summary/d2` — a decomposition I have not seen quoted anywhere:

| Token type | Substitution gap |
|---|---:|
| Background tokens | 1.679e-4 |
| **Foreground tokens** | **4.668e-5** |
| foreground / all ratio | **0.299** |

The gap on *background* tokens is **3.6× larger** than on foreground tokens. Whatever small use the model
makes of its context, it makes it to predict blank regions — not the person. And there are a lot of blank
regions: mean foreground fraction of masked target tokens is **9.62%**.

### 2.3 The anti-collapse metric is measured on an axis nobody evaluates

`summary/d3`, same checkpoint, same forward pass:

| Representation | Effective rank / 384 | Ratio |
|---|---:|---:|
| Context tokens, all mask groups | **381.6** | 99.4% |
| Context tokens, `small_blocks` | 380.7 | 99.1% |
| Context tokens, `large_blocks` | 380.8 | 99.2% |
| Online encoder, clip-pooled | 10.4 | 2.7% |
| **EMA target, pre-norm, clip-pooled — the exact representation the probes consume** | **11.5** | **3.0%** |

A **33× discrepancy between the axis the regularizer guards and the axis the evaluation reads.**
VICReg reports a perfect score while the evaluated representation lives in 11 of 384 dimensions.

### 2.4 The downstream numbers are consistent with all of the above

From `outputs/phase0/baseline/probes-best_loss.csv` and `outputs/phase1/a00-baseline/probes.csv`:

| Probe | Baseline | a00 | Chance |
|---|---:|---:|---:|
| `gait_system` (2-class, subject-held-out) | 0.926 | 0.933 | 0.500 |
| identity, closed-set (318 classes) | 0.093 | 0.106 | 0.0032 |
| **identity, held-out-subject retrieval (80 classes)** | **0.0245** | **0.0294** | **0.0129** |

The retrieval number is the honest biometric measure and it is **1.9–2.3× chance**. There is no working
biometric in any checkpoint. Meanwhile all eleven `phase1` variants land at gait 0.88–0.94: with 80 test
participants, the binomial SE on a 2-class score is ~5.6 pp, so **the entire eleven-run grid is contained
within ±2 SE**. It was underpowered by construction, independently of the failure mode.

### 2.5 The instrumented labels are unusable — but the *assigned factor* is excellent

Recomputed on `gait_parameters.csv` (n=398) and the authors' own AlphaPose estimates:

**Labels are internally inconsistent.** `Velocity_UGS` spans 0.93–**5.09 m/s** (5 m/s is sprinting, for
*usual* gait). `Cadence_UGS` spans 49–223 steps/min. The parse is right — `Step_UGS` vs `Stride_UGS` is
r = 0.962 — so the inconsistency is real:

| Pair | Should be | Label r | Authors' video-estimator r |
|---|---|---:|---:|
| Cadence UGS vs FGS (stable trait) | high | **0.168** | **0.376** |
| Velocity UGS vs FGS | high | **0.228** | **0.355** |
| Cadence vs Velocity, same condition (algebraically linked) | high | **0.278** | **0.634** |

**A video pipeline is more self-consistent than the instrument it is being scored against.** Label-vs-video
agreement is r = 0.393 (Velocity_UGS), 0.374 (Cadence_UGS), 0.107 (Stride_FGS).

**But the intervention itself is near-perfect.** Under the fast-gait instruction:

- **98.2%** of participants increase velocity by the OptoGait labels
- **99.7%** by the video estimator
- **100.0%** increase cadence by the video estimator

> **This is the pivot of the entire strategy: the *measured label* has reliability ≈ 0.14; the *assigned
> factor* has validity ≈ 1.0.** Any paper anchored on regressing OptoGait parameters is capped at
> r ≈ 0.37 and will be beaten by a 30-line periodogram. Any paper anchored on the assigned contrast is
> immune to the whole problem.

One further finding I did not see in the notes, and it is a live prediction: the *within-subject*
speed effect size correlates with sex (r = +0.264) and height (r = +0.247), while *between-subject*
label velocity correlates with neither. **The magnitude of the intervention is itself subject-dependent** —
which predicts the per-subject intervention-direction matrix is *not* rank-1.

---

## Part 3 — Three candidate papers, adversarially assessed

### Candidate A — Counterfactual probing: do observational concept directions transport real interventions?

*(the archived `iclr2027-direction.md` flagship)*

Fit `w_obs` observationally across subjects; compute `Δ_int` as the within-subject paired difference under the
assigned factor; measure `cos(w_obs, Δ_int)`, transport from donor to query subjects, universality via the
singular spectrum of the 398×D matrix of per-subject `Δ_int`, and selectivity across factors whose true causal
structure is known a priori (jacket = appearance not dynamics; direction = viewpoint not dynamics; speed = dynamics).

- **Novelty 6/10.** The idea is right but **CEBaB (NeurIPS 2022)** already built a real-world counterfactual
  benchmark for concept effects, and **Causal Proxy Models (ICML 2023)** followed it. Positioning must be
  narrow and honest: *first with **physical** interventions rather than human-authored text edits, in vision,
  on self-supervised representations.* "First validation of the linear representation hypothesis against real
  counterfactuals" is not available as stated.
- **Feasibility 9/10.** Zero pretraining. Frozen features + linear algebra. Runs on CPU while the GPUs do
  something else.
- **Real weaknesses:** only **two** cleanly assigned factors (direction is a within-video viewpoint split, not
  an assignment); the section-2.5 finding above predicts non-rank-1, i.e. the "universality" experiment
  probably returns the messier answer; and audit papers cap around borderline unless the proposed
  intervention-aligned adapter actually works.

### Candidate B — Channel entropy as a precondition for joint-embedding prediction

*(the archived `iclr2027-channel-entropy-proposal.md`)*

Claim: latent prediction is powered by the nuisance variability it is supposed to discard. Strip appearance
and the mechanism fails silently. Test across the four Health&Gait renderings of identical events.

- **Novelty 8/10.** I found no direct prior art. Closest: InfoMin (NeurIPS 2020), *On Pretraining Data
  Diversity for SSL* (ECCV 2024), RankMe (ICML 2023), the dimensional-collapse line (Jing et al., ICLR 2022).
  None makes this claim for *predictive* objectives, and it reverses InfoMin's prescription.
- **Feasibility 6/10.** Needs manifests + loaders for three more modalities and ~30–40 training runs.
- **The design flaw nobody flagged:** the four channels differ in *resolution, codec, extractor and
  preprocessing artifacts*, not only entropy. A reviewer will say you varied four things at once. And pose
  is JSON keypoints — it cannot enter the same Conv3d patchifier without being rendered, which changes the
  channel again. **Fixable, and the fix is the best part of the idea** — see §4.2.

### Candidate C — Silent failure in latent-prediction SSL: an axis-resolved diagnostic, a mechanism, and a fix

*(mine; it absorbs B as its mechanism section)*

The framing inverts the repo's situation. The eleven collapsed checkpoints stop being an embarrassment and
become **Table 1**. The headline is not "our gait model failed" — it is:

> Latent-prediction SSL can fail in a way that every standard health metric reports as success. We give
> three cheap diagnostics that catch it, a mechanism that explains it, and a fix.

Three named failure modes, each with a verified number from §2 already in hand:

1. **Axis-mismatched anti-collapse** — token rank 381/384 while the *evaluated* clip-pooled rank is 11/384
   (§2.3). VICReg's guarantee holds on an axis nobody probes. Batch-standardized targets (§1.1) hide it further.
2. **Context-independent prediction** — blanking the input costs 0.025% of the loss; cross- minus
   same-subject is 3.3e-6 (§2.1). The predictor degenerates to a positional prior.
3. **Target vacuity** — 90.4% of masked targets are background, and the context-substitution gap is 3.6×
   *larger* on background than foreground tokens (§2.2). The pretext task is mostly "predict that this region
   is blank," which is solvable from position alone. This is the *cause* of 1 and 2.

- **Novelty 7–8/10** for the diagnostic + mechanism combination; RankMe is the nearest neighbour and it is
  about rank as a *predictor of transfer*, not about axis mismatch as a *failure mode*.
- **Feasibility 8/10.** The headline numbers exist today. Everything else is additive.
- **Contribution to ICLR:** an adoptable practice — *report axis-resolved rank and a context-substitution gap
  in every JEPA paper* — plus a fix with a measurable effect.

**Why C over B.** In C, the entropy ladder is the mechanism *section*. If the ladder comes back messy, the
paper still stands on §2's verified diagnostics. In B, the ladder **is** the paper. With 16 days, that
difference is the whole decision.

**Why C over A.** A is safer but lower-ceiling and its novelty is partly pre-claimed by CEBaB. C makes the
existing compute, the existing checkpoints and the existing dataset all load-bearing.

---

## Part 4 — Recommended plan

**Primary: C.** **Hedge: A's day-1 pilot**, which is nearly free and runs on CPU.
Decision point **Aug 21**: if C's causal ladder (§4.2) has not produced a monotone trend by then, A becomes
the paper and C's §2 numbers become A's Section 5.

### 4.1 Paper skeleton

| § | Content | Evidence status |
|---|---|---|
| 1 | Three silent failure modes, named and defined | Written |
| 2 | Diagnostic battery: axis-resolved rank; context-substitution ladder; foreground/background gap decomposition | **Numbers in hand** (§2.1–2.3) |
| 3 | Mechanism: target vacuity → positional prior is near-optimal → no gradient pressure on context → clip-axis collapse, invisible to token-axis VICReg | Needs §4.2 |
| 4 | Causal test: controlled entropy ladder | To run |
| 5 | Generality: battery on public checkpoints, and a second dataset | To run |
| 6 | Fixes and their effect | To run |

### 4.2 The one experimental design change that makes this work

**Do not compare the four Health&Gait channels as your causal evidence.** They confound entropy with
extractor, codec and resolution. Instead build a **monotone entropy ladder inside a single channel**, holding
content, camera, participants and cells exactly fixed:

```
DensePose IUV (24 part labels + UV)     ← highest entropy
  → 24-way part label map, no UV
    → 6-way coarse part map
      → 2-way binary silhouette          ← the current input
        → rendered stick figure from AlphaPose keypoints
          → binary silhouette + calibrated additive texture noise  ← entropy raised back up
```

Every rung is derived from the *same* frames by a deterministic transform you write, so entropy is the only
thing that moves — and the last rung is the crucial one, because it raises entropy **without adding any
task-relevant information**. If collapse tracks entropy, that rung recovers; if it tracks task information,
it does not. That single contrast is the paper's causal claim, and it is 40 lines of preprocessing.

**Mediation, not correlation.** Measure, at initialization and before any training, the clip-pooled effective
rank of the *frozen randomly-initialized* target encoder on each rung. That is the mediator: channel entropy →
initial target diversity → final clip-axis rank → probe accuracy. Fit it as a path model with a permutation
null. This converts a comparison of runs into a mechanism.

### 4.3 The fixes (Section 6 — the part that makes it a method paper, not a complaint)

Each is small, each is independently testable, each targets one failure mode:

1. **Informativeness-weighted targets.** Reweight or resample masked target tokens by foreground content.
   Directly attacks target vacuity. ~15 lines in `masks.py` + `losses.py`.
2. **Clip-axis variance term.** Already implemented (`clip_var_coef`) but never ablated against `token_var_coef`
   in a controlled run. Two configs.
3. **Explicit context-dependence objective.** Add `−λ·(blank_context_loss − loss)` as an auxiliary term: train
   the predictor to be *worse* when its context is removed. This is the novel one — it converts the diagnostic
   into a regularizer, and I have not seen it proposed. Cheap: one extra forward pass with `torch.zeros_like`.
4. **Temporal masking.** Sample masks in the 8×14×14 volume rather than replicating spatially. Fixes the
   embarrassment in §2.1 (shuffle < blank). ~10 lines in `masks.py:41`.

### 4.4 Non-negotiables (inherited, and I endorse every one)

1. Participant-disjoint splits, stated in the main text.
2. A **random-init encoder** row in every table (already implemented — use it).
3. A **non-learned classical baseline** in every table: silhouette-area FFT, bbox-aspect autocorrelation,
   temporal self-similarity matrix. One afternoon; every reviewer asks.
4. **≥3 seeds** on every claim. Add `--seed` to `train.py` first.
5. **≥2 public-weight encoders** in the generality section — V-JEPA 2 ViT-L, I-JEPA or VideoMAE-v2 — running
   the same diagnostic battery. This is what answers "does this matter outside your broken 6-layer model?"
6. **≥1 replication dataset.** Cheapest credible: take a public RGB video subset, run the same silhouette
   extractor, and show the same signature appears in RGB→silhouette *within one dataset*. CASIA-B if
   registration lands in time.
7. Recording-level (not window-level) statistics everywhere.
8. The word "gait" does not appear in the title, the first two sentences of the abstract, or Figure 1.

### 4.5 Schedule

| Dates | Work | Gate |
|---|---|---|
| **Aug 12–14** | Restore full context-substitution harness from `archive/pre-refactor`; fix `probe_identity` split default; restore held-out retrieval probe; add `--seed`; recording-level stats; classical baseline battery. Run A's pilot: frozen V-JEPA 2 features on all cells → `cos(w_obs, Δ_int)` + per-subject Δ spectrum. | Diagnostics reproduce on a current checkpoint |
| **Aug 17–21** | Build the entropy ladder (§4.2) and manifests. Run ladder × 3 seeds. Mediation analysis. Diagnostic battery on V-JEPA 2 / I-JEPA / VideoMAE-v2 frozen. | **Aug 21 go/no-go: is the ladder monotone?** |
| **Aug 24–28** | Fixes 1–4, ablated on the two extreme rungs × 3 seeds. Replication dataset. Random-init and classical baselines everywhere. | Fixes move clip-rank *and* probe accuracy |
| **Aug 31–Sep 4** | Freeze. Figures, robustness, pre-registered claim boundaries. | Experiments stop |
| Sep 5–17 | Writing | |
| Sep 18 / Sep 25 | Abstract / paper | |

**Compute:** the model is small — 6 layers, 384-dim, 1,568 tokens, 3,900 steps ≈ 2–4 A100-hours per run.
Ladder (6 rungs × 3 seeds = 18) + fixes (4 × 2 rungs × 3 seeds = 24) ≈ 42 runs ≈ **85–170 GPU-hours**,
comfortably inside the ~708 available. **GaitLU-1M stays unopened** — 52 GB of multi-part ZIPs plus an
unwritten shard pipeline is 4–8 elapsed days for an axis that is not load-bearing. Do not touch it.

---

## Part 5 — Repo work required, in order

**Before any experiment (Aug 12–14):**

1. `git show archive/pre-refactor:src/cody_jepa/training/diagnostics.py` → restore as `cody_jepa/diagnostics.py`.
   Wire the same-subject / cross-subject / temporal-shuffle conditions and the foreground/background
   decomposition into `evaluate()`, not a side script. This is the paper's Section 2.
2. `evaluation.py:173` — pass `split="test"` to `probe_identity`, and restore the held-out-subject
   nearest-centroid retrieval probe. Without it there is no honest biometric number.
3. `evaluation.py:84` — aggregate to recording level before scoring; report unit counts explicitly.
4. `train.py` — add `--seed`; thread it into config, dataset, mask RNG and torch.
5. Classical baseline script: silhouette-area FFT, bbox-aspect autocorrelation, temporal self-similarity.

**For the mechanism (Aug 17–21):**

6. `build_manifest.py` — generalize past `--modality silhouette`; the tree grammar already supports the
   other three, but pose is JSON and needs a renderer.
7. New `cody_jepa/channels.py` — the six deterministic entropy-ladder transforms of §4.2, each with a
   measured empirical entropy so the ladder is a *measured* axis, not an asserted one.
8. `masks.py:41` — volumetric masking behind a config flag (fix 4).
9. `losses.py` — informativeness weighting (fix 1) and the context-dependence auxiliary (fix 3).

---

## Part 6 — Risks, and what kills each paper

| Risk | Which paper | Mitigation / kill criterion |
|---|---|---|
| "This is a paper about one broken small model" | C | **Mandatory**: public-checkpoint battery + replication dataset. If the signature appears in neither, C is dead — fall back to A on Aug 21. |
| Entropy ladder is non-monotone | C | Paper survives on §2 diagnostics + fixes; mechanism section becomes "we could not isolate entropy," reported honestly. Reviewers accept this if §2 and §6 are strong. |
| CEBaB pre-empts the framing | A | Reposition to *physical* interventions on *self-supervised* representations; cite CEBaB and Causal Proxy Models in the first paragraph, not the related-work dump. |
| Per-subject Δ spectrum is flat | A | This is a **result**, not a failure — and §2.5's sex/height correlation predicts it. Pre-register the prediction before running. |
| OptoGait labels used anywhere as an anchor | any | Don't. Reliability ≈ 0.14, ceiling r ≈ 0.37, already saturated by a heuristic. Report it as *methodology* (measurement reliability in representation evaluation), never as an attack on the dataset authors. |
| Archived diagnostics describe pre-fix checkpoints | C | **True and must be disclosed.** The §2 numbers come from a model trained with the uncropped pipeline. Re-run the battery on a post-fix checkpoint in week 1 and report both. |
| fps unknown | any temporal claim | Avoid absolute-time claims, or measure fps empirically from the media and state the method. |

## Part 7 — What not to do

- **Do not open GaitLU-1M.** 4–8 elapsed days for a non-load-bearing axis.
- **Do not rerun the phase1 grid.** Eleven variants inside ±2 SE of each other; it varied parameters that
  cannot touch the failure mode, and it would find nothing again.
- **Do not build a Koopman / operator / phase-composition paper this cycle.** Koopman-JEPA (AAAI 2026) and
  Koopman Dreamer (Jul 2026) already occupy it, and the enabling repo work — temporal masking, relative Δ
  encoding, ≥32-frame clips — does not fit in 16 days. Next cycle's project.
- **Do not claim "first large-scale SSL on gait silhouettes."** FoundationGait (arXiv 2512.00691) took it.
- **Do not report a `gait_system` accuracy without a random-init row next to it.** A 2-class score of 0.93
  that is flat across every variant is not evidence of representation quality, and the control now exists
  in the code.

---

## Sources

- [ICLR 2027 Dates and Deadlines](https://iclr.cc/Conferences/2027/Dates)
- [Health & Gait: a dataset for gait-based analysis (Scientific Data, 2025)](https://www.nature.com/articles/s41597-024-04327-4) · [Zenodo record](https://zenodo.org/records/14039922) · [code](https://github.com/AVAuco/healthgait)
- [CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior (NeurIPS 2022)](https://arxiv.org/abs/2205.14140)
- [Causal Proxy Models for Concept-Based Model Explanations (ICML 2023)](https://proceedings.mlr.press/v202/wu23b/wu23b.pdf)
- [RankMe: Assessing Downstream Performance of Pretrained SSL Representations by Their Rank (ICML 2023)](https://arxiv.org/html/2210.02885)
- [VICReg (ICLR 2022)](https://arxiv.org/pdf/2105.04906)
- [I-JEPA (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf) · [V-JEPA (ICLR 2024)](https://openreview.net/pdf?id=WFYbBOEOtv) · [V-JEPA 2](https://arxiv.org/html/2506.09985v1) · [vjepa2 weights/code](https://github.com/facebookresearch/vjepa2)
- [The JEPA Predictor: A Transferable Operator for Occluded Feature Completion (2026)](https://arxiv.org/html/2607.16274v1)
- [Enhancing JEPAs with Spatial Conditioning](https://arxiv.org/html/2410.10773v1)
- Repo-internal: `REVIEW-2026-08-12.md`, `archive/pre-refactor:notes/iclr2027-direction.md`, `archive/pre-refactor:notes/iclr2027-channel-entropy-proposal.md`, `archive/pre-refactor:notes/eval-horizon-resonance.md`, `outputs/phase0/baseline/context_use.{csv,metadata.json}`, `outputs/phase1/*/probes.csv`, `data/healthgait/raw/Health_Gait/{gait_parameters,gait_parameters_estimation,patients_measures}.csv`, `data/healthgait/diagnostics/healthgait_manifest_summary.csv`
