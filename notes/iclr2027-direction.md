# ICLR 2027 direction review — brainstorm, adversarial review, and recommendation

Date: 2026-08-12. Deadlines: abstract Sep 18, paper Sep 25. Experiments must stop ~Sep 4,
so there are roughly **16 working days of experiments**.

Three independent adversarial reviewers were run: a novelty hunter (prior-art search),
a feasibility/statistics skeptic (which actually executed pilot analyses on the repo data),
and a simulated ICLR review panel. This document is the synthesis and the recommendation.

---

## 1. Facts that were verified, and that change the plan

These are not opinions. Each was checked against the repo or the shipped data.

### 1.1 The Health&Gait instrumented labels are mostly noise (verified directly)

Correlations computed on `data/healthgait/raw/Health_Gait/gait_parameters.csv` (n=398):

| Pair | r | What it should be |
| --- | ---: | --- |
| `Step_UGS` vs `Stride_UGS` | **0.962** | ~1.0 — sanity check, columns parse correctly |
| `Cadence_UGS` vs `Cadence_FGS` | **0.168** | high; cadence is a stable individual trait |
| `Velocity_UGS` vs `Velocity_FGS` | **0.228** | high, same reason |
| `Cadence_UGS` vs `Velocity_UGS` | **0.278** | high; `v = cadence x step_length / 2` by definition |

Ranges are physiologically impossible: `Velocity_UGS` spans 0.93–**5.09 m/s** (5 m/s is
sprinting, for *usual* gait speed); `Cadence_UGS` spans 49–223 steps/min. The Step/Stride
check at r=0.962 proves the parsing is right, so the low correlations are real.

The feasibility reviewer went further and estimated the reliability ceiling. A 30-line
non-learned baseline (height-normalised leg-spread width -> detrend -> Hann -> periodogram
peak) reaches r = 0.37 with `Cadence_UGS` **with zero calibration**, and matches the
dataset authors' own AlphaPose-based estimator (`gait_parameters_estimation.csv`, r = 0.374).
Split-half over independent videos of the same subject gives estimator reliability 0.936,
and the two independent video pipelines agree with **each other** at r = 0.946. By the
attenuation formula this implies the OptoGait label reliability is ~**0.139**, i.e. the
maximum attainable correlation for *any* perfect video method is r ~ 0.372, R^2 ~ 0.14 —
and that ceiling is already saturated by a heuristic and by published prior art shipped in
the same folder.

**Consequence: no paper may use the OptoGait gait parameters as its validation anchor.**
Verify the split-half numbers yourself before citing them, but the internal inconsistencies
above are enough on their own.

### 1.2 The predictor was never trained on a temporal task

`src/cody_jepa/masks/multiblock.py` samples masks in the 14x14 **spatial** grid and
replicates each selected cell across all 8 temporal indices. Context and target both span
the full temporal extent of every clip. The predictor is a spatial inpainter with bilateral
temporal context; it has never been asked what happens at `t+delta` given `t`.

`predictor.py` registers `pos` as a fixed absolute `sincos_3d_position_embedding(8,14,14)`
buffer with no relative encoding, so `delta` is not a model parameter and the maximum
horizon is 7 tubelets = **0.47 s**.

`results/phase1_summary.csv`: `wrong_context_gap ~ 1e-4` on the baseline runs — shuffling
the context changes prediction loss by 0.03%. The predictor is nearly a constant map.

### 1.3 The representations are collapsed

Effective rank 5–11 out of 384 on 9 of 11 trained runs, against the repo's own
`min_effective_rank_ratio: 0.05` gate. Only `b02` (clip-level VICReg, rank 75) passes, and
it halves identity accuracy. Best held-out identity retrieval across all checkpoints is
**4.8%** — there is no working biometric in any current model.

### 1.4 One clip is about one step, not one stride

`num_frames=16, tubelet_size=2` -> 8 temporal tokens = 0.533 s at 30 fps. Median cadence
116 steps/min -> step period 0.517 s -> **0.52 gait cycles per clip**. Left/right asymmetry,
the most clinically informative gait feature, is structurally invisible. Also, `fps=30.0`
is a value typed into `build_manifest.py --fps`; the Health&Gait documentation never states
a frame rate.

### 1.5 GaitLU-1M is not prepared

`data/gaitlu-1m/` holds ~52 GB of unopened multi-part ZIPs (~2.04M entries, ~1.02M `.pkl`).
`slurm/prepare-gaitlu-shards.sbatch` expects `gaitlu-000.tar.gz … gaitlu-099.tar.gz`, which
do not exist and which no script in the repo produces. Extraction + repack is realistically
4–8 elapsed days including queue time and one retry.

### 1.6 The planned 28-model study is a designed-in null

`results/gfc-a00-baseline-raw-retain-all/summary.json`: `learned_minus_shortcut = 0.0433`,
95% CI **[0.0027, 0.0822]**, n=76 participants. `configs/eval/gfc_healthgait.json` declares
a powered effect of **0.0625** — larger than the entire learned-vs-shortcut signal a fully
trained model produces. The confirmatory contrast is a difference of differences *inside*
that 0.043 margin, on n=8 blocks. It cannot resolve. Also, the `confirmation` split declared
in the eval config does not exist in `silhouette_gfc_candidate_seed0.csv` (only `train`/`val`).

### 1.7 Silhouettes are uncropped

`dataset.py` resizes raw 960x540 frames to 112x112 with no person crop or tracking
normalisation. Mean foreground area ~3.1%; signed horizontal centroid drift ~0.96 frame
widths per recording. Any latent trajectory is dominated by translation and apparent-scale
change, not limb oscillation.

---

## 2. Adversarial verdicts on the six candidate ideas

| Idea | Novelty | Feasibility | Panel decision | Most damaging citation |
| --- | ---: | ---: | --- | --- |
| I1 operator/spectral probing | 6/10 | 3/10 | borderline, lean reject | Koopman Invariants in JEPAs, **AAAI 2026** (arXiv 2511.09783) |
| I2 universal gait manifold | 3/10 | 3/10 | reject | DeepPhase, SIGGRAPH 2022 **Best Paper** |
| I3 phase-composition SSL | 4/10 | 2/10 | borderline, lean accept if scoped | Koopman Dreamer, arXiv 2607.19719 (**Jul 2026**) |
| I4 orbit-quotient descriptors | 2/10 | 6/10 | borderline | DFT temporal modelling, arXiv 1603.06182 (**2016**) |
| I5 identity/health frontier | 5/10 | 2/10 | reject | GDN, Pattern Recognition 2026 (Koopman static/dynamic split) |
| I6 iso-catalog allocation | 4/10 | 3/10 | reject (unanimous) | Pretraining Data Diversity for SSL, ECCV 2024 |

Three specific kills worth internalising:

- **I1's headline is a tautology.** If the input is periodic with period T, the latent
  trajectory of *any* deterministic encoder — including a random-init one — is periodic
  with period T, so the dominant eigenvalue angle *must* be 2*pi/T. Recovering cadence from
  a spectrum is not evidence of learning. Any version of I1 needs random-init and
  raw-pixel-FFT rows in Table 1 on day one.
- **I3's diagnosis is factually wrong for 2026.** V-JEPA 2, TD-JEPA and Temporal-Distance
  JEPA all use k-step recursive consistency. "JEPA predictors are never required to be
  consistent operators" will be corrected in a reviewer's first paragraph. And Koopman
  Dreamer (three weeks old) already ships 2x2 rotation–scaling blocks with bounded radii
  plus one-step consistency and multi-step rollout losses.
- **I6 costs your whole compute budget on the least novel idea**, uses a bespoke unpublished
  metric, and is statistically guaranteed to report a null.

Also correct one premise: **FoundationGait** (arXiv 2512.00691) pretrains a silhouette gait
foundation model on a 12-dataset corpus that includes GaitLU-1M. "First large-scale SSL on
gait silhouettes" is no longer available as positioning.

---

## 3. Recommended flagship

### Counterfactual probing: do observational concept directions transport real interventions?

**The gap.** Everything the field claims about concept directions, steering vectors,
disentanglement and linear probes is fit on *observational* variation across examples. The
interpretability literature openly concedes that probes are correlational (see the causal
probing line: RLACE, amnesic probing, "How Reliable are Causal Probing Interventions?",
IJCNLP 2025). The field's response has been to intervene **on the representation**, because
you cannot re-run reality with one factor changed. In vision this has never been tested
against a real-world intervention at all.

**The asset — and this, not OptoGait, is what makes this data special.** Health&Gait is a
**within-subject randomised 2x2x2 design**: the same person, same camera, same session,
walking at instructed usual vs fast speed, with and without a jacket, in each direction.
398 participants x 8 cells. These are *assigned* experimental factors, not measured labels,
so they are completely immune to the label-noise problem in §1.1 — and the manipulation
demonstrably worked at the population level (median velocity 1.55 -> 2.28 m/s).

This gives something almost no vision dataset offers: **genuine paired counterfactuals**.

**The experiments.**

1. **Alignment.** For each factor f, compute `w_obs` (probe direction fit across subjects)
   and `Delta_int` (mean within-subject paired difference). Report `cos(w_obs, Delta_int)`,
   and the harder *transport* test: take `Delta_int` estimated from held-out donor subjects,
   apply it to a query, and retrieve the correct factorial cell. **This is exactly what the
   existing GFC evaluator already does** — the infrastructure is written.
2. **Universality (the linear representation hypothesis, tested with real counterfactuals).**
   Is there one speed direction, or 398? Take the 398 x D matrix of per-subject `Delta_int`,
   look at its singular spectrum and participation ratio against a permutation null. A rank-1
   spectrum means a universal concept direction exists; a flat spectrum means concept
   directions are subject-conditional. **Neither answer is currently known, and both are
   publishable.**
3. **Selectivity, with a known causal structure.** Jacket changes appearance but not
   dynamics. Direction changes viewpoint but not dynamics. Speed changes dynamics. So you
   have factors whose true entanglement structure is known a priori — a rare calibration
   target for any disentanglement metric. Measure the interference between `Delta_speed`,
   `Delta_jacket`, `Delta_direction`.
4. **Reliability layer.** ICC(2,1) of the representation across the 8 repeated cells,
   minimal detectable change, and reliability-ceiling-corrected effect sizes. This turns
   §1.1 from a liability into a contribution: *R^2 is the wrong number; a representation
   scoring R^2 = 0.8 can have ICC = 0.3 and be useless as an instrument.* It also lets you
   report the label-reliability finding as methodology rather than as an attack on the
   dataset authors.
5. **Scope, to defuse the generality objection.** Run the whole matrix on frozen public
   encoders — V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), VideoMAE-v2, per-frame
   DINOv2, GaitBase/GaitSSB — plus your JEPA and a **random-init control**. Replicate on
   **CASIA-B**, which is itself a within-subject factorial (124 subjects x {NM, BG, CL} x 11
   views) that the whole field already has. Optionally add a rendered synthetic control where
   the true intervention direction is known by construction.
6. **The fix (turns an audit into a method).** An *intervention-aligned readout*: a cheap
   adapter fit on a subset of subjects' paired differences so that observational directions
   transport. Show it improves GFC transport and out-of-distribution generalisation.

**Why this is an ICLR paper and not a gait paper.** The claim is about probing methodology
and the linear representation hypothesis. Gait is the apparatus. Do not put "gait" in the
title, the first two sentences of the abstract, or Figure 1.

**Why it fits in 16 days.** The core requires **zero new pretraining**. It is frozen
feature extraction, linear algebra, and the GFC evaluator you already wrote.

**Why it is unusually safe.** Both outcomes are papers. If directions misalign, the headline
is "observational directions do not transport interventions — probe-based claims in video
need revision." If they align, the headline is "the first validation of the linear
representation hypothesis in vision against real physical counterfactuals." You do not need
the result to break a particular way. With 16 days, that property is worth more than a
higher ceiling.

**Main risk.** Between-subject speed variation is confounded with height, BMI and sex, all
of which are readable from a silhouette. That is what makes the misalignment prediction
plausible — but it also means every alignment number must be reported with those variables
partialled out, and with `shortcut_duration_seconds` and `shortcut_horizontal_centroid_drift`
(both already flagged in the repo) controlled. Splits must be participant-disjoint.

**Where GaitLU/JEPA fits — strictly optional, strictly secondary.** If extraction and a
temporal-masking fix land by ~Aug 21, add one clean secondary axis: does the pretraining
corpus or sampling policy change interventional alignment? That reduces I6 from 28 models to
4–6 and gives it a far better outcome variable than GFC-minus-control. It must not be
load-bearing.

---

## 4. Fallbacks

**B — Reliability-first evaluation (safest, lower ceiling).** Standalone version of layer 4.
Import clinical measurement science into representation evaluation: ICC, minimal detectable
change, Bland–Altman against the instrument, ceiling-corrected R^2, plus the demonstration
that a headline Scientific Data ground truth has reliability ~0.14 while two independent
video pipelines agree with each other at r = 0.95. Nearly free computationally. Reads as a
critique paper, which caps it around borderline, but it will certainly produce a result.

**C — Operator / composition (highest ceiling, highest risk).** Only viable if temporal
masking, relative/delta position encoding and >=32-frame clips are all fixed fast, which is
17–22 engineering days by the feasibility estimate — it does not fit. Keep it as the next
cycle's project. If any part is attempted now, the random-init and raw-signal spectral
baselines must be row 1 of Table 1, and Koopman-JEPA (AAAI 2026) and Koopman Dreamer
(Jul 2026) must be cited and beaten.

---

## 5. Kill list

- **I2** — DeepPhase (SIGGRAPH 2022 Best Paper) already learns an unsupervised shared phase
  manifold for locomotion; the Gait Deviation Index has been the clinical deviation-from-
  template score since 2008.
- **I4 as a standalone** — the "mean pooling is the 0th Fourier coefficient" insight is the
  abstract of a 2016 paper, and Frame2Freq (2026) is the modern version. Keep it as a
  feature set inside the flagship if useful.
- **I5 as framed** — no model in the repo has a working biometric (4.8% retrieval), so there
  is no privacy axis to trade off; and six of the eleven "health" targets are anthropometric
  measurements, so both axes are the same variable viewed twice.
- **I6 at 28-model scope** — kill this week with the one-day check in §6.

---

## 6. Week-1 falsification pilots (run all three in parallel, 3 days)

**P1 — kill or save the 28-model campaign. 1 day, no new training.**
Compute `learned_minus_shortcut` for all 11 existing checkpoints and take the SD across
models that differ only in nuisance hyperparameters. That SD is the noise floor for `D`.
Ask whether a plausible allocation effect exceeds `2*SD/sqrt(8)`. Given the observed margin
is 0.043 with SE 0.020, expect no. This decides whether to spend 708 GPU-hours.

**P2 — test the flagship's entire premise. 1 day.**
Extract frozen V-JEPA 2 features for all 8 factorial cells x 398 participants of
Health&Gait. Compute, per factor: `cos(w_obs, Delta_int)`; the singular spectrum of the
398 x D matrix of per-subject `Delta_int`; and the same two quantities for a random-init
encoder. If `cos` is ~0.9 for every model *and* the spectrum is rank-1, the finding is the
positive version and the framing changes — you need to know that on day 3, not day 25.

**P3 — reliability audit. 0.5 day.**
ICC(2,1) of frozen features across the 8 repeated cells per participant; re-derive the
label split-half reliability in §1.1 independently. This is the layer that must hold
regardless of which paper you write.

**Also do, cheaply, before committing (0.5 day):** the classical baseline battery —
random-init encoder, silhouette-area FFT, bounding-box aspect autocorrelation, temporal
self-similarity matrix. Every reviewer will ask for these, and they cost an afternoon.

---

## 7. Schedule

| Dates | Work |
| --- | --- |
| Aug 12–14 | P1, P2, P3 + baseline battery. Go/no-go on flagship. Formally kill I6 at 28-model scope. |
| Aug 17–21 | Full matrix: 5 encoders x 3 factors, transport tests through GFC, permutation nulls, confound controls (height, BMI, sex, age, duration, centroid drift), participant-disjoint splits. |
| Aug 24–28 | CASIA-B replication. Intervention-aligned adapter. Synthetic known-direction control. Optional GaitLU/JEPA secondary axis only if ready by Aug 21. |
| Aug 31–Sep 4 | Freeze analysis, figures, robustness, pre-registered claim boundaries. Stop experiments. |
| Sep 5–17 | Writing. |
| Sep 18 | Abstract. |
| Sep 19–24 | Anonymisation, reproducibility, final checks. |
| Sep 25 | Submission. |

---

## 8. Non-negotiables for whichever paper is written

1. Participant-disjoint splits, stated in the main text.
2. A random-init encoder row in every table.
3. A non-learned classical baseline in every table.
4. Height / BMI / sex / age partialled out of every anthropometric claim.
5. At least two public-weight encoders, not just the in-house 6-layer ViT.
6. At least one replication dataset (CASIA-B is the cheapest).
7. The word "gait" absent from the title.

---

## Sources

- [Koopman Invariants as Drivers of Emergent Time-Series Clustering in JEPAs (AAAI 2026)](https://arxiv.org/abs/2511.09783)
- [Koopman Dreamer](https://arxiv.org/abs/2607.19719) · [SJEPA](https://arxiv.org/abs/2608.04060) · [seq-JEPA](https://arxiv.org/abs/2505.03176) · [Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337)
- [Self-Supervised Learning of Structured Dynamics from Videos / ProbeMotion](https://arxiv.org/abs/2607.21576) · [A Control Theory of Predictability in Latent World Models](https://arxiv.org/abs/2607.10362)
- [DeepPhase (SIGGRAPH 2022)](https://dl.acm.org/doi/10.1145/3528223.3530178) · [FLD (ICLR 2024)](https://arxiv.org/abs/2402.13820) · [CycleCL (WACV 2024)](https://arxiv.org/abs/2311.03402)
- [DFT for Video Classification (2016)](https://arxiv.org/abs/1603.06182) · [Frame2Freq (2026)](https://arxiv.org/html/2602.18977v1)
- [Gait disentanglement + Koopman (Pattern Recognition 2026)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326005844) · [GaitProtector](https://arxiv.org/abs/2605.12431)
- [On Pretraining Data Diversity for SSL (ECCV 2024)](https://arxiv.org/abs/2403.13808) · [Automatic Data Curation for SSL](https://arxiv.org/pdf/2405.15613)
- [GaitSSB / GaitLU-1M (TPAMI 2023)](https://arxiv.org/abs/2206.13964) · [FoundationGait](https://arxiv.org/abs/2512.00691) · [Health&Gait (Sci Data 2025)](https://www.nature.com/articles/s41597-024-04327-4)
- [How Reliable are Causal Probing Interventions? (IJCNLP 2025)](https://aclanthology.org/2025.ijcnlp-long.47.pdf) · [Latent Causal Probing](https://arxiv.org/html/2407.13765v1)
- [V-JEPA 2 weights](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) · [vjepa2 repo](https://github.com/facebookresearch/vjepa2)
