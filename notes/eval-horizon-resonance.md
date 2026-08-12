# Evaluation: "Horizon resonance and the choice of time coordinate"

Date: 2026-08-12. Evaluated against verified repo facts in `notes/iclr2027-direction.md` §1
and against `configs/train/gaitlu_hierarchy_full.json`.

**Verdict: the mechanism is real and the Δt-vs-Δφ hypothesis is genuinely novel, but four of
the idea's five load-bearing supports fail on facts already established in this repo. As
written it is Fallback C wearing a new hat, and it does not fit 16 days. There is a good
paper inside it, and it is a different paper.**

Scores: mechanism plausibility 8/10 · novelty 7/10 · falsifiability 9/10 ·
feasibility by Sep 4 **2/10** · validity of stated supporting evidence **1/10**.

---

## 1. The retro-explanation is false, and it was the cheapest part of the pitch

The pitch claims the comb "retro-explains three runs of yours without new machinery": rank
collapse, the vanishing wrong-context gap, and reliance on static shape. It cannot, because
**there is no horizon in any existing run.**

`masks/multiblock.py` samples masks in the 14×14 *spatial* grid and replicates each selected
cell across all 8 temporal indices. Context and target both span the full temporal extent of
every clip. `predictor.py` uses a fixed absolute `sincos_3d_position_embedding(8,14,14)` with
no relative encoding; Δ is not a model input. Every trained checkpoint has Δ ≡ 0 with
bilateral temporal context.

So:

- `wrong_context_gap ≈ 1e-4` has a simpler and better-supported explanation: a spatial
  inpainter with co-temporal neighbours on both sides of the masked tube barely needs the
  context at all. Shuffling context across the batch leaves the *within-clip* temporal
  neighbours intact. This is a masking-geometry artifact, not a resonance artifact.
- Effective rank 5–11 with `var_coef: 1.0` already on is a collapse-dynamics result, not a
  horizon result.
- "The model leans on static shape" follows directly from §1.7 (uncropped 960×540 → 112×112,
  ~3.1% foreground, ~0.96 frame-widths of centroid drift per recording).

This matters beyond bookkeeping. R1's defense was "the contribution is the *measured* comb."
If the pitch's own claimed measurements are misattributed, R1's objection stands unanswered
and the paper starts from zero evidence rather than three runs of free support. Delete the
retro-explanation paragraph; do not put a softened version in the paper. A reviewer with the
codebase would catch it.

## 2. R4's cheap generality experiment is the tautology this project already killed

The "single highest-leverage experiment" — run the clock-explainable-fraction diagnostic on
frozen V-JEPA 2, periodic vs aperiodic clips, two days, no pretraining — measures *prediction
error* as a function of Δ on periodic input.

§2 of the direction review kills I1 on exactly this: if the input is periodic with period T,
the latent trajectory of **any** deterministic encoder, including random-init, is periodic
with period T. A frozen encoder will therefore show low prediction error at Δ ≈ kT on
periodic clips and not on aperiodic ones, regardless of what it learned. The comb in *error*
is a property of the signal, not of the objective.

The claim that actually matters — that the objective **exerts no training pressure** at
Δ ≈ kT — is a statement about gradients during optimization. It is untestable on a frozen
checkpoint. So the cheap experiment tests a tautology and the non-tautological experiment
requires the full training sweep. R4's escape hatch does not exist.

(A random-init V-JEPA row would expose this in Table 1 on day one, which is precisely why
non-negotiable #2 exists.)

## 3. The double dissociation is anchored to labels ruled out as an anchor

The second half validates rate-vs-shape reallocation against OptoGait: cadence and velocity
as the rate group, step/stride length and support times as the geometry group.

§1.1, verified on `gait_parameters.csv` (n=398): `Cadence_UGS` vs `Cadence_FGS` r = 0.168.
`Velocity_UGS` vs `Velocity_FGS` r = 0.228. `Velocity_UGS` ranges to 5.09 m/s for *usual*
gait. Estimated label reliability ≈ 0.139, implying a ceiling of r ≈ 0.372 / R² ≈ 0.14 for
any perfect video method — already saturated by a 30-line periodogram heuristic and by the
dataset authors' own AlphaPose estimator. The standing conclusion is explicit: **no paper may
use the OptoGait gait parameters as its validation anchor.**

A predicted double dissociation measured across two groups of variables whose *within-subject
test–retest* correlation is 0.17–0.23 is not pre-registrable. Both arms will land inside noise
and the null will be uninterpretable. Worse, the rate group is the one that fails hardest, and
the rate arm is the arm the Δt condition is supposed to win.

**Repairable, and the repair is cheap.** Use the video-derived cadence estimator instead: §1.1
puts its split-half reliability at 0.936 with two independent pipelines agreeing at r = 0.946.
That is a usable rate axis. Expect the circularity objection ("you are predicting a quantity
computed from the same pixels") and answer it structurally: the encoder is frozen, the probe
is linear, and the question is what information is *linearly available*, not whether it is
present. For the shape axis there is no rescued real label — that group has to come from the
synthetic oracle, where limb excursion and asymmetry are exact by construction.

## 4. The collapse confound is not addressed and may be fatal to the comb measurement

The comb is a claim about the depth of the prediction term's minimum as a function of Δ. But
the prediction term has a second, Δ-independent global minimum: a constant encoder. Zero loss
at every horizon. So the comb cannot be read off raw loss curves without anti-collapse
pressure — and anti-collapse pressure is exactly what fills the notches, because the variance
and covariance terms supply gradient to E at Δ = kT where the prediction term supplies none.

The measured notch depth is therefore a function of `var_coef` / `cov_coef`, not a property of
the objective. Any comb figure needs a regularizer-strength axis, which multiplies the run
count.

And the current config already runs `var_coef: 1.0`, `cov_coef: 0.04` — and still collapses to
rank 5–11 of 384. Collapse dynamics currently dominate whatever horizon effect exists. On this
codebase, a comb sweep would be measuring collapse.

## 5. The resonance premise assumes registration the pipeline does not have

The mechanism requires E(x_ctx) ≈ E̅(x_tgt) at Δ ≈ T. At one stride the subject has translated
roughly 1.4 m. With uncropped 960×540 frames resized to 112×112, no person crop, and ~0.96
frame-widths of centroid drift per recording (§1.7), the target clip at Δ = T is the same pose
at a substantially different image location and apparent scale. Silhouettes at Δ = 0 and
Δ = T are *not* approximately equal in pixel space, and a ViT without translation invariance
will not make them equal in latent space.

Consequence: notch depth is a monotone function of how well the input is registered. That is
not a fatal flaw — it is arguably an interesting finding, "the comb appears only once you
normalize translation" — but it demotes the headline. The claim stops being "horizon placement
is the dominant control knob for video JEPAs" and becomes "for tracked, scale-normalized,
quasi-periodic input, horizon placement is a dominant knob." Narrower, and it hands a reviewer
the generality objection back after R4 spent the cheap fix.

Cropping and tracking normalization is a prerequisite, not an option, and it is engineering
time not in the budget.

## 6. Cost, honestly counted

R3's defense ("horizon is the gap, not the span; token count is constant") is correct and is
the single strongest technical argument in the document. But constant token count per step
does not make the study cheap, because the cost is in the number of *runs*.

`gaitlu_hierarchy_full.json`: 128,000 steps, batch 16 × 4 accumulation. A comb figure at 8 Δ
values × 3 seeds is 24 pretraining runs. The dissociation at 2 conditions × 3 seeds is 6 more.
Add the regularizer-strength axis from §4 and it doubles. Before that: temporal masking
rewrite, relative/Δ position encoding, ≥32-frame (realistically ≥64-frame) clip sampling,
person crop + tracking normalization, and an unsupervised Hilbert phase estimator.

The feasibility reviewer already priced the first three at 17–22 engineering days and
concluded Fallback C "does not fit." This proposal is Fallback C plus a phase estimator plus a
synthetic renderer plus a CASIA-B baseline plus a V-JEPA diagnostic, in 16 days, against a
Sep 4 experiment freeze. R2 accepted the feasibility charge and then cut two items. Two is not
enough.

## 7. What is actually good here, and should survive

Three things, in order of value:

1. **Δt vs Δφ conditioning is a real, novel, well-posed hypothesis.** "In time, the natural
   coordinate for predictor conditioning is the signal's intrinsic phase, not the frame index"
   is a clean idea, it is the correct temporal analogue of I-JEPA's positional conditioning,
   and the prior-art scan is right that CycleCL / SimPer / DeepPhase / FLD all treat phase as
   the *target* rather than as a nuisance to be given away. I did not find this inverted
   framing either. Keep it.
2. **The comb is a sharp, quantitative, falsifiable prediction** with a stated kill criterion.
   That is rarer than it should be, and it is the reason this idea is worth arguing with
   rather than filing.
3. **The synthetic 2D articulated walker** is the most undervalued item in the plan. R2 lists
   it as a concession; it is actually the load-bearing instrument.

One caveat on (1) that the adversarial panel missed. Computing Δφ from silhouette area/width
via Hilbert phase *is* a hand-built cadence extractor. The Δt arm must infer ω from pixels;
the Δφ arm receives it from an external estimator. A reviewer will say the work moved outside
the model and that freeing capacity is trivially expected. The defense has to be quantitative
and asymmetric: show that shape information **increases** in absolute terms, not merely that
rate information decreases. A pure decrease is consistent with "we deleted a task." Only a
genuine increase supports "capacity reallocates."

## 8. Recommendation

Do not run this as the flagship. Run the counterfactual-probing flagship, which requires zero
pretraining and is already resourced.

If you want this idea in the 2027 cycle rather than the next one, invert the burden:

- **Move the comb entirely to the synthetic walker.** Exact ω, exact registration, exact limb
  excursion and asymmetry, no collapse ambiguity because you can run tiny models to
  convergence, and no label-reliability problem because the labels are constructed. The full
  Δ × seed × regularizer-strength grid becomes hours, not weeks. Every confound in §4 and §5
  disappears by construction. This is where the mechanism gets *established*.
- **Spend real data on exactly one confirmatory pair**: Δt vs Δφ at a single well-chosen
  horizon, 3 seeds each, scored on the reliability-0.936 video-derived cadence for rate and on
  nothing else from OptoGait. Six runs, not thirty.
- **Drop the frozen V-JEPA 2 generality diagnostic** unless it is redesigned to escape §2. As
  specified it produces a tautology with a random-init control that matches.
- **Drop the scaling axis entirely**, per the 2504.07598 warning. Correct call in the original.

That gives "mechanism established in a controlled system, confirmed once in the wild" — a
normal, defensible paper structure — instead of "mechanism claimed, comb measured through
three confounds on a codebase whose predictor has no time axis."

**Kill criterion for the salvaged version:** if the synthetic walker at exact period T, exact
registration, and swept regularizer strength shows notch depth below ~10% of the
non-resonant-horizon probe accuracy, the comb is not a dominant knob and the paper is the
Δφ-conditioning paper alone. Decide that on synthetic data in week one, before any real-data
run is launched.

## 9. Corrections to fold back into the source document

- Remove the claim that existing runs support the comb (§1 above). They cannot; Δ ≡ 0.
- R4's response is invalidated by the project's own I1 kill. Either redesign or withdraw it.
- The OptoGait rate/geometry split described as "clean" is the label set §1.1 ruled out.
- R6's "3 seeds on the comb figure and the dissociation only" is right in spirit but the comb
  figure is where the run count lives; 3 seeds × 8 horizons is not a rounding error.
