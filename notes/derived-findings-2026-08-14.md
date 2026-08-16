# Findings I derived from repo artifacts (not in any of the three source docs)

## F1. Clip-pooled effective rank predicts the honest biometric probe (rho = 0.89)

Computed by me from `outputs/phase1/*/features.npz` (9,390 x 384 EMA-target pre-norm pooled
features per run) plus each run's `probes.csv`. Effective rank = exp(entropy of normalized
covariance eigenvalues), float64.

| run | clip_var_coef | pooled eff. rank /384 | held-out retrieval (chance .0129) | closed-set id | gait 2-class |
|---|---|---:|---:|---:|---:|
| a02-lr3e-4 | 0 | 4.96 | .0282 | .0766 | .9316 |
| a03-ema0.995 | 0 | 5.41 | .0300 | .0877 | .9300 |
| a05-mask-heavy | 0 | 8.25 | .0288 | .1074 | .9290 |
| a01-lr3e-5 | 0 | 9.08 | .0282 | .0978 | .9311 |
| a06-pred-depth3 | 0 | 9.73 | .0306 | .1026 | .9332 |
| a00-baseline | 0 | 9.83 | .0294 | .1063 | .9327 |
| a04-mask-light | 0 | 9.99 | .0306 | .1031 | .9375 |
| b01-mask-light | 0 | 11.85 | .0325 | .1026 | .9252 |
| a07-clip-var | 1.0 | 19.76 | .0404 | .0771 | .8926 |
| b00-clip-var | 1.0 | 27.08 | .0343 | .0558 | .8857 |
| b02-mask-light-clip-var | 1.0 | **77.98** | **.0484** | .0579 | .8841 |

Correlations across the 11 runs:
- pooled rank vs held-out retrieval: Spearman rho = **0.890**, p = 0.0002 (Pearson r = 0.915)
- pooled rank vs closed-set identity: rho = -0.327, p = 0.33
- pooled rank vs gait 2-class: rho = -0.564, p = 0.071

Interpretation: the probe that everyone reports (2-class gait system, 0.88-0.94) is **negatively**
correlated with representation quality measured any other way. The open-set retrieval probe is the
only one that tracks rank.

## F2. `claude-iclr-analysis.md` is wrong that the phase1 grid found nothing

It says "all eleven phase1 variants land at gait 0.88-0.94 ... the entire eleven-run grid is
contained within +/-2 SE". True for the 2-class probe only. On pooled rank the grid spans
5.0 to 78.0 (16x). On retrieval it spans .0282 to .0484 (1.7x). The grid did find something;
it was measured with the wrong readout.

## F3. The "axis mismatch" arm was already run, and the fix works

`engine.py:288-294`: `clip_var_coef = config.get("clip_var_coef", config.get("var_coef", 0.0))`,
`token_var_coef = config.get("token_var_coef", 0.0)`.
In every phase1 config, `clip_var_coef` is present explicitly (0.0 or 1.0) and `token_var_coef`
is absent (-> 0.0). So `var_coef: 1.0` in those configs is **dead**: the a-series runs had
**no anti-collapse regularization at all**, and the clip-var runs had clip-axis VICReg only.
So the repo's real contrast is regularizer-off vs clip-axis-on, not token vs clip.

The archived `jepa-v4` checkpoint that produced token rank 381.6 / pooled rank 11.5 was trained
with token-axis VICReg (`r3` note in `context_use.metadata.json`: "VICReg is applied separately
per group" on context tokens). That is where the 33x axis discrepancy comes from.

Consequence: the three-way ladder is complete and already sits on disk.
- token-axis VICReg -> token rank 381/384 (green), pooled rank 11.5/384 (collapsed)
- no regularizer -> pooled rank 5-12
- clip-axis VICReg -> pooled rank 20-78, retrieval up 1.6x

## F4. Codex's blank-input bug claim is REFUTED for this codebase

codex-iclr-ideas.md sec 2.1 / sec 10: "engine.py:161 constructs a blank after normalization by
setting the tensor to zero. With the default normalization, zero is raw mid-gray."

`cody_jepa/data.py:178` is the only normalization: `np.asarray(image, dtype=np.float32) / 255.0`.
There is no mean subtraction or std division anywhere in the package (grepped). So zero **is**
black. `engine.py:161 blank_video = torch.zeros_like(video)` is correct.
Also `context_use.metadata.json` states `condition_definitions/blank = "all-zero pixels before
input normalization"`, so the archived harness was correct too.
This objection should be dropped from the plan.

## F5. same-subject minus cross-subject is not just small, it is the wrong sign

`summary/d1_contrasts/same_subject_minus_cross_subject`: mean = +4.95e-6,
`positive_subject_fraction` = 0.5375, `effect_size_dz` = 0.093.
The gap for same-subject context (1.596e-4) is *larger* than for cross-subject (1.563e-4),
i.e. swapping in the same person's other trial is very slightly worse than swapping in a
stranger. Indistinguishable from zero, and the direction contradicts any identity sensitivity.

Also worth noting and not noted anywhere: **blank (9.53e-5) < cross-subject (1.56e-4)**.
Removing the input entirely hurts *less* than giving the predictor a wrong but real input.
That is the signature of a positional prior that a wrong layout actively disrupts.

## F6. Statistical power of the retrieval probe

`identity_heldout_retrieval`: train 240, val 1,632, 80 classes, chance 0.0129.
1,632 windows over 80 subjects, ~3 windows/recording -> ~544 independent recordings.
Window-level SE at p=0.03 is 0.0042; subject-clustered SE is larger. The a00 -> b02 move
(.0294 -> .0484) is roughly 4.5 window-level SE. Suggestive at one seed, not conclusive.
Needs >=3 seeds and subject-level bootstrap before it is a headline.

## F7. The four modalities are NOT all co-registered (verified on disk myself)

`claude-iclr-ideas.md` line 26 and 420 claim "four co-registered renderings at 960x540".
Measured on `data/healthgait/raw/Health_Gait`, recording PA000/FGS/WJ_1:

| modality | resolution | mode | frames | directory grammar |
|---|---|---|---:|---|
| silhouette (YOLOv8) | 960x540 | L | 96 | `silhouette/PA###/<UGS,FGS>/<WoJ,WJ>_<1,2>_YOLOV8/` |
| semantic seg (DensePose) | 960x540 | RGB | 96 | `semantic_segmentation/PA###/<UGS,FGS>/<WoJ,WJ>_<1,2>_DensePose/` |
| optical flow GMFlow | **480x270** | RGB | **121** | `optical_flow/PA###/<UGS,FGS>/<GMFLOW,TVL1>/<WoJ,WJ>_<1,2>/` |
| optical flow TVL1 | **480x270** | RGB | **121** | same |
| pose (AlphaPose) | JSON, 205 KB | - | - | `pose/PA###/<UGS,FGS>/<WoJ,WJ>_<1,2>_AlphaPose.json` |

Silhouette and DensePose ARE frame-aligned and same resolution. Optical flow is half
resolution, has a different frame count, and uses a different directory grammar, so it is
neither spatially nor temporally co-registered without resampling.

Leaf counts: silhouette / segmentation / pose = 3,130 recordings each.
optical_flow = 1,592 at the `PA###/<speed>/<method>` level = 398 x 2 x 2, consistent.
3,130 recordings / 2 directions = 1,565 source videos (claude says 1,565, codex says 1,564).

Consequence: the "four-rung privacy ladder" and any "identical physical event, four renderings"
claim must be narrowed to **two** genuinely co-registered rungs (silhouette, DensePose) plus
two that require an alignment step. This is fixable with resampling but it is real work and it
must be disclosed.

## F8. THE HEADLINE PROBE IS A STOPWATCH (new, decisive)

Every document treats `gait_system` accuracy (0.88-0.94, chance 0.50) as the repo's one
working downstream result. I ran the classical baseline that all three documents call for
and none of them ran, using the precomputed shortcut columns already sitting in
`data/healthgait/manifests/silhouette_gfc_candidate_seed0.csv` (3,130 recordings, 398
subjects, the same subject-disjoint split, 0 subject overlap).

**One threshold on how long the walk took:**
- rule: predict FGS if `shortcut_duration_seconds < 3.267` (threshold fit on train only)
- held-out-subject accuracy = **0.9519**, subject-clustered bootstrap 95% CI **[0.9245, 0.9755]**, 80 subjects
- mean duration: UGS 4.09 s, FGS 2.75 s

**Logistic regression comparisons (same split):**
| features | val acc |
|---|---:|
| walk duration alone (1 feature) | 0.9567 |
| all 9 handcrafted shortcut features | 0.9583 |
| all 5 silhouette-area statistics | 0.6987 |
| centroid drift alone | 0.5849 |
| silhouette area mean alone | 0.5128 |
| **best JEPA (a04-mask-light)** | **0.9375** |
| majority | 0.5000 |

**Every JEPA checkpoint in the repo falls inside or below the CI of a stopwatch.**

Honest caveat that must be stated in any writeup: duration is a recording-level property and
the JEPA only sees a 16-frame (0.53 s at 30 Hz) window, so this is not a like-for-like
comparison of inputs. That does not weaken the conclusion, it sharpens it: the *label* is
essentially "did this person cross the corridor quickly", so a 0.93 from a representation is
not evidence that the representation encodes anything about how the body moves. The probe
cannot distinguish a good representation from a bad one, which is exactly what F1 shows
empirically (rank vs gait_system: rho = -0.56).

Implication for all three source documents: `gait_system` must be demoted from headline
metric to shortcut-contaminated control, and the open-set retrieval probe promoted.

## F9. fps IS documented and IS on disk

`claude-iclr-analysis.md` sec 1.2 claims "Frame rate is unknown... Any claim in absolute
seconds rests on an assumption". The manifest `silhouette_gfc_candidate_seed0.csv` carries an
`fps` column: constant 30.0 across all 3,130 recordings. The published paper also states
30 Hz for capture (with an internal inconsistency: its gait-estimation section says 29.97).
Absolute-time claims are available. A 16-frame clip is 0.53 s.

## F10. RANDOM-INIT CONTROL: the rank numbers measure the model, not the architecture

Both adversarial reviewers named this as the decisive missing control. I ran it.
Encoder = `outputs/phase1/b02-mask-light-clip-var/best_loss.pt`, val split, 1,872 clips,
identical measurement code for both rows, float64 eigendecomposition.

| representation | random-init | trained |
|---|---:|---:|
| clip-pooled, pre-norm | **2.54** | **23.60** |
| token, pre-norm | **12.31** | **63.39** |
| token, post-LayerNorm | 12.33 | 64.85 |

Three objections die here:

1. "Token rank ~381 is just the fixed sinusoidal position basis, a random encoder would show
   it too." **Refuted.** Random-init token rank is 12.3, not 381. Token rank is model-dependent.
2. "Pooled rank ~11 is a pooling artifact; any model would show it." **Refuted.** Pooling the
   random-init model gives 2.54; the trained model gives 23.6. Identical pooling operator,
   identical data, 9x apart. Pooled rank measures the model.
3. "The 33x gap is really a pre-norm vs post-norm LayerNorm mismatch." **Mostly refuted.**
   LayerNorm moves token rank by 2% (63.39 -> 64.85).

Caveat to state plainly: my token measurement is over full-view tokens subsampled to 200k
rows, whereas the archived 381.6 was over masked *context* tokens after LayerNorm, per mask
group, over 447k-734k rows. These are different measurement contracts and the numbers are not
directly comparable. What is comparable is the random-vs-trained contrast within my contract.

Second caveat, and it is a real methodological point worth putting in the paper: effective
rank depends on the population it is estimated over. The same b02 checkpoint gives pooled
rank 78.0 over the 9,390-row train+val export (398 subjects) and 23.6 over the 1,872-row
val-only export (80 subjects). Any rank number without a stated estimation population is
uninterpretable. That is itself part of the "readout contract" argument.

## F11. Duration-matching collapses the pace probe across all 11 checkpoints

Binned recordings into 0.25 s duration bins, kept equal UGS/FGS counts per bin in the val
split, re-scored every checkpoint's linear probe (trained on the full train split).

| run | pooled rank | contaminated | duration-matched |
|---|---:|---:|---:|
| a02-lr3e-4 | 4.96 | .9322 | .6313 |
| a03-ema0.995 | 5.41 | .9295 | .5758 |
| a05-mask-heavy | 8.25 | .9284 | .5404 |
| a01-lr3e-5 | 9.08 | .9306 | .5556 |
| a06-pred-depth3 | 9.73 | .9332 | .5758 |
| a00-baseline | 9.83 | .9327 | .5808 |
| a04-mask-light | 9.99 | .9380 | .5808 |
| b01-mask-light | 11.85 | .9257 | .5404 |
| a07-clip-var | 19.76 | .8932 | .5202 |
| b00-clip-var | 27.08 | .8851 | .5455 |
| b02-mask-light-clip-var | 77.98 | .8841 | .5101 |
| stopwatch | - | .9519 | .5556 |

Mean matched accuracy 0.560, range 0.510 to 0.631, chance 0.500.
Matched val = 198 windows over 39 subjects (from 1,872 over 80).

Two consequences:
1. Essentially ALL of the reported 0.93 was the duration shortcut.
2. The clean probe does NOT flip the rank correlation. rank vs matched probe rho = -0.618
   (p = 0.043), slightly more negative than rank vs contaminated (-0.564). At 39 subjects
   this is underpowered and should not be interpreted as a real negative relationship, but
   it cannot be used as support either.

Conclusion: the walking-pace label is unusable on this dataset in either form. Open-set
retrieval is the only readout here with enough classes and power. This is the strongest
argument that Proposal 1 needs a second domain with a genuinely predictable,
shortcut-resistant label.

## F12. Pretraining loss does not predict transfer; pooled rank does

Across the 11 phase1 runs (best value over each run's logged history):

| predictor of held-out retrieval | Spearman rho | p |
|---|---:|---:|
| clip-pooled effective rank | **+0.890** | 0.0002 |
| best validation loss | +0.306 | 0.360 |
| best training loss | +0.187 | 0.582 |

The repo selects checkpoints by `best_loss.pt`. That criterion is uninformative about
representation quality, and the sign is if anything backwards.

## F13. THE DECOMPOSITION (this is the novelty upgrade)

Law of total covariance for tokens x_{i,t} with equal tokens per clip:
    Sigma_token = Sigma_between + Sigma_within,   Sigma_pooled = Sigma_between exactly.
So the token-axis health metric is an entropy over eigenvalues of a SUM, only one term of
which a pooled probe can ever consume. Define the between-clip share
    beta = tr(Sigma_between) / tr(Sigma_token).

Measured on the val split, 1,872 clips, 2,935,296 tokens, identical code per row:

| model | tr(Sb) | tr(Sw) | **beta** | erank(St) | erank(Sb) | erank(Sw) |
|---|---:|---:|---:|---:|---:|---:|
| random init | 0.0131 | 53.045 | **0.00025** | 12.32 | 2.54 | 12.31 |
| a00, no regularizer | 0.0088 | 13.978 | **0.00063** | 60.32 | 10.86 | 60.23 |
| b02, clip-axis variance on | 1.7316 | 14.904 | **0.104** | 63.39 | 23.60 | 51.53 |

Four things fall straight out:

1. For the unregularized trained model, **99.94 percent** of token-axis variance is
   within-clip. The standard health metric is almost entirely measuring something no pooled
   probe can see.
2. **erank(Sigma_token) is numerically almost erank(Sigma_within)**: 60.32 vs 60.23 for a00.
   The token-axis metric IS the within-clip metric to two significant figures.
3. Training raises token rank from 12.3 to 60.3 while beta stays at 0.0006. The metric looks
   5x healthier while the share visible to any probe does not move. That is the silent
   failure, quantified in one number.
4. beta spans 0.00025 to 0.104, a factor of **400**, across models on identical data with an
   identical pooling operator. This kills the "it is just pooling" objection quantitatively:
   if it were pooling, beta would be constant.

beta is also directly controllable: the clip-axis variance term raises it 165x.
Note the fix improves but does not solve. Even b02 leaves 90 percent of token variance
invisible to the probe.

NEAREST-NEIGHBOUR RISK: LiDAR (arXiv 2312.04000) uses a discriminant-style within/between
ratio where "within" is augmentation-induced. Structurally analogous. The paper must engage
it head-on: LiDAR's within-class scatter comes from augmentations of one sample, ours comes
from token position inside one clip, and the two are different objects for masked predictive
architectures. Do not discover this in review.

---

# Round 2 findings, 2026-08-14 (after adversarial review of the revised proposal)

## F14. The mechanism is prior art. The measurement is not. This inverts the paper's ordering.

Verified by an adversarial prior-art agent, with arXiv IDs confirmed:

| paper | what it establishes | evidence it uses | leaves open |
|---|---|---|---|
| **PCP-MAE**, arXiv 2408.08753, NeurIPS 2024 Spotlight | Feeding masked-patch centres to the decoder lets it reconstruct well *without* the encoder, so "the reconstruction objective does not necessarily rely on representations of the encoder, thus preventing the encoder from learning semantic representations" | downstream accuracy only | any variance geometry; any metric claim |
| **MPL-MAE**, arXiv 2606.31570 | formalises positional leakage as a constraint lower-bounding how well masked coordinates are recoverable from positional embeddings alone | gradient-magnitude reliance scores | variance decomposition; metric validity |
| **Causal-JEPA**, arXiv 2602.11389 (incl. LeCun, Balestriero) | object-level masking prevents shortcut solutions | counterfactual VQA, planning | mostly adjacent, removes little |
| **V-JEPA 2.1**, arXiv 2603.14482 | context tokens lack incentive to encode local information because the loss is applied only to masked tokens | downstream task performance | no variance split, no rank metric |

**Consequence: the earlier recommendation to make the mechanism primary was wrong.**
Positional conditioning starving the encoder is established 2024 prior art. What is
unclaimed is the measurement consequence: the exact annihilation argument linking mean
pooling to metric invalidity, with beta as the correction. No paper found states that mean
pooling annihilates within-clip variance exactly, and none defines
beta = tr(Sigma_between)/tr(Sigma_token) for SSL evaluation.

The mechanism experiments stay, but reframed as a mediation analysis connecting a *known*
cause to a *novel* consequence, citing PCP-MAE and MPL-MAE as having established the cause
in another modality.

**Two additional threats found:**
- *Anchoring the Eigengap*, arXiv 2605.08764: effective rank "is label-agnostic, so it
  captures both signal and noise, and can be artificially inflated". Independent correction
  on the same logic, different nuisance and different fix. Removes novelty from the general
  statement "effective rank is inflated by task-irrelevant variance".
- Whetten et al., arXiv 2409.10787: RankMe "fails as a standalone surrogate for downstream
  performance" in speech. So rank-metric fallibility is already known; we cannot claim to
  observe it first.

**The V-JEPA 2.1 direction conflict, which must be resolved in the paper:** they argue
context tokens *under*-encode local structure; this claim says tokens *over*-encode
position. These are compatible only if positional decodability and local content fidelity
are distinct properties of the same tokens. Assert that and test it, or one account is wrong.

Also: the widely repeated "context tokens act as registers" line is a **paraphrase, not a
quotation** from V-JEPA 2.1. Do not quote it.

## F15. "Mean pooling annihilates within-clip variance" needs a leakage floor

Sigma_between is the covariance of the *empirical* clip means, and each empirical mean
carries within-clip sampling noise. Under an i.i.d.-tokens null the floor is
tr(Sigma_within)/T. With T = 1568 (the archived contract):

| model | tr(Sb) | tr(Sw) | i.i.d. floor tr(Sw)/T | observed / floor |
|---|---:|---:|---:|---:|
| random init | 0.0131 | 53.045 | 0.0338 | **0.39** |
| a00, no regulariser | 0.0088 | 13.978 | 0.0089 | **0.99** |
| b02, clip-axis variance | 1.7316 | 14.904 | 0.0095 | **182** |

Readings:
1. Random init sits **below** the i.i.d. floor, which is impossible under i.i.d. tokens.
   So tokens within a clip are strongly non-independent, and the reason is exactly the
   mechanism: the fixed position encoding is identical in every clip, so it loads onto
   Sigma_within and contributes **exactly zero** to Sigma_between. The i.i.d. floor
   therefore OVERSTATES leakage here. The objection turns into evidence for the mechanism.
2. a00 nevertheless sits at 0.99x that overstated floor, so **a00's between-clip signal is
   not safely distinguishable from pooling noise**. Do not headline a00's beta without a
   permutation null.
3. b02 sits at 182x the floor, so its signal is real under any plausible null.

Against the empirical random-init floor (beta = 0.00025): a00 is 2.5x, b02 is 416x.

**Required in the paper:** report beta against a stated null, either the i.i.d. floor or a
clip-label permutation null, and never report beta bare.

## F16. The buckets are NOT zero-sum, and the toy example must not imply they are

b02 versus a00: tr(Sigma_between) rises 0.0088 -> 1.7316 while tr(Sigma_within) ALSO rises
13.978 -> 14.904. Both grew. Raising beta did not require destroying within-clip structure.
This matters because a dense-prediction reviewer will otherwise object that the proposal
wants to delete the variance their task consumes.

## F17. Effective rank is not "how spread out" and the scalar toy cannot illustrate it

Effective rank is scale-invariant and counts how many directions are occupied. A
one-dimensional worked example has effective rank 1 by construction, so it cannot
demonstrate the metric's blindness.

The actual proof is already measured: for a00, erank(Sigma_token) = 60.32 and
erank(Sigma_within) = 60.23. To two significant figures the health statistic **is** the
within-clip statistic. Use that, not a toy.

A worked example with both terms nonzero, for teaching the identity only:
Ana 90 and 70 (mean 80), Ben 40 and 20 (mean 30), grand mean 55.
total 725 = between 625 + within 100, exactly, for any numbers.

## F18. POSITION EXPLAINS 92 PERCENT OF WITHIN-CLIP VARIANCE (measured, not hypothesised)

Every clip contributes the same fixed grid of 1,568 token positions, so tokens form a
balanced two-way layout x[i,p] = mu + a[i] + b[p] + e[i,p]. The trace decomposition is exact:
tr(Sigma_token) = tr(Sigma_clip) + tr(Sigma_position) + tr(Sigma_residual), and
Sigma_within = Sigma_position + Sigma_residual. Measured on the same 1,872 held-out clips
(`notes/scripts/position_share.py`):

| model | total | position | residual | between-clip | position share of within |
|---|---:|---:|---:|---:|---:|
| random init | 53.058 | 51.146 | 1.899 | 0.0131 | **96.4%** |
| a00, no regulariser | 13.987 | 12.822 | 1.156 | 0.0088 | **91.7%** |
| b02, clip-axis variance | 16.636 | 13.832 | 1.072 | 1.7316 | **92.8%** |

As a share of ALL token variance:

| model | position | within-residual | between-clip |
|---|---:|---:|---:|
| random init | 96.4% | 3.6% | 0.025% |
| a00, no regulariser | **91.7%** | 8.3% | **0.063%** |
| b02, clip-axis variance | 83.1% | 6.4% | **10.4%** |

Three consequences, and the second one is inconvenient.

**1. The primary claim is now quantified and strong.** The standard token-axis health metric
for a00 is computed on a quantity that is 91.7 percent a deterministic function of position.
The position main effect is by construction identical in every clip, so it contributes
exactly zero to what a mean-pooled probe consumes. Between-clip variance is 0.063 percent of
the total.

**2. The predictor-conditioning mechanism has much less headroom than assumed.** Position
share is already 96.4 percent at random initialisation, before any training. Training moves
it only to 91.7 percent. So the dominance is mostly the encoder's own additive sin-cos
positional embedding, not something target-position conditioning creates. The hypothesis must
be restated: does target-position conditioning *slow the decay* of position dominance during
training? Prediction becomes "removing target-position conditioning drives position share
below 91.7 percent", not "adding it raises position share". This is a sharper and more honest
test, and it materially raises the burden of proof on the mechanism half.

**3. It fixes the leakage objection from F15.** Only the residual behaves like sampling noise,
because the position main effect is shared across clips and generates none. So the correct
i.i.d. floor is tr(Sigma_residual)/T, not tr(Sigma_within)/T.

| model | naive floor | observed / naive | corrected floor | observed / corrected |
|---|---:|---:|---:|---:|
| random init | 0.03383 | 0.39 | 0.00121 | **10.8** |
| a00 | 0.00891 | 0.99 | 0.00074 | **12.0** |
| b02 | 0.00951 | 182 | 0.00068 | **2,533** |

The naive floor overstates leakage by roughly twelvefold. Under the corrected floor every
model's between-clip signal is real, including a00, which the naive floor appeared to
eliminate. Report beta against the corrected floor and say why the naive one is wrong.

**Net effect on the paper: promote the measurement claim, demote the mechanism claim.**
Combined with F14 (PCP-MAE already owns positional conditioning as a cause), the ordering
that survives is: measurement primary, mechanism supporting and explicitly harder to win.

## F19. CORRECTION to F1: the rho = 0.890 headline is population-dependent

I computed token effective rank, pooled effective rank, and beta for all eleven checkpoints
on one matched population (the 1,872-clip val split), and compared against the pooled rank I
had computed earlier from the 9,390-clip train+val export.

| run | token rank | pooled rank (1,872) | pooled rank (9,390) | beta | retrieval |
|---|---:|---:|---:|---:|---:|
| a01 | 36.95 | 9.08 | 9.08 | 0.00087 | .0282 |
| a02 | 123.59 | 4.59 | 4.96 | 0.00201 | .0282 |
| a05 | 78.62 | 11.89 | 8.25 | 0.00051 | .0288 |
| a00 | 60.32 | 10.86 | 9.83 | 0.00063 | .0294 |
| a03 | 21.55 | 5.95 | 5.41 | 0.00043 | .0300 |
| a04 | 55.14 | 10.59 | 9.99 | 0.00073 | .0306 |
| a06 | 59.71 | 10.69 | 9.73 | 0.00063 | .0306 |
| b01 | 112.93 | 14.42 | 11.85 | 0.00052 | .0325 |
| b00 | 46.76 | 6.52 | 27.08 | 0.04159 | .0343 |
| a07 | 40.88 | 5.13 | 19.76 | 0.02971 | .0404 |
| b02 | 63.39 | 23.60 | 77.98 | 0.10409 | .0484 |

Spearman against held-out retrieval, n = 11:

| metric | rho | p |
|---|---:|---:|
| token effective rank, val | **-0.100** | 0.77 |
| pooled effective rank, val (1,872) | +0.260 | 0.44 |
| **pooled effective rank, train+val (9,390)** | **+0.890** | **0.0002** |
| beta, val (1,872) | +0.446 | 0.17 |

**The two estimates of the same pooled quantity agree with each other at only rho = +0.382
(p = 0.25).** Changing nothing but the estimation population moves the correlation with
transfer from highly significant to nothing.

**The bias is systematic, not random.** The three clip-axis-variance runs, which are the
genuinely high-rank models, are underestimated 3.3x to 4.2x on the small population
(a07 5.13 vs 19.76; b00 6.52 vs 27.08; b02 23.60 vs 77.98) while the a-series barely move.
Effective rank is a spectral quantity, so estimating it needs N well above d. Val gives 4.9
samples per dimension; train+val gives 24.5. At low ratios the estimate is truncated, and the
truncation bites hardest on exactly the models the paper cares about.

**Three consequences.**

1. **F1's rho = 0.890 must be reported with its estimation population, and it is not robust
   to a reasonable alternative choice.** I previously presented it without that caveat. Do
   not print it bare.
2. **Token rank is uninformative about transfer (rho = -0.100).** This is good for the thesis
   and is the cleanest result in the table.
3. On a matched population the ordering is beta (0.446) > pooled rank (0.260) > token rank
   (-0.100), which is the predicted direction, but none is significant at n = 11. So the
   pilot does not establish that beta beats pooled rank. Q1 is genuinely open.

**This suggests a better motivation for beta than the axis argument.** Beta is a trace ratio,
so it needs only sums of variances and converges at 1/sqrt(N) largely independent of
dimension. Effective rank needs the whole eigenspectrum and needs N >> d. So beta should be
dramatically more sample-efficient, which would explain why it could beat pooled rank rather
than merely equal it, and it is a principled and testable advantage rather than hygiene.
Test it directly: subsample the estimation population and plot the stability of each metric.
