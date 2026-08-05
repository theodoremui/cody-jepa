# Stage B results: the health-utility tradeoff persists

## Status and scope

The 7/27 research update concluded that Stage A had found one family of configurations
with useful downstream features and another with a healthier pooled representation, but
not one configuration with both. It proposed three 100-epoch Stage B checks. The
portable repository contains results for two of them:

| Run | Configuration | Stage A antecedent | Question |
|---|---|---|---|
| `b01-mask-light` | light masking, no clip-variance penalty | `a04-mask-light` | Does the downstream-preserving light-mask behavior persist at 100 epochs? |
| `b02-mask-light-clip-var` | light masking plus clip-variance penalty | `a04-mask-light` + `a07-clip-var` | Can pooled breadth, context sensitivity, and downstream utility coexist? |

Both runs used seed 0, the historical Health&Gait training split, 100 epochs, and 3,900
optimizer steps. They are new duration-matched runs, not continuations of the Stage A
checkpoints. The planned `b00-clip-var` result is absent from the compact result files,
so Stage B does not independently confirm the a07 configuration at 100 epochs.

The source analysis is
[04-writeup-stage-b-results.ipynb](../notebooks/04-writeup-stage-b-results.ipynb).
It validates the experiment contract against `results/checkpoint_histories.json`,
cross-checks selected values against the stored trajectories, and reconstructs every
figure below from aggregate checked-in results.

## Result in one paragraph

Stage B did not find one configuration with both properties. **b01 optimized the JEPA
loss while remaining narrow and nearly insensitive to the historical wrong-context
intervention. b02 substantially broadened the pooled representation and increased that
intervention response, but its speed and closed-set identity measures weakened, and it
underperformed the declared shortcut in the later legacy completion test.** The
clip-variance penalty therefore produced a real mechanistic change, but broader and
more context-sensitive features were not generally better downstream features.

## Training health

![Stage B loss, cosine similarity, effective rank, and wrong-context trajectories](images/stage-b-health-trajectories.png)

The 50 stored validation points per run show stable, distinct regimes:

- **b01 optimizes steadily.** Its validation loss reaches `0.3823` at the selected
  epoch 88 and remains `0.3823` at epoch 100. Effective rank plateaus near 11 of 384,
  and the wrong-context gap remains about `0.00035`.
- **b02 broadens steadily.** At the selected healthy epoch 84, effective rank is
  `75.19` of 384 and the wrong-context gap is `0.1136`. At epoch 100 those values remain
  similar (`75.87` and `0.1134`), while validation loss remains about `0.555`.
- **Checkpoint selection is not driving the qualitative conclusion.** Both endpoints
  are near their selected values. Longer training does not give b01 b02-like breadth,
  and it does not give b02 b01-like prediction loss.

The wrong-context measure here is historical: another participant's context replaces
the correct context, and the raw loss difference is averaged within and then across
subjects. It is not the revised proposal's normalized, geometry-matched intervention
on the common GaitLU holdout.

## Selected-checkpoint comparison

![Stage A antecedents and Stage B selected checkpoints](images/stage-b-selected-checkpoints.png)

| Run | Selected epoch | Validation loss | Effective rank | Rank ratio | Wrong-context gap | Relative gap | Closed-set identity | Held-out identity | Speed balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `a00-baseline` | 40 | 0.4772 | 9.83 | 2.56% | 0.000134 | 0.028% | 10.63% | 2.94% | 93.27% |
| `a04-mask-light` | 40 | 0.4912 | 9.99 | 2.60% | 0.000266 | 0.054% | 10.31% | 3.06% | 93.75% |
| `a07-clip-var` | 15 | 0.5111 | 25.22 | 6.57% | 0.05227 | 10.23% | 7.71% | 4.04% | 89.26% |
| `b01-mask-light` | 88 | 0.3823 | 10.86 | 2.83% | 0.000352 | 0.092% | 10.26% | 3.25% | 92.52% |
| `b02-mask-light-clip-var` | 84 | 0.5547 | 75.19 | 19.58% | 0.11363 | 20.49% | 5.79% | 4.84% | 88.41% |

### b01: longer light-mask training improves optimization, not representation regime

Relative to a04, b01:

- lowers selected validation loss by 22.2%, from `0.4912` to `0.3823`;
- raises effective rank by only 8.7%, from `9.99` to `10.86`, leaving 97.2% of the
  nominal 384-dimensional breadth unrealized under this ratio;
- raises the raw wrong-context gap from `0.000266` to only `0.000352`;
- leaves closed-set identity essentially unchanged (`10.31%` to `10.26%`);
- improves held-out retrieval by 0.18 percentage points (`3.06%` to `3.25%`); and
- lowers speed balanced accuracy by 1.23 points (`93.75%` to `92.52%`).

The answer to b01's question is therefore narrow: the light-mask model trains to a much
lower loss over 100 epochs, but longer training does not repair pooled collapse or
create substantial context dependence. The Stage A downstream profile largely
persists.

### b02: the pooled-feature repair works, but it is not a general solution

Relative to a07, b02:

- nearly triples effective rank, from `25.22` to `75.19`;
- more than doubles the raw wrong-context gap, from `0.05227` to `0.11363`;
- doubles the relative gap, from `10.23%` to `20.49%`;
- improves held-out identity retrieval by 0.80 points, from `4.04%` to `4.84%`;
- reduces closed-set identity by 1.91 points, from `7.71%` to `5.79%`;
- reduces speed balanced accuracy by 0.85 points, from `89.26%` to `88.41%`; and
- raises validation loss by 8.5%, from `0.5111` to `0.5547`.

Relative to b01, b02 has 6.9 times the effective rank and a 323-fold larger raw
wrong-context gap, but 4.47 points lower closed-set identity and 4.11 points lower speed
balanced accuracy. This is the clearest Stage B result: clip-variance regularization
changes the representation in the intended direction, yet the change does not dominate
the unregularized light-mask model across capabilities.

## Legacy Grounded Factorial Completion

The GFC summaries were produced after the 7/27 update. They compare a00, b01, and b02,
but they use `legacy_donor_excluded_v1`, not GFC-v2:

- 308 complete participants from the historical training group fit adapters and
  normalizers;
- 76 complete participants from the historical validation group are evaluated;
- each participant contributes 24 queries;
- both donors are removed, leaving a six-cell gallery; and
- the shortcut does not use the revised, fully matched three-head control.

![Legacy donor-excluded development GFC comparison](images/stage-b-legacy-gfc.png)

Under the declared historical `raw_retain_all` normalization:

| Checkpoint | Learned top-1 | Shortcut top-1 | Learned - shortcut | 95% participant bootstrap |
|---|---:|---:|---:|---:|
| A00 baseline | 69.79% | 65.46% | +4.33 pp | [+0.27, +8.22] pp |
| B01 light mask | 63.76% | 65.46% | -1.70 pp | [-5.98, +2.47] pp |
| B02 light mask + clip variance | 57.51% | 65.46% | -7.95 pp | [-12.17, -3.78] pp |

B01 does not separate from the shortcut under any stored normalization. B02 is below
the shortcut for `raw_retain_all` and `raw_effective_rank`; its
`pca_effective_rank` interval reaches just above zero but has a negative point estimate.
Meanwhile, changing normalization materially changes the estimated a00 advantage.

This gives a descriptive rank inversion across three checkpoints: b02 has the broadest
pooled features, largest historical context gap, and best held-out identity retrieval,
yet the worst legacy completion score. It demonstrates that those diagnostics cannot
stand in for a real-target recombination test. It does not estimate how data scale
affects representations, and it cannot validate or invalidate GFC-v2.

## Why these results are preliminary evidence only

These runs differ from the revised study in every major evidentiary role:

| Historical Stage B | Revised proposal |
|---|---|
| Encoders trained on Health&Gait | Encoders train only on GaitLU |
| One seed and selected checkpoints | Five four-rung ladders, 20 primary runs |
| Epoch-selected checkpoint | Final-step checkpoint primary |
| Horizontal flipping probability 0.5 | Flipping disabled because direction is evaluated |
| Raw Health&Gait context substitution | Normalized near-geometry substitution on common held-out GaitLU sequences |
| Six-cell donor-excluded legacy gallery | Full eight-cell GFC-v2 gallery |
| Possible target-source reuse | Both donors must differ from target `source_video_id` |
| Comparator-specific dimensional changes | Matched three-head learned and cue controls |
| Development result | Prospectively locked outcome cohort after analysis freeze |

The 7/27 label “health-gate winner” is therefore historical shorthand, not evidence that
a07 or b02 is the correct base configuration for the revised scaling experiment.
Similarly, a00's legacy advantage cannot be promoted into the revised study's abstract
or figures.

## Revised next steps

![Stage B evidence to revised study execution](images/stage-b-revised-next-steps.svg)

### 1. Finish and freeze the instrument

- Keep all eight cells in the primary gallery and use fractional top-1 and average-rank
  MRR for ties.
- Enforce that both complementary donors differ from the target's `source_video_id`.
- Validate the exact full-gallery oracle spectrum: 12.5% with no factors, 25% with one,
  50% with any two, and 100% with all three.
- Give learned and acquisition-cue inputs the same three ridge heads, labels, output
  dimensions, normalization, gallery, and tie policy.
- Add the hard and soft independent-factor completion controls and prespecified ridge
  sensitivities.

### 2. Prepare the GaitLU-only scaling data

- Validate decoding and silhouettes, remove exact duplicates, preserve known source
  groups, and audit near duplicates where metadata are missing.
- Reserve the common 10,000-sequence group-disjoint context and health holdout.
- Build five seeded nested ladders near 2,500, 25,000, 250,000, and the eligible maximum,
  recording actual counts and checksums.

### 3. Lock training before any outcome is opened

- Choose one architecture, mask policy, objective, augmentation policy, and exposure
  budget for all 20 runs. Stage B says not to choose among them using one diagnostic.
- Disable horizontal flipping because direction is an evaluated factor.
- Apply the throughput gate uniformly, then use the final checkpoint as primary.

### 4. Freeze the analysis package

Before unblinding, record the protocol version, gallery policy, query count, pool and
cohort checksums, exposure, seeds, exclusions, thresholds, figure templates, reference
checkpoints, and analysis-freeze commit. The 80-person cohort may fit adapters and
calibrators; the 318-person outcome cohort remains locked.

### 5. Evaluate all axes together

For every rung, report learned GFC-v2 top-1/MRR, matched cue control, hard/soft completion,
three factor probes, normalized context reliance, token and pooled rank, fixed
cross-condition identity rank-1/MRR, training health, and throughput. This prevents a
b02-like improvement on one axis from being mislabeled as global progress.

### 6. Make the prespecified run-level decision

The primary estimand is full-minus-small learned GFC-v2 top-1 across five ladders. One
of 16 queries is 6.25 percentage points. Report the result as meaningful positive,
positive but small, equivalent at that resolution, or inconclusive using the frozen
interval rules. Participant resampling is a sensitivity; it cannot replace the five
trained-model replicates.

## Claims boundary

Current evidence supports that:

- light masking and clip-variance regularization produce distinct representation
  regimes;
- clip variance repairs pooled breadth and increases the historical context-substitution
  response;
- optimization loss, breadth, context sensitivity, speed decoding, identity, and legacy
  completion rank these checkpoints differently; and
- those disagreements motivate the revised full-gallery, session-safe scaling audit.

Current evidence does not support that:

- more unique GaitLU data improves GFC-v2, context reliance, or identity;
- b01 or b02 is a confirmed compositional representation;
- GFC-v2 exceeds independent-factor controls;
- the locked outcome cohort confirms the preliminary result;
- the representation is causally disentangled; or
- any model is appropriate for clinical or identity-sensitive deployment.

## Reproduction

Run the notebook from the repository root:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook \
  --output-dir /tmp/cody-jepa-stage-b \
  notebooks/04-writeup-stage-b-results.ipynb
```

By default, exact plotted values are written below
`results/generated/writeup-stage-b/`, and writeup images are written to `docs/images/`.
`CODY_JEPA_REPRO_OUTPUT_DIR` and `CODY_JEPA_WRITEUP_IMAGE_DIR` can override those
destinations for a clean reproduction.
