# Results

## Current evidence boundary

The checked-in aggregate results answer a narrow preliminary question: do existing single-stream checkpoints receive the same ranking from prediction loss, representation diagnostics, context-use diagnostics, and linear probes? They do not.

No empirical Grounded Factorial Completion result has been run from the compact pipeline yet. The repository therefore makes no claim that learned features beat shortcuts, recover instrumented gait structure, or transfer to another dataset.

## Phase 0 baseline

The seed-0 baseline used 2,506 training sequences from 318 participants and 624 validation sequences from 80 disjoint participants. Three deterministic windows per sequence produced 7,518 training feature rows and 1,872 validation rows.

At the epoch-80 best-loss checkpoint, the subject-balanced validation loss was `0.387394`, pooled effective rank was `10.452` out of 384, and the wrong-context gap was `0.000154`. Closed-set identity accuracy was `9.25%`, held-out identity retrieval was `2.45%`, and instructed-speed balanced accuracy was `92.57%`.

The endpoint checkpoint at epoch 100 had nearly the same validation loss and representation breadth. These values establish feasibility for training and feature extraction, not semantic factorization.

The compact source is `results/phase0_summary.json`; the generator writes the active table to [phase0_table.csv](../results/generated/phase0_table.csv).

## Phase 1 checkpoint comparison

The Phase 1 sweep changes learning rate, target-encoder momentum, mask difficulty, predictor depth, and pooled clip-variance regularization. All reported rows use seed 0 and the same subject split.

Different measurements favor different checkpoints:

- `a03-ema0.995` has the lowest selected-checkpoint validation loss among Stage A rows (`0.3818`) but only `1.35%` effective-rank ratio and a near-zero context gap.
- `a05-mask-heavy` has the best Stage A closed-set identity accuracy (`10.74%`).
- `a04-mask-light` has the best Stage A instructed-speed balanced accuracy (`93.75%`).
- `a07-clip-var` increases the effective-rank ratio to `6.57%` and held-out retrieval to `4.04%`, while instructed-speed balanced accuracy falls to `89.26%`.
- `b02-mask-light-clip-var` reaches the broadest pooled representation (`19.58%` effective-rank ratio), largest wrong-context gap (`0.1136`), and best held-out retrieval (`4.84%`) across all reported rows, but its instructed-speed balanced accuracy is `88.41%`.
- `b01-mask-light` has the lowest selected-checkpoint validation loss among Stage B rows (`0.3823`) and retains `92.52%` instructed-speed balanced accuracy, but its effective-rank ratio is only `2.83%`.

These disagreements motivate an observed completion task with an explicit shortcut comparator. They do not show that GFC will prefer a scientifically better checkpoint.

The compact row-level source is `results/phase1_summary.csv`. The active outputs are [phase1_table.csv](../results/generated/phase1_table.csv) and the diagnostic comparison below.

![Phase 1 validation loss and effective-rank ratio](../results/generated/phase1_diagnostics.png)

## Context and pooling diagnosis

The context diagnosis separates token-level diversity from recording-level diversity. The combined context-token representation has effective rank `381.58` out of 384, while the pooled online representation has effective rank `10.44`. Token diversity therefore did not guarantee a broad pooled recording representation.

Replacing context with another participant's clip increased loss by only about `0.000156` on average. Same-participant replacement produced a similar gap (`0.000161`), and temporal shuffling produced a smaller one (`0.0000475`). This does not support a claim that the predictor isolated identity or motion.

Only `9.62%` of masked target tokens were foreground under the diagnostic threshold. The foreground-only gap (`0.0000467`) was smaller than the background-only gap (`0.0001679`). This motivates controls for acquisition and framing cues; it does not prove that any particular shortcut predicts a factor.

The compact source is `results/context_diagnosis.json`.

![Context substitution gaps and token-versus-pooled breadth](../results/generated/context_diagnosis.png)

## Regenerating tables and figures

The paper-result generator reads the compact files directly:

```bash
uv run python scripts/make_paper_results.py \
  --results-dir results \
  --output-dir results/generated
```

It does not consume notebooks or prose reports as data sources.

## What can be claimed now

Current aggregate evidence supports these statements:

- the baseline model trains and exports usable features on the recorded split;
- token-level and pooled representation breadth can disagree substantially;
- context sensitivity, representation breadth, validation loss, and probes select different checkpoints;
- existing results are insufficient to infer compositional factor structure.

Current evidence does not support these statements:

- learned GFC beats the shortcut baseline;
- a positive completion score reflects instrumented gait rather than acquisition cues;
- one checkpoint is universally best;
- CoDy-JEPA has clinical value or causal factor separation;
- findings transfer beyond the present Health&Gait setup.

## Limitations

All Phase 0/1 training and probe rows use one seed. Validation windows share recordings and participants, so their row count is not the inferential sample size. Probe performance measures recoverability, not causality or disentanglement. The Health&Gait factorial design has no repeated recording within a cell, and its instrumented measurements are not synchronized with each video pass.

Future empirical GFC results must report complete-case participant counts, exclusions, participant-level uncertainty, learned and shortcut scores on identical queries, and both configured normalization sensitivities.
