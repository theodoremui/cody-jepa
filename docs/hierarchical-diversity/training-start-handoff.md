# Agent Handoff: Implement Hierarchical-Diversity Training Locally

## Objective

Implement the hierarchical-diversity training experiment in the local repository. When
this task is complete, the user should only need to finish preparing GaitLU-1M, run the
real support audit, generate the real manifests, choose the exposure tier on HAIC, and
start the 32 training jobs.

Work only with local code and synthetic test data. Do not connect to HAIC, inspect
private GaitLU-1M data, run GPU throughput tests, or access Health&Gait outcomes.

Read before editing:

1. `docs/hierarchical-diversity/method.md`
2. `src/cody_jepa/data/gaitlu.py`
3. `src/cody_jepa/data/gaitlu_prepare.py`
4. `src/cody_jepa/cli/train_gaitlu_study.py`
5. `configs/train/gaitlu_scaling.json`

## Scope

Implement only these five deliverables:

1. temporal-anchor sampling with frozen and resampled policies;
2. hierarchy pool, manifest, and registry generation;
3. full, fallback, and smoke training configurations;
4. a hierarchy-specific local training entry point; and
5. focused synthetic tests and a four-cell local smoke run.

Do not implement feature export, GFC evaluation, factorial result aggregation, paper
figures, result rendering, or additional datasets and objectives. Do not add new artifact
tracking, hashing, provenance, tagging, or release-management systems. Leave existing
manifest validation and checkpoint behavior intact where the current training path
already uses them.

Preserve the existing unique-sequence training path. Add hierarchy-specific behavior
without changing its current defaults or deleting its tests.

## Experiment definition

The production experiment contains:

- two sequence-support levels: approximately 2,500 and 250,000 sequences;
- two temporal-window policies: `frozen_random` and `resampled_anchor`;
- eight replicate blocks;
- four cells per block; and
- 32 models total.

All four cells in a replicate use the same encoder, objective, optimization settings,
mask generation, spatial augmentations, initialization seed, and sampled-clip exposure.
Within a fixed sequence-support pair, the window policy is the only intended difference.

Use 16-frame clips. Allowed temporal starts are separated by eight frames:

```text
0, 8, 16, ... through the last valid 16-frame window
```

Hierarchy training sequences must have at least two allowed starts.

## Task 1: Implement temporal-anchor sampling

Update `src/cody_jepa/data/gaitlu.py`.

### Loader interface

Add hierarchy training options to `GaitLULoaderConfig` and pass them into the training
dataset:

```text
train_window_policy
anchor_spacing
replicate_seed
```

The existing unique-sequence caller must retain its current unrestricted random-window
behavior when these options are not supplied. Validation sampling must remain unchanged.

Validate that:

- `anchor_spacing` is positive;
- the hierarchy policies are accepted only for training;
- a hierarchy sequence has at least two allowed anchors; and
- unknown policy names fail clearly.

### Frozen-random policy

For `frozen_random`:

- choose one allowed anchor uniformly from the sequence's anchor set;
- derive the choice deterministically from `sequence_id` and `replicate_seed`;
- reuse it for every draw and epoch; and
- make it independent of manifest row position so the same sequence receives the same
  anchor in nested low and high manifests.

### Resampled-anchor policy

For `resampled_anchor`:

- choose uniformly from the same allowed anchor set on every training draw;
- include epoch, stable sequence identity, and draw index in the temporal random state;
  and
- keep temporal sampling separate from spatial augmentation and sequence sampling.

Do not consume global Python, NumPy, or Torch randomness when selecting temporal
anchors. Existing paired seeds must continue to produce the same sequence indices and
spatial transformations across the two policies.

### Minimum tests

Add focused tests showing that:

- the anchor set is correct for sequences of 16, 24, 32, and 40 frames;
- sequences with fewer than two anchors are rejected for hierarchy training;
- a frozen anchor is unchanged across draws, epochs, workers, and nested manifests;
- resampled anchors stay in the allowed set and vary across repeated draws;
- repeated construction with the same seed returns the same results;
- a changed replicate seed can change the frozen anchor; and
- paired frozen and resampled datasets return the same spatial transformation for the
  same sequence and draw; and
- identical training seeds preserve the same mask-draw sequence across policies.

## Task 2: Implement hierarchy pool and registry generation

Add a hierarchy-specific preparation module, such as
`src/cody_jepa/data/gaitlu_hierarchy.py`, and a small CLI wrapper. It should consume the
existing finalized `inventory.csv`. Use synthetic inventories in local tests.

### Selection behavior

The finalizer must:

1. keep rows already marked exact-content eligible;
2. apply the global requirement of at least two eight-frame anchors;
3. select one common holdout near 10,000 sequences using whole source groups;
4. construct eight independently ordered training pools;
5. select a low pool near 2,500 and high pool near 250,000 for each replicate;
6. require the high pool to strictly contain the low pool; and
7. write one low and one high manifest per replicate.

Each manifest is reused by both window policies. This produces 16 training manifests,
not 32 duplicated manifests.

Production defaults must be configurable so local tests can use small synthetic targets:

```text
replicate seeds: 0 through 7
holdout target: 10,000
low target: 2,500
high target: 250,000
clip length: 16
anchor spacing: 8
training exposure: supplied by the selected configuration
```

### Registry

Write exactly 32 production rows with these fields:

```text
model_label
replicate
sequence_support
window_policy
train_manifest
val_manifest
pool_seed
optimization_seed
replicate_seed
unique_sequences
training_exposure
anchor_spacing
```

For each replicate, write these four cells:

```text
low, frozen_random
low, resampled_anchor
high, frozen_random
high, resampled_anchor
```

Use the same optimization and replicate seed for all four cells in one replicate. Use
the same low manifest for both low cells and the same high manifest for both high cells.

Validate:

- eight replicates and four unique cells per replicate;
- exactly 32 rows;
- sequence counts match the requested finalizer targets and low is smaller than high;
- low groups are a strict subset of high groups;
- manifests match within each window-policy pair;
- all rows use the same training exposure; and
- model labels are unique.

Keep hierarchy outputs in a separate subdirectory such as `prepared/hierarchy/` so the
existing scaling-study outputs remain usable.

### Minimum tests

Using a small synthetic inventory, test:

- filtering by temporal eligibility;
- whole-group holdout selection;
- strict low-within-high nesting;
- manifest reuse across policies;
- eight complete four-cell blocks;
- deterministic output from the same seeds; and
- rejection of insufficient, duplicated, missing, or non-nested cells.

## Task 3: Add training configurations

Add three hierarchy configurations under `configs/train/`.

### Full exposure

Create `gaitlu_hierarchy_full.json` from the current scaling configuration:

```text
batch_size: 16
accumulation_steps: 4
effective batch size: 64
steps: 128000
loader_epoch_examples: 65536
num_epochs: 125
total sampled clips: 8192000
```

Keep the existing architecture, masks, optimizer, augmentations, and horizontal-flip
setting.

### Fallback exposure

Create `gaitlu_hierarchy_half.json`:

```text
batch_size: 16
accumulation_steps: 4
effective batch size: 64
steps: 64000
loader_epoch_examples: 64000
num_epochs: 64
total sampled clips: 4096000
```

Scale step-based warmup from 2,000 to 1,000 updates. Keep the model, loss, masks,
augmentations, and optimizer settings otherwise unchanged. All models will use either
the full configuration or the fallback configuration, never a mixture.

### Local smoke configuration

Create `gaitlu_hierarchy_smoke.json` with one optimizer step, effective batch size 64,
and CPU-compatible settings. It must exercise the real hierarchy loader and training
entry point but must not resemble a primary run output.

Add a small configuration test that verifies the exact sampled-clip totals above.

## Task 4: Add a hierarchy training entry point

Add `src/cody_jepa/cli/train_gaitlu_hierarchy.py`, a wrapper in `scripts/`, and a project
CLI entry such as `cody-jepa-train-gaitlu-hierarchy`.

The command must accept:

```text
--registry
--run-index
--config
--data-root
--output-root
--repo-root
--num-workers
--device
```

It must:

1. load and validate the complete 32-row hierarchy registry;
2. select exactly one row by `--run-index`;
3. verify that configuration exposure equals the registry exposure;
4. pass `window_policy`, `anchor_spacing`, and `replicate_seed` into the GaitLU loader;
5. use the row's optimization seed;
6. write to `output_root/model_label`;
7. refuse to overwrite an existing run directory; and
8. require training to reach the configuration's declared final step before returning
   success.

Reuse the existing training engine and checkpoint code. Do not build a new trainer or
add new provenance infrastructure.

Add tests for:

- valid row selection from a 32-row registry;
- rejection of the old ladder registry;
- rejection of incomplete or duplicate hierarchy cells;
- exposure mismatch;
- propagation of window policy, spacing, replicate seed, and optimization seed; and
- refusal to overwrite an existing output.

## Task 5: Run a local four-cell smoke test

Create a tiny synthetic prepared dataset with enough sequences and frames to exercise
both sequence-support levels and both window policies. Use reduced pool targets in the
synthetic hierarchy finalizer, but preserve the four-cell structure.

Run one smoke step for:

```text
low, frozen_random
low, resampled_anchor
high, frozen_random
high, resampled_anchor
```

Use CPU and separate temporary output directories. Confirm that:

- all four cells load their intended manifest and policy;
- all four complete the declared smoke step;
- the frozen cells reuse their selected anchors;
- the resampled cells draw only from allowed anchors;
- paired policies use the same sequence ordering and spatial transforms; and
- each cell writes a usable final checkpoint.

Do not implement feature export or downstream evaluation as part of this smoke test.

## Local completion criteria

The local implementation is complete when:

- both temporal policies work and pass focused tests;
- the hierarchy finalizer creates 16 manifests and a valid 32-row production registry;
- full, fallback, and smoke configurations have correct exposure arithmetic;
- the hierarchy training CLI can execute any registry row;
- all four synthetic smoke cells finish successfully; and
- the existing unique-sequence loader and training tests still pass.

Run at least:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_gaitlu*.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_train_gaitlu_hierarchy.py' -v
```

## Next steps for the user on HAIC

Do not perform these tasks locally. Return them to the user after implementation:

1. finish converting all 100 GaitLU-1M shards;
2. finalize and review the real exact-content-deduplicated inventory;
3. run the real hierarchical-support audit and stop if it fails;
4. run the new hierarchy finalizer on the real inventory;
5. run the four-cell systems pilot on H100s;
6. measure eight-job shared-storage throughput and choose the full or fallback exposure;
7. regenerate the real 32-row registry with the selected exposure; and
8. launch the 32 jobs in four waves of eight.

## Required handoff response

Report:

1. `LOCAL_HIERARCHY_SETUP_READY: YES` or `LOCAL_HIERARCHY_SETUP_READY: NO`;
2. files added or changed;
3. focused test commands and results;
4. whether the four synthetic smoke cells completed; and
5. any remaining local blocker before the user continues on HAIC.
