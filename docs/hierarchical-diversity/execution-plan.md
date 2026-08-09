# Execution Plan: Hierarchical Video Diversity

This plan tracks the implementation and deadline work for the proposed
[hierarchical-diversity method](method.md). The scientific protocol belongs in the
methods document. This file records what must change in the current repository before
that protocol can be called executable.

## Current implementation gap

The repository currently implements the unique-sequence scaling study:

- GaitLU training always resamples from every valid temporal start;
- pool finalization requires five seeds and four rungs;
- the private registry and training launcher require exactly 20 rows;
- GFC preflight expects five named ladders and four rung names;
- aggregate inference and rendering produce ladder contrasts;
- the 4.096-million-example fallback configuration is not checked in; and
- batch feature-export orchestration for even the 20-model study is not yet available.

The hierarchical protocol must replace these assumptions prospectively. Outcome data
cannot influence which changes are completed or which design is launched.

## Required implementation

### 1. Temporal anchors and loader contract

Update `src/cody_jepa/data/gaitlu.py` to construct starts at 8-frame spacing and expose
two training policies: `frozen_random` and `resampled_anchor`. Frozen selection must use
a versioned stable digest of sequence ID and replicate seed. Resampling must use a
separate temporal stream. Policy pairs must receive identical sampler indices, spatial
parameters, and masks.

Add the anchor spacing, policy, seed version, and eligibility rule to loader
configuration, dataset descriptions, checkpoint metadata, and resume validation.

### 2. Preparation, eligibility, and manifests

Update `src/cody_jepa/data/gaitlu_prepare.py` to apply the global rule of at least two
separated anchors before holdout and pool construction. Build one common holdout near
10,000 group-disjoint sequences, then create eight seeded nested pairs near 2,500 and
250,000 sequences.

The training registry must contain exactly four cells per replicate block and include
`replicate`, `sequence_support`, `window_policy`, actual sequence count, manifest digest,
optimization seed, exposure, anchor spacing, and seed-scheme version. Each manifest is
reused across its two policy cells.

### 3. Training configuration and launch

Generalize `src/cody_jepa/cli/train_gaitlu_study.py` from 20 ladder rows to 32 factorial
rows. Add a hierarchy-compatible 4.096-million-example configuration before the
throughput decision. Update `slurm/train-gaitlu-study.sbatch` to launch four waves of
eight jobs, with two complete replicate blocks per wave. Randomize and record cell-to-GPU
and cell-to-node assignments.

### 4. Feature export and evaluation registry

Add batch feature-export orchestration for all 32 final-step checkpoints. Generalize the
private GFC registry and preflight from named ladders and rungs to eight complete
factorial blocks. Preflight must reject missing cells, duplicate cells, policy mismatches,
manifest mismatches, non-final checkpoints, unequal exposure, and incomplete outcome
archives.

### 5. Factorial inference and rendering

Update `src/cody_jepa/evaluation/gfc/study.py` and
`src/cody_jepa/cli/make_gfc_study_results.py` to emit cell outcomes, replicate-level
simple effects, the primary interaction, the direct allocation contrast, the
completion-gap interaction, ceiling sensitivity, and every frozen decision gate.
Private participant rows remain confined to the existing aggregate boundary.

The renderer must produce a 32-run condition table, eight replicate factorial plots,
the primary interaction interval, simple-effect intervals, completion-gap decisions,
and training-health summaries. It must reject ladder-form or mixed-protocol inputs.

## Verification gates

Before protocol freeze, automated tests must verify:

- anchors are eight frames apart and eligible sequences have at least two anchors;
- a frozen anchor is stable across epochs, draws, workers, and nested manifests;
- resampled anchors change and stay within the same anchor set;
- policy pairs have identical sequence-index, spatial-transform, and mask streams;
- the lower exposure tier passes the fourfold support gate in both sequence pools;
- the registry contains eight complete four-cell blocks and no extra cells;
- resume rejects any policy, anchor, seed, manifest, or exposure mismatch;
- constructed cell outcomes recover the declared interaction sign and every decision
  category;
- hard and soft completion top-1 agree in the complete factorial gallery;
- the completion-gap interval has materially nonzero, equivalent, and unresolved test
  cases; and
- feature export, private preflight, aggregation, and rendering run end to end without
  Health&Gait outcome aggregates.

The systems pilot must run all four cells. The throughput probe must run eight jobs
concurrently on the high pool because a single job cannot reveal shared-storage
contention.

## Compute plan

Eight continuously reserved H100s run two complete replicate blocks per wave. Four
waves complete 32 models. At 8.192 million examples per model, the experiment processes
262.144 million examples. The 60-example-per-second boundary gives about 6.3 elapsed
days. The 4.096-million fallback has the same duration at its 30-example-per-second
boundary.

The reservation must cover primary training and leave at least two days for exact reruns
of documented systems failures. New seeds may not replace failed cells after outcomes
are opened.

## Deadline schedule

| Date | Required exit condition |
|---|---|
| August 8 to 12 | GaitLU inventory complete; global anchor eligibility and support audit pass |
| August 10 to 14 | Loader, finalizer, 32-row registry, checkpoint contract, inference, and renderer implemented |
| August 14 to 16 | Four-cell pilot, eight-job throughput probe, sensitivity check, and outcome-blind dry run pass |
| August 16 | Protocol, code, exposure, margins, and GPU reservation frozen |
| August 17 to 24 | Four primary training waves complete |
| August 25 to 27 | Exact systems-failure reruns and 32-checkpoint feature export complete |
| August 28 to September 6 | Private preflight and outcome-blind analysis rehearsal pass |
| September 7 | Locked outcome aggregates opened once |
| September 7 to 18 | Results, controls, audit, paper, and abstract completed |
| September 19 to 25 | Reproducibility, anonymity, references, and submission checks completed |

## Go or no-go rule

The hierarchical study launches only if every implementation and verification gate
passes by August 16, all eight H100s are reserved continuously, and concurrent storage
meets the throughput tier. Otherwise, the project retains the implemented 20-model
unique-sequence scaling study.

The two protocols cannot be pooled. The project cannot reduce replication, change
anchor spacing, select a different high pool, or redefine substitution after inspecting
Health&Gait outcomes.
