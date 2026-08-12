# CoDy-JEPA

A single-stream video JEPA (Joint-Embedding Predictive Architecture) for self-supervised
representation learning on gait video, reduced to its core parts: the model, the training
engine, dataset loading, and frozen-representation evaluation.

This repository is deliberately a foundation, not a study. Earlier research directions
(Grounded Factorial Completion, the hierarchical-diversity scaling campaign) and their
result artifacts have been removed. Everything before that removal is recoverable from the
`archive/pre-refactor` git tag.

## What's here

```
src/cody_jepa/
  models/       ViT encoder, predictor, attention blocks, 3D sin-cos position embedding
  masks/        Multiblock context/target mask sampling
  training/     Training engine, VICReg + prediction losses, EMA/LR schedules,
                checkpointing, AMP runtime, representation diagnostics
  data/         Manifest schema, frame discovery, Health&Gait and GaitLU-1M loaders
  evaluation/   Frozen feature export, linear probes (gait system, identity)
  cli/          Five entry points wiring the above together
```

The dependency direction is one-way: `evaluation` and `cli` depend on `models`,
`masks`, `training`, and `data`; those four never depend on evaluation or CLI code.

### Model and objective

A Vision Transformer encodes masked video clips; a narrower predictor maps context
embeddings to target-block embeddings produced by an EMA target encoder. The loss combines
prediction error with VICReg variance/covariance terms to prevent representation collapse.
Training is instrumented with representation-health diagnostics (effective rank, cosine
similarity, context-substitution checks) so collapse is visible while it happens rather
than after.

### Datasets

Two loaders, both deterministic and seed-controlled:

- **Health&Gait** — silhouette / optical-flow / segmentation recordings with a
  subject-disjoint manifest. Factors are `speed`, `clothing`, `direction`; the manifest
  also carries non-learned shortcut features so probes can be checked against trivial
  baselines.
- **GaitLU-1M** — seekable, bit-packed tar shards with a fixed-exposure sampler, for
  large-scale pretraining. Shards must be prepared first (`cody-jepa-prepare-gaitlu`).

### Evaluation

Frozen features are exported from a trained checkpoint's target encoder, then evaluated
with linear probes: gait-system classification, closed-set identity, and held-out identity
retrieval. Probes report against majority-class and shortcut-feature baselines.

## Setup

Requires Python 3.10+ and `uv`.

```bash
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
uv sync --frozen
```

Run the tests (no private data required):

```bash
uv run python -m unittest discover -s tests
```

## Usage

Build a Health&Gait manifest from raw recordings:

```bash
uv run cody-jepa-build-manifest --fps 30
```

Train:

```bash
uv run cody-jepa-train \
  --config configs/train/healthgait_baseline.json \
  --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
  --output-dir outputs/run-01
```

Export frozen features and run probes:

```bash
uv run cody-jepa-export-features \
  --checkpoint outputs/run-01/best.pt \
  --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
  --output outputs/run-01/features.npz

uv run cody-jepa-eval-probes \
  --features outputs/run-01/features.npz \
  --output-dir outputs/run-01/probes
```

On SLURM, see `slurm/train-jepa.sbatch` and `slurm/prepare-gaitlu-shards.sbatch`.

## Data and claim boundaries

Raw recordings, manifests, participant tables, feature exports, and checkpoints stay
outside Git. Anything committed should be aggregate, non-identifying, and reproducible
from a script in this repo.
