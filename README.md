# CoDy-JEPA

A single-stream masked video JEPA for self-supervised gait representation learning.

A Vision Transformer encodes visible "context" tubes of a silhouette video; a predictor
maps them to the embeddings of masked "target" tubes produced by an EMA copy of the same
encoder. Nothing is reconstructed in pixel space — the loss lives entirely in embedding
space, with VICReg variance/covariance terms preventing the degenerate constant solution.

The whole thing is ~1,100 lines. Everything in it either trains a model or measures one.

## Structure

```
train.py              Train a JEPA
export_features.py    Export frozen features from a checkpoint
probe.py              Run linear probes on a feature table
build_manifest.py     Build a subject-disjoint manifest from raw frames

cody_jepa/
  models.py           ViT encoder, predictor, 3D sin-cos position embedding
  masks.py            Multi-block tube masking
  data.py             Manifest reading and clip loading
  losses.py           Prediction loss and VICReg
  engine.py           Training loop, schedules, validation, checkpoints
  evaluation.py       Feature export and linear probes
```

## Setup

```bash
uv sync
uv run python -m unittest discover -s tests
```

The test suite runs on a synthetic corpus and needs no private data.

## Usage

**1. Build a manifest.** Walks `<raw-root>/<modality>/<participant>/<speed>/<clothing>_<direction>/`
and holds out whole participants for validation:

```bash
python build_manifest.py --raw-root data/healthgait/raw/Health_Gait --fps 30
```

**2. Train:**

```bash
python train.py --config configs/healthgait.json \
    --manifest data/healthgait/manifest.csv --output-dir outputs/run-01
```

Each epoch prints train loss and, on eval epochs, validation loss and **effective rank** —
the exponentiated entropy of the covariance spectrum. It equals the embedding dimension for
isotropic features and collapses toward 1 when every clip maps to the same direction. Watch
it: a falling effective rank means the run is collapsing, and no amount of falling loss
redeems that.

Writes `last.pt`, `best.pt`, and `history.json`. Resume with `--resume outputs/run-01/last.pt`.

**3. Evaluate.** Export frozen EMA-target features, then probe them:

```bash
python export_features.py --checkpoint outputs/run-01/best.pt \
    --manifest data/healthgait/manifest.csv --output outputs/run-01/features.csv

python probe.py --features outputs/run-01/features.csv
```

Two probes, both reported against a majority-class baseline:

| Probe | What it asks | Protocol |
|---|---|---|
| `gait_system` | Is walking speed linearly decodable? | Held-out participants |
| `identity` | Is the participant linearly decodable? | Held-out source videos |

Both are disjoint by design — `gait_system` never sees a validation participant during
training, and `identity` holds out whole source videos so a score cannot come from matching
the same recording to itself.

## Configuration

One JSON file per run (`configs/healthgait.json`). Model geometry must satisfy
`img_size % patch_size == 0`, `num_frames % tubelet_size == 0`, and `embed_dim % 6 == 0`
(the position embedding splits channels three ways). A test enforces this.

## Data boundaries

Raw recordings, manifests, feature exports, and checkpoints stay out of Git. Anything
committed should be aggregate and reproducible from a script here.

Earlier research directions (Grounded Factorial Completion, hierarchical-diversity scaling)
and the GaitLU-1M loader were removed; they are recoverable from the `archive/pre-refactor`
tag.
