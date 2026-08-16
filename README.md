# CoDy-JEPA

JEPA for self-supervised gait representation learning.

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
and creates subject-disjoint train, tuning, and test splits. By default, they are 80/10/10:

```bash
python build_manifest.py --raw-root data/healthgait/raw/Health_Gait
```

**2. Train:**

```bash
python train.py --config configs/healthgait.json \
    --manifest data/healthgait/manifest.csv --output-dir outputs/run-01
```

Each epoch prints train loss and, on eval epochs, tuning loss, **clip effective rank**,
and the loss delta from replacing context with an all-zero video. Effective rank is the
exponentiated entropy of the covariance spectrum on the same EMA target features exported
to probes. It equals the embedding dimension for isotropic features and collapses toward 1
when every clip maps to the same direction. Watch it: a falling clip rank, or a near-zero
blank-context delta, means the run is solving the wrong problem.

Writes `last.pt`, `best.pt`, and `history.json`. Resume with `--resume outputs/run-01/last.pt`.

**3. Evaluate.** Export frozen EMA-target features, then probe them:

```bash
python export_features.py --checkpoint outputs/run-01/best.pt \
    --manifest data/healthgait/manifest.csv --output outputs/run-01/features.csv \
    --random-init-output outputs/run-01/random-init-features.csv

python probe.py --features outputs/run-01/features.csv \
    --random-init-features outputs/run-01/random-init-features.csv
```

Two probes, both reported against a majority-class baseline:

| Probe | What it asks | Protocol |
|---|---|---|
| `gait_system` | Is walking speed linearly decodable? | Held-out test participants |
| `identity` | Is the participant linearly decodable? | Held-out source videos |

Both are disjoint by design — training uses only training participants, checkpoint selection
uses the tuning participants, and `gait_system` is reported only on test participants.
`identity` holds out whole source videos so a score cannot come from matching the same
recording to itself.

## Configuration

One JSON file per run (`configs/healthgait.json`). Model geometry must satisfy
`img_size % patch_size == 0`, `num_frames % tubelet_size == 0`, and `embed_dim % 6 == 0`
(the position embedding splits channels three ways). A test enforces this.

Silhouette clips are foreground-cropped over the whole temporal window, padded to preserve
aspect ratio, then resized to the model input. The default `var_coef` and `cov_coef`
regularize pooled clip features; set `token_var_coef` and `token_cov_coef` only when you
also want token-axis VICReg.

## Data boundaries

Raw recordings, manifests, feature exports, and checkpoints stay out of Git. Anything
committed should be aggregate and reproducible from a script here. Checkpoints are loaded
in PyTorch's tensor-only mode; use checkpoints from known sources.

Earlier research directions are recoverable from the `archive/pre-refactor` tag.
