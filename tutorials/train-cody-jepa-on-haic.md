# Train the single-stream JEPA prototype on HAIC

This guide runs [`notebooks/single-stream-jepa.ipynb`](../notebooks/single-stream-jepa.ipynb) on a Stanford HAI Compute Cluster GPU. The notebook is a Health&Gait feasibility baseline: it predicts masked latent video features with one stream. It is not the final dual-stream CoDy-JEPA experiment.

## What the notebook validates

The run is successful only if all of these hold:

- JEPA validation loss improves.
- Validation feature variance and effective rank remain nontrivial.
- Shuffling video context increases prediction loss.
- Train and validation subjects remain disjoint.
- `latest.pt` can resume exactly at an epoch boundary.
- `best_loss.pt` preserves the lowest subject-balanced validation loss.
- `best_healthy.pt` preserves the best checkpoint that also passes rank, variance, norm, and wrong-subject-context checks.

Loss alone is insufficient because a collapsed or position-only model can still produce a deceptively smooth curve.

## Current run shape

| Setting | Default | Meaning |
| --- | ---: | --- |
| Physical batch | `16` | Clips resident for one microbatch. |
| Accumulation | `4` | Effective batch is 64 clips. |
| Frames | `16` | Frames per clip. |
| Resolution | `112x112` | Grayscale silhouette input. |
| Tubelet / patch | `2 / 8` | Produces 1,568 video tokens. |
| Epochs | `100` | 39 equal-sized optimizer updates per epoch. |
| Optimizer steps | `3,900` | Exact epoch-boundary stopping point. |
| Validation cadence | every 5 epochs | Three deterministic windows per validation sequence. |
| Checkpoints | every epoch | Atomic `latest.pt`, `best_loss.pt`, and health-gated `best_healthy.pt`. |

The physical batch is intentionally much smaller than the old notebook default. Gradient accumulation preserves an effective batch of 64 without retaining a batch-256 predictor graph. Training drops only the final incomplete physical batch each epoch so every optimizer/EMA update has the same 64-example weight.

## Prepare the cluster workspace

From the HAIC head node:

```bash
cd /hai/scratch/$USER
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
uv sync --frozen
```

The prepared dataset and manifest must exist beneath:

```text
data/healthgait/raw/Health_Gait/silhouette/
data/healthgait/manifests/silhouette_subject_split_seed0.csv
```

Follow [`health-and-gait.md`](health-and-gait.md) if they are missing.

## Run the safety checks first

Request a short interactive GPU allocation using the account and partition assigned to you:

```bash
srun \
  --account=<ACCOUNT> \
  --partition=hai-interactive \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=04:00:00 \
  --pty bash
```

Then run:

```bash
nvidia-smi
uv run python -m unittest discover -s tests -v
```

Execute the notebook once with its default `RUN_FULL_TRAINING = False`. This validates the real manifest, sampled image integrity, subject isolation, and model geometry; it also decodes one deterministic clip from every train and validation sequence to reject blank or static sources, then runs a tiny CPU training loop without starting the expensive job:

```bash
mkdir -p notebook-runs
MPLCONFIGDIR=/tmp/mpl \
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output single-stream-jepa-preflight.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Do not continue if preflight raises an error. In particular, the loader refuses blank identities, split leakage, frame paths outside `data/healthgait`, duplicate frame sources, corrupt sampled images, blank/static sequence probes, and clips that would cross missing-frame gaps.

## Enable the full run

Near the top of the notebook, change only:

```python
RUN_FULL_TRAINING = True
```

Keep `CONFIG["required_device"] = "cuda"`. This makes a missing GPU fail immediately instead of silently starting a multi-day CPU run. Keep `compile=False` for the first successful eager run; compilation is an optional later benchmark.

The flag is defined before dataset construction. Enabling it automatically changes the loader from sampled verification/fingerprinting to decoding and content-hashing every frame. The full job refuses to start unless both production integrity modes are active, so a corrupt nonsampled frame fails before optimization and exact resume is bound to every frame byte.

If `outputs/single-stream-jepa-healthgait-v3/latest.pt` already exists, the notebook refuses to overwrite it unless you explicitly resume or choose a new output directory.

## Submit a batch job

Create `slurm/train-single-stream-jepa.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=single-stream-jepa
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=hai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/single-stream-jepa-%j.out

set -euo pipefail
cd /hai/scratch/$USER/cody-jepa
mkdir -p logs notebook-runs outputs/single-stream-jepa-healthgait-v3

nvidia-smi
MPLCONFIGDIR=/tmp/mpl \
uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output single-stream-jepa-haic.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Submit and monitor it:

```bash
mkdir -p logs notebook-runs
sbatch slurm/train-single-stream-jepa.sbatch
squeue -u "$USER"
tail -f logs/single-stream-jepa-<JOBID>.out
```

## Resume an interrupted run

The notebook writes `latest.pt` only after a complete epoch, so the saved DataLoader, mask, Torch, optimizer, scaler, and EMA states share one exact boundary. Set:

```python
RUN_FULL_TRAINING = True
RESUME_CHECKPOINT = OUTPUT_DIR / "latest.pt"
```

Resume validation rejects a changed architecture, mask policy, optimizer behavior, loader contract, manifest hash, or frame-inventory fingerprint. This prevents accidentally mixing datasets or training policies in one run.

## Inspect results

Expected artifacts:

```text
notebook-runs/single-stream-jepa-haic.executed.ipynb
outputs/single-stream-jepa-healthgait-v3/latest.pt
outputs/single-stream-jepa-healthgait-v3/best_loss.pt
outputs/single-stream-jepa-healthgait-v3/best_healthy.pt
logs/single-stream-jepa-<JOBID>.out
```

Inspect checkpoint metadata without loading it onto the GPU:

```bash
uv run python - <<'PY'
from pathlib import Path
from cody_jepa.single_stream_jepa import load_checkpoint

for name in ("latest.pt", "best_loss.pt", "best_healthy.pt"):
    path = Path("outputs/single-stream-jepa-healthgait-v3") / name
    if not path.exists():
        print(name, "not written yet")
        continue
    checkpoint = load_checkpoint(path)
    print(name, {
        "epoch": checkpoint["completed_epochs"],
        "step": checkpoint["global_step"],
        "best_epoch": checkpoint["best_epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "dataset": checkpoint["data_contract"]["train_dataset"]["dataset_sha256"],
    })
PY
```

Use `best_healthy.pt` for representation probes. `best_loss.pt` is retained for diagnosis because its lower loss may still coincide with collapse or a position-only shortcut. `latest.pt` exists for continuation.

## Stop conditions

Stop and investigate if any of these occur:

- Loss, gradients, or inputs become non-finite. The loop raises before corrupting online or EMA weights.
- Effective rank trends toward 1 or per-feature variance approaches zero.
- Shuffled-context loss gap remains near zero after the initial learning period.
- Validation improves only clip-weighted metrics but not subject-balanced metrics.
- GPU memory is close to capacity. Reduce physical `batch_size` and increase `accumulation_steps` to preserve effective batch size.

Do not use this prototype’s prediction loss as evidence for CoDy-JEPA’s final disentanglement claim. That requires the later stream-specific probes, counterfactual intervention, and transfer evaluation.
