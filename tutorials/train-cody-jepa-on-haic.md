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
  --gres=gpu:h100:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=04:00:00 \
  --pty bash
```

Then run:

```bash
nvidia-smi
uv run python - <<'PY'
import json
import sys
import torch

capability = torch.cuda.get_device_capability()
required_arch = f"sm_{capability[0]}{capability[1]}"
torch_cuda_arch_list = torch.cuda.get_arch_list()
torch.zeros(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(json.dumps({
    "cuda_compute_capability": capability,
    "cuda_device_name": torch.cuda.get_device_name(),
    "cuda_preflight": "passed",
    "python_executable": sys.executable,
    "required_cuda_arch": required_arch,
    "torch_cuda_arch_list": torch_cuda_arch_list,
    "torch_cuda_version": torch.version.cuda,
    "torch_has_required_cuda_arch": required_arch in torch_cuda_arch_list,
    "torch_version": torch.__version__,
}, indent=2, sort_keys=True))
PY
uv run python -m unittest discover -s tests -v
```

If the CUDA probe reports `no kernel image is available for execution on the
device`, compare `cuda_compute_capability` with `torch_cuda_arch_list`:

- If HAIC allocated an H100/Hopper GPU and the required architecture is missing,
  the notebook is loading an incompatible or stale PyTorch build. Repair the
  locked environment and rerun the same probe before decoding the dataset:

```bash
uv sync --frozen --reinstall-package torch --reinstall-package torchvision
```

- If HAIC allocated a newer Blackwell GPU, the current locked PyTorch 2.6.0
  CUDA 12.4 environment is the wrong binary family for that node. Request a
  Hopper/H100 node for this locked run, or update the project pins to a PyTorch
  CUDA 12.8+ build and regenerate `uv.lock` before training.

Do not launch a separate system or Conda Jupyter process for this run. Use the
project environment through `uv run`, and check `python_executable` in the probe
if an interactive notebook still selects the wrong kernel.

After the CUDA probe and tests pass, leave the GPU shell. Execute the notebook
once in a CPU-only Slurm allocation, without `--gres` and without the training
flag. This validates the real manifest, sampled image integrity, subject
isolation, and model geometry. It also decodes one deterministic clip from
every train and validation sequence to reject blank or static sources, then
runs a tiny CPU training loop without occupying an allocated GPU:

```bash
exit
srun \
  --account=<ACCOUNT> \
  --partition=hai-interactive \
  --cpus-per-task=8 \
  --mem=64G \
  --time=01:00:00 \
  --pty bash
cd /hai/scratch/$USER/cody-jepa
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

Set `CODY_JEPA_RUN_FULL_TRAINING=1` in the job environment. Do not edit the
notebook. Full training defaults `CODY_JEPA_RUN_DATA_AUDIT` to `0`, because the
all-sequence audit was already completed above. The training path still checks
the full manifest, every frame name and size, sampled frame bytes, subject
isolation, and a real batch. It then runs one production-size BF16
forward/backward step on the allocated GPU before starting epoch 1.

Keep `CONFIG["required_device"] = "cuda"`. This makes a missing GPU fail
immediately instead of silently starting a multi-day CPU run. Keep
`compile=False` for the first successful eager run; compilation is an optional
later benchmark.

If you later test `compile=True`, run it only after the eager CUDA run has
completed a short checkpoint. `torch.compile` on CUDA uses TorchInductor and
Triton; if Triton is missing or too old, PyTorch raises
`BackendCompilerFailed: Cannot find a working triton installation`. Keep
`compile=False` for production training until this smoke test passes:

```bash
uv run python - <<'PY'
import torch
from torch.utils._triton import has_triton

print({"has_triton": has_triton(), "torch": torch.__version__})
if not has_triton():
    raise SystemExit("Triton is unavailable; keep CONFIG['compile'] = False")

@torch.compile
def add_one(x):
    return x + 1

x = torch.zeros(8, device="cuda")
add_one(x)
torch.cuda.synchronize()
print("torch.compile CUDA smoke test: passed")
PY
```

Exhaustive certification of all 321,247 frames is intentionally separate from
GPU training. To run it as a one-time I/O job, set
`CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT=1` and leave
`CODY_JEPA_RUN_FULL_TRAINING=0`. Do not combine those flags in an H100 job: the
exhaustive mode opens every image and reads every frame byte before model work.

If `outputs/single-stream-jepa/latest.pt` already exists, the notebook refuses to overwrite it unless you explicitly resume or choose a new output directory.

## Submit a batch job

Create `slurm/train-single-stream-jepa.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=single-stream-jepa
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/single-stream-jepa-%j.out

set -euo pipefail
cd /hai/scratch/$USER/cody-jepa
mkdir -p logs notebook-runs outputs/single-stream-jepa

nvidia-smi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$gpu_name" in
  *H100*) ;;
  *) echo "Expected an H100, got: $gpu_name" >&2; exit 2 ;;
esac
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=60 > "logs/gpu-${SLURM_JOB_ID}.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

CODY_JEPA_RUN_FULL_TRAINING=1 \
CODY_JEPA_RUN_DATA_AUDIT=0 \
CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT=0 \
MPLCONFIGDIR=/tmp/mpl \
uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output "single-stream-jepa-${SLURM_JOB_ID}.executed.ipynb" \
  --ExecutePreprocessor.timeout=-1
```

Submit and monitor it:

```bash
mkdir -p logs notebook-runs
sbatch slurm/train-single-stream-jepa.sbatch
squeue -u "$USER"
tail -f logs/single-stream-jepa-<JOBID>.out
tail -f logs/gpu-<JOBID>.csv
```

## Resume an interrupted run

The notebook writes `latest.pt` only after a complete epoch, so the saved
DataLoader, mask, Torch, optimizer, scaler, and EMA states share one exact
boundary. Resume without editing the notebook:

```bash
CODY_JEPA_RUN_FULL_TRAINING=1 \
CODY_JEPA_RESUME_CHECKPOINT=outputs/single-stream-jepa/latest.pt \
MPLCONFIGDIR=/tmp/mpl \
uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output "single-stream-jepa-${SLURM_JOB_ID}.resumed.ipynb" \
  --ExecutePreprocessor.timeout=-1
```

Resume validation rejects a changed architecture, mask policy, optimizer
behavior, loader contract, manifest hash, or frame-inventory fingerprint. The
fingerprint covers every frame name and size plus sampled frame contents. Use
the separate exhaustive audit when a byte-for-byte certification is required.

## Inspect results

Expected artifacts:

```text
notebook-runs/single-stream-jepa-<JOBID>.executed.ipynb
outputs/single-stream-jepa/latest.pt
outputs/single-stream-jepa/best_loss.pt
outputs/single-stream-jepa/best_healthy.pt
logs/single-stream-jepa-<JOBID>.out
logs/gpu-<JOBID>.csv
```

Inspect checkpoint metadata without loading it onto the GPU:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from cody_jepa.single_stream_jepa import load_checkpoint

metadata = {}
for name in ("latest.pt", "best_loss.pt", "best_healthy.pt"):
    path = Path("outputs/single-stream-jepa") / name
    if not path.exists():
        metadata[name] = {"status": "not_written_yet"}
        continue
    checkpoint = load_checkpoint(path)
    metadata[name] = {
        "best_epoch": checkpoint["best_epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "dataset_sha256": checkpoint["data_contract"]["train_dataset"]["dataset_sha256"],
        "epoch": checkpoint["completed_epochs"],
        "step": checkpoint["global_step"],
        "status": "written",
    }
print(json.dumps(metadata, indent=2, sort_keys=True))
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
