# Train CoDy-JEPA On Stanford HAIC GPUs

This guide explains how to run the CoDy-JEPA notebook on the Stanford HAI Compute Cluster, usually called HAIC.

The goal is a simple first GPU run of [`notebooks/01-cody-jepa.ipynb`](../notebooks/01-cody-jepa.ipynb). That notebook trains the current single-stream JEPA baseline on Health&Gait silhouette clips. It is the training starting point for CoDy-JEPA experiments, not the final dual-stream factorized model.

## 1. What You Are Running

The notebook does four main things:

1. Loads Health&Gait silhouette clips from a manifest.
2. Splits each clip into video patch tokens.
3. Masks some tokens and asks a predictor to predict target encoder embeddings.
4. Logs training and validation loss plus cosine similarity.

The current default training config is:

| Field | Value | Meaning |
| --- | ---: | --- |
| `num_frames` | `16` | Each sample is a 16-frame clip. |
| `img_size` | `72` | Each frame is resized to `72x72`. |
| `patch_size` | `24` | Each frame is split into `24x24` patches. |
| `num_tokens` | `144` | `16 * (72 / 24)^2 = 16 * 9` tokens per clip. |
| `batch_size` | `64` | Sixty-four clips are loaded per training step. |
| `num_epochs` | `25` | The training loop can make up to 25 passes over the training split. |
| `steps` | `1500` | Maximum optimizer steps before the loop stops early. |
| `lr` | `1e-4` | Adam learning rate. |
| `ema_tau` | `0.9995` | Target encoder exponential moving average rate. |
| `num_workers` | `8` | Eight CPU worker processes load and decode image batches. |
| `pin_memory` | `True` | DataLoader prepares CPU tensors for faster transfer to CUDA. |

Two details matter for HAIC:

- Training ends when either limit is reached: `num_epochs=25` full passes or `steps=1500` optimizer updates. With the current manifest and `batch_size=64`, the 25-epoch limit will usually happen first.
- With `batch_size=64` and `num_frames=16`, one training step reads `64 * 16 = 1024` silhouette frames before the GPU receives a batch.

The notebook's `train_jepa` function chooses a device automatically. If CUDA is available, it uses the GPU. If CUDA is not available, it falls back to Apple MPS or CPU.

Before using HAIC, make sure the last training cell does not force CPU. This is correct:

```python
result = train_jepa(CONFIG, train_loader, val_loader)
result
```

This is wrong for a GPU run:

```python
result = train_jepa(CONFIG, train_loader, val_loader, device="cpu")
```

## 2. HAIC Concepts In Plain Language

HAIC is a shared GPU cluster. You do not run training directly on the login/head node.

| Term | Simple meaning |
| --- | --- |
| Head node | The machine you SSH into first. Use it for editing files and submitting jobs, not training. |
| Compute node | A machine Slurm gives you for real work. GPU training runs here. |
| Slurm | The scheduler that decides when and where your job runs. |
| `srun` | Starts an interactive job, useful for quick debugging. |
| `sbatch` | Submits a script as a queued batch job, best for real training. |
| `--account` | Your HAIC project or team billing/accounting group. HAIC requires this. |
| `--partition` | The queue you submit to, such as `hai` or `hai-interactive`. |
| `--gres=gpu:1` | Requests one GPU. `gres` means generic resource. |
| Scratch | Large shared storage at `/hai/scratch/$USER`. Put the repo and dataset here. |
| Home | Smaller storage at `/hai/users/$USER`. Do not put large datasets here. |

Use `hai-interactive` for a short setup test. Use `hai` for the real notebook execution.

Official HAIC details can change. Check the Stanford HAIC page before long runs:

<https://legacy.cs.stanford.edu/haic>

## 3. Files You Need On HAIC

You need three things on HAIC:

1. The `cody-jepa` repository.
2. The Health&Gait dataset under `data/healthgait/raw/`.
3. The generated manifest at `data/healthgait/manifests/silhouette_subject_split_seed0.csv`.

The expected dataset shape is:

```text
cody-jepa/
  data/
    healthgait/
      raw/
        Health_Gait/
          silhouette/
            PA000/
              FGS/
                WJ_1_YOLOV8/
                  001.jpg
                  002.jpg
                  ...
      manifests/
        silhouette_subject_split_seed0.csv
  notebooks/
    01-cody-jepa.ipynb
```

If you have not prepared the dataset yet, follow [`tutorials/health-and-gait.md`](health-and-gait.md) first.

## 4. Log In To HAIC

From your laptop:

```bash
ssh <SUNetID>@haic.stanford.edu
```

If you are off campus, connect to Stanford VPN first. HAIC uses SUNetID and Duo.

After logging in, move to scratch storage:

```bash
cd /hai/scratch/$USER
```

If the repo is not there yet:

```bash
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
```

If the repo is already there:

```bash
cd /hai/scratch/$USER/cody-jepa
git pull
```

You also need your HAIC Slurm account name. This is not always the same as your SUNetID. If you do not know it, ask your group admin or HAIC support. Some clusters expose account information with:

```bash
sacctmgr show assoc user=$USER format=Account,Partition
```

If that command is unavailable or hard to interpret, do not guess. The wrong account name is one of the most common reasons `srun` and `sbatch` fail.

## 5. Start A Short Interactive GPU Session

Use an interactive session only to check setup. Do not leave long notebook training in an interactive shell.

Replace `<ACCOUNT>` with your HAIC Slurm account:

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

When the command succeeds, your shell is running on a compute node with a GPU.

Check the GPU:

```bash
nvidia-smi
```

You should see an NVIDIA GPU listed.

## 6. Install The Python Environment

The repo uses `uv` to create and run the Python environment.

Check whether `uv` exists:

```bash
uv --version
```

If `uv` is missing, install it in your user account:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Install the project dependencies:

```bash
cd /hai/scratch/$USER/cody-jepa
uv sync
```

The repo pins Linux `x86_64` installs to the PyTorch CUDA 12.4 wheels:

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`

This matters on HAIC because the current NVIDIA driver reports CUDA 12.4 compatibility. Newer PyTorch CUDA 13 builds can see that a GPU exists but cannot initialize CUDA on this driver.

Verify PyTorch can see CUDA:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected output:

```text
torch: 2.6.0+cu124
torch cuda build: 12.4
cuda: True
gpu: <some NVIDIA GPU name>
```

If you see `torch: 2.13.0+cu130`, your checkout has an old `pyproject.toml` or `uv.lock`. Pull the latest repo changes, remove `.venv`, and run `uv sync` again.

If `cuda` is `False`, stop here. A GPU training run would silently fall back to CPU unless the notebook is changed to require CUDA.

## 7. Verify The Dataset And Manifest

Still inside the interactive GPU session:

```bash
cd /hai/scratch/$USER/cody-jepa
test -f data/healthgait/manifests/silhouette_subject_split_seed0.csv
```

If that command prints nothing, the file exists.

Check a few manifest rows:

```bash
uv run python - <<'PY'
from pathlib import Path
from cody_jepa.data import healthgait_manifest_path, preview_manifest, find_repo_root

root = find_repo_root()
manifest = healthgait_manifest_path(root)
print("repo root:", root)
print("manifest:", manifest)
print("exists:", manifest.exists())
for row in preview_manifest(manifest, n=3):
    print(row)
PY
```

Run the fixture tests. These do not need the full dataset, but they confirm the repo environment is healthy:

```bash
uv run python -m unittest discover -s tests -v
```

If the tests pass and CUDA is visible, exit the interactive session:

```bash
exit
```

You are now back on the HAIC head node.

## 8. Confirm What The Notebook Saves

The current notebook already saves the trained state dictionaries after the final training cell. This matters because a batch job ends when notebook execution finishes; anything only stored in Python memory disappears.

Confirm that this cell appears immediately after:

```python
result = train_jepa(CONFIG, train_loader, val_loader)
result
```

The checkpoint cell should look like this:

```python
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cody-jepa-haic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = OUTPUT_DIR / "01-cody-jepa.pt"
torch.save(
    {
        "config": CONFIG,
        "history": result["history"],
        "steps": result["steps"],
        "context_encoder": result["context_encoder"].state_dict(),
        "target_encoder": result["target_encoder"].state_dict(),
        "predictor": result["predictor"].state_dict(),
        "optimizer": result["optimizer"].state_dict(),
    },
    checkpoint_path,
)
print(f"saved checkpoint: {checkpoint_path}")
```

The repo ignores `outputs/` and `*.pt`, so the checkpoint will stay local to HAIC unless you copy it elsewhere.

Important: this cell saves the final weights only. The local notebook experiment had its best validation loss around epoch 4, while the final epoch was worse. The saved `history` lets you see which epoch was best, but it does not restore the model weights from that earlier epoch.

For the first HAIC replication run, keep the default notebook so you can compare the full curve. For a shorter run that stops near epoch 4 with the current `batch_size=64`, use:

```python
CONFIG["num_epochs"] = 4
CONFIG["steps"] = 160
```

Why `160`? The current training split is about 40 batches per epoch at `batch_size=64`, so 4 epochs is about `4 * 40 = 160` optimizer steps. If your manifest changes, recompute this as:

```text
steps = desired_epochs * ceil(number_of_training_clips / batch_size)
```

Longer term, the better fix is to save a best-validation checkpoint inside `train_jepa`.

## 9. Submit The Notebook As A Batch Job

Create a Slurm script from the repo root on the HAIC head node:

```bash
cd /hai/scratch/$USER/cody-jepa
mkdir -p slurm logs notebook-runs outputs/cody-jepa-haic
```

Create `slurm/train-cody-jepa-notebook.sbatch`:

```bash
cat > slurm/train-cody-jepa-notebook.sbatch <<'SH'
#!/bin/bash
#SBATCH --job-name=cody-jepa
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=hai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cody-jepa-%j.out

set -euo pipefail

cd /hai/scratch/$USER/cody-jepa

echo "job id: ${SLURM_JOB_ID:-unknown}"
echo "host: $(hostname)"
echo "started: $(date)"

nvidia-smi

uv sync

uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available; refusing to run notebook on CPU")
PY

uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/01-cody-jepa.ipynb \
  --output-dir notebook-runs \
  --output 01-cody-jepa-haic.executed.ipynb \
  --ExecutePreprocessor.timeout=-1

echo "finished: $(date)"
SH
```

Edit the file and replace `<ACCOUNT>` with your HAIC account.

Submit the job:

```bash
sbatch slurm/train-cody-jepa-notebook.sbatch
```

Slurm will print a job id:

```text
Submitted batch job 123456
```

## 10. Watch The Job

List your queued and running jobs:

```bash
squeue -u $USER
```

Watch the log file:

```bash
tail -f logs/cody-jepa-123456.out
```

Replace `123456` with your job id.

Expected training output looks like:

```text
epoch 01 | step 0040 | train loss ... | val loss ...
epoch 02 | step 0080 | train loss ... | val loss ...
...
```

Those step counts are approximate for the current manifest and `batch_size=64`. If your data split changes, the step count per epoch changes too.

The first `nvidia-smi` in the Slurm script runs before the notebook starts. It is normal for that early check to show `No running processes found`; it only proves that the GPU is allocated and currently idle. The GPU should show a Python process after the notebook reaches the full training cell.

Stop watching the log with `Ctrl-C`. This does not cancel the job.

Cancel the job only if needed:

```bash
scancel 123456
```

## 11. Collect The Results

After the job finishes, check these files:

```bash
ls -lh notebook-runs/
ls -lh outputs/cody-jepa-haic/
ls -lh logs/
```

The important outputs are:

| File | Purpose |
| --- | --- |
| `notebook-runs/01-cody-jepa-haic.executed.ipynb` | The executed notebook with printed metrics and plots. |
| `outputs/cody-jepa-haic/01-cody-jepa.pt` | The trained model checkpoint. |
| `logs/cody-jepa-<jobid>.out` | Slurm stdout/stderr log for debugging. |

Inspect the checkpoint:

```bash
uv run python - <<'PY'
import torch

ckpt = torch.load("outputs/cody-jepa-haic/01-cody-jepa.pt", map_location="cpu")
print("steps:", ckpt["steps"])
print("history:")
for row in ckpt["history"]:
    print(row)
PY
```

For the current experiment, watch both loss and cosine similarity. In the local notebook run, the best validation loss occurred around epoch 4. If your HAIC run shows the same pattern, treat the best validation epoch as the useful checkpoint and do not assume the final epoch is best.

## 12. Copy Results Back To Your Laptop

From your laptop, not from HAIC:

```bash
mkdir -p cody-jepa-haic-results

scp <SUNetID>@haic.stanford.edu:/hai/scratch/<SUNetID>/cody-jepa/logs/cody-jepa-123456.out cody-jepa-haic-results/
scp <SUNetID>@haic.stanford.edu:/hai/scratch/<SUNetID>/cody-jepa/notebook-runs/01-cody-jepa-haic.executed.ipynb cody-jepa-haic-results/
scp <SUNetID>@haic.stanford.edu:/hai/scratch/<SUNetID>/cody-jepa/outputs/cody-jepa-haic/01-cody-jepa.pt cody-jepa-haic-results/
```

Replace `<SUNetID>` and `123456`.

Do not copy raw Health&Gait frames unless your data-use agreement and storage location allow it.

## 13. Common Problems

### `sbatch: error: Batch job submission failed`

Likely causes:

- `<ACCOUNT>` was not replaced.
- You used the wrong HAIC account.
- Your account does not have access to the requested partition.
- Your requested time, memory, or GPU count exceeds the partition limit.

Check your account and partition names:

```bash
sacctmgr show assoc user=$USER format=Account,Partition
sinfo
```

If those commands are restricted or confusing, ask HAIC support or your group admin for your Slurm account name.

### `CUDA is not available`

Likely causes:

- You ran the command on the head node.
- Your job did not request `--gres=gpu:1`.
- PyTorch installed without CUDA support.
- Your environment still has the old `torch 2.13.0+cu130` install.
- The job landed in a CPU-only allocation.

Check:

```bash
hostname
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On HAIC, the expected PyTorch line is:

```text
2.6.0+cu124 12.4 True
```

If you see `2.13.0+cu130`, refresh the environment:

```bash
rm -rf .venv
uv sync
```

### The notebook runs on CPU

Search the notebook for forced CPU usage:

```bash
grep -n 'device="cpu"' notebooks/01-cody-jepa.ipynb
```

The notebook has a small smoke test that intentionally uses `device="cpu"`. That is fine. The important full-training cell should be:

```python
result = train_jepa(CONFIG, train_loader, val_loader)
```

If the final full-training cell includes `device="cpu"`, remove that argument.

### The manifest is missing

Build it from the repo root:

```bash
uv run python scripts/build_healthgait_manifest.py
```

This requires the extracted dataset under `data/healthgait/raw/Health_Gait/`.

### The job ends but there is no checkpoint

Use the latest repo version and confirm the checkpoint-saving cell from Section 8 executed after training. If the cell is missing or appears before `result = train_jepa(...)`, fix the notebook and rerun the batch job.

### The job is slow even on GPU

Possible causes:

- The DataLoader still has to read and decode 1024 JPEG frames for every `batch_size=64` training step.
- Frames are loaded as JPEGs one clip at a time.
- The notebook runs diagnostics and visualizations before training.
- The model is small, so the GPU may not be fully utilized.

The current notebook sets `num_workers=8` and `pin_memory=True`, which is the right first HAIC setting for `--cpus-per-task=8`. If the GPU is still idle most of the time, the bottleneck is probably JPEG loading or the pre-training notebook cells, not CUDA visibility.

For the first run, correctness matters more than utilization. After the first successful run, consider moving repeated diagnostics out of the training notebook, caching clips in a more training-friendly format, or converting the notebook code into a script.

## 14. A Clean First Experiment Checklist

Before submitting:

- [ ] Repo is under `/hai/scratch/$USER/cody-jepa`.
- [ ] Health&Gait is extracted under `data/healthgait/raw/Health_Gait`.
- [ ] `data/healthgait/manifests/silhouette_subject_split_seed0.csv` exists.
- [ ] `uv sync` succeeds on a compute node.
- [ ] `torch.cuda.is_available()` prints `True` inside a GPU allocation.
- [ ] The final training cell does not force `device="cpu"`.
- [ ] The checkpoint-saving cell exists after training and writes to `outputs/cody-jepa-haic/`.
- [ ] `LOADER_CONFIG` uses `num_workers=8` and `pin_memory=True`.
- [ ] The Slurm script has your real `--account=<ACCOUNT>`.

After the job:

- [ ] The log shows `cuda: True`.
- [ ] The log shows epoch metrics.
- [ ] `notebook-runs/01-cody-jepa-haic.executed.ipynb` exists.
- [ ] `outputs/cody-jepa-haic/01-cody-jepa.pt` exists.
- [ ] You copied the log, executed notebook, and checkpoint to a safe location.

## 15. Recommended Next Improvements

Once the notebook works on HAIC, improve the experiment in this order:

1. Save the best validation checkpoint, not just the final checkpoint.
2. Add a learning-rate schedule because the current local run worsened after epoch 4.
3. Log GPU memory, step time, and examples per second.
4. Move notebook model code into `src/cody_jepa/` so batch jobs can run a plain Python script.
5. Add command-line config overrides for `batch_size`, `steps`, `lr`, and `ema_tau`.
6. Add a short `--smoke` run that trains for one or two batches and exits.

The first HAIC goal is not maximum throughput. The first goal is a repeatable GPU run with a saved checkpoint and enough logs to understand what happened.

## References

- Stanford HAIC documentation: <https://legacy.cs.stanford.edu/haic>
- Stanford CS Slurm guide: <https://cluster.cs.stanford.edu/slurm/>
- Repo setup and Health&Gait preparation: [`README.md`](../README.md)
- Health&Gait tutorial: [`tutorials/health-and-gait.md`](health-and-gait.md)
