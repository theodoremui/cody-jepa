# Week 2.0: HAIC, uv, and SLURM Basics

This lesson is for the first hour on HAIC. It assumes you know how to use a terminal, but not that you already understand GPU clusters.

The key idea is that an HPC cluster has different kinds of machines. Login nodes are shared front doors for editing files, submitting jobs, and checking status. Compute nodes are where CPU and GPU work runs. Treat this as a hard rule:

```text
Login nodes are not for computation.
```

Use HAIC login nodes for light commands such as `git`, `ls`, `squeue`, `sacct`, and small text edits. Use a compute allocation, a site Jupyter service, or SLURM jobs for dependency sync, notebook execution, extraction, indexing, diagnostics, and training.

## 1. Start With A Clean Mental Model

SLURM is the scheduler. A SLURM job has two parts:

1. Resource requests, such as time, CPUs, memory, and GPUs.
2. Job steps, which are the commands that run after resources are allocated.

Batch scripts are shell scripts with `#SBATCH` lines near the top. SLURM reads those lines before running the script.

```bash
#!/bin/bash
#SBATCH --job-name=cody_smoke
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

hostname
python -V
```

Important: put every `#SBATCH` directive before the first real shell command. SLURM ignores later directives after the first non-comment, non-whitespace line.

## 2. Discover HAIC Before Requesting Resources

Do not copy a partition name from someone else's script. Discover what your account can use.

Run these on the login node:

```bash
hostname
whoami
groups
```

Find accounts and associations:

```bash
sacctmgr show assoc user=$USER format=Cluster,Account,Partition,QOS%30
```

Find partitions and node summaries:

```bash
sinfo
sinfo -o "%20P %8a %10l %10D %30G"
scontrol show partition
```

Find GPU-related features if your HAIC environment exposes them:

```bash
sinfo -o "%20P %30G %80f"
```

Find storage and quota information. Exact commands vary by site, so try the ones that exist:

```bash
df -h
quota -s
lfs quota -h "$HOME" 2>/dev/null || true
echo "$SCRATCH"
echo "$TMPDIR"
```

Find modules and CUDA options:

```bash
module avail 2>&1 | head
module spider cuda 2>/dev/null || module avail cuda 2>&1
module spider python 2>/dev/null || module avail python 2>&1
```

Run `nvidia-smi` inside a GPU allocation, not as a login-node workload. If the login node happens to have it installed, that does not mean you should run GPU code there.

## 3. Set Portable Project Variables

Use environment variables instead of hardcoded cluster paths. Put these in a local shell profile or in the top of your batch scripts after the `#SBATCH` block.

```bash
export CODY_JEPA_ROOT="${CODY_JEPA_ROOT:-$HOME/cody-jepa}"
export CODY_JEPA_DATA="${CODY_JEPA_DATA:-$SCRATCH/cody-jepa-data}"

export GAITLU_ROOT="$CODY_JEPA_DATA/gaitlu-1m"
export GAITLU_ARCHIVE_DIR="$GAITLU_ROOT/archives"
export GAITLU_EXTRACTED_DIR="$GAITLU_ROOT/raw"
export GAITLU_MANIFEST_DIR="$GAITLU_ROOT/manifests"
export GAITLU_DIAGNOSTICS_DIR="$GAITLU_ROOT/diagnostics"
export GAITLU_PROBE_EXPORT_DIR="$GAITLU_ROOT/probe_exports"

mkdir -p \
  "$GAITLU_ARCHIVE_DIR" \
  "$GAITLU_EXTRACTED_DIR" \
  "$GAITLU_MANIFEST_DIR" \
  "$GAITLU_DIAGNOSTICS_DIR" \
  "$GAITLU_PROBE_EXPORT_DIR"
```

If your cluster does not define `$SCRATCH`, choose the storage location recommended by HAIC after checking quota and performance guidance. Do not commit that absolute path.

## 4. Clone And Sync With uv

On a HAIC login node, clone the repo and inspect the Python pin:

```bash
git clone <your-repo-url> "$CODY_JEPA_ROOT"
cd "$CODY_JEPA_ROOT"
cat .python-version
```

Then request a small compute allocation before syncing dependencies. The lock file includes the notebook stack plus PyTorch, so the first sync can download large wheels.

```bash
salloc \
  --job-name=cody_uv_sync \
  --time=00:30:00 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=16G

cd "$CODY_JEPA_ROOT"
uv sync --locked
uv run python - <<'PY'
import sys
import numpy
import pandas
import torch
print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
PY
exit
```

Use this pattern for every Python command:

```bash
uv run python your_script.py
uv run jupyter nbconvert \
  --to notebook \
  --execute tutorials/notebooks/week2_02_index_splits_loader.ipynb \
  --ExecutePreprocessor.kernel_name=python3
```

Run JupyterLab through HAIC's recommended Jupyter service or inside an allocated compute session, not as a long-running login-node process.

Do not add separate package-manager install steps. If a library is missing, add it to the project with `uv add` from the repo root and commit `pyproject.toml` plus `uv.lock`.

## 5. Submit A Tiny SLURM Smoke Job

Create a `logs` directory:

```bash
mkdir -p "$CODY_JEPA_ROOT/logs"
```

Save this as a local script such as `scripts/haic_smoke.sbatch`. The tutorial does not commit it because your account, partition, and QOS are site-specific.

```bash
#!/bin/bash
#SBATCH --job-name=cody_haic_smoke
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${CODY_JEPA_ROOT:?Set CODY_JEPA_ROOT before sbatch}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-no_slurm_job_id}"
echo "cwd=$(pwd)"
uv run python - <<'PY'
import sys
import torch
print(sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
```

Submit it after choosing any account or partition flags HAIC requires:

```bash
sbatch scripts/haic_smoke.sbatch
```

If HAIC requires an account or partition, pass it at submission time after discovery:

```bash
sbatch --account=<your_account> --partition=<your_partition> scripts/haic_smoke.sbatch
```

Check live status:

```bash
squeue -u "$USER"
```

Check completed resource usage:

```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,ReqMem
```

The smoke job proves that the repo, `uv`, Python, and basic SLURM flow work before you spend time on the dataset.

## 6. Common Beginner Mistakes

| Mistake | Safer habit |
| --- | --- |
| Running extraction on the login node | Put extraction into an `sbatch` script. |
| Copying an old partition name | Discover partitions with `sinfo` and account associations first. |
| Adding shell variables inside `#SBATCH` lines | Pass site-specific settings at `sbatch` time or write concrete directives. |
| Requesting the largest GPU and longest wall time for every job | Start with small smoke jobs and scale requests after `sacct` evidence. |
| Installing packages outside the project | Use `uv add`, `uv sync`, and `uv run`. |
| Running JupyterLab on a login node | Use HAIC's Jupyter service or an allocated compute session. |

## Week 2 Gate For This Lesson

This lesson does not require the real GaitLU data. It passes when:

1. You can run `uv sync --locked` on HAIC inside a compute allocation or site-managed Jupyter environment.
2. A tiny `sbatch` job starts and writes logs.
3. You can inspect that job with `squeue` while it runs and `sacct` after it finishes.
4. The companion notebook proves the synthetic Week 2 data gate: reproducible splits, reproducible validation batches, and `[B, T, C, H, W]` batch shape.
