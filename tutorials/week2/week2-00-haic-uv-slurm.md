# Week 2.0: HAIC, uv, and SLURM Basics

This lesson is for your first hour on HAIC or Sherlock. It assumes only that you can open a terminal. It does not assume that you already understand SSH, GPU clusters, SLURM, storage, partitions, or `uv`.

The main idea is simple: a cluster has different kinds of machines, and each kind has a different job.

```text
Login nodes are for light coordination.
Compute nodes are for computation.
```

Use HAIC or Sherlock login nodes for light commands such as `git`, `ls`, `pwd`, `hostname`, `squeue`, `sacct`, and small text edits. Use a compute allocation, a site Jupyter service, or SLURM jobs for dependency sync, notebook execution, extraction, indexing, diagnostics, and training.

Before every command block, this tutorial tells you where to run the command and which directory you should be in. Keep that habit. It prevents the two most common beginner mistakes: running heavy work on a login node and running a command from the wrong directory.

## 1. SSH Mental Model

SSH means Secure Shell. It gives you a terminal on a remote machine.

Before SSH, your terminal commands run on your laptop. After SSH, your terminal commands run on the remote cluster login node.

If your administrator tells you to use Sherlock, the Stanford resources overview shows this login host:

```text
sherlock.stanford.edu
```

If your administrator gives you a HAIC-specific hostname, use that hostname instead. The rest of the tutorial uses `<haic-login-host>` as a placeholder for whichever login host you were told to use.

Run this from any directory on your laptop:

```bash
laptop$ ssh <sunetid>@<haic-login-host>
```

Read the command piece by piece:

| Piece | Meaning |
| --- | --- |
| `laptop$` | A label in this tutorial showing that the command starts on your laptop. Do not type `laptop$`. |
| `ssh` | The program that opens a secure terminal connection to another machine. |
| `<sunetid>` | Your HAIC or Stanford username. Replace the placeholder, including the angle brackets. |
| `@` | Separates the username from the computer you want to connect to. |
| `<haic-login-host>` | The cluster login server hostname. For Sherlock, use `sherlock.stanford.edu`. For HAIC, use the hostname your administrator gives you. |

After the connection succeeds, your prompt may look different. This tutorial labels the remote login prompt as `haic-login$`.

If your real prompt looks like this, you are on the login node already:

```text
tedmui@haic:~$
```

The text before `$` is your shell prompt. `tedmui` is the username. `haic` is the machine name or cluster frontend name. `~` means your home directory. None of those words is a command to type.

Run these on the HAIC or Sherlock login node from any directory:

```bash
haic-login$ hostname
haic-login$ pwd
haic-login$ exit
```

Each line has one job:

| Line | Meaning |
| --- | --- |
| `hostname` | Print the name of the machine you are currently using. This helps you confirm that you are on the cluster rather than your laptop. |
| `pwd` | Print the current directory on that machine. Use this before commands that depend on location. |
| `exit` | Close the remote SSH session and return to your laptop shell. |

If SSH asks whether to trust the host the first time, stop and verify the hostname and fingerprint against the official cluster instructions. If SSH asks for a password or second factor, that is authentication for your remote account. Stanford's resources overview notes that Sherlock SSH requires two-factor authentication. Do not paste dataset passwords into SSH prompts.

One more distinction matters:

```text
SSH:   laptop terminal -> cluster login node
SLURM: cluster login node -> cluster compute node
```

SSH gets you into the cluster front door. SLURM gives you resources for real work.

## 2. Stanford Resource Names You Will See

Stanford's SDSS-CC resources overview describes Sherlock as a shared Stanford HPC cluster. It lists public Sherlock partitions named `normal`, `gpu`, `dev`, `bigmem`, and `owners`. It also says SDSS users may be able to submit to a `serc` partition, and that Oak storage may be available for groups at paths like `/oak/stanford/schools/ees/{PI SUNetID}`.

Read those names carefully:

| Name | Plain meaning | What you should do |
| --- | --- | --- |
| `normal` | Public general-purpose compute partition. | Good first place to look for CPU jobs if your account can use it. |
| `gpu` | Public GPU partition. | Use only for jobs that need GPUs. |
| `dev` | Public development partition. | Useful for short tests if your account can use it. |
| `bigmem` | Public large-memory partition. | Use only when ordinary memory is not enough. |
| `owners` | Access to idle owner resources. | Use only if your account and job policy allow it. |
| `serc` | SDSS-related partition described in the Stanford overview. | Do not assume access. Check your account association first. |
| `Oak` | Larger research storage, separate from home and scratch. | Use only if your PI or group has access and gives you the correct path. |
| `$SCRATCH` | Your personal temporary scratch directory on Sherlock. | Use it for large temporary project data and job I/O. |

The important rule is not to memorize these names. The important rule is to discover which names your account can use before submitting a job.

## 3. SLURM Mental Model

SLURM is the scheduler. You tell SLURM what resources you need, and SLURM decides where and when the job runs.

A SLURM job has two parts:

1. Resource requests, such as time, CPUs, memory, and GPUs.
2. Job steps, which are the commands that run after resources are allocated.

Batch scripts are shell scripts with `#SBATCH` lines near the top. SLURM reads those `#SBATCH` lines before running the script.

Save this example only after you understand the shape. It is a template, not a finished site-specific script:

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

Read it line by line:

| Line | Meaning |
| --- | --- |
| `#!/bin/bash` | Run this script with the Bash shell. |
| `#SBATCH --job-name=cody_smoke` | Name the job `cody_smoke` so it is easy to find in `squeue`, `sacct`, and log filenames. |
| `#SBATCH --time=00:10:00` | Request ten minutes of wall-clock time. The format is hours:minutes:seconds. |
| `#SBATCH --ntasks=1` | Request one task. For these tutorials, one task is enough. |
| `#SBATCH --cpus-per-task=2` | Give that one task two CPU cores. |
| `#SBATCH --mem=8G` | Request 8 GB of memory. |
| `#SBATCH --output=logs/%x-%j.out` | Write standard output to `logs/`. `%x` becomes the job name and `%j` becomes the job ID. |
| `#SBATCH --error=logs/%x-%j.err` | Write standard error to a matching log file. |
| `set -euo pipefail` | Make the shell stop on common mistakes: failed commands, unset variables, and failed pipeline pieces. |
| `hostname` | Print the compute node name after the job starts. |
| `python -V` | Print the Python version available inside the job. Later we use `uv run python` instead. |

Important: put every `#SBATCH` directive before the first real shell command. SLURM ignores later directives after the first non-comment, non-whitespace line.

## 4. Discover HAIC Before Requesting Resources

Do not copy a partition name, account name, or QOS setting from someone else's script. First discover what your account can use.

Run this after SSH login, from any directory on the HAIC or Sherlock login node:

```bash
haic-login$ hostname
haic-login$ whoami
haic-login$ groups
```

These lines identify your current session:

| Line | Meaning |
| --- | --- |
| `hostname` | Print the HAIC machine name. |
| `whoami` | Print the username that HAIC sees for this shell. |
| `groups` | Print the Unix groups attached to your account. Groups sometimes control project storage or cluster access. |

Run this on the HAIC or Sherlock login node from any directory:

```bash
haic-login$ sacctmgr show assoc user=$USER format=Cluster,Account,Partition,QOS%30
```

This asks SLURM which accounts, partitions, and QOS settings your current user can request. `$USER` is an environment variable that already contains your remote username.

Run these on the HAIC or Sherlock login node from any directory:

```bash
haic-login$ sinfo
haic-login$ sinfo -o "%20P %8a %10l %10D %30G"
haic-login$ scontrol show partition
```

These commands inspect partitions and nodes:

| Line | Meaning |
| --- | --- |
| `sinfo` | Show SLURM partitions and whether nodes are idle, allocated, mixed, or down. |
| `sinfo -o "%20P %8a %10l %10D %30G"` | Show selected partition fields in a wider, easier-to-read format. |
| `scontrol show partition` | Print detailed partition settings when `sinfo` is too compressed. |

If your administrator specifically mentions the `serc` partition, run this on the login node from any directory:

```bash
haic-login$ sinfo --Node --long --partition=serc
```

This follows the Stanford resources overview suggestion for viewing `serc` partition information. If the command says the partition is unknown or unavailable, do not force it into your scripts. Use the partitions shown by your own `sinfo` and account association output.

If HAIC or Sherlock exposes GPU information through SLURM, run this on the login node from any directory:

```bash
haic-login$ sinfo -o "%20P %30G %80f"
```

This asks `sinfo` to show partition names, generic resources such as GPUs, and node features. Run `nvidia-smi` only inside a GPU allocation. If the login node happens to have `nvidia-smi` installed, that does not mean you should run GPU code there.

Now check storage. Run these on the HAIC or Sherlock login node from any directory:

```bash
haic-login$ df -h
haic-login$ quota -s
haic-login$ lfs quota -h "$HOME" 2>/dev/null || true
haic-login$ echo "$SCRATCH"
haic-login$ echo "$TMPDIR"
haic-login$ echo "$OAK"
```

The storage commands are safe discovery commands:

| Line | Meaning |
| --- | --- |
| `df -h` | Show mounted filesystems and free space. `-h` means human-readable units such as GB and TB. |
| `quota -s` | Show your storage quota if HAIC exposes classic Unix quotas. |
| `lfs quota -h "$HOME" 2>/dev/null || true` | Try a Lustre quota check for your home directory. Hide the error if `lfs` is not available, and keep going. |
| `echo "$SCRATCH"` | Print the path stored in `$SCRATCH`. On Sherlock, this is the right starting point for large temporary project data and job I/O. |
| `echo "$TMPDIR"` | Print the path stored in `$TMPDIR`. This is usually temporary job-local or session-local storage. |
| `echo "$OAK"` | Print the path stored in `$OAK`, if your group has Oak access and the environment defines it. |

For this project, keep source code in `$HOME` or another small-code location. Put large GaitLU archive files, extracted data, and intermediate outputs under `$SCRATCH` or approved project storage. Use Oak only if your group has access and the data should live longer than scratch policy allows.

Finally, inspect available software modules. Run these on the HAIC or Sherlock login node from any directory:

```bash
haic-login$ module avail 2>&1 | head
haic-login$ module spider cuda 2>/dev/null || module avail cuda 2>&1
haic-login$ module spider python 2>/dev/null || module avail python 2>&1
```

These commands ask the module system what software can be loaded. `2>&1` sends error output to the same stream as normal output so you can read both. `||` means "if the first command fails, try the second command."

## 5. Set Portable Project Variables

Use environment variables instead of hardcoded cluster paths. A variable gives a short name to a path, and later commands can use `$NAME` instead of repeating the full path.

Run this on the HAIC or Sherlock login node after you have chosen the cluster storage location. You can run it from any directory, but the directories it creates will be on the cluster.

This version checks that `$SCRATCH` exists before using it. That prevents the common mistake where an empty `$SCRATCH` turns into the invalid path `/cody-jepa-data`.

```bash
haic-login$ export CODY_JEPA_ROOT="${CODY_JEPA_ROOT:-$HOME/cody-jepa}"

haic-login$ if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH" ] && [ -w "$SCRATCH" ]; then
  export CODY_JEPA_DATA="${CODY_JEPA_DATA:-$SCRATCH/cody-jepa-data}"
else
  export CODY_JEPA_DATA="${CODY_JEPA_DATA:-$HOME/cody-jepa-data}"
fi

haic-login$ export GAITLU_ROOT="$CODY_JEPA_DATA/gaitlu-1m"
haic-login$ export GAITLU_ARCHIVE_DIR="$GAITLU_ROOT/archives"
haic-login$ export GAITLU_EXTRACTED_DIR="$GAITLU_ROOT/raw"
haic-login$ export GAITLU_MANIFEST_DIR="$GAITLU_ROOT/manifests"
haic-login$ export GAITLU_DIAGNOSTICS_DIR="$GAITLU_ROOT/diagnostics"
haic-login$ export GAITLU_PROBE_EXPORT_DIR="$GAITLU_ROOT/probe_exports"

haic-login$ mkdir -p \
  "$GAITLU_ARCHIVE_DIR" \
  "$GAITLU_EXTRACTED_DIR" \
  "$GAITLU_MANIFEST_DIR" \
  "$GAITLU_DIAGNOSTICS_DIR" \
  "$GAITLU_PROBE_EXPORT_DIR"
```

Read the block in order:

| Line | Meaning |
| --- | --- |
| `export CODY_JEPA_ROOT=...` | Set the repo location. If `CODY_JEPA_ROOT` is already set, keep it. Otherwise use `$HOME/cody-jepa`. |
| `if [ -n "${SCRATCH:-}" ] ...` | Check that `$SCRATCH` is set, exists, and is writable before using it. |
| `export CODY_JEPA_DATA=...` | Set the project data location. Prefer `$SCRATCH/cody-jepa-data` when scratch is available. Fall back to `$HOME/cody-jepa-data` only for small smoke tests. |
| `export GAITLU_ROOT=...` | Set the top-level GaitLU-1M data directory. |
| `export GAITLU_ARCHIVE_DIR=...` | Set the directory for encrypted archive files. |
| `export GAITLU_EXTRACTED_DIR=...` | Set the directory for extracted `.pkl` sequence files. |
| `export GAITLU_MANIFEST_DIR=...` | Set the directory for generated index and split CSV files. |
| `export GAITLU_DIAGNOSTICS_DIR=...` | Set the directory for diagnostic images, tables, and plots. |
| `export GAITLU_PROBE_EXPORT_DIR=...` | Set the directory for dummy or real probe-export tables. |
| `mkdir -p ...` | Create all needed directories. `-p` means "also create parent directories, and do not fail if the directory already exists." |

If your cluster does not define `$SCRATCH`, choose the storage location recommended by your administrator after checking quota and performance guidance. Do not commit that absolute path to the repo.

After setting the variables, check them before creating directories:

```bash
haic-login$ echo "CODY_JEPA_ROOT=[$CODY_JEPA_ROOT]"
haic-login$ echo "CODY_JEPA_DATA=[$CODY_JEPA_DATA]"
haic-login$ echo "GAITLU_ARCHIVE_DIR=[$GAITLU_ARCHIVE_DIR]"
```

The data path should start with a writable location such as `$SCRATCH`, `$HOME`, or an approved project path. It should not start with `/cody-jepa-data`.

These exports affect only the current shell unless you add them to a shell profile or repeat them in batch scripts. In an `sbatch` script, put them after the `#SBATCH` block and before commands that use the variables.

## 6. Clone The Repo And Sync Dependencies With uv

First clone the repo. Run this on the HAIC or Sherlock login node from any directory:

```bash
haic-login$ git clone <your-repo-url> "$CODY_JEPA_ROOT"
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ cat .python-version
```

The lines mean:

| Line | Meaning |
| --- | --- |
| `git clone <your-repo-url> "$CODY_JEPA_ROOT"` | Copy the project code from its Git remote into the cluster directory stored in `CODY_JEPA_ROOT`. Replace `<your-repo-url>` with the real Git URL. |
| `cd "$CODY_JEPA_ROOT"` | Enter the project directory on the cluster. Most repo commands in the tutorials assume this directory. |
| `cat .python-version` | Print the Python version expected by the repo. |

Do not run the first dependency sync as a login-node workload. The lock file includes the notebook stack plus PyTorch, so the first sync can download and install large wheels.

Request a small interactive compute allocation from the HAIC or Sherlock login node. You can run `salloc` from any directory, but after allocation you should move to the repo root:

```bash
haic-login$ salloc \
  --job-name=cody_uv_sync \
  --time=00:30:00 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=16G
```

Read this request line by line:

| Line | Meaning |
| --- | --- |
| `salloc \` | Ask SLURM for an interactive compute allocation. The backslash continues the command onto the next line. |
| `--job-name=cody_uv_sync \` | Name the allocation so it is easy to identify. |
| `--time=00:30:00 \` | Request 30 minutes. |
| `--ntasks=1 \` | Request one task. |
| `--cpus-per-task=2 \` | Request two CPU cores. |
| `--mem=16G` | Request 16 GB of memory. |

When the allocation starts, you are allowed to run heavier setup commands. This tutorial labels that shell as `haic-compute$`.

Run these from the repo root on the HAIC or Sherlock compute node:

```bash
haic-compute$ cd "$CODY_JEPA_ROOT"
haic-compute$ uv sync --locked
haic-compute$ uv run python - <<'PY'
import sys
import numpy
import pandas
import torch
print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
PY
haic-compute$ exit
```

The commands mean:

| Line | Meaning |
| --- | --- |
| `cd "$CODY_JEPA_ROOT"` | Enter the repo root before running `uv`. |
| `uv sync --locked` | Install exactly the dependency set recorded in `uv.lock`. `--locked` prevents accidental lock-file changes. |
| `uv run python - <<'PY'` | Run Python inside the project environment and feed it the lines until the closing `PY` marker. |
| `import ...` lines | Import the main packages so missing installs fail immediately. |
| `print(...)` lines | Print package versions so the log records what environment ran. |
| `PY` | End the inline Python program. Do not indent this marker. |
| `exit` | Leave the compute allocation and return to the HAIC login shell. |

Use this pattern for every Python command in the repo:

```bash
haic-compute$ cd "$CODY_JEPA_ROOT"
haic-compute$ uv run python your_script.py
```

`uv run` matters because it runs Python inside the repo's managed environment. A plain `python your_script.py` may use the wrong Python or the wrong packages.

To execute a notebook non-interactively, run this from the repo root inside a compute allocation or site-managed Jupyter environment:

```bash
haic-compute$ uv run jupyter nbconvert \
  --to notebook \
  --execute tutorials/notebooks/week2_02_index_splits_loader.ipynb \
  --ExecutePreprocessor.kernel_name=python3
```

Each line has a purpose:

| Line | Meaning |
| --- | --- |
| `uv run jupyter nbconvert \` | Run Jupyter's notebook conversion tool inside the repo environment. |
| `--to notebook \` | Write another notebook as output, rather than HTML or PDF. |
| `--execute ...ipynb \` | Execute the named notebook from top to bottom. |
| `--ExecutePreprocessor.kernel_name=python3` | Use the `python3` kernel. |

Run JupyterLab through HAIC's recommended Jupyter service or inside an allocated compute session, not as a long-running login-node process.

Do not add separate package-manager install steps. If a library is missing, add it to the project from the repo root with `uv add <package>` and commit both `pyproject.toml` and `uv.lock`.

## 7. Submit A Tiny SLURM Smoke Job

The smoke job proves that the repo, `uv`, Python, and basic SLURM flow work before you spend time on the dataset.

First create the local `scripts/` and `logs/` directories inside the cluster repo checkout. Run this on the HAIC or Sherlock login node from the repo root:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ mkdir -p scripts logs
```

`cd "$CODY_JEPA_ROOT"` enters the repo on the cluster. `mkdir -p scripts logs` creates both directories if they do not already exist. `scripts/` will hold your site-specific batch script, and `logs/` will hold SLURM output and error files.

Now create a local script named `scripts/haic_smoke.sbatch` in your cluster repo checkout. The tutorial does not commit this file because your account, partition, and QOS are site-specific.

Save this content in `scripts/haic_smoke.sbatch`:

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

The script lines mean:

| Line | Meaning |
| --- | --- |
| `#!/bin/bash` | Run the script with Bash. |
| `#SBATCH --job-name=cody_haic_smoke` | Give the job a recognizable name. |
| `#SBATCH --time=00:10:00` | Request ten minutes. |
| `#SBATCH --ntasks=1` | Request one task. |
| `#SBATCH --cpus-per-task=2` | Request two CPU cores. |
| `#SBATCH --mem=8G` | Request 8 GB of memory. |
| `#SBATCH --output=logs/%x-%j.out` | Write standard output to the repo's `logs/` directory. |
| `#SBATCH --error=logs/%x-%j.err` | Write standard error to the repo's `logs/` directory. |
| `set -euo pipefail` | Stop the script when common shell errors happen. |
| `cd "${CODY_JEPA_ROOT:?Set CODY_JEPA_ROOT before sbatch}"` | Enter the repo root. The `:?` form stops with a clear error if `CODY_JEPA_ROOT` is unset. |
| `echo "host=$(hostname)"` | Print the compute node name into the log. |
| `echo "job=${SLURM_JOB_ID:-no_slurm_job_id}"` | Print the SLURM job ID if SLURM set it. |
| `echo "cwd=$(pwd)"` | Print the directory where the job is running. |
| `uv run python - <<'PY'` | Run an inline Python check inside the repo environment. |
| `import sys` | Load Python's system module so the script can print the Python version. |
| `import torch` | Load PyTorch so dependency problems fail in the smoke job. |
| `print(...)` lines | Record Python, PyTorch, and CUDA availability in the job log. |
| `PY` | End the inline Python program. |

Submit the script from the repo root on the HAIC or Sherlock login node:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ sbatch scripts/haic_smoke.sbatch
```

If HAIC or Sherlock requires an account or partition, pass it at submission time after discovery:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ sbatch --account=<your_account> --partition=<your_partition> scripts/haic_smoke.sbatch
```

The command submits the script to SLURM. The job runs later on a compute node, not inside your SSH login shell. Replace `<your_account>` and `<your_partition>` with values you discovered with `sacctmgr` and `sinfo`.

Check live status from any directory on the HAIC or Sherlock login node:

```bash
haic-login$ squeue -u "$USER"
```

Check completed resource usage from any directory on the HAIC or Sherlock login node:

```bash
haic-login$ sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,ReqMem
```

Replace `<job_id>` with the number printed by `sbatch`. `sacct` tells you whether the job completed, how long it ran, and how much memory it used.

## 8. Common Beginner Mistakes

| Mistake | Safer habit |
| --- | --- |
| Running extraction on the login node | Put extraction into an `sbatch` script or an interactive compute allocation. |
| Running `uv sync` on the login node | Use `salloc`, then run `uv sync --locked` from `$CODY_JEPA_ROOT` on the compute node. |
| Copying an old partition name | Discover partitions with `sinfo` and account associations first. |
| Assuming `serc` access because it appears in Stanford docs | Check your own account association and `sinfo` output before using `--partition=serc`. |
| Using `$HOME` for full GaitLU data | Use `$SCRATCH` or approved project storage for large temporary data. |
| Letting empty `$SCRATCH` become `/cody-jepa-data` | Use the safe variable block and inspect `echo "$CODY_JEPA_DATA"` before `mkdir`. |
| Adding shell variables inside `#SBATCH` lines | Pass site-specific settings at `sbatch` time or write concrete directives. |
| Requesting the largest GPU and longest wall time for every job | Start with small smoke jobs and scale requests after `sacct` evidence. |
| Installing packages outside the project | Use `uv add`, `uv sync`, and `uv run` from the repo root. |
| Running JupyterLab on a login node | Use HAIC's Jupyter service or an allocated compute session. |
| Forgetting which machine you are on | Run `hostname` and `pwd` before commands that change files or start work. |

## Week 2 Gate For This Lesson

This lesson does not require the real GaitLU data. It passes when:

1. You can run `uv sync --locked` on HAIC or Sherlock inside a compute allocation or site-managed Jupyter environment.
2. A tiny `sbatch` job starts and writes logs.
3. You can inspect that job with `squeue` while it runs and `sacct` after it finishes.
4. The companion notebook proves the synthetic Week 2 data gate: reproducible splits, reproducible validation batches, and `[B, T, C, H, W]` batch shape.
