# Week 2.2: Index, Splits, And Loader

This lesson turns extracted GaitLU-1M `.pkl` sequences into a reproducible data contract.

A data contract is a promise about what files exist, what columns describe them, how splits are chosen, and what tensor shape the loader returns. Training should not start until this contract is stable.

The contract has four parts:

1. An index manifest with one row per sequence.
2. A deterministic split manifest.
3. A notebook-local PyTorch dataset that returns one clip with shape `[T, C, H, W]`.
4. A `DataLoader` batch with shape `[B, T, C, H, W]`.

The notebook uses synthetic `.pkl` files by default. On HAIC or Sherlock, run it with `GAITLU_NOTEBOOK_MODE=real` so it refuses to create synthetic fixtures and only scans the legally extracted GaitLU pickle root.

## 1. Load Real Pickles Only After Legal Extraction

The real GaitLU archive is password-protected and access-controlled. Do not load real `.pkl` files until all four conditions are true:

1. The dataset agreement process is complete.
2. The official password has been obtained through the approved channel.
3. Extraction has happened on HAIC, Sherlock, or approved cluster storage.
4. Indexing runs in a SLURM job or controlled interactive compute allocation.

In default synthetic mode, the notebook creates tiny `.pkl` files under `data/gaitlu-1m/synthetic_raw/` so every code path can be tested without touching the real archive. In `GAITLU_NOTEBOOK_MODE=real`, it refuses to create fixtures and scans only `GAITLU_EXTRACTED_DIR`.

## 2. Build The Index Manifest

The index manifest is a CSV table. Each row describes one sequence pickle. It does not copy the video data.

Each row should contain:

| Column | Meaning |
| --- | --- |
| `sequence_id` | Stable SHA-1 based ID derived from the relative pickle path. This gives each sequence a reproducible name. |
| `relative_pkl_path` | Path from the discovered pickle root to the pickle file. This avoids storing machine-specific absolute paths. |
| `shard_0`, `shard_1`, `shard_2` | The first path tokens, useful as proxy grouping keys. |
| `shard_path` | Parent directory path relative to the pickle root. |
| `num_frames` | Frame count after loading and normalizing the array shape. |
| `height` | Frame height. |
| `width` | Frame width. |
| `dtype` | Source array dtype. |
| `read_ok` | Boolean read status. `true` means the pickle loaded and normalized successfully. |
| `error` | Empty when `read_ok` is true, otherwise a short error string. |

The manifest is metadata, but it can still be sensitive. Treat it as restricted unless the dataset agreement explicitly permits sharing. Gait is biometric, and path shards, diagnostics, and latent exports can still reveal information about the source collection. Keep generated manifests under `data/gaitlu-1m/manifests/` locally, or under the cluster path stored in `$GAITLU_MANIFEST_DIR`, until the project has a clear sharing policy.

## 3. Normalize Sequence Arrays

Pickle payloads may be plain arrays, lists, or dictionaries. The first job is to normalize each payload into this shape:

```text
[T, H, W]
```

The letters mean:

| Letter | Meaning |
| --- | --- |
| `T` | Time steps, or frames. |
| `H` | Frame height. |
| `W` | Frame width. |

Then the dataset adds a channel dimension and returns:

```text
[T, C, H, W]
```

For silhouettes, use `C = 1` because each frame is a single-channel mask. The first loader does not need augmentation, resizing, labels, or training tricks. It needs correctness and reproducibility.

## 4. Split With Stable Hashing

Do not use Python's built-in `hash()` for splits. It is intentionally salted between processes and can change across runs.

Use a stable hash such as SHA-1:

```python
import hashlib

def stable_int(text, seed=0):
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:16], 16)
```

Read the function line by line:

| Line | Meaning |
| --- | --- |
| `import hashlib` | Load Python's standard hashing library. |
| `def stable_int(text, seed=0):` | Define a function that turns text and a seed into a reproducible integer. |
| `payload = f"{seed}:{text}".encode("utf-8")` | Combine the seed and text, then convert the result to bytes because SHA-1 hashes bytes. |
| `return int(hashlib.sha1(payload).hexdigest()[:16], 16)` | Hash the bytes, take the first 16 hexadecimal characters, and convert them into an integer. |

GaitLU-1M is access-controlled silhouette data from public videos, and gait remains a biometric signal. There is no guaranteed subject ID in the path. Split by the highest stable path shard available, such as `shard_0`, and document it as a proxy split key.

This is not a true subject split. It is a leakage-reduction proxy until better metadata exists.

For small smoke tests, choose validation groups by sorted stable hash rank so both train and validation are nonempty.

## 5. Sample Windows Reproducibly

Training and validation have different needs.

Training windows may use seeded random starts because training benefits from seeing different windows over time. Validation windows must be deterministic so validation loss curves are comparable across runs.

Validation should depend only on:

```text
sequence_id, seed, clip_length
```

The notebook-local dataset follows this rule:

```text
training:
  start = seeded random window from sequence_id, seed, and epoch

validation:
  start = stable SHA-1 window from sequence_id, seed, and clip_length
```

If a sequence is shorter than `clip_length`, repeat the last frame until the clip is long enough. This keeps every returned clip the same length without inventing new motion.

## 6. Loader Contract

A single sample returns a dictionary:

```python
{
    "video": video_tensor,       # [T, C, H, W]
    "sequence_id": sequence_id,
    "split": split,
    "proxy_label": proxy_split_key,
}
```

Each field has one purpose:

| Field | Meaning |
| --- | --- |
| `video` | The clip tensor. Its shape must be `[T, C, H, W]`. |
| `sequence_id` | The stable ID for the source sequence. |
| `split` | The split name, such as `train` or `val`. |
| `proxy_label` | The path-shard proxy key. It is not a verified subject label. |

A `DataLoader` stacks several samples together. The batch must satisfy:

```text
batch["video"].shape == [B, T, C, H, W]
```

`B` is batch size. PyTorch adds this dimension when it collates individual samples.

Use `num_workers=0` in the first notebook to remove worker-seeding complexity. Increase workers later after the single-process contract passes.

## 7. HAIC Batch Template For Indexing

After the notebook works on synthetic data, move the full index pass into a SLURM job.

Create a local script named `scripts/gaitlu_index.sbatch` in your cluster repo checkout.

First make sure the repo has local `scripts/` and `logs/` directories. Run this from the repo root on the HAIC or Sherlock login node:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ mkdir -p scripts logs
```

`scripts/` will hold your site-specific batch script. `logs/` will hold SLURM output and error files.

Save this content in `scripts/gaitlu_index.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=gaitlu_index
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${CODY_JEPA_ROOT:?Set CODY_JEPA_ROOT before sbatch}"

export GAITLU_EXTRACTED_DIR="${GAITLU_EXTRACTED_DIR:?Set GAITLU_EXTRACTED_DIR}"
export GAITLU_MANIFEST_DIR="${GAITLU_MANIFEST_DIR:?Set GAITLU_MANIFEST_DIR}"
export GAITLU_NOTEBOOK_MODE=real

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

uv run jupyter nbconvert \
  --to notebook \
  --execute tutorials/notebooks/week2_02_index_splits_loader.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --output-dir "$GAITLU_MANIFEST_DIR" \
  --output "week2_02_index_splits_loader.executed.ipynb"
```

The script lines mean:

| Line | Meaning |
| --- | --- |
| `#!/bin/bash` | Run the script with Bash. |
| `#SBATCH ...` lines | Request job name, time, task count, CPU cores, memory, and log paths. Keep these lines before real shell commands. |
| `set -euo pipefail` | Stop on common shell errors. |
| `cd "${CODY_JEPA_ROOT:?...}"` | Enter the repo root and fail clearly if the variable is unset. |
| `export GAITLU_EXTRACTED_DIR=...` | Require the path to legally extracted `.pkl` data. |
| `export GAITLU_MANIFEST_DIR=...` | Require the output directory for index and split manifests. |
| `export GAITLU_NOTEBOOK_MODE=real` | Tell the notebook not to create synthetic fixtures. |
| `uv sync --locked` | Install the locked environment on the compute node. |
| `uv run python - <<'PY' ... PY` | Print package versions into the job log. |
| `uv run jupyter nbconvert ...` | Execute the indexing notebook non-interactively. |
| `--output-dir "$GAITLU_MANIFEST_DIR"` | Write the executed notebook to the manifest directory. |
| `--output "week2_02_index_splits_loader.executed.ipynb"` | Give the executed notebook a clear output name. |

Submit and monitor the script from the repo root on the HAIC or Sherlock login node:

```bash
laptop$ ssh <sunetid>@<haic-login-host>
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ sbatch scripts/gaitlu_index.sbatch
haic-login$ squeue -u "$USER"
haic-login$ sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

Read the commands as a path from laptop to compute:

| Command | Meaning |
| --- | --- |
| `ssh <sunetid>@<haic-login-host>` | Open a terminal on the HAIC or Sherlock login node. Replace both placeholders. |
| `cd "$CODY_JEPA_ROOT"` | Move to the project directory on the cluster before submitting a repo script. |
| `sbatch scripts/gaitlu_index.sbatch` | Ask SLURM to run the indexing script on a compute node. |
| `squeue -u "$USER"` | Watch your pending and running jobs from the login node. |
| `sacct -j <job_id> ...` | Inspect a completed job. Replace `<job_id>` with the number printed by `sbatch`. |

For the full dataset, consider turning the notebook code into a script after the tutorial is understood. The notebook exists to make the data contract inspectable first.

## Week 2 Gate For This Lesson

This lesson passes when:

1. Rebuilding the split manifest from the same index and seed gives exactly the same table.
2. Recreating the validation loader gives exactly the same first validation batch.
3. The batch shape is `[B, T, C, H, W]`.
