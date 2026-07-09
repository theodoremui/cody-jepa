# Week 2.2: Index, Splits, And Loader

This lesson turns extracted GaitLU-1M `.pkl` sequences into a reproducible data contract.

The contract has four parts:

1. An index manifest with one row per sequence.
2. A deterministic split manifest.
3. A notebook-local PyTorch dataset that returns `[T, C, H, W]`.
4. A `DataLoader` batch with shape `[B, T, C, H, W]`.

The notebook uses synthetic `.pkl` files by default. On HAIC, run it with `GAITLU_NOTEBOOK_MODE=real` so it refuses to create synthetic fixtures and only scans the legally extracted GaitLU pickle root.

## 1. Load Real Pickles Only After Legal Extraction

The real GaitLU archive is password-protected and access-controlled. Do not load real `.pkl` files until:

1. The dataset agreement process is complete.
2. The official password has been obtained through the approved channel.
3. Extraction has happened on HAIC storage.
4. You are running indexing in a SLURM job or controlled interactive compute allocation.

In its default synthetic mode, the notebook creates tiny `.pkl` files under `data/gaitlu-1m/synthetic_raw/` so every code path can be tested without touching the real archive. In `GAITLU_NOTEBOOK_MODE=real`, it refuses to create fixtures and scans only `GAITLU_EXTRACTED_DIR`.

## 2. Build The Index Manifest

Each row should contain:

| Column | Meaning |
| --- | --- |
| `sequence_id` | Stable SHA-1 based ID derived from the relative pickle path. |
| `relative_pkl_path` | Path relative to the discovered pickle root. |
| `shard_0`, `shard_1`, `shard_2` | The first path tokens, useful as proxy grouping keys. |
| `shard_path` | Parent directory path relative to the pickle root. |
| `num_frames` | Frame count after loading and normalizing the array shape. |
| `height` | Frame height. |
| `width` | Frame width. |
| `dtype` | Source array dtype. |
| `read_ok` | Boolean read status. |
| `error` | Empty when `read_ok` is true, otherwise a short error string. |

The manifest is metadata, not copied data. Treat it as restricted unless the dataset agreement explicitly permits sharing. Gait is biometric, and path shards, diagnostics, and latent exports can still reveal information about the source collection. Keep generated manifests under `data/gaitlu-1m/manifests/` until the project has a clear sharing policy.

## 3. Normalize Sequence Arrays

Pickle payloads may be plain arrays, lists, or dictionaries. Normalize them to:

```text
[T, H, W]
```

Then the dataset converts each clip to:

```text
[T, C, H, W]
```

For silhouettes, use `C = 1`. The first loader does not need augmentation, resizing, or labels. It needs correctness and reproducibility.

## 4. Split With Stable Hashing

Do not use Python's built-in `hash()` for splits. It is intentionally salted between processes and can change across runs.

Use a stable hash such as SHA-1:

```python
import hashlib

def stable_int(text, seed=0):
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:16], 16)
```

GaitLU-1M is access-controlled silhouette data from public videos, and gait remains a biometric signal. There is no guaranteed subject ID in the path. Split by the highest stable path shard available, such as `shard_0`, and document it as a proxy split key. This is not a true subject split. It is a leakage-reduction proxy until better metadata exists.

For small smoke tests, choose validation groups by sorted stable hash rank so both train and validation are nonempty.

## 5. Sample Windows Reproducibly

Training windows may use seeded random starts. Validation windows must be deterministic from:

```text
sequence_id, seed, clip_length
```

That makes validation loss curves comparable across runs.

The notebook-local dataset follows this rule:

```text
training:
  start = seeded random window from sequence_id, seed, and epoch

validation:
  start = stable SHA-1 window from sequence_id, seed, and clip_length
```

If a sequence is shorter than `clip_length`, repeat the last frame until the clip is long enough.

## 6. Loader Contract

A single sample returns:

```python
{
    "video": video_tensor,       # [T, C, H, W]
    "sequence_id": sequence_id,
    "split": split,
    "proxy_label": proxy_split_key,
}
```

A `DataLoader` batch returns:

```text
batch["video"].shape == [B, T, C, H, W]
```

Use `num_workers=0` in the first notebook to remove worker-seeding complexity. Increase workers later after the single-process contract passes.

## 7. HAIC Batch Template For Indexing

After the notebook works on synthetic data, move the full index pass into a SLURM job:

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

For the full dataset, consider turning the notebook code into a script after the tutorial is understood. The notebook exists to make the data contract inspectable.

## Week 2 Gate For This Lesson

This lesson passes when:

1. Rebuilding the split manifest from the same index and seed gives exactly the same table.
2. Recreating the validation loader gives exactly the same first validation batch.
3. The batch shape is `[B, T, C, H, W]`.
