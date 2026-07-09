# Week 2.3: Diagnostics And Probe Exports

This lesson checks whether the loader is producing plausible motion clips and writes a dummy probe export schema for later CoDy-JEPA latents.

Diagnostics are not decoration. They catch broken extraction, bad sorting, repeated frames, empty silhouettes, mislabeled splits, and non-reproducible batches before training hides the problem behind a loss curve.

The workflow is:

```text
manifest + loader -> diagnostic tables and images -> dummy probe export schema
```

Run real-data diagnostics on HAIC or Sherlock, not on your laptop, because diagnostics read extracted sequences and may produce participant-derived media.

## 1. Keep Diagnostics Under Ignored Data Paths

For local synthetic smoke tests, use:

```text
data/gaitlu-1m/diagnostics/
data/gaitlu-1m/probe_exports/
```

On HAIC, use:

```text
$GAITLU_DIAGNOSTICS_DIR
$GAITLU_PROBE_EXPORT_DIR
```

Do not commit rendered participant media, contact sheets, frame grids, full manifests with sensitive paths, or latent exports.

Treat diagnostics and latent exports as restricted unless the dataset agreement explicitly permits sharing. Gait remains biometric even when the source data is a silhouette sequence.

Before running real diagnostics, confirm the output paths. Run this from any directory on the HAIC or Sherlock login node:

```bash
haic-login$ echo "$GAITLU_DIAGNOSTICS_DIR"
haic-login$ echo "$GAITLU_PROBE_EXPORT_DIR"
```

Each line should print a cluster storage path. If either line is empty, set the variables before submitting diagnostics.

## 2. Metadata Summaries

Start with table summaries before looking at images.

Useful summaries include:

```text
number of sequences
read_ok count
read error count
frame count distribution
height and width distribution
split counts
proxy split key counts
```

These summaries answer basic questions:

| Summary | What it catches |
| --- | --- |
| `number of sequences` | Accidentally indexing too few files or the wrong root. |
| `read_ok count` | Whether most pickle files loaded successfully. |
| `read error count` | Corrupt files, unexpected payloads, or wrong paths. |
| `frame count distribution` | Very short clips, unusually long clips, and possible partial extraction. |
| `height and width distribution` | Unexpected frame shapes or decoding mistakes. |
| `split counts` | Empty train or validation splits. |
| `proxy split key counts` | A split key that is too coarse, too fine, or unexpectedly imbalanced. |

For real GaitLU-1M, scan for long tails. Very short clips, zero-sized arrays, or unusual frame shapes should be visible before training.

## 3. Batch Contact Sheets

A contact sheet takes a few evenly spaced frames from a batch item and places them in a grid.

Use it to answer:

1. Are frames sorted in time?
2. Are silhouettes visible?
3. Are channels and intensity scaling correct?
4. Is padding obvious on short clips?

For real data, save only a small controlled sample in the ignored diagnostics directory. Do not commit the image.

## 4. Frame-Difference Maps

A frame-difference map averages absolute differences between adjacent frames:

```text
mean(abs(x[t + 1] - x[t]))
```

Read the expression as:

| Piece | Meaning |
| --- | --- |
| `x[t + 1]` | The next frame. |
| `x[t]` | The current frame. |
| `x[t + 1] - x[t]` | The pixel-wise change from one frame to the next. |
| `abs(...)` | Ignore direction and keep only change magnitude. |
| `mean(...)` | Average the changes into a compact map or scalar. |

This catches two common bugs:

1. Low or zero motion because every frame is duplicated.
2. Excessive flicker because frame order is wrong or arrays are decoded incorrectly.

## 5. Motion-Energy Histograms

Motion energy is a simple scalar per sequence or clip:

```text
mean(abs(x[t + 1] - x[t]))
```

Plot the distribution. Then inspect low-motion and high-motion examples.

Low motion may be normal standing, excessive padding, repeated frames, or failed extraction. High motion may be a valid fast walk, camera jitter, wrong frame order, or corrupted decoding. The histogram tells you which examples deserve visual inspection.

## 6. Dummy Probe Export Schema

Week 2 should create a dummy table with the same columns later probes expect. This makes Week 8 evaluation easier because the export contract already exists.

Use columns like:

```text
sequence_id
split
proxy_label
s_attr_0, s_attr_1, ...
s_dyn_0, s_dyn_1, ...
```

The columns mean:

| Column | Meaning |
| --- | --- |
| `sequence_id` | Stable sequence ID from the manifest. |
| `split` | Split name, such as `train` or `val`. |
| `proxy_label` | Path-shard proxy key. It is not a verified subject label. |
| `s_attr_0, s_attr_1, ...` | Placeholder static-attribute dimensions. Later weeks replace these with frozen model outputs. |
| `s_dyn_0, s_dyn_1, ...` | Placeholder dynamic-motion dimensions. Later weeks replace these with frozen model outputs. |

In Week 2, `s_attr` and `s_dyn` are deterministic dummy vectors derived from simple clip statistics and stable seeded noise. In later weeks, they will be frozen model outputs.

Keep in mind that `proxy_label` is not a subject label. For GaitLU-1M, it is only the best available path-shard proxy unless better metadata is added.

## 7. HAIC Batch Template For Diagnostics

Run diagnostics in SLURM for the real dataset.

Create a local script named `scripts/gaitlu_diag.sbatch` in your cluster repo checkout.

First make sure the repo has local `scripts/` and `logs/` directories. Run this from the repo root on the HAIC or Sherlock login node:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ mkdir -p scripts logs
```

`scripts/` will hold your site-specific batch script. `logs/` will hold SLURM output and error files.

Save this content in `scripts/gaitlu_diag.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=gaitlu_diag
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${CODY_JEPA_ROOT:?Set CODY_JEPA_ROOT before sbatch}"

export GAITLU_NOTEBOOK_MODE=real
export GAITLU_MAX_DIAGNOSTIC_ROWS="${GAITLU_MAX_DIAGNOSTIC_ROWS:-64}"

uv sync --locked
uv run jupyter nbconvert \
  --to notebook \
  --execute tutorials/notebooks/week2_03_diagnostics_probe_exports.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --output-dir "$GAITLU_DIAGNOSTICS_DIR" \
  --output "week2_03_diagnostics_probe_exports.executed.ipynb"
```

The script lines mean:

| Line | Meaning |
| --- | --- |
| `#!/bin/bash` | Run the script with Bash. |
| `#SBATCH ...` lines | Request job name, time, task count, CPU cores, memory, and log paths. Keep these before real shell commands. |
| `set -euo pipefail` | Stop on common shell errors. |
| `cd "${CODY_JEPA_ROOT:?...}"` | Enter the repo root and fail clearly if the variable is unset. |
| `export GAITLU_NOTEBOOK_MODE=real` | Prevent the notebook from creating synthetic fixtures when using real data. |
| `export GAITLU_MAX_DIAGNOSTIC_ROWS="${GAITLU_MAX_DIAGNOSTIC_ROWS:-64}"` | Limit the first real diagnostic pass to 64 rows unless you set a different value. |
| `uv sync --locked` | Install the locked environment on the compute node. |
| `uv run jupyter nbconvert ...` | Execute the diagnostics notebook from top to bottom. |
| `--output-dir "$GAITLU_DIAGNOSTICS_DIR"` | Write the executed notebook under the diagnostics directory. |
| `--output "week2_03_diagnostics_probe_exports.executed.ipynb"` | Give the executed notebook a clear output name. |

Submit and monitor it from the repo root on the HAIC or Sherlock login node:

```bash
laptop$ ssh <sunetid>@<haic-login-host>
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ sbatch scripts/gaitlu_diag.sbatch
haic-login$ squeue -u "$USER"
haic-login$ sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

These commands mean:

| Command | Meaning |
| --- | --- |
| `ssh <sunetid>@<haic-login-host>` | Connect from your laptop to the HAIC or Sherlock login node. |
| `cd "$CODY_JEPA_ROOT"` | Enter the project directory on the cluster before submitting the repo script. |
| `sbatch scripts/gaitlu_diag.sbatch` | Submit the diagnostics notebook execution to SLURM. |
| `squeue -u "$USER"` | Check whether the diagnostics job is waiting or running. |
| `sacct -j <job_id> ...` | Check exit status, elapsed time, and memory after completion. |

Start with a controlled subset. Only broaden diagnostics after the loader gate passes and the first images look plausible.

## Week 2 Gate For This Lesson

This lesson passes when:

1. Metadata summaries agree with the split manifest.
2. Contact sheets and frame-difference maps look plausible on a small sample.
3. Dummy probe exports contain `s_attr`, `s_dyn`, `sequence_id`, `split`, and proxy labels.
4. The companion notebook proves reproducible splits, reproducible validation batches, and `[B, T, C, H, W]` batch shape.
