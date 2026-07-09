# Week 2.3: Diagnostics And Probe Exports

This lesson checks whether the loader is producing plausible motion clips and writes a dummy probe export schema for later CoDy-JEPA latents.

Diagnostics are not decoration. They catch broken extraction, bad sorting, repeated frames, empty silhouettes, mislabeled splits, and non-reproducible batches before training hides the problem behind a loss curve.

## 1. Keep Diagnostics Under Ignored Data Paths

Use:

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

## 2. Metadata Summaries

Start with table summaries:

```text
number of sequences
read_ok count
read error count
frame count distribution
height and width distribution
split counts
proxy split key counts
```

For real GaitLU-1M, scan for long tails. Very short clips, zero-sized arrays, or unusual frame shapes should be visible before training.

## 3. Batch Contact Sheets

A contact sheet takes a few evenly spaced frames from a batch item and places them in a grid. Use it to answer:

1. Are frames sorted in time?
2. Are silhouettes visible?
3. Are channels and intensity scaling correct?
4. Is padding obvious on short clips?

For real data, save only a small controlled sample in the ignored diagnostics directory.

## 4. Frame-Difference Maps

A frame-difference map averages absolute differences between adjacent frames:

```text
mean(abs(x[t + 1] - x[t]))
```

It helps catch two common bugs:

1. Low or zero motion because every frame is duplicated.
2. Excessive flicker because frame order is wrong or arrays are decoded incorrectly.

## 5. Motion-Energy Histograms

Motion energy is a simple scalar per sequence or clip:

```text
mean(abs(x[t + 1] - x[t]))
```

Plot the distribution. Then inspect low-motion and high-motion examples. Low motion may be normal standing or failed extraction. High motion may be a valid fast walk, camera jitter, or corrupted ordering.

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

In Week 2, `s_attr` and `s_dyn` are deterministic dummy vectors derived from simple clip statistics and stable seeded noise. In later weeks, they will be frozen model outputs.

Keep in mind that `proxy_label` is not a subject label. For GaitLU-1M, it is only the best available path-shard proxy unless better metadata is added.

## 7. HAIC Batch Template For Diagnostics

Run diagnostics in SLURM for the real dataset:

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

Start with a controlled subset. Only broaden diagnostics after the loader gate passes.

## Week 2 Gate For This Lesson

This lesson passes when:

1. Metadata summaries agree with the split manifest.
2. Contact sheets and frame-difference maps look plausible on a small sample.
3. Dummy probe exports contain `s_attr`, `s_dyn`, `sequence_id`, `split`, and proxy labels.
4. The companion notebook proves reproducible splits, reproducible validation batches, and `[B, T, C, H, W]` batch shape.
