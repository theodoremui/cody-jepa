# Train the single-stream JEPA prototype on HAIC

This guide runs [`notebooks/single-stream-jepa.ipynb`](../notebooks/single-stream-jepa.ipynb) on one NVIDIA H100 GPU in the Stanford HAI Compute Cluster. The notebook is a Health&Gait feasibility experiment that learns masked video representations from a single stream. It is not the final dual-stream CoDy-JEPA model.

The workflow has three separate phases:

1. Verify the Python environment and CUDA runtime in a short interactive H100 session.
2. Validate the dataset in a CPU-only Slurm session.
3. Use the pipeline entry point to submit training, validation, feature export,
   all probes, and report generation as one Slurm job.

Keeping these phases separate prevents slow dataset checks from consuming H100 time. It also makes failures easier to locate.

## Retained results and new-run names

The retained working result uses the same directory name in HAIC and locally:

| Experiment | HAIC directory | Local directory | Retained notebook |
| --- | --- | --- | --- |
| Stabilized VICReg baseline (job 91108) | `outputs/jepa-v4` | `outputs/jepa-v4` | `haic-results/job_91108.ipynb` |

The job 90881 and 91023 notebooks and the local `outputs/jepa-v3/` copy were
deleted because they are failed historical attempts, not research comparison
artifacts. Run IDs 90881 and 91023 may still appear in diagnostic comments or
the job 91108 writeup to explain the earlier collapse, but they no longer name
files that should exist. Use a fresh versioned directory, beginning with
`outputs/jepa-v5`, for any new run so the retained `jepa-v4` result is never
overwritten.

## What a successful run must show

A falling loss is not enough. A model can produce a smooth loss curve while learning collapsed or position-only features. Treat the run as successful only when all of the following conditions hold:

- Validation loss improves.
- The normalized, full-view online/context encoder features retain nontrivial variance and effective rank.
- A seeded one-to-one wrong-subject permutation over the full validation set increases prediction loss on a subject-balanced basis.
- Training and validation subjects remain disjoint.
- `latest.pt` resumes from an exact epoch boundary when the dataset is unchanged.
- `best_loss.pt` records the lowest subject-balanced validation loss.
- `best_healthy.pt`, when written, records the best checkpoint that also passes every representation-health check.

## Training configuration

The notebook currently uses these production settings:

| Setting | Value | Meaning |
| --- | ---: | --- |
| Physical batch size | `16` | Clips processed in one GPU microbatch. |
| Gradient accumulation | `4` | Four microbatches produce an effective batch of 64 clips. |
| Frames per clip | `16` | Number of grayscale silhouette frames in each input clip. |
| Image resolution | `112 x 112` | Spatial size after preprocessing. |
| Tubelet size | `2` | Number of adjacent frames in each temporal token. |
| Patch size | `8 x 8` | Spatial area represented by each token. |
| Tokens per clip | `1,568` | Token count produced by the configured video geometry. |
| Epochs | `100` | Complete passes through the training split. |
| Optimizer steps | `3,900` | Exactly 39 optimizer updates per epoch. |
| Validation cadence | Every 5 epochs | Validation uses three deterministic windows per sequence. |
| Checkpoint cadence | Every epoch | `latest.pt` is written after each complete epoch. |
| Numeric precision | BF16 | Reduced precision supported efficiently by H100 GPUs. |
| Compilation | Disabled | The first production run uses eager PyTorch execution. |

The physical batch size is intentionally smaller than the original notebook setting. Gradient accumulation preserves an effective batch of 64 without keeping a batch-256 predictor graph in GPU memory.

## 1. Prepare the HAIC workspace

Run setup commands from the HAIC head node. Do not run training, Jupyter, or other compute-heavy processes on the head node.

For a new checkout:

```bash
cd /hai/scratch/$USER
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
uv sync --frozen
```

For an existing checkout:

```bash
cd /hai/scratch/$USER/cody-jepa
git pull --ff-only
uv sync --frozen
```

Always launch Python and Jupyter through `uv run`. This ensures that the notebook uses the project environment instead of a system or Conda installation.
Manage dependencies only with `uv add`, `uv remove`, `uv lock`, and `uv sync`;
do not use pip, Conda, Poetry, or system-Python installation commands in this workflow.

### Confirm that the dataset exists

The notebook expects these paths:

```text
data/healthgait/raw/Health_Gait/silhouette/
data/healthgait/manifests/silhouette_subject_split_seed0.csv
```

Check them before requesting compute resources:

```bash
test -d data/healthgait/raw/Health_Gait/silhouette
test -f data/healthgait/manifests/silhouette_subject_split_seed0.csv
wc -l data/healthgait/manifests/silhouette_subject_split_seed0.csv
```

The current manifest should contain 3,131 lines: one header and 3,130 sequence rows. If the data is missing, follow [`health-and-gait.md`](health-and-gait.md) before continuing.

## 2. Verify the H100 and Python environment

Request a short interactive H100 allocation:

```bash
srun \
  --account=mind \
  --partition=hai-interactive \
  --gres=gpu:h100:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=04:00:00 \
  --pty bash
```

After the interactive shell opens, return to the project directory:

```bash
cd /hai/scratch/$USER/cody-jepa
```

### Check the allocated GPU

```bash
nvidia-smi
```

The device name should contain `H100`. Do not continue with this locked environment if Slurm assigns a B200 or another GPU family.

### Check the PyTorch CUDA runtime

Run a real CUDA operation and print the active Python and PyTorch configuration:

```bash
uv run python - <<'PY'
import json
import sys
import torch

device = torch.device("cuda")
capability = torch.cuda.get_device_capability(device)
required_arch = f"sm_{capability[0]}{capability[1]}"
torch_cuda_arch_list = torch.cuda.get_arch_list()

probe = torch.zeros(1, device=device)
probe.add_(1)
torch.cuda.synchronize(device)

print(json.dumps({
    "cuda_compute_capability": capability,
    "cuda_device_name": torch.cuda.get_device_name(device),
    "cuda_preflight": "passed",
    "python_executable": sys.executable,
    "required_cuda_arch": required_arch,
    "torch_cuda_arch_list": torch_cuda_arch_list,
    "torch_cuda_version": torch.version.cuda,
    "torch_has_required_cuda_arch": required_arch in torch_cuda_arch_list,
    "torch_version": torch.__version__,
}, indent=2, sort_keys=True))
PY
```

For an H100, expect the device name to contain `H100`, the compute capability to be `(9, 0)`, and the required architecture to be `sm_90`.

### Run the test suite

```bash
MPLCONFIGDIR=/tmp/mpl \
uv run python -m unittest discover -s tests -v
```

Do not submit the training job until the CUDA probe and all tests pass.

### Repair an incompatible PyTorch installation

If the probe reports `no kernel image is available for execution on the device`, compare `required_cuda_arch` with `torch_cuda_arch_list`.

If the allocated GPU is an H100 but `sm_90` is missing, reinstall the locked PyTorch packages:

```bash
uv sync --frozen \
  --reinstall-package torch \
  --reinstall-package torchvision
```

Then rerun the CUDA probe. Do not continue until it passes.

When the GPU and test checks are complete, leave the interactive GPU shell:

```bash
exit
```

## 3. Run the dataset preflight without a GPU

The dataset preflight validates every manifest row and decodes one deterministic clip from every training and validation sequence. It detects subject leakage, invalid paths, duplicate sources, corrupt sampled images, blank sequences, static sequences, and missing-frame gaps.

This work is CPU and storage intensive, so run it without requesting a GPU:

```bash
srun \
  --account=mind \
  --partition=hai-interactive \
  --cpus-per-task=8 \
  --mem=64G \
  --time=01:00:00 \
  --pty bash
```

Inside the CPU-only shell, execute the notebook with explicit preflight settings:

```bash
cd /hai/scratch/$USER/cody-jepa
mkdir -p notebook-runs

CODY_JEPA_RUN_FULL_TRAINING=0 \
CODY_JEPA_RUN_DATA_AUDIT=1 \
CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT=0 \
MPLCONFIGDIR=/tmp/mpl \
uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output single-stream-jepa-preflight.ipynb \
  --ExecutePreprocessor.timeout=1800
```

The preflight can take about 10 minutes on HAIC storage. A successful run writes:

```text
notebook-runs/single-stream-jepa-preflight.ipynb
```

Open or inspect that executed notebook and confirm that it reports:

- 2,506 training sequences.
- 624 validation sequences.
- 318 training subjects.
- 80 validation subjects.
- No subject overlap.
- A completed CPU smoke-training step.

Do not submit the GPU job if the preflight raises an exception.

When the preflight passes, leave the CPU-only shell:

```bash
exit
```

## 4. Configure a new training run

Choose three fresh, distinct destinations: checkpoints, evaluation artifacts,
and the versioned report. Never reuse a directory from an existing experiment.
The pipeline rejects every `outputs/jepa-v3` path and all writes under retained
baseline evidence in `outputs/jepa-v4`.

The current compatibility identifiers remain
`MODEL_ARCHITECTURE = "cody-jepa-single-stream-masked-v3"` and
`CHECKPOINT_SCHEMA = 3`. Run-directory names such as `outputs/jepa-v5` identify
experiments, not model architectures or checkpoint schemas; do not bump either
constant merely because a new run is launched.

Select the checkpoint policy before submission. Use `best_healthy.pt` when the
experiment is predeclared to require the health gate, or `best_loss.pt` for a
loss-selected comparison. The workflow never silently falls back when the
declared checkpoint is absent.

### Validate the batch script

```bash
bash -n slurm/train-single-stream-jepa.sbatch
```

No output means that the shell syntax is valid.

## 5. Submit the train-to-report job

From the HAIC head node, run the single documented entry point:

```bash
uv run python scripts/run_phase0_pipeline.py submit \
  --run-dir outputs/jepa-v5 \
  --artifact-dir outputs/pipeline/jepa-v5 \
  --report reports/jepa-v5.md \
  --checkpoint-name best_loss.pt \
  --success-criterion "Named scientific change: pass the predeclared Phase 1 health gate"
```

The command creates scheduler prerequisites and submits
[`slurm/train-single-stream-jepa.sbatch`](../slurm/train-single-stream-jepa.sbatch).
It returns immediately with the numeric job ID. The batch worker invokes the
same pipeline in `run` mode, so training, completed-run validation, feature
export, probes, and report generation all remain inside the H100 allocation.
Do not run those compute-heavy stages on the head node.

Check its state:

```bash
squeue -j "$JOB_ID" -o "%.18i %.2t %.10M %R"
```

Common states are:

| State | Meaning |
| --- | --- |
| `PD` | The job is pending and waiting for resources or quota. |
| `R` | Slurm has started the batch script on a compute node. |
| `CG` | The process has exited and Slurm is completing cleanup. |
| `CD` | The job completed successfully. |
| `F` | The job failed. Inspect the Slurm log for the cause. |

Queue time is not predictable. It depends on H100 availability, the `mind` account quota, and other jobs in the `hai` partition.

## 6. Monitor startup and training

The batch script creates two logs:

```text
logs/single-stream-jepa-<JOB_ID>.out
logs/gpu-<JOB_ID>.csv
```

Watch the Slurm log in one terminal:

```bash
tail -F "logs/single-stream-jepa-${JOB_ID}.out"
```

Watch GPU utilization in another terminal:

```bash
tail -F "logs/gpu-${JOB_ID}.csv"
```

The GPU log records utilization and memory once per minute. Low utilization during initial dataset construction is normal. With the full data audit disabled, expect the production BF16 CUDA preflight to begin within about 5 to 10 minutes after the job enters state `R`. Investigate if the job remains in state `R` for more than 15 minutes without any increase in GPU memory or utilization.

The notebook performs one real, production-sized forward and backward step before epoch 1. This step verifies CUDA compatibility, BF16 execution, data transfer, finite loss, finite gradients, and peak GPU memory.

`nbconvert` captures cell output inside the executed notebook. It does not stream every epoch metric to the Slurm log. During training, use the GPU CSV and checkpoint timestamps as the primary progress signals:

```bash
ls -lh "outputs/jepa-v5"
```

`latest.pt` appears after the first complete epoch and is replaced atomically after each later epoch.

For scheduler details, run:

```bash
scontrol show job "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

To cancel the job:

```bash
scancel "$JOB_ID"
```

## 7. Inspect completed outputs

A completed run should produce:

```text
notebook-runs/single-stream-jepa-<JOB_ID>.executed.ipynb
outputs/jepa-v5/latest.pt
outputs/jepa-v5/best_loss.pt
outputs/pipeline/jepa-v5/features.npz
outputs/pipeline/jepa-v5/features.npz.metadata.json
outputs/pipeline/jepa-v5/probes/probe_metrics.json
reports/jepa-v5.md
reports/jepa-v5.json
logs/single-stream-jepa-<JOB_ID>.out
logs/gpu-<JOB_ID>.csv
```

`best_healthy.pt` appears only after a validation checkpoint passes every representation-health criterion:

```text
outputs/jepa-v5/best_healthy.pt
```

Inspect checkpoint metadata without loading model tensors onto the GPU:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

from cody_jepa.single_stream_jepa import load_checkpoint

output_dir = Path("outputs/jepa-v5")
metadata = {}

for name in ("latest.pt", "best_loss.pt", "best_healthy.pt"):
    path = output_dir / name
    if not path.exists():
        metadata[name] = {"status": "not_written"}
        continue

    checkpoint = load_checkpoint(path)
    metadata[name] = {
        "best_epoch": checkpoint["best_epoch"],
        "best_healthy_epoch": checkpoint["best_healthy_epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "completed_epochs": checkpoint["completed_epochs"],
        "dataset_sha256": checkpoint["data_contract"]["train_dataset"]["dataset_sha256"],
        "global_step": checkpoint["global_step"],
        "status": "written",
    }

print(json.dumps(metadata, indent=2, sort_keys=True))
PY
```

Use the checkpoints as follows:

- `latest.pt` is for resuming an interrupted run.
- `best_loss.pt` is useful for diagnosis and loss-based comparisons.
- `best_healthy.pt` is the preferred checkpoint for representation probes.

If `best_healthy.pt` was not written, the run did not pass the health contract.
Do not silently substitute `best_loss.pt`; report the failed health metrics and
start a corrected run instead.

Archive a completed result only after inspecting it. For job 91108, the curated
copy is `haic-results/job_91108.ipynb`; strings inside it still record the HAIC
paths used when it executed and should not be rewritten after the fact.

When a notebook run contains errors or is unsafe to compare directly:

1. Check whether it completed an epoch and wrote a checkpoint. A partially
   executed notebook without a checkpoint is diagnostic evidence only.
2. Inspect checkpoint schema and `best_healthy_epoch`. A null healthy epoch
   means the run failed the representation-health contract; do not silently use
   `best_loss.pt` as an equivalent result.
3. If a compatible checkpoint survives, use `uv run python
   scripts/run_phase0_pipeline.py evaluate ...` to re-export frozen features,
   rerun every probe, and write a hashed report under current code.
4. If the schema is incompatible or no checkpoint survives, record the failure
   reason and exclude the run from metric tables and direct plot comparisons.
5. Keep only a concise failure summary or external job log when the full failed
   notebook adds no reproducibility value. Do not restore the deleted 90881 or
   91023 notebooks merely to make an unsafe comparison.

## 8. Resume an interrupted run

The train-to-report command intentionally accepts only fresh run directories;
this makes accidental continuation or overwrite impossible. Resume remains a
separate recovery operation and must use an epoch-boundary `latest.pt` with the
same architecture, masks, optimizer behavior, loader configuration, manifest
hash, and frame-inventory fingerprint. Do not edit the production Slurm wrapper
in place or resume job 91108. After an audited resume finishes, evaluate its
declared checkpoint with the pipeline's `evaluate` command into a fresh artifact
directory and report.

## 9. Run exhaustive data certification when needed

Normal training checks the full manifest, every frame name and size, and sampled frame contents. Checkpoints also preserve the random states needed for an exact epoch-boundary resume. Exact continuation assumes that the underlying dataset has not changed.

For byte-level dataset certification, run a separate CPU-only notebook job with:

```bash
CODY_JEPA_RUN_FULL_TRAINING=0 \
CODY_JEPA_RUN_DATA_AUDIT=1 \
CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT=1 \
MPLCONFIGDIR=/tmp/mpl \
uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output single-stream-jepa-exhaustive-audit.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The current dataset contains 321,247 frames. Exhaustive mode opens every image and reads every frame byte, so it can take much longer than the standard preflight. Do not combine exhaustive certification with an H100 training allocation.

## 10. Test `torch.compile` only after eager training works

Keep `CONFIG["compile"] = False` for the first successful run. CUDA compilation adds TorchInductor and Triton as possible failure points without being required for correctness.

If you later want to benchmark compilation, test the runtime first in an interactive H100 session:

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
print("torch.compile CUDA smoke test passed")
PY
```

Do not enable compilation for the production notebook until this test passes and the eager run has produced a valid checkpoint.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Job remains in `PD` | Run `squeue -j "$JOB_ID" -o "%.18i %.2t %.10M %R"`. The final column explains whether the job is waiting for resources, priority, or quota. |
| Job enters `R` but no Slurm log appears | Confirm that `logs/` existed before submission and inspect `scontrol show job "$JOB_ID"`. |
| GPU utilization stays near zero for more than 15 minutes | Inspect the Slurm log, confirm all three run-mode variables, and check whether dataset construction is blocked on HAIC storage. |
| Job receives a non-H100 GPU | Keep `--gres=gpu:h100:1`. The batch script intentionally exits when the device name does not contain `H100`. |
| CUDA reports no compatible kernel image | Reinstall the locked PyTorch packages and confirm that `sm_90` appears in `torch.cuda.get_arch_list()`. |
| Pipeline refuses an existing run directory | Choose a new versioned run directory. Resume is a separate audited recovery operation; never delete or overwrite an existing checkpoint to make the fresh-run workflow proceed. |
| GPU runs out of memory during the BF16 preflight | Reduce the physical `batch_size` and increase `accumulation_steps` by the same factor to preserve the effective batch size. Recheck that the number of microbatches remains divisible by the accumulation count. |
| Executed notebook is missing while the job is running | This is expected. `nbconvert` writes the final executed notebook after execution ends. Use the GPU CSV and checkpoints to monitor an active run. |
| Job stops before `latest.pt` appears | No complete epoch was saved. Start a new run in the same empty directory or choose another fresh output directory. |

## Scientific stop conditions

Stop the run and investigate if any of these conditions occur:

- Loss, gradients, or inputs become non-finite.
- Effective rank trends toward 1.
- Per-feature variance approaches zero.
- The shuffled-context loss gap remains near zero after the initial learning period.
- Validation improves only the clip-weighted metric and not the subject-balanced metric.
- GPU memory usage approaches the H100 capacity.

The training loop raises on non-finite values before updating the online or target encoder with corrupted gradients.

Finally, do not treat this prototype's prediction loss as evidence for the final CoDy-JEPA disentanglement claim. That claim requires stream-specific probes, counterfactual intervention, and transfer evaluation.
