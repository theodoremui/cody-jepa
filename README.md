# CoDy-JEPA

## Learning motion without memorizing who moves

CoDy-JEPA (Counterfactual-Dynamical Joint-Embedding Predictive Architecture) is a research project for learning motion representations from unlabeled video. Its goal is to represent **how an articulated system moves** while limiting how much identity, appearance, or environment information leaks into that motion representation.

The method is intended for people walking, hands gesturing, robot arms reaching, and other systems whose structure changes slowly while their state changes over time.

> **Project status:** this repository contains the CoDy-JEPA research design, a strict Health&Gait silhouette data pipeline, and a tested single-stream masked-JEPA prototype. The prototype validates the training infrastructure and representation-learning signal; it is not yet the final dual-stream, counterfactual CoDy-JEPA model. The runnable pipeline prepares reproducible `[B, T, C, H, W]` batches, validates subject-disjoint data provenance, trains with resumable checkpoints, monitors collapse and context dependence, and evaluates frozen learned representations with linear probes.

## Navigate this repository

Use this table as the shortest path to the part of the project you need:

| Goal | Start here |
| --- | --- |
| Understand the research question and proposed dual-stream method | [Why CoDy-JEPA?](#why-cody-jepa), [Method](#method), and [Evaluation](#evaluation) |
| Install the locked environment and run tests | [Quick start](#quick-start) |
| Download Health&Gait and create the subject-disjoint manifest | [Prepare Health&Gait](#prepare-healthgait) and [`scripts/build_healthgait_manifest.py`](scripts/build_healthgait_manifest.py) |
| Inspect batches, sampling, and motion diagnostics | [Explore the data pipeline](#explore-the-data-pipeline) and [`notebooks/healthgait_manifest_loader.ipynb`](notebooks/healthgait_manifest_loader.ipynb) |
| Run the safe single-stream smoke test or configure training | [Run the single-stream prototype](#run-the-single-stream-prototype) and [`notebooks/single-stream-jepa.ipynb`](notebooks/single-stream-jepa.ipynb) |
| Train on Stanford HAIC with an H100 | [`tutorials/train-cody-jepa-on-haic.md`](tutorials/train-cody-jepa-on-haic.md) and [`slurm/train-single-stream-jepa.sbatch`](slurm/train-single-stream-jepa.sbatch) |
| Export a checkpoint and measure learned information | [Single-stream frozen-feature probes](#single-stream-frozen-feature-probes) |
| Find a model, loader, script, test, or result | [Code and repository guide](#code-and-repository-guide) |
| Review the existing split-leakage evidence | [`haic-results/split-sanity-check.md`](haic-results/split-sanity-check.md) |

## Why CoDy-JEPA?

Video models can solve the wrong problem for the right score. If one person always appears in one room and walks at one speed, a model may associate the room or identity with speed instead of learning gait phase and cadence. This is **shortcut learning**: the model exploits a correlation that works on a familiar split but fails when the person, camera, or environment changes [6].

CoDy-JEPA addresses this failure mode with two latent summaries:

- `S_attr`, the **attribute stream**, represents comparatively stable information such as body proportions, clothing, viewpoint, background, robot morphology, or tool shape.
- `S_dyn`, the **dynamics stream**, represents changing state such as pose transitions, gait phase, cadence, velocity, contact state, or gesture phase.

This split is an inductive bias, not a claim that structure and motion are naturally independent. Body shape can constrain gait, for example. Unsupervised data alone cannot guarantee a uniquely disentangled representation [10], so the separation must be tested with leakage probes and transfer experiments. The counterfactual intervention is motivated by the broader goal of learning representations organized around causal factors rather than incidental correlations [11].

![Stable and dynamic factorization](images/05-v3-factorization.svg)

## Method

CoDy-JEPA combines temporal token routing, counterfactual token swapping, latent future prediction, and redundancy reduction.

### 1. Encode and route video tokens

For a clip $x_{1:T}$, a video encoder $f_\theta$ produces spatiotemporal tokens

$$
z_{t,p}=f_\theta(x_{1:T})_{t,p},
$$

where $t$ is time and $p$ identifies a patch, joint, body part, or learned token slot. The temporal variation of each slot is

$$
\bar z_p=\frac{1}{T}\sum_{t=1}^{T}z_{t,p}, \qquad
v_p=\frac{1}{T}\sum_{t=1}^{T}\lVert z_{t,p}-\bar z_p\rVert_2^2.
$$

Low-variation tokens are routed toward the attribute encoder $g_a$. High-variation tokens are routed toward the dynamics encoder $g_d$:

$$
S_{\text{attr}} = g_a(\left\lbrace z_{t,p}: v_p \le \tau_a \right\rbrace), \qquad
S_{\text{dyn}}(1:t) = g_d(\left\lbrace z_{s,p}: s \le t,\ v_p \ge \tau_d \right\rbrace).
$$

In a gait clip, torso shape may enter the stable stream while alternating foot and knee positions enter the dynamic stream. Temporal variation is only an initial routing signal: a static room can still be a nuisance, and slow movement can still be dynamics. The probes described below determine whether the learned split is useful.

### 2. Break identity-motion shortcuts

The core intervention is **Cross-Instance Token-Swapping Intervention (CI-TSI)**. Given clips A and B from the same broad domain, CoDy-JEPA combines A's stable context with B's motion history:

$$
C_{A\leftarrow B}=[S_{\mathrm{attr}}(A),S_{\mathrm{dyn}}(B,1{:}t)].
$$

The predictor must forecast B's future dynamics:

$$
\hat S_{\mathrm{dyn}}(B,t+k)=q_\psi(C_{A\leftarrow B},k).
$$

If A supplies the body and B supplies the stride rhythm, the association “Person A usually walks slowly” no longer solves the task. The predictor must retain motion from B and use A only as context.

![Counterfactual prediction objective](images/06-v3-objective.svg)

### 3. Predict representations, not pixels

As in JEPA-style learning [1, 2], the target is produced by a slowly updated target encoder. The predictor matches a normalized latent future rather than reconstructing RGB frames:

$$
\mathcal{L}_{\text{pred}} =
\left\lVert \mathrm{norm}(\hat{S}_{\text{dyn}}(B,t+k)) -
\mathrm{sg}\left(\mathrm{norm}(\bar{S}_{\text{dyn}}(B,t+k))\right)\right\rVert_2^2 .
$$

Here, `sg` means stop-gradient. Latent prediction lets the objective focus on predictable semantic and physical structure instead of every pixel. This differs from MAE and VideoMAE, whose training targets are reconstructed image or video content [4, 5].

### 4. Separate the streams without collapse

Prediction alone does not prevent both streams from storing the same information. CoDy-JEPA therefore penalizes their statistical dependence with the Hilbert-Schmidt Independence Criterion (HSIC) [7]. For a batch of $n$ clips, let $K$ and $L$ be kernel Gram matrices for `S_attr` and `S_dyn`, and let $H=I_n-\frac{1}{n}\mathbf1\mathbf1^\top$:

$$
\mathrm{HSIC}(S_{\text{attr}},S_{\text{dyn}})=
\frac{1}{(n-1)^2}\mathrm{tr}(KHLH).
$$

Minimizing HSIC discourages the streams from organizing examples in the same way. VICReg-style variance and covariance terms keep dimensions active and reduce within-stream redundancy [8, 9]. The complete objective is

$$
\begin{gathered}
\mathcal{L} = \mathcal{L}_{\text{pred}} \\
{}+ \lambda_h \mathrm{HSIC}(S_{\text{attr}},S_{\text{dyn}}) \\
{}+ \lambda_v \mathcal{L}_{\text{var}} \\
{}+ \lambda_c \mathcal{L}_{\text{cov}} .
\end{gathered}
$$

Each term has a distinct job: predict future motion, reduce cross-stream overlap, prevent constant representations, and avoid redundant latent dimensions.

## Evaluation

Training loss alone cannot show that the streams learned the intended information. After pretraining, freeze both encoders and fit linear or shallow probes:

| Probe target | Desired `S_attr` result | Desired `S_dyn` result |
| --- | --- | --- |
| Subject identity or body shape | High | Low |
| Robot morphology or tool shape | High | Low |
| Gait phase, cadence, or speed | Low | High |
| Action or contact state | Low | High |
| Camera, room, or dataset source | Measure and control | Low |

The central measurement is the **leakage gap**: `S_dyn` should predict motion variables well and identity variables poorly. Transfer probes should be trained on one set of subjects, views, or robot embodiments and evaluated on held-out ones.

### Single-stream frozen-feature probes

The trained single-stream baseline has a two-stage probe runner. Feature export
restores only the EMA target encoder, freezes it, switches it to evaluation mode,
and runs under `torch.inference_mode()`. Each deterministic clip window becomes
one row containing its manifest metadata and the mean-pooled, pre-final-LayerNorm
target tokens:

```bash
uv run python scripts/export_single_stream_features.py \
  --checkpoint outputs/single-stream-jepa/best_loss.pt \
  --output outputs/single-stream-jepa/frozen_features.npz \
  --device cuda

uv run python scripts/eval_probes.py \
  --features outputs/single-stream-jepa/frozen_features.npz \
  --output-dir outputs/single-stream-jepa/probes
```

The evaluator reports three protocols in both JSON and CSV: a sequence-disjoint
closed-set identity classifier over training subjects, nearest-centroid identity
retrieval over separately enrolled validation subjects, and a `gait_system`
linear classifier trained on training subjects and evaluated on subject-disjoint
validation subjects. Balanced accuracy is the primary gait metric. The exporter
accepts `.csv` or compressed `.npz`; both carry a JSON provenance sidecar with
the checkpoint hash and exact feature formula.

The proposed ablations are:

1. A single-stream V-JEPA-style predictor and the motion-content separation studied by MC-JEPA [3].
2. A dual-stream model without CI-TSI.
3. CoDy-JEPA without HSIC.
4. CoDy-JEPA without variance and covariance safeguards.

A strong result combines competitive future-dynamics prediction, less wrong-stream leakage, and better low-label transfer. A result in which swapping reduces identity leakage but harms motion prediction is also informative because it exposes the separation-performance tradeoff directly.

![Evaluation matrix](images/07-v3-evaluation.svg)

## Quick start

### Requirements

- Git
- [uv](https://docs.astral.sh/uv/)
- Python 3.10 or newer; `uv` will create and manage the project environment

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
uv sync --frozen
```

This repository uses `uv` exclusively for Python environments and dependencies.
Run project tools through `uv run`; change dependencies with `uv add` or
`uv remove`, regenerate the lock with `uv lock`, and verify existing environments
with `uv sync --frozen`. Do not add pip, Conda, Poetry, or ad-hoc system-Python
installation steps to project documentation or notebooks.

Run the test suite:

```bash
uv run python -m unittest discover -s tests -v
```

The tests use generated fixtures, so they do not require the Health&Gait dataset.

### Prepare Health&Gait

The dataset is not distributed with this repository. Download the [Health&Gait v1.0 dataset from Zenodo](https://doi.org/10.5281/zenodo.14039922). The full dataset is a 26.8 GB multipart archive: `Health_Gait.z01` through `Health_Gait.z25` plus `Health_Gait.zip`. Keep every part in the same directory and extract from `Health_Gait.zip` into `data/healthgait/raw/`.

After extraction, silhouette trials should follow this structure:

```text
data/healthgait/raw/Health_Gait/
└── silhouette/
    └── PA000/
        ├── FGS/
        │   └── WJ_1_YOLOV8/
        │       ├── 001.jpg
        │       └── ...
        └── UGS/
            └── ...
```

`FGS` and `UGS` denote fast and usual gait speed. Once the frames are in place, build a deterministic, subject-disjoint train/validation manifest:

```bash
uv run python scripts/build_healthgait_manifest.py
```

The script writes:

- `data/healthgait/manifests/silhouette_subject_split_seed0.csv`
- metadata summaries under `data/healthgait/diagnostics/`

The manifest records one trial per row. It contains the subject, modality, gait system, trial, frame directory, frame count, and split. Keeping the split in a manifest makes the experiment auditable and prevents the same subject from appearing in both training and validation.

### Load a batch

```python
from pathlib import Path

from cody_jepa.data import HealthGaitLoaderConfig, build_healthgait_loaders_from_config

root = Path.cwd()
config = HealthGaitLoaderConfig(
    manifest_csv=root / "data/healthgait/manifests/silhouette_subject_split_seed0.csv",
    repo_root=root,
    clip_length=16,
    image_size=(224, 224),
    batch_size=4,
    seed=0,
)

train_loader, val_loader = build_healthgait_loaders_from_config(config)
batch = next(iter(train_loader))

print(batch["video"].shape)  # [B, T, C, H, W] = [4, 16, 1, 224, 224]
print(batch["video"].min().item(), batch["video"].max().item())  # 0.0 to 1.0
```

The default policy selects deterministic pseudo-random windows for training and center windows for validation. A training loop should call `train_loader.dataset.set_epoch(epoch)` at the start of each epoch to change training windows reproducibly. Subject and trial metadata are provided for diagnostics; they are not labels for the self-supervised objective.

### Explore the data pipeline

Launch the notebook:

```bash
uv run jupyter lab notebooks/healthgait_manifest_loader.ipynb
```

The notebook walks through:

- manifest validation and metadata summaries;
- train and validation datasets;
- `[B, T, C, H, W]` DataLoader batches;
- deterministic temporal-window sampling;
- clip contact sheets and frame-difference diagnostics;
- motion-energy summaries.

Learned representations are exported and evaluated with the frozen-feature probe workflow described above, after a checkpoint has been trained.

### Run the single-stream prototype

[`notebooks/single-stream-jepa.ipynb`](notebooks/single-stream-jepa.ipynb) is the experiment controller. The reusable model, masking, evaluation, and checkpoint code lives in [`src/cody_jepa/single_stream_jepa.py`](src/cody_jepa/single_stream_jepa.py); the notebook configures that code rather than duplicating it.

Open the notebook through the locked environment:

```bash
uv run jupyter lab notebooks/single-stream-jepa.ipynb
```

With no environment flags, `Run All` validates the configured Health&Gait data and executes a one-step synthetic CPU smoke test. A real training run requires a CUDA worker and is deliberately opt-in. These are the notebook's runtime controls:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `CODY_JEPA_RUN_FULL_TRAINING` | `0` | Set to `1` to run the full CUDA training path after the real-batch CUDA preflight. |
| `CODY_JEPA_RUN_DATA_AUDIT` | `1` for preflight, `0` for full training | Runs the all-sequence clip-quality audit separately from expensive GPU training. |
| `CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT` | `0` | Verifies and hashes every frame. Use as a separate CPU/I/O certification job. |
| `CODY_JEPA_OUTPUT_DIR` | `outputs/single-stream-jepa` | Selects the checkpoint directory. A new run refuses to overwrite an existing `latest.pt`. |
| `CODY_JEPA_RESUME_CHECKPOINT` | unset | Resumes from an explicit epoch-boundary checkpoint after validating its model and data contract. |

The baseline uses multiblock tube masking, an online context encoder, an exponential-moving-average target encoder, and a predictor. Its training path supports BF16, gradient accumulation, optional `torch.compile`, target batch standardization, and VICReg-style variance/covariance safeguards. Validation reports prediction metrics, feature variance and effective rank, plus a seeded wrong-subject context-shuffle gap. It writes `latest.pt`, `best_loss.pt`, and, only when the representation-health checks pass, `best_healthy.pt`.

For the production H100 settings, submission commands, monitoring, resume procedure, output inspection, and troubleshooting, follow the [HAIC training guide](tutorials/train-cody-jepa-on-haic.md). The checked-in [Slurm script](slurm/train-single-stream-jepa.sbatch) is the source of truth for scheduler resources and environment flags.

## Data-pipeline guarantees

`HealthGaitManifestDataset` validates the full manifest before yielding samples. It rejects:

- missing required columns or unsupported split names;
- missing frame directories;
- differences between declared and actual frame counts;
- trials shorter than the requested clip length;
- empty training or validation splits; and
- subjects shared between training and validation.

Each sample is a normalized grayscale tensor shaped `[T, C, H, W]` plus its source metadata. Frames are loaded only when requested, so the full dataset is not held in memory.

## Code and repository guide

### End-to-end code flow

```text
Health&Gait frames
    │
    ├─ scripts/build_healthgait_manifest.py
    │      └─ subject-disjoint CSV + metadata summaries
    │
    ├─ cody_jepa.data
    │      └─ validated datasets → deterministic DataLoaders → diagnostics
    │
    ├─ notebooks/single-stream-jepa.ipynb
    │      └─ cody_jepa.single_stream_jepa → training → checkpoints
    │
    ├─ scripts/export_single_stream_features.py
    │      └─ frozen EMA-target features + provenance sidecar
    │
    └─ scripts/eval_probes.py
           └─ identity and gait-system metrics in JSON + CSV
```

The notebooks are entry points for exploration and experiment orchestration. The reusable and tested implementation is under `src/cody_jepa/`; command-line workflows live under `scripts/`.

### Tracked repository layout

```text
cody-jepa/
├── .gitignore
├── .python-version                  # Local Python version hint
├── README.md                        # Research overview and repository entry point
├── pyproject.toml                   # Package metadata and locked dependency policy
├── uv.lock                          # Exact cross-platform dependency resolution
├── images/                          # Seven research, method, objective, and evaluation SVGs
├── notebooks/
│   ├── healthgait_manifest_loader.ipynb  # Data-contract and diagnostics walkthrough
│   └── single-stream-jepa.ipynb          # Safe smoke test and full-training controller
├── scripts/
│   ├── build_healthgait_manifest.py      # Scan frames and write the subject split
│   ├── export_single_stream_features.py  # Checkpoint → deterministic feature table
│   └── eval_probes.py                    # Feature table → linear-probe reports
├── slurm/
│   └── train-single-stream-jepa.sbatch   # HAIC H100 batch job
├── src/cody_jepa/
│   ├── __init__.py
│   ├── single_stream_jepa.py             # Masking, ViT, predictor, training, eval, resume
│   ├── probes.py                         # Feature I/O, provenance, and probe protocols
│   └── data/
│       ├── __init__.py                   # Public data-pipeline exports
│       ├── dataset.py                    # Strict manifest dataset and frame loading
│       ├── healthgait.py                 # Loader config, builders, and quality audit
│       └── healthgait_diagnostics.py     # Summaries, plots, and motion diagnostics
├── tests/
│   ├── test_healthgait_dataset.py        # Sampling, manifests, provenance, augmentation
│   ├── test_healthgait_diagnostics.py    # Summary and motion-diagnostic artifacts
│   ├── test_probes.py                    # Frozen export, formats, and probe protocols
│   ├── test_single_stream_jepa.py        # Model, masks, training, health, checkpoints
│   ├── test_single_stream_notebook.py    # Notebook safety and orchestration invariants
│   └── test_uv_policy.py                 # Locked-environment policy checks
├── tutorials/
│   └── train-cody-jepa-on-haic.md        # Full H100 operator runbook
└── haic-results/
    ├── single-stream-jepa-90881.executed.ipynb  # Archived executed experiment
    ├── split-sanity-check.md                    # Split leakage investigation
    └── split_nearest_pairs_top12.png            # Nearest-pair evidence image
```

### Source modules and change map

| Area | Read or modify | Verify with |
| --- | --- | --- |
| Manifest rules, frame discovery, temporal windows, augmentation, or sample metadata | `src/cody_jepa/data/dataset.py` and `src/cody_jepa/data/healthgait.py` | `tests/test_healthgait_dataset.py` |
| Dataset summaries, contact sheets, difference maps, or motion-energy checks | `src/cody_jepa/data/healthgait_diagnostics.py` | `tests/test_healthgait_diagnostics.py` |
| Mask generation, video transformer, predictor, schedules, training, validation, or checkpoint resume | `src/cody_jepa/single_stream_jepa.py` | `tests/test_single_stream_jepa.py` |
| Frozen feature schema, checkpoint restoration, identity protocols, or gait-system probe | `src/cody_jepa/probes.py` | `tests/test_probes.py` |
| Experiment configuration and the boundary between CPU audit and CUDA training | `notebooks/single-stream-jepa.ipynb` | `tests/test_single_stream_notebook.py` |
| Dependency or environment policy | `pyproject.toml`, `uv.lock`, and notebook shell commands | `tests/test_uv_policy.py` |
| HAIC resource requests and job lifecycle | `slurm/train-single-stream-jepa.sbatch` | `bash -n slurm/train-single-stream-jepa.sbatch` and the HAIC tutorial |

### Data, outputs, and archived evidence

The tracked repository is intentionally small. A fresh clone does not contain Health&Gait frames or trained checkpoints.

| Path | Lifecycle |
| --- | --- |
| `data/healthgait/raw/` | Local dataset extracted from Zenodo; ignored by Git. |
| `data/healthgait/manifests/` and `data/healthgait/diagnostics/` | Reproducible local products of the manifest and diagnostic workflows; the whole `data/` tree is ignored. |
| `outputs/`, `checkpoints/`, `runs/`, `*.pt`, `*.pth`, `*.ckpt` | Local training and evaluation products; ignored by Git. |
| `logs/` and `notebook-runs/` | Created by the Slurm workflow for scheduler/GPU logs and executed notebooks. Treat them as run artifacts and review them before committing. |
| `haic-results/` | Curated, tracked evidence from completed HAIC experiments. It is for inspection, not imported by training code or tests. |

When adding behavior, keep reusable logic in `src/cody_jepa/`, keep notebooks thin, add the corresponding focused test, and run the full suite before submitting changes:

```bash
MPLCONFIGDIR=/tmp/mpl uv run python -m unittest discover -s tests -v
```

## References

[1] Mahmoud Assran et al. [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243). 2023.

[2] Adrien Bardes et al. [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). 2024.

[3] Adrien Bardes, Jean Ponce, and Yann LeCun. [MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698). 2023.

[4] Kaiming He et al. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). 2021.

[5] Zhan Tong et al. [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602). 2022.

[6] Robert Geirhos et al. [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780). 2020.

[7] Arthur Gretton et al. [Kernel Methods for Measuring Independence](https://www.jmlr.org/papers/v6/gretton05a.html). 2005.

[8] Adrien Bardes, Jean Ponce, and Yann LeCun. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). 2021.

[9] Jure Zbontar et al. [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230). 2021.

[10] Francesco Locatello et al. [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359). 2018.

[11] Bernhard Schölkopf et al. [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107). 2021.
