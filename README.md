# CoDy-JEPA

## Learning motion without memorizing who moves

CoDy-JEPA (Counterfactual-Dynamical Joint-Embedding Predictive Architecture) is a research project for learning motion representations from unlabeled video. Its goal is to represent **how an articulated system moves** while limiting how much identity, appearance, or environment information leaks into that motion representation.

The method is intended for people walking, hands gesturing, robot arms reaching, and other systems whose structure changes slowly while their state changes over time.

> **Project status:** this repository contains the CoDy-JEPA research design, a strict Health&Gait silhouette data pipeline, and a tested single-stream masked-JEPA prototype. The prototype validates the training infrastructure and representation-learning signal; it is not yet the final dual-stream, counterfactual CoDy-JEPA model. The runnable pipeline prepares reproducible `[B, T, C, H, W]` batches, validates subject-disjoint data provenance, trains with resumable checkpoints, monitors collapse and context dependence, and exports a placeholder schema for later representation probes.

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
uv sync
```

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
- motion-energy summaries; and
- placeholder `S_attr` and `S_dyn` probe exports.

The placeholder exports contain deterministic clip statistics, not learned CoDy-JEPA representations. They establish the table schema that later model checkpoints should produce.

## Data-pipeline guarantees

`HealthGaitManifestDataset` validates the full manifest before yielding samples. It rejects:

- missing required columns or unsupported split names;
- missing frame directories;
- differences between declared and actual frame counts;
- trials shorter than the requested clip length;
- empty training or validation splits; and
- subjects shared between training and validation.

Each sample is a normalized grayscale tensor shaped `[T, C, H, W]` plus its source metadata. Frames are loaded only when requested, so the full dataset is not held in memory.

## Repository layout

```text
cody-jepa/
├── images/                         # Method and evaluation diagrams
├── notebooks/
│   └── healthgait_manifest_loader.ipynb
├── scripts/
│   └── build_healthgait_manifest.py
├── src/cody_jepa/data/             # Dataset, loaders, and diagnostics
├── tests/                          # Fixture-based data-pipeline tests
├── pyproject.toml                  # Project metadata and dependencies
└── README.md
```

Large datasets, model outputs, and checkpoints are intentionally excluded from version control.

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
