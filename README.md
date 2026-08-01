# CoDy-JEPA

CoDy-JEPA is a research codebase for asking whether video representations preserve factors that can be recombined correctly. Its main evaluation family is **Grounded Factorial Completion (GFC)**: two non-target recordings supply complementary representation blocks, and their mixed query must retrieve the real recording with the requested factor combination.

The maintained Health&Gait implementation and checked-in preliminary results use the legacy GFC protocol: a complete participant has eight observed cells and contributes 24 queries, both donors are removed from each gallery, and learned features are compared with duration, frame-count, image-plane-motion, and foreground-area shortcuts. The prospectively specified GFC-v2 study instead keeps the full eight-cell gallery and uses 16 session-safe queries; it has not yet produced outcome results.

The repository contains model-training code, aggregate Phase 0/1 diagnostics, and development-split GFC results for three selected checkpoints. It does **not** yet contain a confirmation-split GFC result, an external gait-measurement result, or a cross-dataset result.

## Setup

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required. All Python commands and dependency changes use uv.

```bash
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
uv sync --frozen
```

Run the test suite without private data:

```bash
uv run python -m unittest discover -s tests -v
```

## Repository map

The maintained implementation lives under `src/cody_jepa/`:

- `models/`: encoder, predictor, attention blocks, positional embeddings, and model factories.
- `masks/`: context and target mask policies.
- `training/`: optimization, losses, diagnostics, checkpoints, runtime setup, and the training engine.
- `data/`: the Health&Gait manifest schema, frame discovery, datasets, loaders, and diagnostics.
- `evaluation/features.py`: frozen-feature export and the validated feature-table boundary.
- `evaluation/probes/`: identity and gait-system protocols.
- `evaluation/gfc/`: GFC scoring, normalization, inference, configuration, and orchestration.
- `cli/`: installed command-line entry points. Files in `scripts/` are compatibility wrappers.

Configs are grouped by purpose under `configs/train/` and `configs/eval/`. The top-level `single_stream_jepa.py`, `probes.py`, and `gfc*.py` modules preserve older imports; new code should use the packages above.

## Research workflow

Health&Gait is not redistributed. After obtaining access from the dataset provider, place the extracted release under `data/healthgait/raw/Health_Gait/`; `data/` remains excluded from Git. Build the subject-disjoint manifest:

```bash
uv run python scripts/build_healthgait_manifest.py --fps 30
```

The acquisition is nominally 30 Hz. Verify the actual frame rate for the local release
and pass it explicitly because duration is one of the shortcut controls.

Train the single-stream baseline, export recording features, and evaluate the diagnostic probes:

```bash
uv run python scripts/train.py \
  --config configs/train/healthgait_baseline.json \
  --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
  --output-dir outputs/training-baseline

uv run python scripts/export_features.py \
  --checkpoint outputs/training-baseline/best_loss.pt \
  --manifest data/healthgait/manifests/silhouette_subject_split_seed0.csv \
  --output outputs/training-baseline/features.csv

uv run python scripts/eval_probes.py \
  --features outputs/training-baseline/features.csv \
  --output-dir outputs/training-baseline/probes
```

Run GFC from the exported window-level feature table. The runner averages its three rows per recording before fitting or scoring; required factor, split, learned-feature, and shortcut columns are described in [the method](docs/method.md).

If you have a legacy deterministic NPZ export, upgrade it to the validated GFC feature
boundary without changing its float32 feature values:

```bash
uv run cody-jepa-prepare-gfc-features \
  --legacy-features outputs/training-baseline/legacy-features.npz \
  --manifest data/healthgait/manifests/gfc_manifest.csv \
  --output outputs/training-baseline/features.npz
```

The upgrade requires exactly three distinct window rows per recording and records only
input basenames and SHA-256 digests in its provenance metadata.

```bash
uv run python scripts/run_gfc.py \
  --features outputs/training-baseline/features.csv \
  --config configs/eval/gfc_healthgait.json \
  --model-label training-baseline-best-loss \
  --split development \
  --output-dir outputs/gfc-development \
  --aggregate-output-dir results/gfc-development
```

The command above runs the primary `raw_retain_all` analysis. Run the two declared sensitivities into separate directories:

```bash
uv run python scripts/run_gfc.py \
  --features outputs/training-baseline/features.csv \
  --config configs/eval/gfc_healthgait.json \
  --model-label training-baseline-best-loss \
  --split development \
  --normalization raw_effective_rank \
  --output-dir outputs/gfc-development-raw-er \
  --aggregate-output-dir results/gfc-development-raw-er

uv run python scripts/run_gfc.py \
  --features outputs/training-baseline/features.csv \
  --config configs/eval/gfc_healthgait.json \
  --model-label training-baseline-best-loss \
  --split development \
  --normalization pca_effective_rank \
  --output-dir outputs/gfc-development-pca-er \
  --aggregate-output-dir results/gfc-development-pca-er
```

Regenerate active aggregate tables and figures:

```bash
uv run python scripts/make_paper_results.py \
  --results-dir results \
  --output-dir results/generated
```

Detailed participant and optional query outputs stay under ignored `outputs/`; only aggregate `summary.json` and `summary.csv` files are copied into `results/` for paper generation.

## Documentation

- [Method](docs/method.md): GFC construction, controls, normalization, scoring, and inference.
- [Data](docs/data.md): Health&Gait access, layout, splits, privacy, and preprocessing boundaries.
- [Results](docs/results.md): current aggregate evidence and the claims it does and does not support.
- [Paper draft](docs/paper.md): concise manuscript skeleton for the revised ICLR 2027 study.
- [Research proposal](docs/proposal.md): accessible motivation, study design, preliminary evidence, and proposed confirmation work.
- [Result files](results/README.md): compact inputs to the paper-result generator.

## Data and claim boundaries

Keep raw frames, archives, participant tables, checkpoints, and participant-level feature exports outside Git. Publish only aggregate, non-identifying results. In the checked-in legacy analysis, all fitted transformations use training participants only, windows are aggregated to recordings before scoring, donors are excluded from galleries, and participants receive equal weight in cohort inference. The revised full-gallery protocol is documented separately and remains prospective.

Current results are single-seed diagnostics on one Health&Gait split. They motivate GFC because prediction loss, representation breadth, context sensitivity, and probe scores prefer different checkpoints. They do not establish semantic factorization, causal gait structure, clinical validity, or transfer beyond Health&Gait.
