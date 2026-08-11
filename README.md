# CoDy-JEPA

CoDy-JEPA is a research codebase for studying where useful video diversity comes from in
self-supervised representation learning.

The current baseline is the hierarchical-diversity approach. It keeps model architecture,
training exposure, optimizer settings, evaluation rules, and analysis rules fixed, then
moves a fixed catalog of sampling opportunities between two places in the video hierarchy:
more source sequences, or more phase-separated origins inside fewer sequences. The method
is the contribution. The current repository instance applies it to private gait video data,
but the design is meant to carry forward to future JEPA studies that need the same clean
separation between training support, evaluation support, and claim boundaries.

The main evaluator family is Grounded Factorial Completion, or GFC. It measures whether a
frozen representation supports recombining factor information under a locked factorial
retrieval protocol. The detailed evaluator contract lives in the method documentation.

## Current Status

The repository contains the code and documentation needed to convert and audit private
training shards, define the active study gates, and run the reusable evaluator pieces. The
active 28-row registry finalizer, phase-aware loader, and production launcher are still
implementation gates before the documented study can train end to end. The repository does
not contain private datasets, trained study checkpoints, participant-level results, or a
locked outcome result.

Checked-in development diagnostic tables are not active study outcomes. Treat them as
development evidence only.

## Setup

You need Python 3.10 or newer and `uv`.

```bash
git clone https://github.com/theodoremui/cody-jepa.git
cd cody-jepa
uv sync --frozen
```

Run the test suite without private data:

```bash
uv run python -m unittest discover -s tests -v
```

## Repository Map

- `src/cody_jepa/` contains the training, data preparation, evaluation, and command-line
  packages.
- `configs/` contains training and evaluation configuration files.
- `scripts/` contains compatibility wrappers for installed command-line tools.
- `slurm/` contains HAIC launchers for data preparation and training jobs.
- `docs/` explains the active research method, execution path, and data preparation
  workflow.
- `tutorials/` teaches the mathematical, machine-learning, statistical, and engineering
  background needed to understand the study.
- `tests/` contains unit and smoke tests that run without private data.

## Research Workflow

The active workflow has five parts.

1. Prepare a private video corpus into validated, seekable training records.
2. Freeze a phase catalog and allocation registry for the hierarchical-diversity study.
3. Train paired model blocks at one fixed exposure tier.
4. Export frozen features and score them with the GFC evaluator and matched controls.
5. Report aggregate contrasts, uncertainty, diagnostics, and claim boundaries.

In this repository, GaitLU is the current pretraining corpus and Health&Gait is the current
held-out factorial evaluator. Future JEPA studies can reuse the same separation of
concerns: one source trains the encoder, another held-out factorial setting evaluates what
the frozen representation supports.

## Documentation

Start with the documentation map in [docs/README.md](docs/README.md). The active method is
under [docs/hierarchical-diversity](docs/hierarchical-diversity/README.md), and the shared
GaitLU preparation runbook is [docs/gaitlu_training.md](docs/gaitlu_training.md).

Use [tutorials/README.md](tutorials/README.md) if the tensor, JEPA, GFC, inference, or
reproducibility vocabulary is unfamiliar.

## Data and Claim Boundaries

Raw recordings, private manifests, participant tables, feature exports, checkpoints, and
participant-level results stay outside Git. Public artifacts must be aggregate and
non-identifying.

The active study can support narrow claims about how the hierarchical organization of
training support affects frozen representations under the declared protocol. It does not
establish a universal video-scaling law, identify people, validate downstream use, or prove
unsupervised disentanglement.
