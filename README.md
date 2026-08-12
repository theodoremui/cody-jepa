# CoDy-JEPA

CoDy-JEPA is a research codebase for self-supervised video representation learning and
factorial evaluation. The previous active research direction has been removed from the
documentation so the repository can be used as a clean starting point for the next
direction.

The repository currently preserves code, configurations, tests, and checked-in aggregate
results. It does not define a current study proposal, active tutorial track, or training
claim.

## Latest Checked-In Results

The current public evidence lives in [results](results/README.md). These files are
aggregate, non-identifying outputs from prior development runs:

- Phase 0 diagnostic summary for a single-stream masked JEPA run.
- Phase 1 diagnostic table for 11 development runs across architecture and training
  variants.
- Context-substitution and foreground-pooling diagnostics for the Phase 0 best-loss
  checkpoint.
- Legacy GFC development-split comparisons for selected checkpoints.
- Historical planning records that should be treated as archived context, not as an active
  research direction.

These results are useful for understanding what the current code has produced so far. They
are not final claims and should not be presented as a locked outcome study.

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
- `results/` contains aggregate result artifacts that are safe to track in Git.
- `docs/` is intentionally minimal while the next research direction is being chosen.
- `tests/` contains unit and smoke tests that run without private data.

## Data and Claim Boundaries

Raw recordings, private manifests, participant tables, feature exports, checkpoints, and
participant-level results stay outside Git. Public artifacts should stay aggregate,
non-identifying, and traceable to reproducible scripts or documented analysis steps.

At this point, use the repository as an implementation and result archive. Define the next
scientific question before adding new method documentation or tutorials.
