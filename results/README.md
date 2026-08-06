# Aggregate research results

This directory contains compact, non-identifying inputs to the paper-result generator.

- `phase0_summary.json` records the baseline split, feature settings, and aggregate checkpoint/probe metrics.
- `phase1_summary.csv` compares the Phase 1 diagnostic sweep at each selected checkpoint.
- `context_diagnosis.json` records the aggregate context-substitution, foreground, and representation-rank diagnosis.
- `checkpoint_histories.csv` contains one flattened row per recorded training epoch for the maintained Phase 0 and Phase 1 runs.
- `checkpoint_histories.json` records the history schema, CSV hash, source-checkpoint hashes, run configurations, mask definitions, data contracts, and checkpoint-selection metadata.
- `gfc-*` directories contain historical legacy aggregate `summary.json` and `summary.csv` files. Private GFC-v2 run outputs remain under ignored `outputs/`; only the three aggregate study files may enter a later result-only commit.
- `generated/phase0_table.csv` and `generated/phase1_table.csv` are the paper-ready diagnostic tables.
- `generated/phase1_diagnostics.png` and `generated/context_diagnosis.png` are the active diagnostic figures.
- `generated/legacy_gfc_table.csv` and `generated/legacy_gfc_comparison.png` are added after a protocol-tagged legacy GFC summary exists.

Regenerate the active result presentation with:

```bash
uv run python scripts/make_paper_results.py \
  --results-dir results \
  --output-dir results/generated
```

Do not pass GFC-v2 files to that historical generator. After unblinding, the separate
study renderer accepts only the aggregate summary and tables produced by the frozen
study summarizer:

```bash
uv run cody-jepa-make-gfc-study-results \
  --aggregate-dir outputs/gfc-v2-study/aggregate \
  --output-dir outputs/gfc-v2-study/paper
```

Its only inputs are `outcome_summary.json`, `run_table.csv`, and
`ladder_contrasts.csv`; its outputs are two paper CSVs plus PDF and PNG versions of the
five-ladder scaling figure. Participant rows, identity-bearing keys, feature paths, and
checkpoint paths are rejected.

When the ignored local checkpoints are available, refresh the portable notebook histories with:

```bash
uv run python scripts/export_checkpoint_histories.py
```

The notebooks read only tracked files below `results/`; they do not require model checkpoints, feature arrays, or participant-level probe exports.

Execute clean copies of every notebook without changing tracked generated figures:

```bash
CODY_JEPA_REPRO_OUTPUT_DIR=outputs/notebook-reproduction/artifacts \
  uv run jupyter nbconvert --execute --to notebook \
  --output-dir outputs/notebook-reproduction notebooks/*.ipynb
```

The checked-in Phase 0/1 values are prior aggregate diagnostics, not empirical GFC results. Participant-level rows, queries, and raw Health&Gait data do not belong in this directory; the documented runner writes them below ignored `outputs/`.
