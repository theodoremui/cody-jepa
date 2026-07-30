# Aggregate research results

This directory contains compact, non-identifying inputs to the paper-result generator.

- `phase0_summary.json` records the baseline split, feature settings, and aggregate checkpoint/probe metrics.
- `phase1_summary.csv` compares the Phase 1 diagnostic sweep at each selected checkpoint.
- `context_diagnosis.json` records the aggregate context-substitution, foreground, and representation-rank diagnosis.
- `gfc-*` directories contain only the aggregate `summary.json` and `summary.csv` copied by `scripts/run_gfc.py --aggregate-output-dir`.
- `generated/phase0_table.csv` and `generated/phase1_table.csv` are the paper-ready diagnostic tables.
- `generated/phase1_diagnostics.png` and `generated/context_diagnosis.png` are the active diagnostic figures.
- `generated/gfc_table.csv` and `generated/gfc_comparison.png` are added after a GFC summary exists.

Regenerate the active result presentation with:

```bash
uv run python scripts/make_paper_results.py \
  --results-dir results \
  --output-dir results/generated
```

The checked-in Phase 0/1 values are prior aggregate diagnostics, not empirical GFC results. Participant-level rows, queries, and raw Health&Gait data do not belong in this directory; the documented runner writes them below ignored `outputs/`.
