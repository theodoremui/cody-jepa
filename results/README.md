# Results

This folder holds compact, non-identifying outputs that are safe to track in Git. Treat it
as the current public record of prior development evidence, not as a final outcome package
for an active research direction.

## Current Result Artifacts

- `phase0_summary.json` records the Phase 0 single-stream masked JEPA diagnostic run. The
  best-loss checkpoint is epoch 80 with validation loss 0.3874, effective-rank ratio
  0.0272, held-out retrieval accuracy 0.0245, and gait balanced accuracy 0.9257.
- `phase1_summary.csv` records 11 Phase 1 development runs. The table covers Stage A and
  Stage B variants, selected checkpoints, validation loss, effective-rank ratio,
  context-shuffle gap, identity retrieval, and gait balanced accuracy.
- `context_diagnosis.json` records context-substitution, foreground-pooling, and
  representation-rank diagnostics for the Phase 0 best-loss checkpoint.
- `checkpoint_histories.csv` and `checkpoint_histories.json` collect training curves and
  checkpoint metadata for the development runs.
- `generated/phase0_table.csv`, `generated/phase1_table.csv`, and the generated figures
  provide compact tables and plots derived from the summary files.
- `generated/legacy_gfc_table.csv` and the `gfc-*` folders contain legacy
  development-split GFC comparisons for selected checkpoints.
- `generated/writeup-*` folders contain historical writeup tables and figures. Some files
  describe planned studies or readiness checks. Keep them as archive context only unless a
  future direction explicitly reuses them.

## How To Read These Results

The checked-in outputs are development evidence. They can help diagnose representation
rank, context sensitivity, retrieval behavior, and gait-label probe behavior in prior
runs. They do not establish a final scientific claim.

Do not treat these artifacts as:

- private-data releases;
- participant-level outputs;
- locked outcome results;
- evidence for a currently active research proposal;
- proof that a future research direction should use the same method.

## Adding New Results

Add only aggregate, non-identifying files that are small enough for Git and traceable to
the scripts or commands that produced them. Keep raw data, checkpoints, feature arrays,
query-level outputs, participant tables, and local paths outside this folder.

When adding a new result, name the protocol, describe the split, and make clear whether the
artifact is exploratory, diagnostic, or a locked outcome.
