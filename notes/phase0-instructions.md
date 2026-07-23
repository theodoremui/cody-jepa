**Role** You are an expert software engineer and AI research scientist.

**Task** You are to carefully and systematically implement Phase 0 of `tutorials/cody-jepa-plan.md`.

Ultrathink on how to thoroughly and systematically make the following changes in order to convert the current work in cody-jepa from promising infrastructure into a trustworthy, reproducible experimental base:

- Treat job 91108 and outputs/jepa-v4/ as read-only baseline evidence. Do not restore retired jobs or recreate outputs/jepa-v3/.
- Re-export features from outputs/jepa-v4/best_loss.pt and outputs/jepa-v4/latest.pt into distinct files, rerun all three supported probes under current code, and record checkpoint and feature-table hashes. Use the comparison to lock the canonical baseline checkpoint for all later ablation tables.
- Freeze the evaluation protocol: manifest hash, split counts, metadata schema, probe seed, feature formula, and checkpoint identifier, all recorded in the baseline report.
- Add one documented orchestration entry point that runs the full pipeline: submit or run training, validate the completed checkpoint, export features, run probes, produce a compact report. Preserve the Slurm boundary required by HAIC.

Your code should be clean, have no errors, and utilize best software engineering and AI research practices.

Use codex:adversarial-review to review your implementation, and fix all suggested actions and corrections.

For managing libraries and dependencies, we should use uv exclusively: update all docs and notebooks.

Use fan out subagents and dynamic workflows to orchestrate your tasks.