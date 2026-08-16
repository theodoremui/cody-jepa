# Research proposals for ICLR 2027

A synthesis of `claude-iclr-analysis.md`, `claude-iclr-ideas.md`, and `codex-iclr-ideas.md`, put through adversarial review and checked against primary sources and the files in this repository.

Prepared August 14, 2026. ICLR 2027 abstracts are due September 18, 2026 and papers on September 25, 2026.

## Start here

**[Overview and evaluation](00-overview-and-evaluation.md)** is the anchor document. It records which claims from the three source documents survived checking, what new evidence came out of the repository's own artifacts, how each proposal scores against the official ICLR criteria and against what recently got accepted, and which ideas should be dropped.

## The proposals

Each is standalone and each carries an ICLR core plus an extension to the Stanford HAI ambient intelligence work and an extension to Scott Delp's balance assessment work.

| | Proposal | Status |
|---|---|---|
| 1 | **[The readout problem](01-readout-problem.md)** | Recommended submission. Written to be readable with first-year linear algebra and probability. **[One-page summary](01-readout-problem-onepager.md).** |
| 2 | **[Paired-condition geometry](02-paired-condition-geometry.md)** | Strong alternative. Higher ceiling, higher risk. |
| 3 | **[Minimum sufficient state](03-minimum-sufficient-state.md)** | Protocol folds into 1. Biomechanics half is the best Delp approach. |
| 4 | **[The contained personal baseline](04-personal-baseline.md)** | Blocked on longitudinal data. Best HAI proposal on a longer horizon. |

## Figures

Vector graphics live in [`images/`](images/). Every diagram is hand-authored SVG with no external dependencies, sized for inline reading at 750 points wide.

## The six things worth knowing before reading anything else

1. The dataset has no randomised assignment. Usual pace was always recorded before fast pace, and the jacket was the participant's own and optional. Call these paired conditions, never interventions.
2. The two-class walking-pace probe that all three source documents treat as the one working result is beaten by a single threshold on how long the walk took, 0.952 against 0.938, on the same held-out participants. Matching the evaluation set on duration collapses every checkpoint to a mean of 0.560 against chance 0.500.
3. The eleven-run sweep that was written off as finding nothing spans a factor of sixteen in clip-pooled effective rank, and that rank correlates with the honest open-set probe at Spearman 0.89, while pretraining loss correlates at 0.187.
4. A randomly initialised encoder does not reproduce the rank numbers, so they measure the trained model rather than the architecture. That control has been run.
5. For a trained encoder with no variance regulariser, 91.7 percent of all token variance is a deterministic function of position, and 0.063 percent is the between-clip part that a pooled probe can read.
6. That position dominance is already 96.4 percent at random initialisation. So the measurement claim in Proposal 1 is strong and its mechanism claim is weak, which is why the two were reordered.

Everything numeric above was recomputed from artifacts already in this repository. The working notes are in [`../notes/derived-findings-2026-08-14.md`](../notes/derived-findings-2026-08-14.md) and the scripts are in [`../notes/scripts/`](../notes/scripts/).
