# Research documentation

The repository contains two alternative ICLR research directions. They share GaitLU
pretraining infrastructure, Health&Gait evaluation roles, and GFC-v2, but they ask
different questions and require different model registries. They must not be combined
after outcomes are observed.

## Unique-sequence scaling study

This is the study currently implemented by the 20-row training and evaluation pipeline.
It varies the number of unique GaitLU sequences across four nested pools while holding
sampled-sequence exposure fixed. A completion-gap interval and equality assertion remain
required pre-freeze analysis additions.

- [Proposal](unique-sequence-scaling/proposal.md)
- [Methods](unique-sequence-scaling/method.md)
- [Data roles and preprocessing](unique-sequence-scaling/data.md)
- [GaitLU preparation and 20-model training runbook](gaitlu_training.md)
- [Evidence available before the primary study](unique-sequence-scaling/results.md)

## Hierarchical-diversity study

This is a proposed replacement that is not yet implemented. It crosses sequence-pool
size with temporal-window resampling policy. It asks whether these sources of training
support have different effects on donor-based factor-composition retrieval than on
independent-factor prediction, and whether multiple windows from a small pool can match
one frozen window from a 100×-larger pool.

- [Proposal](hierarchical-diversity/proposal.md)
- [Methods](hierarchical-diversity/method.md)
- [Execution plan](hierarchical-diversity/execution-plan.md)
- [Training-start agent handoff](hierarchical-diversity/training-start-handoff.md)

The hierarchical study requires a new frozen-random window policy, a 32-model registry,
factorial inference, and new execution tests. Until those changes pass the documented
gate, the existing runbook applies only to the unique-sequence scaling study.

## Shared boundaries

GaitLU trains encoders. Health&Gait fits evaluation heads and scores frozen encoders.
No Health&Gait recording updates a primary encoder. The public documentation contains
only aggregate, non-identifying evidence. Raw data, participant tables, embeddings,
participant-level results, and identity-capable checkpoints remain private.
