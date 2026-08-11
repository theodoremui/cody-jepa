# Research Documentation

This folder is the navigation point for the active CoDy-JEPA research direction. It keeps
the current method, execution path, and data-preparation workflow in one place.

## Where to Start

- [Hierarchical-diversity guide](hierarchical-diversity/README.md) gives the top-level
  picture and routes readers to the detailed files.
- [Research proposal](hierarchical-diversity/proposal.md) explains the scientific question
  and why the allocation contrast matters.
- [Method](hierarchical-diversity/method.md) fixes the treatment, evaluator, estimand,
  interpretation rules, and privacy boundary.
- [Execution plan](hierarchical-diversity/execution-plan.md) lists the gates, work
  packages, training-start checklist, and schedule.
- [GaitLU preparation](gaitlu_training.md) explains the current private-corpus preparation
  workflow used by this study instance.

## How the Documents Fit Together

The proposal explains why the study exists. The method says exactly what result would be
allowed to mean. The execution plan says what must pass before training and analysis can
begin. The preparation runbook covers the shared data conversion path before any
study-specific registry is frozen.

The technical tutorials live outside this folder in [tutorials](../tutorials/README.md).
They are the best entry point for readers who want to learn the mathematical, statistical,
and engineering concepts before reading the protocol.

## Shared Boundaries

The research contribution is the controlled hierarchical-diversity approach and its
factorial evaluation logic. GaitLU and Health&Gait name the current implementation
instance, not the whole idea.

Private data, participant-level artifacts, feature exports, and identity-capable
checkpoints stay outside Git. Public documentation should explain aggregate evidence,
reproducible protocols, and claim limits.
