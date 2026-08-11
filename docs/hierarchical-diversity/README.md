# Hierarchical diversity: active study guide

## What this study is

The active study asks where video diversity should live. With fixed clip exposure and the same nominal catalog of sequence-origin atoms, should training use many sequences with one phase origin each, or fewer sequences with several phase-separated origins?

| Allocation | Sequences | Origins per sequence | Nominal catalog |
| --- | ---: | ---: | ---: |
| Breadth | 250,000 | 1 | 250,000 |
| Balanced | 125,000 | 2 | 250,000 |
| Phase depth | 62,500 | 4 | 250,000 |
| Nearby jitter diagnostic | 62,500 | 4 | 250,000 |

The first three are eight paired allocation blocks. Nearby jitter appears in four blocks and checks whether phase separation matters beyond local temporal variation. The complete experiment has 28 models.

## Which document answers which question

| Need | Read |
| --- | --- |
| Why this is a useful ICLR representation-learning question | [proposal.md](proposal.md) |
| Exact eligibility, phase construction, registry, outcomes, and claims | [method.md](method.md) |
| Gates, software work, training checklist, and calendar | [execution-plan.md](execution-plan.md) |
| How raw GaitLU data become prepared inputs | [GaitLU preparation](../gaitlu_training.md) |
| Earlier rejected designs and review history | [archive](../archive/hierarchical-diversity/) |

## Non-negotiable interpretation limits

This is a controlled case study in GaitLU silhouettes with one JEPA objective. U times k controls nominal cardinality, not information. GFC measures supervised donor-based factor recombination, not unsupervised disentanglement. Three path points do not establish a universal scaling law. The study makes no clinical or balance-assessment claim.

## Status

Do not launch training until the phase, catalog, metric, power, software, and systems gates in [execution-plan.md](execution-plan.md) pass. The experiment and analysis freeze by September 4, 2026, preserving September 5 through September 25 for ICLR writing.

