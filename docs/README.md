# Research documentation

## Start here

The active research direction is the **iso-catalog phase-allocation study**. It asks whether, under fixed training exposure and fixed nominal sequence-origin catalog size, representation learning differs when video diversity is placed across more sequences or across phase-separated views of fewer sequences.

Read in this order:

1. [Revised study overview](hierarchical-diversity/README.md)
2. [Research proposal](hierarchical-diversity/proposal.md)
3. [Frozen method](hierarchical-diversity/method.md)
4. [Execution plan](hierarchical-diversity/execution-plan.md)

The active study has 28 models: eight paired blocks of breadth, balanced, and phase-depth allocation, plus a nearby-jitter diagnostic in four prespecified blocks. It supersedes the earlier low/high support by frozen/resampled-anchor design.

## Shared GaitLU operations

[GaitLU preparation](gaitlu_training.md) explains the shared path from the private raw release to validated, indexed, deduplicated prepared data. It deliberately stops before constructing an experiment-specific training registry.

## Legacy fallback: unique-sequence scaling

The repository retains an earlier unique-sequence-scaling study as a fallback and reusable baseline. It is not the active ICLR framing.

- [Data roles and preprocessing](unique-sequence-scaling/data.md)
- [Method](unique-sequence-scaling/method.md)
- [Original proposal and evidence record](archive/unique-sequence-scaling/)

## Archive

The archive preserves reasoning and results that informed the revised direction. These files are not active protocol sources.

- [Hierarchical-diversity design reviews and handoff](archive/hierarchical-diversity/)
- [Legacy unique-sequence study materials](archive/unique-sequence-scaling/)
- [Stage B preliminary results](archive/stage-b-results.md)
- [GFC-v2 adversarial review record](archive/gfc-v2-adversarial-review.md)

## Future translation

[Future clinical applications](future-clinical-applications.md) describes possible ambient-intelligence, biomechanics, and balance-assessment follow-on work. It does not make clinical claims for the active representation study.

## Shared boundaries

GaitLU trains encoders. Health&Gait fits evaluation heads and scores frozen encoders. No Health&Gait recording updates a primary encoder. Public documentation contains only aggregate, non-identifying evidence. Raw data, participant tables, embeddings, participant-level results, and identity-capable checkpoints remain private.
