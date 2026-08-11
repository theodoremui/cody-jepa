# Execution plan: 28-model iso-catalog phase-allocation study

## Objective

Execute the revised study in [proposal.md](proposal.md) and [method.md](method.md), then stop experiments and analysis by September 4, 2026. The ICLR 2027 abstract deadline is September 18 and the paper deadline is September 25, so September 5 through September 25 are reserved for writing only.

## Starting point

The repository contains a working but superseded 32-model hierarchy scaffold based on low and high sequence support with frozen or resampled anchors. It is useful infrastructure, not the protocol. The revised study needs a phase catalog, a 28-row registry, a phase-aware loader, validated outcome instrument, and matching analysis path. Do not run the old registry as evidence for the new question.

## Gates

| Gate | Deadline | Required evidence | Failure action |
| --- | --- | --- | --- |
| Phase audit | Aug 13 | Frozen estimator, common `k=4` eligibility, blinded manual check, semantic separation from jitter | Stop the phase branch and retain the instrument fallback |
| Catalog audit | Aug 14 | Source groups, near-duplicate clusters, feasible common `M`, nested-pool rule | Lower `M` once prospectively or use fallback |
| Metric and power audit | Aug 15 | Synthetic GFC/control cases and legacy or development MDE estimate | Do not launch an underpowered contrast |
| Software dry run | Aug 17 | 28-row registry, loader, provenance, analysis, figures, synthetic controls | Revert to the separate unique-sequence study |
| Systems pilot | Aug 18 | Four-cell GPU pilot and eight-job storage probe at checkpoint cadence | Select frozen half exposure or cancel |
| Training completion | Aug 29 | All 28 rows plus only permitted exact systems reruns | No seed replacement or opportunistic cell dropping |
| Analysis lock | Sep 4 | Final checkpoints, aggregate export, figures, audit report | Enter paper-only period |

## Work packages

### Build the phase catalog

- Implement deterministic stride-period estimation from the frozen silhouette signal.
- Store eligibility, stable base-phase rotation, nested `k=1/2/4` origins, and nearby-jitter origins in a hashed catalog.
- Audit bounds, overlap, phase coverage, trajectory separation, and failure reasons.
- Build source-group and near-duplicate cluster summaries before pool selection.

### Replace the registry and loader contract

- Define `breadth`, `balanced`, `phase_depth`, and `nearby_jitter` cells.
- Require `unique_sequences`, `origins_per_sequence`, `nominal_catalog_size`, `origin_policy`, `phase_catalog_digest`, source-group digest, and cluster summary in every row.
- Generate eight blocks for the three primary cells and nearby jitter only in four prespecified blocks.
- Pair optimization, sequence, spatial, and mask streams. Only semantic origin construction differs in the phase-versus-jitter comparison.
- Make resume and export fail closed on provenance mismatch.

### Lock the evaluator

- Implement the matched eight-gallery continuous margin for GFC and independent completion.
- Run synthetic positive and negative controls before outcomes.
- Freeze aggregation, competing gallery items, top-1, MRR, missing-data handling, and the eight paired `P_r` contrasts.
- Implement the supporting cross-context factor-transport geometry diagnostic.

### Train and report

- Select full or half exposure only through the throughput rule.
- Train paired block waves with cell-to-hardware randomization.
- Save complete checkpoint provenance and produce the locked figure with all replicate trajectories, the balanced point, and paired jitter diagnostics.
- Report nominal catalog cardinality beside measured semantic diagnostics.

## Calendar

| Dates | Deliverable |
| --- | --- |
| Aug 11–13 | Phase and near-duplicate audits, common eligibility decision |
| Aug 13–15 | Catalog freeze, metric synthetic tests, MDE audit |
| Aug 14–17 | Registry, loader, provenance, analysis, and figure dry run |
| Aug 18–25 | Primary training waves |
| Aug 26–29 | Permitted systems reruns, final export, audit completion |
| Aug 30–Sep 4 | Locked aggregate analysis and figures |
| Sep 5–17 | Paper only |
| Sep 18 | Abstract submission |
| Sep 19–24 | Paper only: anonymization, reproducibility, final checks |
| Sep 25 | Paper submission |

## Non-goals for this cycle

- No second dataset, RGB replication, architecture sweep, objective sweep, or `k=8` ladder on the critical path.
- No claim of a general allocation frontier from three correlated path points.
- No post-outcome changes to eligibility, phase thresholds, jitter offsets, metric, or allocations.
- No clinical or balance-assessment conclusion from this representation study.

## Training-start checklist

Do not submit a training job until all artifacts below are frozen and validated.

1. A versioned phase catalog records sequence eligibility, period estimate, confidence,
   base phase, nested semantic origins, nearby-jitter origins, and a digest.
2. A source-group and near-duplicate audit establishes the common pool-size rule.
3. The registry has exactly 28 rows: eight breadth, balanced, and phase-depth rows per
   block, plus nearby jitter only in the four prespecified blocks.
4. Every registry row records allocation, sequence count, origins per sequence, nominal
   catalog size, origin policy, manifest digest, phase-catalog digest, source-group digest,
   exposure, seeds, stream versions, and checkpoint rule.
5. The continuous GFC and independent-completion instrument passes synthetic controls.
6. The exposure tier is selected once by the throughput rule and applies to every row.

For the phase-depth versus nearby-jitter pair, sequence draws, base phases, masks, spatial
transforms, optimization seeds, exposure, and checkpoint rules are paired. Only phase
separation versus local jitter may change. Resume, feature export, and evaluation reject
any provenance mismatch rather than inferring missing metadata.

Stop rather than patch around failures when phase separation fails, the common eligible pool
cannot be built, the metric fails synthetic controls, the MDE is not useful, throughput is
unstable, or provenance cannot be validated. Permitted reruns are exact systems failures
defined before training, never seed replacement or selective extra training.
