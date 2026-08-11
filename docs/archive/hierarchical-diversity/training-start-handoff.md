# Training-start handoff: phase-allocation protocol

## Do not start training until this checklist is complete

This handoff is for the revised 28-model iso-catalog study. The old 32-model low/high by frozen/resampled registry is superseded and must not be used for this experiment.

### Required frozen artifacts

- A versioned phase catalog containing sequence eligibility, period estimate, confidence, base phase, `k=1/2/4` semantic origins, nearby-jitter origins, and a digest.
- A source-group and near-duplicate cluster audit with a frozen common pool-size rule.
- A 28-row registry: eight blocks of `breadth`, `balanced`, and `phase_depth`, with `nearby_jitter` in four prespecified blocks.
- An exposure decision of exactly 8,192,000 or 4,096,000 clips applied to every row.
- Synthetic validation of the continuous GFC and independent-completion scores.
- An analysis configuration defining participant aggregation, target-margin competitor rule, top-1, MRR, the eight `P_r` contrasts, and figures.

### Required registry fields

Each row must include at least:

```text
block, allocation, unique_sequences, origins_per_sequence,
nominal_catalog_size, origin_policy, train_manifest_digest,
phase_catalog_digest, source_group_digest, cluster_summary_digest,
planned_exposure, effective_batch, completed_updates,
optimization_seed, replicate_seed, sequence_stream_version,
phase_stream_version, spatial_stream_version, mask_stream_version,
checkpoint_rule, configuration_digest
```

`nominal_catalog_size` must equal `unique_sequences × origins_per_sequence`. The first three allocations have eight blocks; nearby jitter has exactly the four blocks chosen before outcomes. Resume, export, and evaluation must reject mismatches rather than infer missing values.

### Pairing rules

Within a block, all primary cells share optimization and replicate seeds. The phase-depth and nearby-jitter cells share sequence draws, base phases, spatial transforms, masks, exposure, and checkpoint rule. Their only intended difference is phase-separated versus nearby origin construction.

Where feasible, pools follow `62,500 ⊂ 125,000 ⊂ 250,000`. If source-group constraints prevent exact counts, use the frozen common-`M` rule and record realized counts. Do not silently alter a cell after outcome access.

### Preflight commands and outputs

The implementation must provide outcome-blind commands to:

1. Build and audit the phase catalog.
2. Create and validate the registry and nested pool relations.
3. Run synthetic evaluator controls.
4. Run a four-cell pilot that includes one jitter comparison.
5. Run an eight-job throughput and checkpoint-storage probe.
6. Train exactly the rows declared in the registry.
7. Export only final-step checkpoints with complete provenance.
8. Produce aggregate figures from locked analysis code.

The audit report must show phase confidence, origin coverage, overlap, trajectory distance, nominal and effective cluster counts, actual clips, completed updates, checkpoint digests, and failures. A successful job without this evidence is not an accepted row.

### Stop conditions

Stop instead of patching around failures when phase separation fails, the common eligible pool cannot be constructed, the metric fails synthetic controls, the MDE is not useful, production throughput is unstable, or provenance cannot be validated. Permitted reruns are exact systems failures defined before training, never seed replacement or selective extra training.
