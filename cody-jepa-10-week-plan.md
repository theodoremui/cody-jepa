# CoDy-JEPA 10-Week Execution Plan

## Purpose and Success Criteria

This plan turns CoDy-JEPA into a focused 10-week research build. The goal is a working self-supervised video representation system that separates stable attributes from changing dynamics in articulated motion, then evaluates whether that separation is real.

CoDy-JEPA should not be judged by prediction loss alone. The project succeeds only if the learned streams show the expected asymmetric information pattern:

| Evaluation target | Expected `S_attr` behavior | Expected `S_dyn` behavior |
| --- | --- | --- |
| Subject identity, morphology, embodiment, persistent condition | Strong | Weak |
| Gait phase, action state, pose transition, velocity, trajectory | Weak | Strong |
| Camera, background, session artifacts | Limited | Limited |
| Held-out user, condition, view, or domain transfer | Useful as context | Stronger than baselines for motion transfer |

The strongest supported claim would be: CoDy-JEPA improves separation of stable attributes and changing dynamics relative to a single-stream JEPA baseline and dual-stream baselines without counterfactual token swapping or independence pressure. A weaker but still useful outcome is a well-instrumented negative result that identifies which component fails.

## Core Technical Components

The minimum build has eight parts.

1. A video data pipeline that produces fixed-length clips, metadata-aware splits, diagnostics, and probe exports.
2. A single-stream JEPA baseline that predicts future target embeddings from context embeddings without pixel reconstruction.
3. Temporal-variation token partitioning that routes low-variation tokens toward stable attributes and high-variation tokens toward dynamics.
4. A dual-stream model with `S_attr` for stable attributes and `S_dyn` for dynamics.
5. CI-TSI, the Cross-Instance Token-Swapping Intervention, which combines stable tokens from one clip with dynamic tokens from a same-domain second clip.
6. Correct CI-TSI target alignment: when dynamics come from clip `X_j`, the prediction target must be the future dynamics embedding of `X_j`, not the future of the stable-token source.
7. HSIC regularization to reduce cross-stream dependence between `S_attr` and `S_dyn`.
8. VICReg-style variance and covariance terms to prevent collapse while HSIC pushes the streams apart.

The model should be instrumented from the start. Track prediction loss, latent norm, per-dimension variance, covariance or effective rank, HSIC, probe performance, leakage gaps, and transfer performance.

## Ten-Week Execution Plan

| Week | Technical goal | Required build or experiment output | Gate for continuing or pivoting |
| --- | --- | --- | --- |
| 1 | Define the experiment precisely. Choose the first articulated domain, labels or proxies, baselines, and success metrics. | One-page experiment spec covering dataset, train and validation split, probe split, transfer split, baseline set, leakage metrics, and collapse diagnostics. | Continue only when success is measurable. If labels are weak, define proxy probes such as subject ID, temporal order, speed bucket, motion-energy phase, view, or condition. |
| 2 | Build a trustworthy data pipeline. Convert raw videos into reproducible training windows and metadata-aware evaluation splits. | Loader returning `[B, T, C, H, W]` batches, deterministic validation sampling, metadata summaries, split manifests, batch visualization, frame-difference or motion-energy diagnostics, and dummy probe exports. | Continue when batches and splits are reproducible. If the full dataset is fragile, reduce to a controlled subset and preserve split discipline. |
| 3 | Train the single-stream JEPA baseline. Establish that future-embedding prediction can learn noncollapsed video features. | Compact video encoder, context and target windows, stop-gradient or EMA target branch, predictor head, embedding loss, short training run, and baseline diagnostics. | Continue when loss decreases and latent variance remains nontrivial. If the encoder is unstable, simplify architecture before adding factorization. |
| 4 | Implement temporal-variation token masks. Use token changes over time as a weak cue for stable and dynamic evidence. | Token sequence extraction, variation scores such as squared frame-to-frame token differences, quantile masks `M_s` and `M_d`, mask-size statistics, score histograms, and optional token overlays. | Continue when stable and dynamic token subsets can be exported consistently. If masks are noisy, keep quantile masks and rely on diagnostics rather than building learned masking yet. |
| 5 | Train a dual-stream no-intervention baseline. Verify that `S_attr` and `S_dyn` can be extracted separately before adding CI-TSI. | Shared or separate token backbone, stable-token path into `S_attr`, dynamic-token path into `S_dyn`, dual-stream prediction objective, stream-level variance diagnostics, and saved latent exports. | Continue when both streams train without immediate collapse. If `S_dyn` carries all useful signal, rebalance capacity or mask thresholds before adding swaps. |
| 6 | Add CI-TSI. Break identity-motion shortcuts by pairing stable tokens from one clip with dynamic tokens from another compatible clip. | Same-domain pair sampler, counterfactual context construction, swap logging, rejection rules for empty masks, smoke tests for tensor shape and source tracking, and a CI-TSI training mode. | Continue only when target alignment is correct: dynamic tokens from `X_j` predict future dynamics from `X_j`. If raw token swapping is brittle, swap projected stable and dynamic summaries first. |
| 7 | Add HSIC and VICReg-style stability terms. Reduce stream overlap without allowing trivial constant representations. | Linear HSIC between `S_attr` and `S_dyn`, optional RBF HSIC later, variance and covariance penalties, weighted objective, regularization sweeps, and curves for HSIC, variance, covariance, and prediction loss. | Continue when HSIC is controlled and latent variance stays nonzero. If regularization damages prediction or probes, reduce weights and keep a learnable prediction regime. |
| 8 | Run frozen probes and core ablations. Test whether factorization worked rather than assuming it from architecture. | Frozen linear or shallow probes for `S_attr -> structure`, `S_attr -> motion`, `S_dyn -> structure`, and `S_dyn -> motion`; ablations for single-stream JEPA, dual-stream no CI-TSI, CI-TSI only, no HSIC, and no VICReg. | Continue when leakage gaps can be compared across variants. If compute is limited, keep four rows: single-stream, dual-stream no swap, CI-TSI only, and full CoDy-JEPA. |
| 9 | Test transfer. Measure whether `S_dyn` improves generalization under held-out users, conditions, views, or domains. | Low-label transfer protocol, frozen-feature transfer probes, held-out split results, and comparison against the strongest baseline from Week 8. | Continue when at least one credible transfer result exists. If transfer is noisy, report confidence intervals or seed variation and avoid broad generalization claims. |
| 10 | Consolidate evidence and state the strongest honest claim. Turn experiments into a research artifact. | Final report with method, data, implementation details, diagnostics, probe results, ablations, transfer tests, failures, limitations, and next steps. | Finish when the claim is supported by evidence. If results are mixed, make the central claim diagnostic rather than promotional. |

## Minimum Viable Experiment Set

The minimum viable CoDy-JEPA experiment should fit into a small but defensible run:

| Category | Minimum requirement |
| --- | --- |
| Dataset | One articulated video domain with metadata or proxy labels for stable and dynamic factors. |
| Baselines | Single-stream JEPA, dual-stream without CI-TSI or HSIC, CI-TSI only, full CoDy-JEPA. |
| Token partition | Quantile-based temporal-variation masks with diagnostics. |
| CI-TSI | Same-domain paired clips with logged stable source, dynamic source, and target source. |
| Regularization | Linear HSIC plus VICReg-style variance; covariance term if affordable. |
| Probes | `S_attr -> structure`, `S_attr -> motion`, `S_dyn -> structure`, `S_dyn -> motion`. |
| Transfer | One held-out user, condition, view, or domain test with limited labels. |
| Diagnostics | Prediction loss, HSIC, latent variance, covariance or effective rank, leakage gap, and transfer score. |

This set is enough to decide whether CI-TSI and independence pressure improve separation. Extra seeds, learned masks, RBF HSIC, additional datasets, nearest-neighbor visualizations, and larger models are useful only after the minimum experiment is stable.

## Decision Gates and Risks

| Gate | Timing | Pass condition | Pivot if it fails |
| --- | --- | --- | --- |
| Data gate | End of Week 2 | Loader, metadata, and splits are reproducible. | Shrink to a controlled subset or use proxy labels. |
| Baseline gate | End of Week 3 | Single-stream JEPA learns noncollapsed features. | Simplify encoder, target branch, or clip sampling. |
| Partition gate | End of Week 4 | Stable and dynamic masks are consistent enough for routing. | Use simpler quantiles, temporal smoothing, or frame-difference proxies. |
| Stream gate | End of Week 5 | `S_attr` and `S_dyn` train and export separately. | Rebalance stream capacity or revise mask thresholds. |
| Swap gate | End of Week 6 | CI-TSI trains with correct source and target alignment. | Swap pooled representations before raw tokens. |
| Regularization gate | End of Week 7 | HSIC decreases or stays controlled without collapse. | Lower HSIC weight and keep VICReg variance active. |
| Evidence gate | End of Week 9 | Probes and transfer produce interpretable comparisons. | Reduce claims to the components that are actually supported. |

Main risks are predictable. Token partitions may follow background rather than body motion. CI-TSI pairs may be semantically incompatible. HSIC may remove information that is genuinely shared between embodiment and motion. VICReg-style terms may be too weak to prevent collapse or too strong for a small model. Probe labels may measure shortcuts rather than intended factors. Each risk must be audited directly instead of hidden behind aggregate prediction loss.

## Final Deliverables

By the end of Week 10, produce:

- CoDy-JEPA training code with configurable single-stream, dual-stream, CI-TSI, HSIC, and VICReg-style variants.
- Reproducible data splits and metadata diagnostics.
- Saved `S_attr` and `S_dyn` latent exports for probes.
- Probe and transfer results for the minimum experiment set.
- Ablation results showing the contribution of CI-TSI, HSIC, dual streams, and VICReg-style safeguards.
- A final report that states the strongest supported claim, the failed or ambiguous results, and the next experiment to run.

The report should be written even if the result is negative. A clear failure mode, such as CI-TSI reducing identity leakage while hurting motion prediction, is a useful research outcome because it identifies the tradeoff the next version must solve.
