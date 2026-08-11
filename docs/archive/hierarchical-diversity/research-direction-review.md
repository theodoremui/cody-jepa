# Research-direction review: revised hierarchical diversity study

**Decision, August 11, 2026:** pursue an iso-catalog phase-allocation study if the outcome-blind gates pass. Do not launch the superseded low/high support by frozen/resampled-anchor experiment as the main ICLR submission.

## The central idea in plain language

Suppose two models process the same number of training clips. One gets one clip from many walking sequences. The other gets several deliberately different points in the gait cycle from fewer sequences. It is tempting to call these equally diverse if the product of sequences and origins is the same. They are not necessarily equally informative.

The revised study makes that distinction testable. It holds clip exposure and nominal catalog size constant, then moves the same catalog across the video hierarchy. The key diagnostic asks whether four separated gait-cycle origins behave differently from four nearby start offsets around the same origin. That separates semantic temporal coverage from temporal randomness.

## Why this is a stronger ICLR question

The old design asked whether windows could replace sequences. Its most dramatic comparison also changed recurrence by orders of magnitude and did not verify that its start indices represented different gait content. It was vulnerable to the simple reply that a regular start grid, repeated tensors, or silhouettes themselves explained the result.

The revised study asks whether a flat count of video examples hides a representation-relevant distinction in where diversity enters training. It is a controlled phenomenon study. The result becomes interesting only if it is tied to a capability that simple factor accuracy does not fully explain.

Health&Gait provides that capability through source-disjoint recombination of speed, clothing, and direction. GFC is not treated as intrinsic compositionality. It is explicitly a supervised, donor-based retrieval instrument, paired with an independent-factor completion control on the same continuous scale.

## Exact design

| Allocation | `U` sequences | `k` phase origins | `U × k` |
| --- | ---: | ---: | ---: |
| Breadth | 250,000 | 1 | 250,000 |
| Balanced | 125,000 | 2 | 250,000 |
| Phase depth | 62,500 | 4 | 250,000 |
| Nearby jitter | 62,500 | 4 | 250,000 |

The path uses the first three allocations in eight paired blocks. Nearby jitter appears in four blocks selected before outcome access. The total is 28 models. `U × k` is nominal cardinality, not equal semantic information, so every result is accompanied by measured phase coverage, overlap, trajectory separation, and near-duplicate cluster counts.

For every sequence, estimate stride period from a frozen silhouette signal, use one common `k=4`-eligible corpus, choose a base phase by stable hash, and make `k=1`, `k=2`, and `k=4` origin sets nested. Semantic `k=4` uses quarter-cycle origins. Nearby jitter uses four small symmetric offsets around exactly the same base phase. All other streams are paired.

## Primary result and required evidence

For query `q`, GFC uses the continuous gallery margin

$$
m(q)=d(q,\text{best non-target})-d(q,\text{true target}).
$$

Use the same construction for independent factor completion. Within each model, subtract the independent-completion margin from GFC margin. Across each paired block, compare phase depth with breadth. This asks whether allocation changes donor-based recombination more than separate factor recovery.

The strongest claim needs all of these:

1. The phase audit confirms that semantic origins are genuinely more separated than jitter.
2. GFC margin, top-1, and MRR agree in direction.
3. Phase depth differs from nearby jitter in the prespecified diagnostic.
4. The GFC residual differs between phase depth and breadth with useful eight-block precision.
5. A locked factor-transport geometry diagnostic agrees with the behavioral result.

Any missing condition narrows the claim. A raw allocation result can still be publishable as careful sampling evidence. It is not enough for a strong representation-mechanism headline.

## Relation to prior work

The general allocation question is not new. Ghadiyaram et al. compared video breadth and temporal duration under different budgets; MAE-ST studies repeated sampling; TCLR uses within-video temporal clips; and compositional-generalization work has shown that factor recognition and novel combinations can dissociate. The contribution must therefore avoid claiming that no one studied breadth, clips per video, or composition.

The narrower wedge is a naturalistic video SSL experiment that holds exposure and nominal sequence-origin cardinality fixed while moving diversity between hierarchy levels, uses a phase versus nearby-jitter semantic control, and reads out source-disjoint recombination against matched independent completion.

## Feasibility and fallback

This is feasible only because it reuses the existing JEPA, Health&Gait GFC, paired training, and provenance scaffolding. The new implementation burden is real: phase catalog generation, revised registry and loader, continuous matched control, synthetic metric tests, and analysis rewrite. It is not feasible to add a second dataset, full capacity sweep, objective sweep, and dose-response ladder before ICLR.

The phase audit is the decisive early gate. If semantic separation cannot be verified by August 13, do not relabel ordinary start-index variation as phase depth. Fall back to a GFC instrument and measurement study on the existing unique-sequence ladder, or defer the hierarchy claim.

## Timeline

- Aug 11–13: phase, duplicate, and common-eligibility audits.
- Aug 13–15: metric synthetic controls and MDE audit.
- Aug 14–17: registry, loader, evaluator, provenance, and figure dry run.
- Aug 18–29: training and permitted exact systems reruns.
- Aug 30–Sep 4: locked analysis and results freeze.
- Sep 5–25: writing only. Abstract Sep 18; paper Sep 25.

## Scope

The study can inform ambient-intelligence and biomechanical sensing research by clarifying when repeated views of a motion cycle are worth retaining during representation learning. It does not establish a clinical endpoint, validate balance assessment, or become a world model. A later counterfactual biomechanical world-model project could use representations and phase audits from this work, but would require explicit dynamics, interventions, and predictive evaluation beyond the present study.

