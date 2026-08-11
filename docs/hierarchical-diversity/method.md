# Method: iso-catalog phase allocation in video self-supervised learning

## Claim boundary

The study tests whether the hierarchical location of video diversity changes learned representations under fixed compute and fixed nominal catalog cardinality. It compares new sequences with phase-separated views of fewer sequences. It does not estimate a universal sequence-to-clip exchange rate.

All inference is conditional on GaitLU silhouettes, the JEPA objective, the chosen architecture, and the fixed corpus. It describes reproducibility over paired optimization, stable phase rotations, and pool ordering. Sequence IDs must not be called people, walks, cameras, or environments without verification.

## Treatment construction

### Common eligibility

Estimate stride period and confidence for each validated sequence using a documented frozen silhouette signal, such as width or area autocorrelation. Apply one outcome-blind confidence, coverage, and clip-validity rule to make a common corpus that supports four candidate origins. This prevents one cell from receiving cleaner or longer sequences.

Before selecting pools, group known source groups and outcome-blind near-duplicate feature clusters. Freeze a common available count `M` and selection rule. If source-group constraints make the target counts infeasible, lower `M` once for every cell and record that decision.

### Nested phase origins

For sequence `i` in replicate block `r`, stable hashing selects a uniform base phase `b_ir`. The semantic origins are quarter-cycle separated and nested:

$$
O_{i,r}^{(1)}=\{b_{i,r}\},\qquad
O_{i,r}^{(2)}=\{b_{i,r},b_{i,r}+1/2\},
$$

$$
O_{i,r}^{(4)}=\{b_{i,r},b_{i,r}+1/4,b_{i,r}+1/2,b_{i,r}+3/4\}\pmod 1.
$$

Nearby jitter uses four distinct symmetric small offsets around `b_ir`. It shares the base phase distribution, sequence draws, spatial transforms, masks, exposure, and all nuisance streams with semantic `k=4`. Jitter offsets, rounding, boundary handling, and weights are selected by outcome-blind audit and then frozen.

Audit a stratified sample manually before outcomes. Report phase confidence, origin coverage, realized starts, window overlap, pose-trajectory separation, nominal sequence count, and effective near-duplicate cluster count. Failure ends the phase-allocation branch.

### Allocation registry

| Cell | `U` | `k` | Purpose |
| --- | ---: | ---: | --- |
| `breadth` | 250,000 | 1 | New-sequence extreme |
| `balanced` | 125,000 | 2 | Intermediate path point |
| `phase_depth` | 62,500 | 4 | Phase-separated extreme |
| `nearby_jitter` | 62,500 | 4 | Mechanism diagnostic |

Every row has `U × k = 250,000`. Within each block, source pools are nested whenever the frozen source-group rule permits: `62,500 ⊂ 125,000 ⊂ 250,000`. A row records allocation, sequence count, origins per sequence, nominal catalog size, origin policy, phase-catalog digest, source-group digest, cluster summary, every stream version, and checkpoint provenance.

Eight blocks include breadth, balanced, and phase depth. Four prespecified blocks include nearby jitter. The registry therefore has `8 × 3 + 4 = 28` models. The jitter comparison is deliberately lower precision and is not used as a fourth allocation-path point.

### Constant conditions

Architecture, objective, optimizer, schedule, masks, spatial transformations, batch size, checkpoint selection, and sampled-clip exposure are fixed. One outcome-blind systems gate selects 8,192,000 or 4,096,000 clips for every model. With cardinality 250,000, planned recurrence is 32.77 or 16.38 draws per nominal atom. Report recurrence because equal cardinality does not mean equal information.

## Outcome instrument

Health&Gait GFC v2 uses source-disjoint donors and a complete eight-item factorial gallery. It is supervised alignment followed by donor-based factor recombination retrieval. It is not an unsupervised disentanglement score.

For query `q`, use continuous target margin:

$$
m(q)=d(q,g^-_q)-d(q,g^+_q),
$$

where `g⁺` is the true target and `g⁻` is the nearest frozen non-target competitor. Positive margin means the target wins. Freeze block normalization, competitor rule, aggregation, tie policy, and distance scale. Aggregate within participant before model-level inference.

Independent completion uses the same eight-gallery continuous margin. If that construction fails synthetic controls, use development-fitted shared-temperature gallery NLL instead. Validate the selected score on perfect factorized recovery, independent noisy recovery, missing factors, donor attraction, acquisition shortcut, collapse, and confidence rescaling. Top-1 and MRR must agree in direction with the margin for a directional headline.

## Confirmatory contrast

Let `G_{r,a}` be participant-averaged GFC margin and `C_{r,a}` independent-completion margin for block `r` and allocation `a`. Define

$$
D_{r,a}=G_{r,a}-C_{r,a},
$$

and the primary paired model-level contrast

$$
P_r=D_{r,\mathrm{phase\_depth}}-D_{r,\mathrm{breadth}}.
$$

Analyze the eight `P_r` values using the frozen small-sample procedure, show all values, and require a useful minimum-detectable-effect result before launch. Participant and query observations improve cell precision but do not create additional trained-model replicates.

Report the raw breadth-to-phase-depth GFC contrast, the balanced path point, the four-block phase-depth versus jitter contrast, and the direction of margin, top-1, and MRR. A development-estimated factor-transport geometry check, evaluated on locked participants for parallelism and donor-vector closure, is supporting mechanism evidence only.

## Interpretation rules

Allowed: hierarchical organization of nominal video support mattered, or no difference was resolved at the declared precision; phase-separated origins differed from matched jitter; and an allocation effect did or did not exceed independent factor recovery.

Not allowed: `U × k` equalizes information; phase origins are independent examples; GFC proves intrinsic composition; three points establish a law, frontier, or transferable exchange rate; or a silhouette result automatically transfers to RGB or clinical assessment.

## Lock requirements

Before outcome access, freeze phase estimation, eligibility, common `M`, source and cluster construction, registry, exposure, metric, synthetic tests, primary contrast, analysis code, figures, and failure rules. Resume, evaluation, and export fail closed on any mismatch in manifest, phase catalog, streams, exposure, checkpoint, or registry digests. Release aggregate allowlisted results and instrument code, never participant identifiers, raw recordings, embeddings, or private paths.
