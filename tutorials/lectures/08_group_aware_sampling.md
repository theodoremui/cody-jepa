# 08. Group-aware sampling and shortcut learning

![Overview of a group-aware sampling pipeline](../images/08_group_aware_sampling.svg)

## Why this lesson matters

Imagine a video dataset with 40 participants and 100 clips per participant. A random row split produces 4,000 clip rows and may report excellent test accuracy. Yet clips from the same participant share body shape, room background, camera, and recording session. If each participant appears in both training and test sets, the model can recognize participants or acquisition conditions instead of learning the intended task.

This lesson asks a simple question before any model is trained: **Which observations are genuinely new evidence?** The answer determines data partitions, temporal sampling, transformation policy, lineage records, and the meaning of evaluation.

## Prerequisites

You should know training, validation, and test splits, basic probability, and Python indexing. [Lesson 07](07_gradient_updates_and_schedules.md) explains how sampled batches drive parameter updates.

## Learning goals

By the end of this lesson, you will be able to:

1. Identify the independent group for a generalization claim.
2. Construct and verify group-disjoint partitions.
3. Construct fixed 16-frame anchor support and measure overlap.
4. Compare frozen-random and resampled temporal policies.
5. Derive expected realized support at a fixed exposure.
6. Isolate an intervention with paired, named random streams.
7. Record lineage from source observation to model tensor.
8. Diagnose acquisition artifacts and shortcut features.

## 1. Begin with the intended generalization

The correct group key depends on what "new" means.

- To generalize to unseen people, group by participant.
- To generalize to unseen recording sessions, group by session.
- To generalize to unseen hospitals, group by hospital or site.
- To generalize to new households, participant grouping may be too narrow.

Let $G(i)$ be the group identifier of observation row $i$. A train-test split is group-disjoint when

$$
\{G(i): i\in I_{\mathrm{train}}\}
\cap
\{G(i): i\in I_{\mathrm{test}}\}
{}={}
\varnothing.
$$

The set $I_{\mathrm{train}}$ contains training-row indices. The set $I_{\mathrm{test}}$ contains test-row indices. The empty intersection means no group identifier appears in both partitions.

### Mental model

Rows are containers. Groups are evidence units. Splitting containers is safe only when containers from the same evidence unit stay together.

### Group keys are design metadata

The group identifier is usually metadata, not a model feature. A participant ID can be essential for safe splitting even when it must never enter the predictor.

Keep dependency identifiers through preprocessing. A window tensor may no longer contain a filename, but its manifest should still identify participant, session, and source sequence. Losing these identifiers makes leakage difficult to audit.

Real datasets often contain nested dependencies:

$$
\text{site}
\rightarrow
\text{participant}
\rightarrow
\text{session}
\rightarrow
\text{sequence}
\rightarrow
\text{window}.
$$

A participant-disjoint split blocks participant and lower-level overlap. It does not test unseen sites when every site appears in all partitions. Choose the highest dependency level required by the claim.

## 2. Why row-level random splits leak

Let $x_{g,r}$ be repeat $r$ from group $g$. A simple decomposition is

$$
x_{g,r}
{}={}
s_{g,r}+a_g+\varepsilon_{g,r}.
$$

The term $s_{g,r}$ is intended signal. The vector $a_g$ is a stable group-specific artifact, such as background or device response. The residual $\varepsilon_{g,r}$ is observation noise.

If training and test sets contain the same group $g$, both contain $a_g$. A flexible model can use that artifact to recognize the group. The test score then measures performance on new repeats from known groups, not performance on unseen groups.

![Row splitting leaks group artifacts while group splitting blocks them](../images/08_group_split_leakage.svg)

This problem does not require duplicate files. Two videos can differ in every frame and still share recognizable participant or acquisition information.

## 3. Split groups first

The safest workflow is:

1. Define the group key from the scientific claim.
2. Deduplicate and audit group identifiers.
3. Assign whole groups to train, validation, or test.
4. Create clips, windows, and augmentations inside each partition.
5. Fit data-dependent preprocessing on training groups only.

~~~python
import numpy as np

rng = np.random.default_rng(8)
unique_groups = np.unique(group_ids)
shuffled = rng.permutation(unique_groups)

n_train = int(0.70 * len(shuffled))
n_valid = int(0.15 * len(shuffled))
train_groups = set(shuffled[:n_train])
valid_groups = set(shuffled[n_train:n_train + n_valid])
test_groups = set(shuffled[n_train + n_valid:])

assert train_groups.isdisjoint(valid_groups)
assert train_groups.isdisjoint(test_groups)
assert valid_groups.isdisjoint(test_groups)
~~~

After assigning groups, <code>np.isin</code> creates vectorized row masks. Store the group assignments so a later run cannot silently reshuffle the evaluation set.

## 4. Stratification happens at group level

Class balance can be difficult when groups have different labels or sizes. Row-level stratification can violate group disjointness.

If every group has one label, stratify group identifiers by that label. If a group contains several labels, define a group-level summary or use a constrained assignment algorithm. State the rule before examining test performance.

No split can guarantee perfect balance when the number of groups is small. Report group counts and row counts separately.

### Weighting defines the population average

Suppose group $g$ contributes $n_g$ windows with losses $\ell_{g,1},\ldots,\ell_{g,n_g}$. A global row average is

$$
R_{\mathrm{row}}
{}={}
\frac{\sum_g\sum_{r=1}^{n_g}\ell_{g,r}}
{\sum_g n_g}.
$$

This estimates loss for a randomly selected row. Groups with long recordings receive more weight.

An equal-group average is

$$
R_{\mathrm{group}}
{}={}
\frac{1}{G}
\sum_{g=1}^{G}
\left(
\frac{1}{n_g}
\sum_{r=1}^{n_g}\ell_{g,r}
\right).
$$

The integer $G$ is the number of groups. This estimates loss for a randomly selected group followed by an observation within that group.

Neither estimand is universally correct. Group-disjoint splitting prevents overlap, while group-aware weighting prevents prolific groups from silently dominating. State which population draw the metric represents.

## 5. Define temporal support before drawing windows

Validation and test windows should remain deterministic so every model receives the same
evaluation inputs. A training intervention needs an equally exact support definition.
Let sequence $i$ contain $n_i$ contiguous frames. For the 16-frame training clip length
$T=16$, the number of valid integer starts is

$$
W_i=n_i-T+1.
$$

The hierarchical-diversity intervention does not use every valid start. It uses anchors
spaced by eight frames:

$$
\mathcal A_i=
\{0,8,16,\ldots,8\lfloor(W_i-1)/8\rfloor\},
\qquad
K_i=|\mathcal A_i|.
$$

Here $\mathcal A_i$ is the supported anchor set and $K_i$ is its size. The Python slice
for anchor $a$ is `sequence[a:a + T]`. It contains 16 frames because the stop index is
excluded. One global capability rule, $K_i\ge2$, is applied after basic validation and
before holdout selection or construction of any replicate pool. Applying the same rule
once prevents conditions from quietly using different eligibility criteria.

Adjacent anchors are eight frames apart, so their 16-frame windows share eight frames.
The overlap fraction is

$$
\frac{T-8}{T}=\frac{8}{16}=0.5.
$$

This is 50 percent overlap, not two independent observations. To describe a more
conservative support scale, also count anchors at 16-frame spacing:

$$
\mathcal B_i=
\{0,16,32,\ldots,16\lfloor(W_i-1)/16\rfloor\},
\qquad
Q_i=|\mathcal B_i|.
$$

The count $Q_i$ measures non-overlapping anchored windows that begin on this coarser
grid. Report both $K_i$ and $Q_i$. Neither count turns windows from the same sequence
into new participants, sessions, or source videos.

## 6. Frozen-random and resampled policies change temporal support

The frozen-random policy chooses one anchor uniformly for each stable
`(sequence_id, replicate_seed)` pair. Once selected, that anchor stays fixed across
epochs and repeated sequence draws. Across randomization, every anchor in $\mathcal A_i$
has probability $1/K_i$. Within one realized run, however, the sequence always returns
the same anchor, so its conditional distribution is a point mass.

The word "random" therefore describes how the frozen anchor is assigned, not what
happens on every training draw. A center window would not be equivalent because gait
phase or recording quality can vary with temporal position. Uniform frozen assignment
avoids making one position special in expectation.

Use a versioned stable hash to derive the frozen seed from the sequence ID and replicate
seed. Python's built-in `hash` is process dependent and must not be used. Manifest row
position is also unsafe because inserting another sequence would change later anchors.
The stable identity rule gives **nested-manifest stability**: if a low-support manifest
is a prefix of a larger manifest, every shared sequence keeps the same frozen anchor.

~~~python
import hashlib
import json

def stable_uint64(namespace, *parts):
    payload = json.dumps(
        [namespace, *parts], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")

def frozen_anchor(sequence_id, replicate_seed, anchors):
    seed = stable_uint64("temporal-frozen-v1", sequence_id, replicate_seed)
    rng = np.random.default_rng(seed)
    return int(rng.choice(anchors))
~~~

The resampled policy instead selects uniformly from $\mathcal A_i$ on every draw. Its
versioned seed includes the base seed, virtual epoch, stable sequence or sample identity,
draw index, and temporal-stream version. Recreating a virtual epoch therefore recreates
its anchors even when workers finish in a different order.

Frozen and resampled policies have the same uniform marginal distribution over anchors.
They differ in dependence across repeated draws. The frozen condition can realize at
most one anchor for a given sequence and replicate. The resampled condition can realize
up to $K_i$. This distinction isolates access to within-sequence temporal support.

## 7. Pair nuisance streams and fix sampled exposure

A temporal intervention should not accidentally change spatial crops or JEPA masks.
Create separate named streams for:

- ordered sequence draws,
- temporal anchors,
- spatial transformations,
- prediction masks.

Within a sequence-support pair, frozen and resampled runs share the ordered sequence,
spatial, and mask streams. Only the temporal stream follows a different policy. They
also share initialization, optimizer settings, effective batch size, and checkpoint
rule. Tests should compare the paired records directly rather than assuming that equal
base seeds produce equal nuisance draws.

The total sampled-example exposure $C$ is fixed across conditions. A repeated draw still
counts toward $C$, even when the frozen policy returns the same sequence-window pair.
This is the point of the comparison: available support changes while optimizer exposure
does not. Use the final planned checkpoint for every cell. Downstream outcomes cannot
choose a different epoch, seed, or rerun.

Separate generators prevent call-order coupling. With one global generator, one extra
temporal draw in the resampled condition would shift every later crop and mask. Named
streams let the sequence draw and nuisance parameters remain paired even though the
temporal anchor changes.

## 8. Expected realized support quantifies treatment strength

Suppose a pool contains $U$ sequences and training makes $C$ sequence draws uniformly
with replacement. The frozen policy supports one sequence-window pair per sequence. Its
expected number of distinct realized pairs is

$$
E_F(U)=
U\left[1-\left(1-\frac{1}{U}\right)^C\right].
$$

The expression in brackets is the probability that a particular sequence is drawn at
least once. For the resampled policy, pair $(i,a)$ has probability $1/(UK_i)$ on each
draw. Summing its visit probability over all supported pairs gives

$$
E_R(U)=
\sum_{i=1}^{U}
K_i\left[1-\left(1-\frac{1}{UK_i}\right)^C\right].
$$

These are occupancy expectations, not claims of semantic independence. They quantify
how many supported `(sequence_id, window_start)` pairs the sampler is expected to visit.
For numerical stability at large $C$, software can evaluate
`-expm1(C * log1p(-p))` instead of `1 - (1 - p)**C`.

Before training, reconstruct every nested low and high pool for every replicate from the
deduplicated inventory. For each pool, report $W_i$, $K_i$, $Q_i$, $E_F$, $E_R$, the
expected fraction of draws that revisit an already counted sequence-anchor pair, and mean
overlap among distinct anchor pairs. Evaluate treatment separation at the lower permitted
exposure, $C=4{,}096{,}000$, so the gate also holds if the larger tier is selected.

If $E$ is the expected number of distinct sequence-anchor pairs after $C$ draws, the
expected repeated-draw fraction used in this audit is $1-E/C$. Compute it separately with
$E_F$ and $E_R$. For mean overlap, enumerate unordered pairs of different anchors within
each sequence, divide their shared-frame count by 16, and average those fractions across
the pool. These are required diagnostics, not extra launch gates.

The prospective launch gates are median $K_i\ge4$ and $E_R/E_F\ge4$ in every low and
high pool. These frozen gates show that resampling creates a substantial support
contrast. They do not prove that additional anchors are independent or useful. If a
gate fails, revise or cancel the design before examining downstream outcomes rather than
lowering the threshold afterward.

## 9. Transformations should respect time

Suppose a video window has shape $(T,H,W,C)$:

- $T$ is the number of frames.
- $H$ and $W$ are image height and width.
- $C$ is the number of color channels.

A random horizontal flip should normally draw one Boolean value for the whole window. Drawing a new flip per frame creates artificial flicker.

~~~python
def consistent_flip(window, rng):
    flip = bool(rng.random() < 0.5)
    if flip:
        return np.flip(window, axis=2).copy(), flip
    return window.copy(), flip
~~~

The call to <code>np.flip</code> can return a negative-stride view. The copy creates contiguous storage that <code>torch.from_numpy</code> can safely consume.

![Window starts and transform parameters have different sampling scopes](../images/08_windows_and_transforms.svg)

Consistency is not a universal rule. Independent sensor noise per frame can be appropriate if it reflects the real data-generating process. The transformation scope should match plausible variation.

### Transform units and label validity

A crop has coordinates and size measured in pixels. A speed perturbation has units of frames or seconds. A color transform acts on channels. Recording these parameters makes augmentation scientifically interpretable.

Some labels transform with the input. A horizontal flip can swap left and right labels. A temporal reversal can change movement direction. Visual consistency is not sufficient; labels must remain valid under the transformation.

## 10. Data lineage makes every tensor traceable

Lineage answers: "Where did this exact model input come from?"

A useful record contains:

- immutable source identifier,
- group identifier,
- partition name,
- original sequence length,
- replicate and sequence-support condition,
- window start and stop,
- temporal policy and stream version,
- separate sequence, spatial, and mask stream versions,
- sampled transform parameters,
- manifest digest and planned exposure,
- preprocessing version,
- label source and label version.

The workflow is

$$
\text{source}
\rightarrow
\text{group partition}
\rightarrow
\text{window}
\rightarrow
\text{transform}
\rightarrow
\text{tensor}.
$$

Store decisions as structured data, not only in filenames. CSV, JSON Lines, or Parquet is sufficient. A stable sample identifier can hash canonical lineage fields.

### Lineage as an executable contract

Given a lineage record and source-data version, a deterministic loader should reconstruct the same pre-transform window. For random transforms, store sampled parameters or a stable per-sample seed.

Useful assertions check that:

- every sample's group belongs to its declared partition,
- start and stop indices are within source bounds,
- window length matches the model contract,
- every start belongs to the sequence's fixed anchor set,
- a shared sequence keeps its frozen anchor across nested manifests,
- paired conditions share sequence, spatial, and mask draws,
- preprocessing version is known,
- no sample identifier appears in more than one partition.

These checks turn provenance from documentation into a tested interface.

## 11. Shortcut learning

A shortcut is a feature that predicts labels in collected data but does not represent the intended construct.

Let $Y$ be the target label and $A$ a suspected artifact. In training data, the artifact can be predictive:

$$
P_{\mathrm{train}}(Y\mid A)
\ne
P_{\mathrm{train}}(Y).
$$

At deployment, the association may weaken or reverse. The model follows the easiest reliable signal available to its objective, even when that signal is scientifically irrelevant.

Common shortcuts include:

- site or device,
- compression quality,
- room background,
- text overlays,
- duration and padding,
- missingness patterns,
- preprocessing order,
- identity-bearing filenames.

### Worked scenario

Suppose all positive-class videos were captured on device A and all negative-class videos on device B. A device classifier can achieve perfect label accuracy without reading motion. A random row split preserves the device-label association. Testing on both devices for both labels breaks it.

Shortcuts are relative to a claim. Device identity is a shortcut for a biological movement claim, but it may be the intended signal in device-quality monitoring. Name the intended construct before labeling a feature spurious.

Removing one known artifact-label correlation does not prove that the model uses the intended mechanism. Other correlated artifacts can remain.

## 12. Shortcut diagnostics

No single diagnostic proves that shortcuts are absent. Combine several:

1. Compare row-level and group-disjoint scores.
2. Train an artifact-only baseline from site, device, duration, or missingness.
3. Alter a suspected artifact while preserving intended content.
4. Report metrics within each acquisition subgroup.
5. Inspect nearest neighbors for shared background rather than shared semantics.
6. Permute labels within groups to test what group identity alone supports.

A counterfactual perturbation is useful only when it changes the suspected artifact without accidentally changing the intended signal.

### Interpreting a split gap

Suppose row-split accuracy is 95 percent and participant-disjoint accuracy is 62 percent. The gap is evidence that row-level evaluation benefited from participant-linked information. It does not prove that all 33 percentage points came from one specific artifact.

Inspect performance by group, device, and site. Averages can hide a model that succeeds only under one acquisition condition.

## 13. Efficiency notes

- Use <code>np.unique</code> to encode group identifiers once.
- Use <code>np.isin</code> for vectorized partition masks.
- Store window indices instead of duplicated window arrays when reconstruction is cheap.
- Use <code>Tensor.unfold</code> for equal-length sliding-window views.
- Use group-balanced samplers when prolific groups would dominate batches.
- Copy a window before any in-place transform if it aliases source storage.
- Precompute anchor arrays and $K_i$ once per immutable inventory.
- Evaluate occupancy with stable `log1p` and `expm1` operations at large exposure.

## 14. Common failure modes

1. **Grouping too narrowly:** recording splits leak participant identity.
2. **Preprocessing before splitting:** test statistics influence training.
3. **Random evaluation windows:** metric noise obscures model changes.
4. **Framewise random crops:** augmentation jitter replaces motion.
5. **A seed without a manifest:** source and transform choices remain hidden.
6. **Row-weighted evaluation:** participants with more repeats dominate.
7. **Repeated test inspection:** the test set becomes tuning feedback.
8. **Manifest-row hashing:** nested pools assign different anchors to the same sequence.
9. **One generator for every draw:** temporal choices shift crop and mask randomness.
10. **Calling anchors independent:** overlap and shared source causes are ignored.
11. **Unequal exposure:** the support intervention is mixed with additional optimization.

## 15. Exercises

1. A dataset has 100 participants and 20 windows per participant. Which count controls participant-level uncertainty?
2. Find $\mathcal A_i$, $K_i$, and $\mathcal B_i$ for $n_i=48$ and $T=16$.
3. Why do frozen-random and resampled policies have the same marginal anchor distribution but different realized support?
4. Why must spatial transforms and masks use streams separate from the temporal policy?
5. Design one lineage record for a resampled training window.

### Brief solutions

1. Under a participant-independence assumption, the nominal independent-unit count is 100. The 2,000 windows are correlated repeats.
2. $W_i=33$, so $\mathcal A_i=\{0,8,16,24,32\}$, $K_i=5$, and $\mathcal B_i=\{0,16,32\}$.
3. Both choose every anchor with probability $1/K_i$ across randomization, but frozen-random repeats one assigned anchor within a run while resampling can visit all $K_i$ anchors.
4. Separate streams keep nuisance draws paired when the temporal policy makes a different number or pattern of random calls.
5. Include source and group IDs, split, replicate, policy, stream versions, start, stop, transform parameters, manifest digest, planned exposure, and preprocessing version.

## Recap

Group-aware sampling aligns evaluation with the intended independent unit. The fixed anchor
set defines temporal support, while frozen-random and resampled policies change how much
of that support a run can realize. Occupancy expectations measure the planned contrast
at fixed exposure. Stable identity hashing, nested-manifest checks, and paired nuisance
streams isolate the intervention. Temporal transformations still need a scientifically
plausible scope, lineage makes every tensor auditable, and shortcut diagnostics test
whether performance depends on acquisition artifacts.

## Next lesson

[09: Eigenspectra and effective rank](09_eigenspectra_and_effective_rank.md) studies the geometry of the representations produced by this pipeline.

## Continue in the notebook

[Open the executable lesson 08 notebook](../implementations/08_group_aware_sampling.ipynb) to compare row and group splits, verify temporal policies, audit expected support, and build a lineage record.
