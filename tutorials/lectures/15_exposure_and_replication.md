# 15. Exposure, replication, and variance decomposition

![Overview of exposure and replication](../images/15_exposure_and_replication.svg)

## Prerequisites

You should understand grouped observations, means, variance, and confidence intervals.
Review [14. Paired contrasts, uncertainty, and decision thresholds](14_paired_inference.md)
for analysis units and participant-level uncertainty.

## Learning goals

By the end of this lesson, you will be able to:

1. distinguish unique source units, generated rows, and training exposures;
2. describe nested sampling pools and their implied weights;
3. separate randomization, sampling, and analysis units;
4. distinguish participant replication from model-seed replication;
5. interpret variance components in nested and crossed designs;
6. estimate a cluster design effect as an intuition tool;
7. identify leakage and complete-case selection; and
8. lock a design prospectively and state its generalization limits.

## 1. Motivating scenario: 3,600 windows from 20 people

A video study begins with 20 participants. Each participant contributes three sessions,
each session contributes three clips, and every clip produces 20 overlapping windows. The
training table now has 3,600 rows. Training for 50 epochs presents 180,000 row exposures.

How much evidence is there about new people? The answer is not 180,000 and not 3,600. If
the 20 people were independently sampled, the study still contains only 20 independent
participant sampling units. Windows and epochs can improve how well the model learns from
those people, but they do not manufacture new participants. If the people form a convenience
sample, even the 20-person population claim needs to be narrowed accordingly.

The mental model is photocopying. More copies make a document easier to distribute and
read repeatedly. They do not create more independent authors.

## 2. Vocabulary: count the right objects

A **source unit** is an original lineage item from which rows are derived, such as a
participant, session, or recording. Calling something a source unit does not make it
statistically independent. A **row** is one item in the model's input table. A source can
produce many rows through windowing or augmentation. An **exposure** occurs each time a row
is presented during optimization.

A **replicate** is an independently repeated unit at a level relevant to the claim. A newly
and independently sampled participant is a participant replicate. A separately trained
model run with an independently generated random seed is a model-run replicate for
algorithmic randomness. Ten crops of the same frame are repeated measurements, not ten
participant replicates.

These counts answer different questions:

- source count describes sampled breadth;
- row count describes stored training examples;
- exposure count describes computational reuse; and
- replicate counts describe uncertainty across declared random dimensions.

## 3. Nested identifiers and shapes

Let $p$ index participants, $s$ sessions within a participant, $c$ clips within a session,
and $w$ windows within a clip. A row can be identified by the tuple `(p, s, c, w)`.

For a balanced design with $P$ participants, $S$ sessions per participant, $C$ clips per
session, and $W$ windows per clip, row count is

$$
N_{\mathrm{row}}=P\times S\times C\times W.
$$

Each symbol is a count at one nested level. With 20, 3, 3, and 20, the product is 3,600.
Real datasets are often unbalanced, so do not infer counts from a rectangular shape when
some participants or sessions contribute more than others. Keep explicit identifiers.

![Participants contain sessions, clips, and windows](../images/15_nested_units.svg)

A tidy manifest has one row per model input and columns such as `participant_id`,
`session_id`, `clip_id`, `window_start`, and `source_id`. These identifiers are part of
the statistical design, not optional bookkeeping.

## 4. Unique observations versus repeated exposure

Suppose $N_{\mathrm{source}}$ source items each create $A$ augmented rows and training uses
$E$ epochs. Then

$$
N_{\mathrm{row}}=A N_{\mathrm{source}},\qquad
N_{\mathrm{exposure}}=E N_{\mathrm{row}}.
$$

Increasing $A$ or $E$ changes optimization. It does not change $N_{\mathrm{source}}$.
This distinction is essential when a learning curve is plotted against "sample size."
State whether the horizontal axis counts sources, rows, or exposures.

Repeated exposure can still be valuable. It allows stochastic augmentation, visits rare
examples more often, and gives an optimizer time to converge. The mistake is interpreting
its benefit as new independent population coverage.

### Conceptual checkpoint

Changing from 20 to 100 epochs multiplies exposure by five. It leaves participant count,
session count, source diversity, and the participant-level standard-error denominator
unchanged.

## 5. Nested sampling pools imply weights

Suppose participant $p$ contributes $n_p$ rows with values $y_{pi}$. A global row mean is

$$
\bar y_{\mathrm{row}}=\frac{1}{\sum_p n_p}
\sum_p\sum_{i=1}^{n_p}y_{pi}.
$$

This gives participant $p$ weight proportional to $n_p$. It estimates the mean of a
randomly selected row from the collected row pool.

A participant-balanced mean first computes

$$
\bar y_p=\frac{1}{n_p}\sum_{i=1}^{n_p}y_{pi}
$$

and then averages participants:

$$
\bar y_{\mathrm{participant}}=\frac{1}{P}\sum_{p=1}^{P}\bar y_p.
$$

This gives every participant weight $1/P$ regardless of row count. It estimates the mean
for a uniformly selected sampled participant. Neither estimator is automatically correct.
The target population and estimand determine the weighting rule.

### Worked weighting example

Participant A has two scores, both 0. Participant B has eight scores, all 10. The row mean
is 8 because B supplies 80 percent of rows. The participant-balanced mean is 5 because the
two participant means, 0 and 10, receive equal weight.

The same logic can be applied at several levels: average windows within clips, clips within
sessions, sessions within participants, and then participants. Each stage declares which
level receives equal weight.

## 6. Randomization, sampling, and analysis units

The **randomization unit** is independently assigned to an experimental condition. The
**sampling unit** is independently drawn from a target population. The **analysis unit**
contributes an independent contrast or random effect to uncertainty estimation.

These units can coincide, but they need not. In a participant-randomized trial, participant
is the randomization unit. Frames are repeated outcomes. In a clinic-randomized trial, the
clinic is the randomization unit even when thousands of patients are measured.

Calling a row "independent" because it is stored separately is not enough. Independence
comes from the sampling and assignment process. The analysis must preserve the independent
unit implied by that process rather than promoting nested rows to independent units.

The three units answer different questions. Randomization supports a treatment comparison,
sampling supports a target-population claim, and the analysis unit determines how
uncertainty is computed. In a cluster-randomized study that samples clinics, randomizes
clinics, and measures patients, the clinic can be all three units while patients remain
nested outcomes. In other designs the three levels can differ.

## 7. Participant and model replication are different axes

Machine-learning experiments vary because of sampled participants and because training is
stochastic. Different random initializations, data orders, and augmentations produce
different fitted models.

Let $Y_{pm}$ be a score for participant $p$ evaluated with model seed $m$. The score table
has shape `(P, M)`. A simple crossed random-effects description is

$$
Y_{pm}=\mu+a_p+b_m+e_{pm}.
$$

$\mu$ is the grand mean. $a_p$ is participant deviation, $b_m$ is model-seed deviation,
and $e_{pm}$ is remaining interaction and measurement variation. Participants and model
seeds are crossed because every model is evaluated on every participant.

Calling model seeds a second replication axis does not make them new data samples. They
measure variability of the training procedure conditional on the dataset and protocol.
Each seed should correspond to a genuinely separate training run; merely evaluating the
same fitted weights twice does not create a model replicate.

Assume these deviations have variances $\sigma_a^2$, $\sigma_b^2$, and $\sigma_e^2$.
The variance of the grand mean is approximately

$$
\mathrm{Var}(\bar Y)=\frac{\sigma_a^2}{P}
+\frac{\sigma_b^2}{M}+\frac{\sigma_e^2}{PM}.
$$

This decomposition assumes a complete balanced crossing, centered independent random
effects, and an estimand that averages over the participant and training-run populations
represented by those effects. The residual term includes participant-by-run interaction.
With missing cells, dependence among runs, fixed hand-picked models, or additional repeated
measurements, the formula needs a model that reflects those features.

$P$ is participant count and $M$ is model count. More model seeds reduce the second and
third terms but do not reduce the participant term. More participants reduce the first and
third terms but do not reduce the model-seed term. Independent replication on both axes is
needed when both are part of the generalization claim; a balanced crossing makes the
components easier to identify and estimate.

## 8. Variance components explain shared dependence

For nested measurements, a useful model is

$$
Y_{pst}=\mu+a_p+b_{ps}+e_{pst}.
$$

$Y_{pst}$ is trial $t$ from session $s$ of participant $p$. The participant effect $a_p$
is shared by every row from participant $p$. The session effect $b_{ps}$ is shared by rows
from that participant-session. The residual $e_{pst}$ varies at the trial level.

![Observed variation comes from several levels](../images/15_variance_components.svg)

If the three effects are independent with variances $\sigma_a^2$, $\sigma_b^2$, and
$\sigma_e^2$, total row variance is their sum:

$$
\mathrm{Var}(Y)=\sigma_a^2+\sigma_b^2+\sigma_e^2.
$$

Two rows from the same participant share $a_p$, so they are correlated. Two rows from the
same session share both $a_p$ and $b_{ps}$ and are usually even more correlated.

Variance components can be estimated with mixed-effects models, restricted maximum
likelihood, Bayesian hierarchical models, or method-of-moments formulas in balanced
designs. Components estimated from small numbers of groups are uncertain. A component
clipped to zero by a simple method is not proof that the true variation is absent.

## 9. Intraclass correlation and design effect

For a one-level cluster model with between-cluster variance $\sigma_a^2$ and residual
variance $\sigma_e^2$, the intraclass correlation is

$$
\rho=\frac{\sigma_a^2}{\sigma_a^2+\sigma_e^2}.
$$

$\rho$ is the correlation induced by sharing a cluster effect. With equal cluster size
$m$, a common approximation to variance inflation is

$$
\mathrm{DE}=1+(m-1)\rho.
$$

The letters DE mean design effect. A rough effective sample-size intuition is

$$
N_{\mathrm{eff}}\approx\frac{N_{\mathrm{row}}}{\mathrm{DE}}.
$$

If $m=20$, $\rho=0.5$, and there are 200 rows, then
$\mathrm{DE}=1+19\times0.5=10.5$ and $N_{\mathrm{eff}}\approx19$. This approximation
is a warning about dependence, not a replacement for a cluster-aware model or bootstrap.
Unequal cluster sizes and multiple nesting levels require more careful treatment.

## 10. Leakage follows shared information

Leakage occurs when information unavailable at deployment enters training or model
selection. Group leakage is especially common in windowed data. If windows from one clip
or participant appear in both train and test sets, the model can exploit shared identity,
background, sensor, or neighboring frames.

Split groups before creating windows. Fit normalization, feature selection, imputation,
and augmentation policies using training data only. A participant-disjoint split is needed
for claims about new participants. A site-disjoint split is needed for stronger claims
about new sites.

Duplicate hashes are useful but insufficient. Two files can be different at every byte and
still share a source recording or participant artifact. Lineage identifiers must reflect
the true dependence path.

### Leakage audit questions

Ask whether any test participant, session, clip, or temporal neighbor appears in training.
Ask whether preprocessing statistics saw test rows. Ask whether hyperparameters were
repeatedly changed after inspecting final-test performance. Every "yes" narrows what the
test result can honestly claim.

## 11. Complete-case analysis changes the population

A paired analysis often keeps only participants observed in both conditions. Let $R_i=1$
when participant $i$ is complete and 0 otherwise. The complete-case mean estimates

$$
E[D_i\mid R_i=1],
$$

the mean difference among complete participants. It equals the target population mean only
under missingness assumptions.

The symbol $E$ denotes a population average. The vertical bar means "among units satisfying
the condition," so this expression explicitly limits the target to participants with
$R_i=1$.

If missingness is unrelated to outcomes and covariates, complete cases may remain
representative. If difficult participants are more likely to miss an intervention session,
the retained sample can look easier and the estimate can be biased.

Always report the flow from eligible units to complete units, reasons for missingness, and
baseline differences between included and excluded groups. Sensitivity analysis can fill
missing participant contrasts over a plausible range to show how conclusions change.

### Worked missingness example

Suppose 40 participants are eligible and 8 difficult participants lack intervention data.
The complete-case analysis has 32 participants, not 72 observations. If those 8 would have
shown smaller effects, the complete-case mean overstates population benefit. More bootstrap
replicates cannot recover their unobserved outcomes.

## 12. Prospective locking protects the claim

**Prospective locking** records the analysis before inspecting confirmatory outcomes. A
useful locked manifest includes:

- target population and primary estimand;
- randomization, sampling, and analysis units;
- train, validation, calibration, and test group assignments;
- preprocessing and exclusion rules;
- primary metric and sign convention;
- bootstrap hierarchy, replicate count, and random seed;
- equivalence margins and multiplicity family;
- model-seed count and aggregation rule; and
- stopping and sensitivity-analysis rules.

A versioned file and cryptographic hash make later changes visible. Locking does not make a
bad design good. It separates confirmatory decisions from exploratory learning and prevents
quiet outcome-dependent changes.

## 13. Generalization is multidimensional

"Generalizes" is incomplete without naming what is new. A model can generalize to new
windows from known clips, new clips from known participants, new participants at known
sites, new devices, new sites, or new model-training randomness. These are different axes.

A held-out axis tests transfer to the particular held-out units. A broader population claim
also requires a defensible sampling frame, enough independent breadth, and exchangeability
assumptions or an explicit model. Merely labeling an effect random does not create missing
sites or participants. Participant replication says little about model-seed instability,
and many model seeds say little about a dataset containing one site.

State the supported claim narrowly. For example: "mean performance across new participants
from the same acquisition sites, averaged over the six sampled model seeds." That sentence
reveals both what varies and what remains fixed.

## 14. Worked end-to-end count and uncertainty example

Thirty participants are evaluated with six model seeds. Each participant supplies 100
windows, so there are 3,000 window rows per model and 18,000 evaluated row exposures across
models. The score table for crossed inference is still `(30, 6)` after aggregating windows
to participant-model scores.

Suppose estimated variance components are participant 1.00, model 0.16, and residual 0.36.
The grand-mean variance is

$$
\frac{1.00}{30}+\frac{0.16}{6}+\frac{0.36}{30\times6}
=0.062.
$$

The standard error is $\sqrt{0.062}\approx0.249$. Doubling model seeds reduces the model
term but leaves $1/30$ unchanged. Doubling participants reduces the participant term but
leaves $0.16/6$ unchanged. The component formula makes the replication bottleneck visible.

## 15. Efficiency and API notes

Keep identifier columns in compact integer form and use grouped reductions. NumPy
`np.unique(..., return_inverse=True)` plus `np.bincount` can compute group sums quickly.
Pandas `groupby` is convenient for manifests. Scikit-learn group splitters enforce group
separation, but you must still choose the correct group key.

For variance components, prefer established mixed-model software when inference matters.
Use simulations to verify that aggregation and resampling reproduce the intended hierarchy.
Store random seeds and exact split maps, not only a verbal description.

## 16. Misconceptions and failure modes

1. **"More epochs increase sample size."** They increase exposure, not source replication.
2. **"Every window is independent."** Overlap and shared source effects create dependence.
3. **"Six seeds solve participant uncertainty."** They address a different variance axis.
4. **"Effective sample size is exact."** The design-effect formula is an approximation.
5. **"Complete cases estimate everyone."** They estimate the retained subpopulation unless
   missingness assumptions justify more.
6. **"No duplicate files means no leakage."** Shared lineage can leak without byte copies.
7. **"One held-out split proves all generalization."** It tests only its held-out axes.
8. **"Preregistration prevents every bias."** It clarifies decisions but cannot fix poor
   measurement, attrition, or unrepresentative sampling.

## Exercises

### Exercise 1

One hundred sources create four augmented rows each and train for 25 epochs. Find source,
row, and exposure counts.

**Brief solution:** 100 sources, 400 rows, and 10,000 exposures.

### Exercise 2

For 10 rows per cluster and intraclass correlation 0.3, compute the design effect.

**Brief solution:** $1+(10-1)\times0.3=3.7$.

### Exercise 3

Why does adding model seeds not remove a participant-variance floor?

**Brief solution:** the term $\sigma_a^2/P$ contains no model count. Only more independent
participants reduce it.

### Exercise 4

A study splits windows randomly while every participant appears in both partitions. What
generalization does the test measure?

**Brief solution:** at best it measures new windows from known participants and sources,
not transfer to unseen participants.

### Exercise 5

What population does a complete-case paired mean directly describe?

**Brief solution:** participants who satisfy the completeness rule, written
$E[D_i\mid R_i=1]$.

## Recap

Row count, source count, exposure, and replication are different quantities. Nested pools
encode weighting choices, and independent units come from design rather than file layout.
Participant and model seeds contribute separate variance components. Leakage, missingness,
and unsupported extrapolation can invalidate a precise-looking result. A prospective,
unit-aware plan makes the evidence and its limits auditable.

## Continue

- Previous: [14. Paired contrasts, uncertainty, and decision thresholds](14_paired_inference.md)
- Notebook: [15. Exposure, replication, and variance decomposition](../implementations/15_exposure_and_replication.ipynb)
- Curriculum: [Tutorial README](../README.md)
