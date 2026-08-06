# 14. Paired contrasts, uncertainty, and decision thresholds

![Overview of paired inference](../images/14_paired_inference.svg)

## Prerequisites

You should understand means, standard deviations, sampling, and participant-level grouping.
Review [13. Context interventions and identity geometry](13_context_interventions.md) for
matched contrasts and held-out evaluation.

## Learning goals

By the end of this lesson, you will be able to:

1. reduce paired repeated measurements to independent participant contrasts;
2. compute and interpret a paired Student $t$ interval;
3. distinguish participant and hierarchical bootstraps;
4. connect interval width to effect resolution;
5. understand noncentral $t$ power conceptually;
6. separate superiority from equivalence decisions; and
7. apply Holm correction to a declared family of hypotheses.

## 1. Motivating scenario: did the intervention help the same people?

Twenty-four participants each complete a baseline condition and an intervention condition.
Every condition contains ten repeated trials. Participant A generally scores higher than
participant B in both conditions. If we compare all baseline rows with all intervention
rows as though they were unrelated, stable participant differences inflate noise and the
row count exaggerates evidence.

Pairing uses a simpler mental model: give every participant one arrow from baseline to
intervention. Analyze the arrow lengths, not the cloud of raw endpoints. Stable participant
baselines cancel when we subtract within person.

## 2. Units and shapes come before a statistical test

Let $P$ be the number of participants and $T$ the number of repeated trials per condition.
Store baseline measurements in $Y^{(0)}$ with shape `(P, T)` and intervention measurements
in $Y^{(1)}$ with the same shape. Superscripts 0 and 1 label conditions.

For participant $i$, first average trials within each condition:

$$
\bar Y_i^{(c)}=\frac{1}{T}\sum_{t=1}^{T}Y_{it}^{(c)}.
$$

The index $c$ is the condition, $t$ is the trial, and the bar denotes a trial mean.
Define one paired difference per participant:

$$
D_i=\bar Y_i^{(1)}-\bar Y_i^{(0)}.
$$

Positive $D_i$ means improvement under the chosen sign convention. The analysis vector
$D=(D_1,\ldots,D_P)$ has shape `(P,)`. This is the key reduction: hundreds of trial rows
become $P$ participant-level contrasts.

![Paired differences remove participant baselines](../images/14_paired_difference.svg)

The participant is the analysis unit when participants are independently sampled and the
claim concerns participant-average change. If treatment was assigned by classroom, clinic,
or site, the independent randomization unit may be larger. No test can repair a unit chosen
incorrectly at the design stage.

### Conceptual checkpoint

With 24 participants, 2 conditions, and 10 trials per condition, there are 480 raw rows
but only 24 participant contrasts. Repeated trials can estimate each participant's mean
more precisely. They do not create 480 independent participants.

## 3. Why pairing can reduce noise

Write a simple measurement model:

$$
Y_{it}^{(c)}=\mu_c+a_i+e_{it}^{(c)}.
$$

$\mu_c$ is the condition mean, $a_i$ is participant $i$'s stable baseline, and
$e_{it}^{(c)}$ is trial-level deviation. Subtracting condition means within participant
removes $a_i$:

$$
D_i=(\mu_1-\mu_0)+(\bar e_i^{(1)}-\bar e_i^{(0)}).
$$

The paired difference focuses on condition change instead of between-person level. Pairing
helps most when the two condition measurements are positively correlated within person.
Incorrectly pairing unrelated rows can instead create bias or extra noise.

## 4. The paired $t$ interval

Once the $P$ differences are formed, the paired analysis is a one-sample analysis of $D$.
The sample mean difference is

$$
\bar D=\frac{1}{P}\sum_{i=1}^{P}D_i.
$$

The sample standard deviation of participant differences is

$$
s_D=\sqrt{\frac{1}{P-1}\sum_{i=1}^{P}(D_i-\bar D)^2}.
$$

$s_D$ measures how much participant responses vary. The standard error of the mean is

$$
\mathrm{SE}=\frac{s_D}{\sqrt{P}}.
$$

The square-root denominator reflects averaging independent participant contrasts. It must
not use the raw trial-row count.

For confidence level $1-\alpha$, a two-sided Student $t$ interval is

$$
\bar D\ \pm\ t_{1-\alpha/2,\,P-1}\mathrm{SE}.
$$

$t_{1-\alpha/2,\,P-1}$ is a quantile of the Student $t$ distribution with $P-1$ degrees
of freedom. The interval is wider than a normal interval for small samples because the
population standard deviation is estimated.

### Worked numerical interval

Take five participant differences: 0.10, 0.20, 0.15, 0.05, and 0.25. Their mean is 0.15.
The sample standard deviation is about 0.079, so the standard error is
$0.079/\sqrt{5}\approx0.035$.

For a 95 percent interval with 4 degrees of freedom, the critical value is about 2.776.
The half-width is $2.776\times0.035\approx0.098$. The interval is approximately
`[0.052, 0.248]`.

This interval describes uncertainty in the population mean paired difference under the
sampling assumptions. It does not say that 95 percent of participant differences lie in
that range, and it does not assign a 95 percent probability to this fixed interval under
the usual frequentist interpretation.

```python
import numpy as np
from scipy import stats

def paired_t_interval(differences, confidence=0.95):
    d = np.asarray(differences, dtype=float)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    critical = stats.t.ppf((1 + confidence) / 2, df=len(d) - 1)
    return mean, (mean - critical * se, mean + critical * se)
```

The distinction between exact and approximate validity matters. If participant differences
are independent and identically distributed normal draws with nonzero finite variance,
the usual standardized pivot has an exact Student $t$ distribution with $P-1$ degrees of
freedom. Under those assumptions, the displayed interval has exact model-based coverage.

If the differences are independent and identically distributed with finite variance but
are not normal, the same interval is generally an asymptotic approximation justified by
the behavior of the sample mean and variance as $P$ grows. Its small-sample robustness can
fail under strong skew, heavy tails, or influential outliers. Plot the participant
differences and report a sensitivity analysis rather than treating the word "paired" as
an automatic guarantee.

### Unequal trial counts and participant weights

Participants may contribute different numbers of valid trials. Computing each participant's
condition mean from their available trials and then averaging $D_i$ gives every participant
equal weight. Pooling all valid trial differences instead gives participants with more
trials more weight and treats dependent trials as separate analysis units.

Equal participant weighting usually matches a population mean over participants. Precision
weighting can be justified when participant means have very different known measurement
variances, but estimated weights add assumptions and can correlate with participant
difficulty. State the estimand and weighting policy before examining outcomes.

Plot the participant differences. The $t$ interval concerns their mean, not whether the
raw differences form a perfect bell curve. With a moderate number of independent units,
the sample mean can be approximately normal even when individual differences are not.
With very few units, one extreme participant can control both the mean and standard
deviation. Report robust summaries and leave-one-participant-out sensitivity alongside the
primary analysis when such influence is plausible.

## 5. The participant percentile bootstrap

The bootstrap approximates repeated sampling by resampling observed units. For paired
inference, resample participant differences, not individual condition rows.

This participant bootstrap assumes participants are the independent, exchangeable sampling
units for the target population. If clinics or sites were sampled or randomized as intact
clusters, resample those clusters and carry all of their participants together. Resampling
participants inside a cluster as though they were independent does not repair a design at
the cluster level.

One bootstrap replicate draws $P$ indices with replacement from `0` through `P-1` and
computes the mean of those selected differences. Repeat this $B$ times to obtain bootstrap
means $\bar D_1^{\ast},\ldots,\bar D_B^{\ast}$.

A 95 percent percentile interval uses the 2.5th and 97.5th percentiles:

$$
[q_{0.025},q_{0.975}].
$$

$q_p$ is the empirical $p$-quantile of bootstrap means. The method is intuitive and does
not impose a normal shape directly, but it is not assumption-free. Small samples provide
few distinct units, and the basic percentile method can have biased coverage.

```python
def participant_bootstrap(d, replicates, rng):
    d = np.asarray(d)
    indices = rng.integers(0, len(d), size=(replicates, len(d)))
    return d[indices].mean(axis=1)
```

The index array has shape `(B, P)`. This vectorized implementation is fast for moderate
$B$ and $P$. For large products, generate replicates in chunks to limit memory.

## 6. Hierarchical bootstrap for nested trials

Participant-level differences already summarize trials. Sometimes we also want uncertainty
from trial sampling. A hierarchical bootstrap mirrors the sampling structure:

1. resample participants with replacement;
2. inside each selected participant, resample paired trial indices with replacement;
3. recompute that participant's difference; and
4. average the resampled participant differences.

When baseline and intervention trial $t$ share a stimulus or time point, resample the same
trial index in both conditions. Drawing them independently destroys their covariance and
changes the estimand.

The hierarchical bootstrap answers a broader repeated-sampling question than resampling
fixed participant summaries. It is useful only when the nested resampling levels correspond
to real sampling or generalization levels. Resampling arbitrary computational rows can
recreate pseudoreplication rather than solve it.

Within-participant trial resampling also assumes trials are exchangeable draws from the
trial population named by the estimand. Serially adjacent or overlapping trials may need
block resampling, and a fixed set of deliberately chosen stimuli may be better treated as
fixed rather than resampled. The bootstrap hierarchy must mirror how new units could
actually arise.

### Which bootstrap should I use?

Use participant resampling when inference treats each participant contrast as the complete
unit. Add within-participant trial resampling when trials represent a meaningful sampled
population and trial variability belongs in the target uncertainty. State the target
explicitly because the two procedures need not produce the same interval.

## 7. Effect resolution and interval half-width

An experiment may be designed to estimate the mean within a desired half-width $h$. Under
a rough normal approximation,

$$
h\approx z_{1-\alpha/2}\frac{\sigma_D}{\sqrt{P}}.
$$

$\sigma_D$ is the anticipated population standard deviation of paired differences, and
$z_{1-\alpha/2}$ is a normal quantile. Solving for participant count gives

$$
P\approx\left(\frac{z_{1-\alpha/2}\sigma_D}{h}\right)^2.
$$

Halving the target half-width requires about four times as many independent participants.
Adding trials can reduce $\sigma_D$ when participant means are noisy, but it cannot remove
true between-participant variation.

For small planned $P$, replace the normal quantile with a $t$ quantile and solve
iteratively because the quantile itself depends on $P-1$.

## 8. Power and the noncentral $t$ distribution

Power is the probability that a predeclared test rejects its null hypothesis when a
specific alternative is true. Let the population mean paired effect be $\delta$ and the
population standard deviation of differences be $\sigma_D$. The standardized effect is

$$
d=\frac{\delta}{\sigma_D}.
$$

For $P$ independent pairs, the noncentrality parameter is

$$
\lambda=\sqrt{P}\,d=\frac{\sqrt{P}\delta}{\sigma_D}.
$$

$\lambda$ measures how far the test statistic shifts under the alternative. Power is a
tail probability of a noncentral $t$ distribution with $P-1$ degrees of freedom and
noncentrality $\lambda$.

This noncentral $t$ calculation is exact for the classical test under independent,
identically distributed normal differences with standardized effect $d$. For nonnormal
or more complex clustered designs it is a planning approximation; simulation from a
realistic data-generating model may be more defensible.

Power calculations are only as credible as their inputs. An effect selected from a noisy
pilot is often too optimistic. Examine a range of plausible effects and standard deviations.
Power does not measure the probability that the hypothesis is true after seeing data.

Power and precision answer related but distinct planning questions. A power target asks
how often a threshold decision succeeds for one assumed effect. A resolution target asks
how narrow the estimate should be regardless of which effect is observed. When scientific
interpretation depends on effect size rather than only rejection, planning for precision is
often more transparent.

## 9. Superiority asks whether the effect is positive

For a one-sided level $\alpha$ superiority claim, the null allows nonpositive effects and
the alternative is positive. A lower confidence bound is

$$
L_{\mathrm{sup}}=\bar D-t_{1-\alpha,\,P-1}\mathrm{SE}.
$$

Declare superiority when $L_{\mathrm{sup}}>0$. The sign convention must be fixed before
analysis. If lower loss is better, define differences so improvement is positive or reverse
the inequality consistently.

Failure to show superiority means the data did not establish a positive effect at the
chosen threshold. It does not prove that the effect is zero or practically negligible.

## 10. Equivalence asks whether the effect is small enough

Equivalence begins with a practical margin $\Delta>0$. Effects between $-\Delta$ and
$+\Delta$ are considered too small to matter for the stated application.

Two one-sided tests, commonly called TOST, reject both nonequivalence regions. At level
$\alpha=0.05$, an equivalent interval rule is that the two-sided 90 percent confidence
interval lies entirely inside $(-\Delta,+\Delta)$.

![Superiority and equivalence decision regions](../images/14_decision_regions.svg)

The margin must have domain meaning. Choosing it after seeing the interval turns the
decision into a moving target. A narrow interval around a large positive effect can show
superiority but not equivalence. A narrow interval around a small positive effect can show
both superiority and equivalence. These claims are not logical opposites.

### Worked decision example

Suppose a 90 percent interval is `[0.03, 0.17]` and the equivalence margin is 0.20. The
interval is above zero, so a corresponding one-sided superiority test may succeed. It is
also entirely inside `[-0.20, 0.20]`, so equivalence may succeed. The interpretation is a
reliably positive effect that is still practically small under the declared margin.

## 11. Holm correction for multiple hypotheses

Testing many outcomes increases the chance of at least one false rejection. Holm's method
controls the family-wise error rate for a declared family of $m$ hypotheses.

Sort raw p-values:

$$
p_{(1)}\leq p_{(2)}\leq\cdots\leq p_{(m)}.
$$

Compare $p_{(j)}$ with $\alpha/(m-j+1)$ in order. Stop at the first comparison that fails;
that hypothesis and all later ones are not rejected. The ordered adjusted p-values can be
constructed from scaled values $(m-j+1)p_{(j)}$ followed by a cumulative maximum and a
cap at 1.

The hypothesis family must be declared. Combining every exploratory metric into one giant
family can be unnecessarily conservative, while splitting one confirmatory family after
seeing results defeats error control.

### Small Holm example

For raw p-values 0.004, 0.018, 0.041, and 0.30 at $\alpha=0.05$, compare them with
0.0125, about 0.0167, 0.025, and 0.05. The first passes. The second fails, so Holm stops
and only the first hypothesis is rejected.

## 12. Efficiency and reproducibility notes

Compute participant contrasts once and keep participant identifiers beside them. Vectorize
the simple bootstrap, but use chunks when `B*P` indices would be large. A hierarchical
bootstrap often needs loops, which is acceptable when the code mirrors the design clearly.

Use an explicit `numpy.random.Generator` and record its seed. Record bootstrap replicate
count, interval type, direction of improvement, equivalence margin, and multiplicity family.
Monte Carlo endpoints vary slightly across seeds, so use enough replicates for the claimed
precision and report that simulation error exists.

## 13. Misconceptions and failure modes

1. **"Every trial is an independent pair."** Participant-level dependence remains.
2. **"A nonsignificant result proves no effect."** Absence of evidence is not equivalence.
3. **"Bootstrap means assumption-free."** The resampling hierarchy is an assumption.
4. **"More bootstrap replicates fix a small sample."** They reduce Monte Carlo noise, not
   missing population information.
5. **"Power is the chance the alternative is true."** It is a long-run rejection
   probability under a specified alternative.
6. **"Any equivalence margin is acceptable."** The margin must be justified before data.
7. **"Holm can be applied after selecting promising outcomes."** Selection changes the
   family and invalidates the planned error guarantee.

## Exercises

### Exercise 1

Participant differences have standard deviation 0.40 and $P=64$. What is the standard
error of their mean?

**Brief solution:** $0.40/\sqrt{64}=0.05$.

### Exercise 2

If target interval half-width is cut from 0.10 to 0.05 with all else fixed, how does the
approximate participant requirement change?

**Brief solution:** it grows by a factor of four.

### Exercise 3

Why must matched baseline and intervention trial indices be resampled together?

**Brief solution:** separate resampling destroys within-trial covariance that pairing is
intended to preserve.

### Exercise 4

A 90 percent interval is `[-0.08, 0.07]` and $\Delta=0.10$. What can be concluded?

**Brief solution:** it lies entirely inside `[-0.10, 0.10]`, so equivalence succeeds at
the corresponding 0.05 TOST level. It does not show positive superiority.

## Recap

Paired inference begins by creating one contrast per independent unit. The $t$ interval,
participant bootstrap, and hierarchical bootstrap represent different uncertainty models.
Interval width expresses resolution, while power describes rejection probability under a
specified alternative. Superiority compares with zero, equivalence compares with a practical
band, and Holm correction protects a declared family of decisions.

## Continue

- Previous: [13. Context interventions and identity geometry](13_context_interventions.md)
- Notebook: [14. Paired contrasts and uncertainty](../implementations/14_paired_inference.ipynb)
- Next: [15. Exposure, replication, and variance decomposition](15_exposure_and_replication.md)
- Curriculum: [Tutorial README](../README.md)
