# 10. Regularized linear estimation and calibration

![Overview of regularized linear estimation and temperature scaling](../images/10_regularized_linear_estimation.svg)

## Why this lesson matters

Suppose you have learned representation vectors for 2,000 observations. You want to answer two questions:

1. Can a simple linear model recover the target from those features?
2. Can the model's probability scores be used responsibly?

A linear probe looks simple, but its result depends on scaling, category encoding, collinearity, regularization, class weighting, and data separation. Temperature scaling can improve held-out negative log likelihood, but that improvement alone does not prove empirical calibration.

This lesson builds the full pipeline from a mixed-feature table to a regularized estimator and temperature-scaled probabilities.

## Prerequisites

You should know algebra, vectors, matrix multiplication, means, and basic probability. [Lesson 09](09_eigenspectra_and_effective_rank.md) explains eigenvalues and feature geometry.

## Learning goals

By the end of this lesson, you will be able to:

1. Standardize numeric features using training data only.
2. One-hot encode categorical variables and handle unseen levels.
3. Derive ridge regression with an unpenalized intercept.
4. Explain how regularization improves conditioning.
5. Build binary logistic and multiclass softmax models.
6. Use class weights without confusing them with population probabilities.
7. Fit a temperature by held-out negative log likelihood and state what that does not prove.
8. Read `np.einsum` subscripts and implement a multi-output ridge map.
9. Compute class likelihoods in log space and optimize a positive temperature continuously.
10. Build the three leakage-safe two-output factor heads used by GFC-v2.

## 1. Begin with a mixed-feature scenario

Imagine predicting a medical outcome from:

- age in years,
- blood pressure in millimeters of mercury,
- a learned 128-dimensional representation,
- collection site as a categorical variable.

These inputs have different units and structures. Age and blood pressure are numeric. Site is a name, not a number with meaningful distance. Representation coordinates may be correlated.

A model matrix needs one numeric value in each cell. Feature construction determines what a coefficient means, so preprocessing is part of the statistical model.

## 2. Standardization makes numeric scales comparable

Let $X$ be a numeric training matrix with shape $(N,D)$. The integer $N$ is the number of training observations, and $D$ is the number of numeric features.

For feature $j$, compute the training mean:

$$
\mu_j=\frac{1}{N}\sum_{i=1}^{N}X_{ij}.
$$

Compute the training standard deviation:

$$
\sigma_j
{}={}
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(X_{ij}-\mu_j)^2
}.
$$

Transform any observation:

$$
\widetilde X_{ij}
{}={}
\frac{X_{ij}-\mu_j}{\sigma_j}.
$$

The transformed feature is measured in training standard deviations from the training mean. Use the same $\mu_j$ and $\sigma_j$ for validation, calibration, and test observations.

### Why scaling matters for regularization

Suppose one feature is measured in meters and another in millimeters. The millimeter coefficient will be roughly one thousand times smaller for the same physical effect. An L2 penalty sees coefficient size, so its treatment depends on units unless features are standardized.

### Failure at zero variance

If $\sigma_j=0$, feature $j$ is constant in training data. It supplies no variation for fitting. Remove it or use a transformer with an explicit safe-scale policy.

~~~python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
~~~

Fitting the scaler on all data leaks test means and variances into training.

## 3. One-hot encoding respects categories

Suppose site has levels north, central, and south. Encoding them as 0, 1, and 2 would imply that central lies halfway between north and south. That geometry is invented.

One-hot encoding creates indicator features:

$$
\text{north}\mapsto[1,0,0],
\quad
\text{central}\mapsto[0,1,0],
\quad
\text{south}\mapsto[0,0,1].
$$

Each column answers one yes-or-no question.

With an intercept, the three columns sum to the all-ones column. This exact dependence is called the dummy-variable trap. Two common strategies are:

- drop one reference level,
- keep all levels and rely on regularization or a solver convention.

If the north column is dropped, its effect becomes part of the intercept. Other site coefficients describe differences from north.

### Unseen categories

At deployment, a new site may appear. Scikit-learn's <code>handle_unknown="ignore"</code> maps it to an all-zero indicator block. If a reference level was also dropped, the new site becomes indistinguishable from the reference inside that block. This can be acceptable for robustness, but it is not a learned new-site effect.

Use a <code>ColumnTransformer</code> or a pipeline so scaling and encoding are fitted inside each training fold.

## 4. Linear regression begins with a weighted sum

For one observation $x$ with shape $(D,)$, a linear prediction is

$$
\widehat y=x^\mathsf{T}w+b.
$$

The vector $w$ has shape $(D,)$. The scalar $b$ is the intercept. Each coefficient converts one feature unit into prediction units.

For all observations, let $X$ have shape $(N,D)$ and $y$ have shape $(N,)$. Without an intercept for the moment, ordinary least squares minimizes

$$
L(w)=\lVert y-Xw\rVert_2^2.
$$

The residual vector $y-Xw$ has shape $(N,)$. Squaring its Euclidean norm sums squared prediction errors.

Expand the objective:

$$
L(w)
{}={}
y^\mathsf{T}y
-2w^\mathsf{T}X^\mathsf{T}y
+w^\mathsf{T}X^\mathsf{T}Xw.
$$

Differentiate with respect to $w$:

$$
\nabla_w L
{}={}
-2X^\mathsf{T}y
+2X^\mathsf{T}Xw.
$$

At a minimum, set the gradient to zero:

$$
X^\mathsf{T}Xw=X^\mathsf{T}y.
$$

These are the normal equations. If $X^\mathsf{T}X$ is invertible, they have a unique solution.

## 5. Collinearity makes estimation unstable

Suppose two representation features are almost copies:

$$
x_2\approx x_1.
$$

Many coefficient pairs can then produce nearly the same prediction. Increasing the first coefficient and decreasing the second can leave $x_1w_1+x_2w_2$ almost unchanged.

This creates a nearly flat direction in the objective. Tiny changes in data can produce large changes in individual coefficients.

For a symmetric positive-definite matrix $A$, the spectral condition number is

$$
\kappa(A)
{}={}
\frac{\lambda_{\max}(A)}
{\lambda_{\min}(A)}.
$$

The numerator is the largest eigenvalue and the denominator is the smallest eigenvalue, which is strictly positive in this case. A large ratio means some directions are much less constrained than others. If a positive-semidefinite matrix is singular, its usual condition number is infinite. Ignoring its zero eigenvalues would instead compute a pseudo-condition number.

## 6. Ridge regression adds a preference for smaller weights

Ridge minimizes

$$
L_{\mathrm{ridge}}(w)
{}={}
\lVert y-Xw\rVert_2^2
+\alpha\lVert w\rVert_2^2.
$$

The positive scalar $\alpha$ is regularization strength. The penalty is the sum of squared coefficients.

Differentiate:

$$
\nabla_w L_{\mathrm{ridge}}
{}={}
-2X^\mathsf{T}y
+2X^\mathsf{T}Xw
+2\alpha w.
$$

Set the gradient to zero:

$$
(X^\mathsf{T}X+\alpha I)w
{}={}
X^\mathsf{T}y.
$$

The identity matrix $I$ has shape $(D,D)$. Ridge adds $\alpha$ to every eigenvalue of $X^\mathsf{T}X$. Small eigenvalues receive the largest relative change. In this no-intercept derivation, the regularized condition number is

$$
\frac{\lambda_{\max}+\alpha}{\lambda_{\min}+\alpha}.
$$

This remains finite when $\lambda_{\min}=0$. The next section handles an intercept without penalizing it.

![Ridge lifts weakly constrained directions and improves conditioning](../images/10_ridge_conditioning.svg)

### Numerical conditioning example

Suppose eigenvalues of $X^\mathsf{T}X$ are 100 and 0.01. The condition number is

$$
\frac{100}{0.01}=10{,}000.
$$

With $\alpha=1$, the eigenvalues become 101 and 1.01. The new condition number is

$$
\frac{101}{1.01}=100.
$$

Ridge greatly improves stability. The tradeoff is shrinkage bias: coefficients move toward zero.

## 7. Treat the intercept separately

Append a column of ones:

$$
X_a=
\begin{bmatrix}
\mathbf{1} & X
\end{bmatrix}.
$$

The augmented matrix $X_a$ has shape $(N,D+1)$. Let

$$
\theta=
\begin{bmatrix}
b\\
w
\end{bmatrix}
$$

contain the intercept followed by feature coefficients.

Usually, do not penalize $b$. Use a penalty matrix

$$
P=\mathrm{diag}(0,1,\ldots,1).
$$

Solve

$$
(X_a^\mathsf{T}X_a+\alpha P)\theta
{}={}
X_a^\mathsf{T}y.
$$

The leading zero leaves the intercept unpenalized. Penalizing it would shrink the overall outcome level toward zero, which is usually unrelated to model complexity.

~~~python
def ridge_with_intercept(x, y, alpha):
    xa = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(xa.shape[1])
    penalty[0, 0] = 0.0
    system = xa.T @ xa + alpha * penalty
    return np.linalg.solve(system, xa.T @ y)
~~~

Use <code>np.linalg.solve</code> instead of explicitly computing a matrix inverse. Direct solvers are more efficient and usually more stable.

### Multi-output ridge and `np.einsum`

A factor head or multivariate regression can predict $K$ outputs at once. Replace target
vector $y$ with target matrix $Y$ of shape `(N,K)` and coefficient vector $w$ with
coefficient matrix $W$ of shape `(D,K)`. After centering the inputs and targets, solve

$$
(X^\mathsf{T}X+\alpha I)W=X^\mathsf{T}Y.
$$

The left side is one `(D,D)` system shared by all outputs. The right side has shape
`(D,K)`, and `np.linalg.solve` returns all $K$ coefficient columns in one call. The
intercept is recovered from the means rather than penalized:

$$
b=\bar y-\bar xW.
$$

NumPy's `einsum` expresses these contractions with named index letters:

```python
gram = np.einsum("ni,nj->ij", centered_x, centered_x, dtype=np.float64)
rhs = np.einsum("ni,nk->ik", centered_x, centered_y, dtype=np.float64)
weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
predictions = np.einsum("nd,dk->nk", standardized_x, weights, optimize=False) + intercept
```

Each letter names an axis. In `"nd,dk->nk"`, both inputs contain `d`, so the feature axis
is multiplied and summed away. The output retains observation axis `n` and output axis
`k`. In `"ni,nj->ij"`, observation axis `n` is contracted while two feature axes remain,
producing the Gram matrix. Letters are local labels, not fixed NumPy keywords.

The arrow is valuable documentation because it states the output order explicitly. Without
an arrow, NumPy orders surviving indices alphabetically, which can obscure intent. Reusing
a letter inside one operand selects a diagonal, while omitting a letter from the output
sums over it. Those are powerful operations, but an accidental repeated or omitted letter
can compute a plausible array with the wrong meaning.

For these two-operand contractions, ordinary matrix multiplication is equally valid:

```python
gram = centered_x.T @ centered_x
rhs = centered_x.T @ centered_y
predictions = standardized_x @ weights + intercept
```

Prefer `@` for a familiar two-dimensional matrix product. Prefer `einsum` when named axes
make a batched or multi-output contraction easier to audit. Pass an explicit accumulation
`dtype` when precision matters. `optimize=False` fixes the direct contraction choice;
optimized paths can improve larger multi-operand expressions, but their intermediate
order should be benchmarked and tested rather than assumed.

Shape reasoning comes before execution. If `centered_x` is `(N,D)` and `centered_y` is
`(N,K)`, then `"ni,nk->ik"` must be `(D,K)`. Predicting that shape by hand catches many
subscript mistakes before numeric assertions are considered.

For GFC-v2, multi-output means two outputs within one binary factor. It does not mean one
six-output head for all factors. Fit three separate maps,

$$
h_f(z)=zW_f+b_f,
\qquad f\in\{\text{speed},\text{clothing},\text{direction}\},
$$

where each $W_f$ has shape `(D,2)`. The two columns score the two levels of factor $f$.
Separate heads make each score block auditable and let the evaluator normalize each block
with its own development statistics.

## 8. Logistic regression turns scores into probabilities

For binary classification, compute a logit:

$$
z=x^\mathsf{T}w+b.
$$

The sigmoid maps any real number to $(0,1)$:

$$
p
{}={}
\frac{1}{1+\exp(-z)}.
$$

Interpret $p$ as the model's score for class 1. As $z$ becomes large and positive, $p$ approaches one. As $z$ becomes large and negative, $p$ approaches zero.

For label $y\in\{0,1\}$, negative log likelihood is

$$
\ell
{}={}
-y\log p
-(1-y)\log(1-p).
$$

If $y=1$, only $-\log p$ remains. If $y=0$, only $-\log(1-p)$ remains. Confident incorrect scores receive a large penalty.

The logit derivative simplifies to

$$
\frac{\partial \ell}{\partial z}=p-y.
$$

This says the update is driven by the difference between predicted probability and observed label.

### A binary numerical example

Let $z=1.386$. Because $\exp(-1.386)\approx0.25$, sigmoid probability is approximately

$$
p=\frac{1}{1+0.25}=0.8.
$$

If the observed label is $y=1$, negative log likelihood is $-\log(0.8)\approx0.223$. If the label is $y=0$, the loss is $-\log(0.2)\approx1.609$. The same confident score receives a much larger penalty when it is wrong.

Changing the classification threshold from 0.5 changes predicted labels, but it does not refit probabilities. Threshold selection and probability estimation are separate decisions.

## 9. Multiclass softmax

For $K$ classes, compute one logit per class:

$$
z=
\begin{bmatrix}
z_1 & z_2 & \cdots & z_K
\end{bmatrix}.
$$

Softmax gives

$$
p_k
{}={}
\frac{\exp(z_k)}
{\sum_{j=1}^{K}\exp(z_j)}.
$$

Every probability is positive and the probabilities sum to one.

Exponentials can overflow for large logits. Subtract the maximum logit $m=\max_j z_j$:

$$
p_k
{}={}
\frac{\exp(z_k-m)}
{\sum_j\exp(z_j-m)}.
$$

Adding or subtracting the same constant from every logit does not change softmax probabilities.

~~~python
def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)
~~~

## 10. Regularized logistic estimation

Logistic regression has no ordinary closed-form solution. Iterative solvers minimize summed or averaged negative log likelihood plus an L2 penalty.

Different libraries define regularization strength differently. Scikit-learn commonly uses $C$, where smaller $C$ means stronger regularization. Do not move a numeric value between libraries without checking objective scaling.

A linear probe tests linearly accessible information. Failure can mean the representation lacks the information, or that the information is present in a nonlinear form. Success does not prove that the feature is causal or robust.

## 11. Class weighting changes the objective

Suppose class 1 is rare. A weighted objective is

$$
L
{}={}
-\sum_{i=1}^{N}
a_{y_i}\log p_{i,y_i}.
$$

The positive scalar $a_{y_i}$ is the weight for the observed class of example $i$. Inverse-frequency weights increase the contribution of rare-class errors.

Weighting can improve balanced accuracy or recall. It also changes the population emphasized by the loss. Probabilities from a weighted fit need not estimate the original population prevalence.

For binary outcomes, let $\eta(x)=P(Y=1\mid X=x)$ under the target population. At a fixed $x$, a weighted expected log loss is

$$
-a_1\eta(x)\log p
-a_0(1-\eta(x))\log(1-p).
$$

Its unconstrained minimizer is

$$
p_w(x)
{}={}
\frac{a_1\eta(x)}
{a_1\eta(x)+a_0(1-\eta(x))}.
$$

Therefore

$$
\frac{p_w(x)}{1-p_w(x)}
{}={}
\frac{a_1}{a_0}
\frac{\eta(x)}{1-\eta(x)}.
$$

Unequal class weights change the optimal odds by the weight ratio. The raw weighted-model output is naturally interpreted for the reweighted objective, not automatically as $P(Y=1\mid X=x)$ in the original population. Regularization, model misspecification, and sampling design can add further differences, so evaluate or recalibrate on data representative of the intended population.

### Misconception

**Balanced class weights automatically produce better probabilities.**

No. They change decision emphasis. Evaluate probability quality under the target population and sampling design.

## 12. Calibration is a property of repeated predictions

A classifier is calibrated when predictions near probability $q$ are correct about fraction $q$ of the time. For example, among many cases assigned probability 0.8, about 80 percent should be positive.

Accuracy does not measure this property. A model can have high accuracy and excessive confidence.

Empirical calibration needs a reliability diagnostic, such as:

- a reliability curve,
- a Brier score,
- expected calibration error with a declared binning rule,
- uncertainty intervals over these summaries.

No single finite sample can prove perfect calibration.

## 13. Temperature scaling adjusts logit spread

Temperature scaling divides every class logit by one positive scalar $T$:

$$
p_k(T)
{}={}
\frac{\exp(z_k/T)}
{\sum_j\exp(z_j/T)}.
$$

If $T>1$, logits move closer together and probabilities soften. If $0<T<1$, probabilities sharpen. Because division by a positive scalar preserves logit order, top-1 class predictions do not change.

![Temperature changes confidence while preserving class order](../images/10_temperature_scaling.svg)

Fit $T$ by minimizing negative log likelihood on a held-out fitting partition often called calibration data. Negative log likelihood is a proper scoring rule: in expectation, it rewards truthful probability assignments.

Lower empirical negative log likelihood on the fitting partition means that the chosen $T$ scored better there than the compared candidates. It does not prove that probabilities are calibrated.

### A three-class temperature example

Suppose logits are $[3,1,0]$. Class 1 has the largest logit. Dividing by $T=2$ gives $[1.5,0.5,0]$. The ordering is unchanged, but the gaps are smaller, so softmax assigns less probability to class 1 and more to the alternatives.

Dividing by $T=0.5$ gives $[6,2,0]$. The gaps are larger, so the distribution becomes sharper. One temperature controls global confidence spread, not class-specific or region-specific errors.

### Stable log probabilities and continuous temperature fitting

Subtracting the maximum makes exponentiation safer, but a probability can still underflow
to zero when logits are extremely separated. Taking `log(softmax(scores))` then produces
negative infinity. If the final goal is negative log likelihood or multiplication of many
probabilities, remain in log space.

For one score row $z$ and target class $y$, the negative log likelihood is

$$
\ell(z,y)=\log\left(\sum_k\exp z_k\right)-z_y.
$$

SciPy's `logsumexp` evaluates the first term stably without materializing tiny
probabilities. Batched target selection uses paired advanced indices:

```python
from scipy.special import logsumexp

scaled = logits / temperature
losses = logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]
mean_nll = losses.mean(dtype=np.float64)
```

Log probabilities also turn products into sums. If independent factor probabilities are
combined as $p_1p_2p_3$, their log mass is

$$
\log p_1+\log p_2+\log p_3.
$$

Impossible events can be represented as negative infinity. Rankings should operate on log
masses directly rather than exponentiating them into indistinguishable zeros.

A positive temperature can be optimized continuously by introducing unconstrained
coordinate $\eta=\log T$. Bounds $T_{\min}$ and $T_{\max}$ become finite log bounds. A
bounded scalar optimizer then searches a smooth one-dimensional domain:

```python
from scipy.optimize import minimize_scalar

result = minimize_scalar(
    lambda eta: temperature_nll(np.exp(eta)),
    bounds=(np.log(lower), np.log(upper)),
    method="bounded",
)
temperature = float(np.exp(result.x))
```

Check `result.success`, the finiteness of the fitted value, and whether the optimum touches
a configured boundary. A saturated objective may report an interior point whose value is
numerically tied with an edge, so compare the fitted objective with both boundary values.
A boundary solution is often a diagnostic that the allowed range, score scale, or model
behavior needs investigation rather than a value to accept silently.

Continuous optimization avoids making the result depend on an arbitrary temperature grid.
It does not remove the need for a held-out fitting role, a frozen objective, or evaluation
on untouched data.

## 14. Separate training, temperature fitting, and testing

Use three roles:

1. Training data fit preprocessing and model parameters.
2. Calibration data fit the temperature.
3. Test data evaluate the frozen pipeline.

Do not choose $T$ on test labels. Do not assert that test negative log likelihood must improve. A temperature that improves calibration-set NLL can perform worse on one finite test sample because of sampling variation or distribution shift.

Report test scoring rules and reliability diagnostics with uncertainty. Temperature scaling cannot repair incorrect class ordering, missing features, or a shifted label relationship.

### Reliability estimates also vary

A reliability curve groups predictions into probability bins and compares mean confidence with observed frequency. Bin edges, sample size, and class prevalence affect the plot. Sparse bins can look badly miscalibrated or perfectly calibrated by chance.

Report bin counts and an uncertainty method, such as a participant-level bootstrap when observations are clustered. Expected calibration error is a summary of a chosen binning scheme, not a universal property independent of that choice.

The Brier score for a binary outcome is

$$
\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
$$

It is another proper scoring rule. Like NLL, it combines calibration and discrimination behavior. Neither score alone replaces a reliability analysis.

### Conceptual checkpoint

What can you conclude after calibration-set NLL decreases?

You can conclude that the selected temperature improved that empirical scoring rule on the data used to select it. You cannot yet conclude that test NLL improves or that empirical calibration is established.

## 15. A complete leakage-safe workflow

1. Split independent groups into training, calibration, and test partitions.
2. Fit standardization and one-hot encoding on training data.
3. Fit the regularized linear model on transformed training data.
4. Transform calibration data with frozen preprocessing.
5. Fit temperature using calibration logits and labels.
6. Freeze every component.
7. Evaluate scores and reliability on transformed test data.

When data are grouped, use group-aware cross-validation for regularization selection.

### The exact GFC-v2 factor-alignment workflow

The study applies the general workflow above in a precise order. A participant belongs to
one data role. Development-fit participants estimate the ridge maps and every statistic
used by those maps. A separate development-calibration partition fits the shared
temperature. Evaluation participants are untouched until the pipeline is frozen.

Before any head is fitted, one recording representation is formed from three distinct,
deterministic 16-frame windows. The frozen encoder maps each window to one pooled vector.
Convert those three vectors to float64 and average them along the window axis. The result
is one row per recording, not three fitting rows. A complete participant therefore
contributes eight recording rows. The windows improve the measurement of one recording;
they do not create independent observations.

For each factor $f$, form a two-column one-hot target matrix $Y_f$. On development-fit
rows only, compute the population mean and standard deviation of every representation
coordinate. Standardize with `ddof=0`, then fit one two-output ridge system with
$\alpha=1$. Centering the standardized inputs and $Y_f$ before solving gives

$$
W_f=(X_c^\mathsf{T}X_c+\alpha I)^{-1}X_c^\mathsf{T}Y_{f,c},
$$

$$
b_f=\bar Y_f-\bar XW_f.
$$

The intercept is recovered from the means and is not penalized. The inverse notation
describes the solution mathematically. Implement it with `np.linalg.solve`.

The primary penalty is $\alpha=1$. Repeat the complete frozen pipeline with
$\alpha=0.1$ and $\alpha=10$ as prespecified sensitivities. Each sensitivity refits its
three heads and development-only post-map normalizers. It does not choose a better
penalty after evaluation outcomes are known.

Raw mapped scores are $S_f=XW_f+b_f$. The primary post-map normalization is also fitted
only on development-fit rows. For each of the two score coordinates, compute its
development mean $m_f$ and population standard deviation $q_f$. Clamp each standard
deviation to the declared positive floor, then transform every later row by

$$
\widetilde S_f=\frac{S_f-m_f}{q_f},
\qquad
B_f=\frac{\widetilde S_f}{\lVert\widetilde S_f\rVert_2}.
$$

The last operation is rowwise L2 normalization. It gives each speed, clothing, and
direction block a comparable directional scale before the three cosine distances are
averaged. A row whose standardized block has zero or near-zero norm must follow the frozen
zero-norm policy. It must not receive statistics estimated from evaluation rows.

The independent-factor soft control uses the raw scores $S_f$, not the normalized
retrieval blocks $B_f$. Fit one positive temperature $T$ on held-out development-
calibration rows by pooling the three factor negative log likelihoods:

$$
\frac{1}{3N}\sum_{i=1}^{N}\sum_f
\left[
\log\sum_{k=1}^{2}\exp(S_{ifk}/T)-S_{if,y_{if}}/T
\right].
$$

Using one $T$ keeps the confidence scale shared across factors. Temperature fitting may
change probabilities and negative log likelihood, but a positive temperature does not
change a factor's argmax. After this fit, freeze input standardization, all three ridge
heads, all three post-map normalizers, and $T$ before evaluation.

## 16. Efficiency notes

- Use <code>Pipeline</code> and <code>ColumnTransformer</code> to keep preprocessing inside folds.
- Use sparse one-hot matrices for high-cardinality categories.
- Use mature iterative solvers for logistic regression.
- Use <code>np.linalg.solve</code>, QR, or SVD rather than explicit inversion.
- Vectorize stable softmax over the final axis.
- Search temperature on log scale or optimize $\log T$ so positivity is automatic.
- Use `logsumexp` when the analysis consumes log likelihoods or products of probabilities.
- Use `einsum` only when its named-axis notation makes the contraction clearer than `@`.
- Keep a separate calibration partition or use nested cross-validation.

## 17. Common failure modes

1. **Scaling before splitting:** test statistics leak into training.
2. **Integer category codes:** arbitrary numbers create fake distances.
3. **Penalized intercept:** the global outcome level is shrunk.
4. **Explicit inverse:** computation and numerical error increase.
5. **Unstable exponentials:** large logits overflow.
6. **Class weights called calibration:** weighted fitting changes probability meaning.
7. **Temperature fit on test data:** final evaluation becomes optimistic.
8. **NLL improvement called calibration proof:** no reliability evidence was measured.
9. **Ignoring distribution shift:** a fitted temperature may not transfer.
10. **Uninspected optimizer boundaries:** a saturated temperature fit can look successful
    while the configured range contains no meaningful interior optimum.
11. **Incorrect Einstein subscripts:** the result can have a valid shape while contracting
    the wrong scientific axis.
12. **One joint factor head:** a six-output map hides the declared three-block structure.
13. **Normalizing with evaluation rows:** post-map means and scales are fitted parameters.
14. **Calibrating normalized retrieval blocks:** the soft control requires the raw ridge
    scores, while retrieval uses the separately normalized blocks.
15. **Treating windows as fitting rows:** three deterministic windows are averaged into
    one float64 recording representation before factor-head fitting.
16. **Choosing ridge strength from outcomes:** $\alpha=1$ is primary and 0.1 and 10 are
    fixed sensitivities, not outcome-selected alternatives.

## 18. Exercises

1. Why does feature standardization change an L2-regularized solution?
2. Derive the ridge normal equations.
3. Does temperature scaling change top-1 accuracy?
4. What ambiguity arises from combining a dropped reference category with ignored unknown categories?
5. What additional evidence is needed before claiming empirical calibration?
6. Decode `np.einsum("ni,nk->ik", X, Y)` and state the result shape.
7. Why optimize $\log T$ instead of sending an unconstrained $T$ directly to an optimizer?
8. Which quantities may development-calibration rows fit in the GFC-v2 workflow?

### Brief solutions

1. The same coefficient size represents different prediction changes for differently scaled features.
2. Set $-2X^\mathsf{T}y+2X^\mathsf{T}Xw+2\alpha w=0$ and rearrange.
3. No. Positive temperature preserves logit order.
4. Both the reference and an unknown category map to the same all-zero indicator block.
5. Evaluate a declared reliability diagnostic and scoring rules on untouched data, with uncertainty.
6. Axis `n` is summed over; feature axis `i` and output axis `k` remain, so the result has
   shape `(D,K)`.
7. Exponentiating the optimizer coordinate guarantees $T>0$, and finite log bounds encode
   a positive search interval without special invalid-value handling.
8. They fit the one shared temperature. Ridge input statistics, coefficients, intercepts,
   and post-map normalization statistics come from development-fit rows.

## Recap

Standardization and one-hot encoding create a meaningful design matrix. Ridge stabilizes weakly constrained directions and normally excludes the intercept. Multi-output contractions can be written with explicit Einstein subscripts. GFC-v2 fits three separate two-output ridge heads, then fits one post-map normalizer per factor block using development data only. Logistic and softmax models convert linear scores into probabilities, while `logsumexp` preserves likelihood information at extreme scales. Class weights change the fitting population. Temperature scaling changes confidence while preserving class order, but fitting by negative log likelihood is not itself proof of calibration.

## Next lesson

[11: Factorial state spaces](11_factorial_state_spaces.md) studies structured combinations of factors used in representation evaluation.

## Continue in the notebook

[Open the executable lesson 10 notebook](../implementations/10_regularized_linear_estimation.ipynb) to solve ridge regression, fit a weighted classifier, and select a temperature by held-out NLL.
