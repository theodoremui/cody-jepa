# 04. Attention and positional representations

![The parts of a Transformer attention block](../images/04_attention_and_positions.svg)

## Begin with a pronoun and its context

Read the sentence, "The robot lifted the box because it was light." To interpret the
word "it," a reader compares that token with other tokens and uses the surrounding
content to decide what information matters. The comparison is not fixed by distance
alone. A nearby word can be irrelevant, while a farther word can resolve the meaning.

Attention gives a neural network a differentiable version of this selective lookup.
Each token asks a question, scores how well every token can answer, converts the scores
into nonnegative weights, and blends the corresponding information. The weights depend
on the current token content, so the lookup pattern can change from one input to another.

This idea is easier to understand one query at a time. A full attention matrix is only
many copies of the same four-step process performed in parallel.

![One query creates one row of attention](../images/04_attention_row.svg)

The four steps are: make a query, compare it with all keys, normalize the comparison
scores, and average the values. Queries and keys determine **where to read**. Values
determine **what is read**. Keeping these roles separate is the core mental model for
the equations that follow.

## Prerequisites

You should understand vectors, dot products, norms, and tensor shapes. Review
[02. Inner-product geometry](02_inner_product_geometry.md) if needed.

## Learning goals

By the end of this lesson, you will be able to:

1. Derive softmax and scaled dot-product attention.
2. Track query, key, value, score, and output shapes.
3. Explain why scaling by $\sqrt{d_k}$ stabilizes softmax.
4. Build multi-head attention from parallel lower-dimensional heads.
5. Construct sinusoidal positional representations.
6. Explain residual connections, LayerNorm, and GELU in a Transformer block.

## 1. Why attention exists

A token sequence is a tensor $X\in\mathbb{R}^{B\times N\times D}$. A convolution mixes
nearby locations using a fixed local pattern. Attention allows every query token to form
a data-dependent weighted average of value tokens.

For one sequence, imagine asking each token three questions:

- **Query:** What information am I looking for?
- **Key:** What kind of information do I contain?
- **Value:** What information should I contribute if selected?

Learned linear maps create these roles:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

If $X$ has shape $(N,D)$, $W_Q,W_K\in\mathbb{R}^{D\times d_k}$, and
$W_V\in\mathbb{R}^{D\times d_v}$, then $Q,K$ have shape $(N,d_k)$ and $V$ has
shape $(N,d_v)$.

Here $N$ is the number of tokens and $D$ is the input feature width. The matrices
$W_Q$, $W_K$, and $W_V$ are learned parameters. Multiplying by them does not change
the number of tokens. It changes the coordinate system used for asking, matching, and
communicating.

Why use three projections instead of comparing the original tokens directly? A feature
that is useful for deciding relevance need not be the feature we want to copy into the
output. A token might advertise "I refer to an object" through its key while contributing
color, shape, and motion information through its value. Separate maps allow training to
discover this division of labor.

For a batch, prepend axis $B$ to every shape. The same parameter matrices apply to every
sample, and attention never mixes different batch members. This is a computational
batching convention, not a statement that samples were collected independently.

## 2. Softmax converts scores into weights

For scores $z_1,\ldots,z_N$, softmax is

$$
\mathrm{softmax}(z)_j=\frac{\exp(z_j)}{\sum_{r=1}^{N}\exp(z_r)}.
$$

Every output is positive and the outputs sum to 1. They can therefore serve as weights
for an average. Adding the same constant $c$ to every score changes neither the result:

$$
\frac{\exp(z_j+c)}{\sum_r \exp(z_r+c)}=
\frac{\exp(c)\exp(z_j)}{\exp(c)\sum_r \exp(z_r)}=\mathrm{softmax}(z)_j.
$$

Use this invariance for stability. Subtract the largest score before exponentiating:

```python
import numpy as np

def softmax(z, axis=-1):
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)
```

Without the shift, `exp(1000)` overflows. After the shift, the largest exponent is 1.

Softmax preserves order: a larger score receives a larger weight. It also depends on
score differences rather than absolute score level. Scores `[2, 1, 0]` and `[102, 101,
100]` produce the same weights because the second list adds the same constant to every
entry.

Score scale does matter. Multiplying all scores by a large positive number concentrates
weight on the maximum, while shrinking scores toward zero makes weights more uniform.
For two scores, the important quantity is their gap. Softmax turns a larger gap into a
more decisive choice.

**Worked example.** The scores `[0, 0, 0]` become weights `[1/3, 1/3, 1/3]`. If the
scores are `[0, 0, log(2)]`, their exponentials are `[1, 1, 2]`, so the weights are
`[1/4, 1/4, 1/2]`. The third value contributes twice as much as either other value.

## 3. Scaled dot-product attention

The score between query $i$ and key $j$ is their dot product:

$$
S_{ij}=q_i^\top k_j.
$$

All scores are computed by $QK^\top$, which has shape $(N,N)$. Row $i$ compares query
$i$ against every key. Convert each row to weights and average values:

$$
\mathrm{attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

For batched inputs, the shapes are:

| Quantity | Shape |
|---|---|
| $Q,K$ | $(B,N,d_k)$ |
| $V$ | $(B,N,d_v)$ |
| $QK^\top$ | $(B,N,N)$ |
| attention weights | $(B,N,N)$ |
| output | $(B,N,d_v)$ |

```python
scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
weights = softmax(scores, axis=-1)
output = weights @ V
```

Read this expression in three stages. First, $QK^\top$ contains every query-key dot
product. Second, division and row-wise softmax turn each score row into a probability-like
weight row. Third, multiplication by $V$ takes one weighted average of value vectors per
query. The output has one vector for every input query.

The matrix orientation matters. Row $i$ corresponds to query $i$, and column $j$
corresponds to key and value $j$. Softmax must act across columns so each query distributes
one unit of weight across available keys. Normalizing down the query axis would answer a
different and usually unintended question.

**Numerical trace.** Suppose one query gives weights `[0.25, 0.75]` and the two scalar
values are 4 and 12. The attention output is $0.25(4)+0.75(12)=10$. With vector values,
the same weights apply to every feature coordinate. Attention is therefore a content-based
weighted average, not a selection that must choose exactly one token.

## 4. Why divide by the square root

Assume query and key coordinates are independent, mean zero, and variance one.
The dot product is a sum of $d_k$ products:

$$
q^\top k=\sum_{r=1}^{d_k}q_r k_r.
$$

Each product has variance approximately one, so the sum has variance $d_k$ and standard
deviation $\sqrt{d_k}$. As $d_k$ grows, unscaled logits spread out. Softmax becomes
nearly one-hot, and its gradients become small for most entries.

Dividing by $\sqrt{d_k}$ makes the score variance approximately one under this simple
model. It is a variance-control argument, not an arbitrary convention.

The assumptions are an explanatory approximation, not a theorem about trained networks.
Coordinates in a trained query and key can be correlated and need not have unit variance.
The derivation still explains why an unscaled sum tends to grow with feature width at
initialization and why the square root, rather than $d_k$, is the natural correction for
standard deviation.

If a head has width 64, its unscaled dot products have typical scale around 8 under the
simple model. Dividing by $\sqrt{64}=8$ restores a typical scale near one. In multi-head
attention, use the width of one head, not the total model width, because each head forms
its own dot products.

## 5. Masks enforce valid information flow

A mask can prohibit certain query-key pairs. Add a very negative value to forbidden
logits before softmax. In exact arithmetic, adding negative infinity produces zero weight.

```python
scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

Each query row must allow at least one key. A fully masked row asks softmax to normalize
all negative infinities and can produce `NaN`. Boolean masks are less error-prone than
manually managing large constants. PyTorch mask conventions differ across APIs, so read
the function contract and assert a small known example.

A mask represents a structural rule, not a learned preference. A causal mask forbids a
token from reading future tokens. A padding mask forbids reading placeholder positions.
A local mask may restrict reading to a neighborhood. Allowed tokens still compete through
softmax; forbidden tokens receive exactly zero weight in the intended mathematical model.

Masking after softmax is usually wrong. Setting forbidden weights to zero after
normalization makes the remaining row sum to less than one unless it is normalized again.
Adding negative infinity before softmax excludes forbidden entries from the denominator.

**Conceptual checkpoint.** Scores express learned compatibility. Masks express what the
computation is permitted to use. Scaling controls numerical and statistical score range.
Softmax converts the final valid scores into weights. These are four distinct roles.

## 6. Multi-head attention

One attention map may focus on one relationship. Multi-head attention runs $H$ learned
attention operations in parallel. Usually $D=H d_h$.

After one projection, reshape

$$
(B,N,D)\rightarrow(B,N,H,d_h)\rightarrow(B,H,N,d_h).
$$

Each head computes an $(N,N)$ attention matrix. The outputs have shape
$(B,H,N,d_h)$. Transpose and concatenate back to $(B,N,D)$, then apply an output
projection $W_O$.

```python
q = q.reshape(B, N, H, d_h).transpose(1, 2)
scores = q @ k.transpose(-2, -1) / (d_h ** 0.5)
head_output = scores.softmax(dim=-1) @ v
joined = head_output.transpose(1, 2).reshape(B, N, D)
```

The divisibility condition `D % H == 0` should be checked explicitly.

Heads do not receive predefined meanings. They are parallel parameterized subspaces that
can learn different comparison patterns. One head may become sensitive to local motion
while another uses longer-range identity cues, but such descriptions must be supported
by analysis rather than assumed from the architecture.

Splitting into heads keeps the joined width $D$ fixed when $d_h=D/H$. Each head builds
its own $N\times N$ score matrix. Increasing the number of heads therefore changes the
number and width of parallel comparisons, not the total joined feature width.

The final output projection mixes information across heads. Without it, each block of
features would remain tied to one head. The projection lets later layers combine patterns
discovered in different subspaces.

## 7. Attention alone does not know order

![Sinusoidal waves encode positions at multiple scales](../images/04_positional_waves.svg)

Without positional information, permuting tokens permutes outputs in the same way.
The layer knows feature content but not whether a token came first, last, left, or right.

Sinusoidal positional representations assign deterministic waves of different frequencies:

$$
P_{p,2i}=\sin\left(p/10000^{2i/D}\right),
$$

$$
P_{p,2i+1}=\cos\left(p/10000^{2i/D}\right).
$$

Here $p$ is position and $i$ indexes a frequency pair. Add $P$ to token features:

$$
X^{(0)}=X+P.
$$

The shortest wavelengths vary rapidly with position; the longest vary slowly. Together
they give each position a distinctive multiscale signature.

Why is order missing in the first place? If we permute the rows of $X$, the same linear
maps permute $Q$, $K$, and $V$. Query-key comparisons and outputs follow that permutation.
Nothing in content-only self-attention says that row 3 occurred before row 4. This
permutation equivariance is useful for sets but incomplete for language, images, and video.

In the positional equations, $p$ is an integer location from 0 to $N-1$. The index $i$
chooses a frequency pair, and $D$ is the representation width. Even coordinates use a
sine and odd coordinates use a cosine at the same frequency. The base 10000 spreads
wavelengths over a broad range; it is a design convention rather than a law of geometry.

Adding $P$ works because token content and position have the same shape $(N,D)$. After
addition, each vector contains both. Attention can learn projections that use positional
coordinates, content coordinates, or their interactions. Position is not appended as a
separate token and does not change sequence length.

For video, a single flattened position encodes an order chosen by flattening. Other
designs encode time, height, and width separately and combine their representations.
The appropriate design depends on whether the model should distinguish these axes and
how it must generalize to new grid sizes.

```python
def sinusoidal_positions(length, width):
    position = np.arange(length)[:, None]
    even = np.arange(0, width, 2)[None, :]
    rates = np.exp(-np.log(10000.0) * even / width)
    out = np.zeros((length, width), dtype=np.float32)
    out[:, 0::2] = np.sin(position * rates)
    out[:, 1::2] = np.cos(position * rates[:, :out[:, 1::2].shape[1]])
    return out
```

The slicing handles odd widths. In practice, even model widths are conventional.

## 8. Residual connections preserve a direct path

A residual update is

$$
y=x+F(x).
$$

Instead of forcing $F$ to construct the entire output, it learns a correction to $x$.
Gradients have a direct identity path through the addition. Shapes must match, so the
attention output projection returns width $D$.

Residual connections do not guarantee stable training by themselves, but they make deep
composition substantially easier.

Picture $F(x)$ as an edit proposed to the current representation. If the best action is
to preserve $x$, the branch can learn an edit near zero. The addition also gives the
backward computation a direct derivative contribution of one, although other branches
and normalization still affect full optimization behavior.

Residual addition requires identical shapes and compatible meanings. A branch that
changes width needs a projection before addition. Accidentally relying on broadcasting
can produce a tensor of the expected rank while applying the wrong edit to every token.

## 9. Layer normalization

LayerNorm normalizes the features of each token independently. For token vector
$x\in\mathbb{R}^D$:

$$
\mu=\frac{1}{D}\sum_d x_d,\qquad
\sigma^2=\frac{1}{D}\sum_d(x_d-\mu)^2,
$$

$$
\mathrm{LN}(x)_d=\gamma_d\frac{x_d-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_d.
$$

$\gamma$ and $\beta$ are learned per-feature scale and shift. Unlike BatchNorm,
LayerNorm does not use statistics across batch members. It behaves consistently for
small batches and variable sequence lengths.

PyTorch's `nn.LayerNorm(D)` uses the population variance over its normalized axes.

Layer normalization treats one token vector at a time. It subtracts that token's feature
mean and divides by its feature standard deviation. The small positive $\epsilon$ keeps
the denominator finite for a constant token. Learned scales $\gamma_d$ and shifts
$\beta_d$ then restore the ability to represent feature-specific ranges.

Normalization does not erase all information. It removes a common offset and overall
scale across the normalized features before the learned affine transformation. Relative
feature patterns remain. Its placement before or after residual branches changes training
dynamics, which is why "pre-norm" and "post-norm" are meaningful architectural choices.

## 10. GELU and the feed-forward network

A Transformer block also applies a feed-forward network independently to every token:

$$
\mathrm{FFN}(x)=W_2\mathrm{GELU}(W_1x+b_1)+b_2.
$$

GELU is

$$
\mathrm{GELU}(x)=x\Phi(x),
$$

where $\Phi$ is the standard normal cumulative distribution function. It smoothly gates
inputs rather than setting every negative input exactly to zero. PyTorch provides
`torch.nn.functional.gelu`, including an optional fast tanh approximation.

Attention mixes information across token positions. The feed-forward network performs a
different operation: it transforms features independently at each position using shared
weights. The first linear layer usually expands width, GELU introduces a nonlinearity,
and the second layer returns to width $D$ for residual addition.

Without a nonlinearity, two consecutive linear maps collapse into one linear map and add
no new functional depth. GELU makes the transformation input-dependent while remaining
smooth. Its probabilistic definition is intuition for the curve, not a claim that hidden
features are sampled from a Gaussian distribution.

## 11. Putting a pre-norm block together

A common pre-norm architecture is

$$
Y=X+\mathrm{MHA}(\mathrm{LN}(X)),
$$

$$
Z=Y+\mathrm{FFN}(\mathrm{LN}(Y)).
$$

Pre-norm places normalization before each sublayer and leaves an especially direct
residual path. Post-norm architectures place normalization after addition. They are not
interchangeable when loading weights or reproducing training behavior.

Follow the data rather than reading these equations as two opaque formulas. Normalize
$X$, let multi-head attention gather context, and add that contextual edit back to $X$.
Call the result $Y$. Normalize $Y$, transform each token's features with the feed-forward
network, and add that edit back to form $Z$. Both branches preserve `(B,N,D)`.

The block therefore alternates two kinds of computation: communication across tokens and
nonlinear processing within tokens. Residual paths preserve prior representations around
both computations. Stacking blocks repeats this exchange, allowing information gathered
at one layer to influence later queries and transformations.

```python
class Block(torch.nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))
```

`batch_first=True` preserves `(B,N,D)`. `need_weights=False` avoids materializing and
returning attention weights when they are not needed. Recent PyTorch versions can route
eligible calls through optimized scaled-dot-product attention kernels.

The implementation uses the same normalized tensor `n` as query, key, and value input,
which makes this **self-attention**. The module still applies distinct learned projections
internally. In cross-attention, queries come from one sequence while keys and values come
from another, but the score-normalize-mix logic is unchanged.

## 12. Worked example

Let $B=2,N=5,D=12,H=3$. Each head has $d_h=4$.

1. Projected $Q,K,V$ initially have shape `(2,5,12)`.
2. Split heads to get `(2,3,5,4)`.
3. Scores have shape `(2,3,5,5)`.
4. Softmax rows sum to one over the final key axis.
5. Weighted values have shape `(2,3,5,4)`.
6. Concatenating heads restores `(2,5,12)`.
7. The residual addition is valid because input and output shapes match.

There are $2\times3\times5\times5=150$ score scalars in this small batch. If sequence
length grows from 5 to 500 while other dimensions stay fixed, a standard implementation
that materializes the full score tensor uses 10,000 times as many score scalars because
both the query and key axes grow. This quadratic relationship is why long-sequence
attention requires careful memory planning.

For one head and one query, suppose scaled scores are `[0, log(2), 0, 0, 0]`. The
exponentials sum to 6, so the weights are `[1/6, 2/6, 1/6, 1/6, 1/6]`. The second value
has twice the influence of each other value, but the output remains a blend. This small
calculation is the scalar core of the larger tensor operation.

## 13. Efficiency and failure modes

- Prefer `torch.nn.functional.scaled_dot_product_attention` or `nn.MultiheadAttention`
  to handwritten production kernels.
- A materialized score tensor requires $O(BHN^2)$ storage. Fused kernels can avoid
  storing the full matrix, although dense attention still performs $O(BHN^2d_h)$
  score arithmetic.
- Subtracting the maximum is built into `torch.softmax`; do not exponentiate manually.
- A fully masked row can create nonfinite values.
- Scaling by $\sqrt{D}$ instead of $\sqrt{d_h}$ is wrong for multi-head attention.
- Forgetting positional information makes self-attention order-equivariant.
- A transpose followed by `view` may require contiguous storage; `reshape` is safer.
- Use shape assertions at head splitting and joining boundaries.

Attention weights are sometimes displayed as explanations, but a large weight alone
does not prove causal importance or human-interpretable reasoning. The output also depends
on value vectors, residual paths, later layers, and alternative computation routes. Treat
weight visualizations as diagnostics that require supporting evidence.

For efficiency, avoid returning weights unless an analysis needs them. Fused attention
kernels can reduce intermediate memory and improve numerical handling. Their exact API
contracts still matter, especially mask meaning, dropout behavior, and tensor layout.

## Exercises

### Exercise 1

For $D=64$ and $H=8$, what are the per-head width and score shape when $B=4,N=20$?

**Brief solution:** $d_h=8$ and scores have shape `(4,8,20,20)`.

### Exercise 2

Why does `softmax([1000, 1001])` need a shifted implementation?

**Brief solution:** direct exponentials overflow. Subtracting 1001 gives `[-1,0]` and
the same probabilities with finite exponentials.

### Exercise 3

What axis must each attention row sum over to equal one?

**Brief solution:** the final key axis. Each query distributes its weight across keys.

### Exercise 4

One query has scores `[2, 2, 2]` and values `[3, 6, 12]`. What is its scalar attention
output? What happens if 100 is added to every score?

**Brief solution:** equal scores give weights `[1/3,1/3,1/3]`, so the output is 7.
Adding a common constant leaves softmax and the output unchanged.

### Exercise 5

Why can a Transformer distinguish two permutations after positional vectors are added,
even if the token contents are identical?

**Brief solution:** each position contributes a different vector. Permuting content
changes which content-position sums enter the learned projections, so the inputs are no
longer merely an unlabeled set of identical token vectors.

## Recap

Attention compares queries with keys, stabilizes their dot products by the head width,
normalizes scores with softmax, and averages values. Multiple heads learn parallel
relations. Positional representations supply order, while residual connections,
LayerNorm, and GELU form a trainable deep block around attention.

Next: [05. Masked latent prediction and target updates](05_masked_latent_prediction.md).

## Continue in the notebook

Run the [attention and positions notebook](../implementations/04_attention_and_positions.ipynb) before moving to Lesson 05.
