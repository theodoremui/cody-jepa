# 11. Factorial state spaces

![Overview of a factorial state space](../images/11_factorial_state_spaces.svg)

## Prerequisites

You should be comfortable with Python lists, sets, loops, and basic probability. The
ideas build on [10. Regularized linear estimation and calibration](10_regularized_linear_estimation.md),
but this lesson does not require advanced linear algebra.

## Learning goals

By the end of this lesson, you will be able to:

1. distinguish factors, levels, cells, and complete assignments;
2. construct a state space as a Cartesian product;
3. encode binary cells as bit vectors and integers;
4. enumerate factor subsets and complementary assignments;
5. use exact rational arithmetic for finite probabilities; and
6. derive a ceiling imposed by partial information.

## 1. Motivating scenario: what exactly did we test?

Imagine evaluating a video representation under three conditions. Lighting can be day
or night. Camera view can be front or side. Walking speed can be slow or fast. A result
reported only as "night accuracy" is incomplete because it mixes four combinations of
view and speed.

The clean mental model is an address book. Each experimental condition has an address
with one entry per factor. The address `(night, side, fast)` identifies one precise cell.
The full state space is the set of every valid address.

This language prevents a common confusion. A factor is not a cell. "Lighting" is a
factor, "night" is one level of that factor, and `(night, side, fast)` is a cell.

## 2. Vocabulary before formulas

A **factor** is a variable deliberately represented in the design, such as lighting.
A **level** is one allowed value, such as day. A **cell** is one complete choice of a
level for every factor. A **state space** is the collection of all cells.

Let there be $F$ factors. Factor $f$ has a set of levels called $A_f$, and the number of
levels in that set is $L_f$. The subscript $f$ is just a factor index from 1 through $F$.

For the motivating example, $F=3$ and every $L_f=2$. One cell has three entries, so its
shape is naturally a length-3 tuple or a one-dimensional array with shape `(3,)`.

## 3. Cartesian products multiply independent choices

The Cartesian product collects every tuple formed by taking one element from each set:

$$
S=A_1\times A_2\times\cdots\times A_F.
$$

Here $S$ is the complete state space. The multiplication symbol means "combine every
choice," not numeric multiplication of the level names.

The number of cells is the product of the level counts:

$$
|S|=\prod_{f=1}^{F}L_f.
$$

The bars $|S|$ mean the number of elements in $S$. The product says that every choice
for factor 1 can be paired with every choice for factor 2, and so on.

![Three binary factors form eight cells](../images/11_cartesian_product.svg)

In the example, the count is $2\times2\times2=8$. If a fourth factor has three levels,
the count becomes $8\times3=24$. Factorial spaces grow quickly because choices multiply.

```python
from itertools import product

lighting = ["day", "night"]
view = ["front", "side"]
speed = ["slow", "fast"]
cells = list(product(lighting, view, speed))

assert len(cells) == 8
print(cells[0])  # ('day', 'front', 'slow')
```

`itertools.product` is preferable to hand-written nested loops because it works for any
number of factors and makes the Cartesian-product intention explicit.

### Conceptual checkpoint

If factor level counts are 2, 3, and 4, there are $2\times3\times4=24$ cells. Adding one
more binary factor doubles the count. It does not add only two cells.

## 4. A table representation and its shapes

Suppose the state space has $M=|S|$ cells. We can store an enumeration as a table with
shape `(M, F)`. Rows are cells and columns are factors.

For three binary factors, the table shape is `(8, 3)`. Entry `states[m, f]` is the level
of factor $f$ in cell $m$. The row index is not itself a scientific factor. It is only an
enumeration label.

```python
import numpy as np

states = np.array(list(product([0, 1], repeat=3)), dtype=np.int8)
assert states.shape == (8, 3)
assert np.array_equal(states[5], [1, 0, 1])
```

Using small integer arrays is efficient for equality checks and indexing. Keep a separate
mapping from integers to human-readable level names when interpretation matters.

## 5. Binary cells as bit vectors

When every factor has two levels, encode them as 0 and 1. A cell becomes a bit vector

$$
b=(b_0,b_1,\ldots,b_{F-1}),\qquad b_j\in\{0,1\}.
$$

The symbol $b_j$ is the bit for factor position $j$. Indexing from 0 matches Python.
There are $2^F$ possible bit vectors because each of the $F$ positions has two choices.

A bit vector can also be encoded as a nonnegative integer:

$$
q=\sum_{j=0}^{F-1}b_j2^{F-1-j}.
$$

The integer $q$ is the binary number represented by the bits. For $b=(1,0,1)$,
$q=1\times4+0\times2+1\times1=5$.

```python
def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = 2 * value + int(bit)
    return value

assert bits_to_int((1, 0, 1)) == 5
```

This encoding makes cells compact dictionary keys. It is only safe when the factor order
is fixed and documented. Reordering columns changes the integer even when the named
assignment is logically the same.

### Mixed-radix encoding for nonbinary factors

Binary is a special case of a mixed-radix system. Suppose factor $f$ is encoded by an
integer $a_f$ from 0 through $L_f-1$. A complete cell can be mapped to

$$
q=\sum_{f=1}^{F}a_f\prod_{r=f+1}^{F}L_r.
$$

The inner product contains the level counts of every factor to the right of position $f$.
For the final factor, that product is empty and is defined as 1. The formula is the same
idea as place value in base 10, except each position can have a different base.

Consider factor sizes 2, 3, and 4 and assignment $(1,2,3)$. Its code is
$1\times(3\times4)+2\times4+3=23$. Because there are 24 total cells, valid codes run from
0 through 23. This assignment is the final cell in ordinary lexicographic enumeration.

Decoding works from right to left with repeated quotient and remainder operations. Divide
the code by the last factor size. The remainder is the last level and the quotient contains
the remaining prefix. Repeat with the next factor size.

```python
def decode_mixed_radix(code, level_counts):
    values = []
    for count in reversed(level_counts):
        code, value = divmod(code, count)
        values.append(value)
    if code != 0:
        raise ValueError("code is outside the state space")
    return tuple(reversed(values))

assert decode_mixed_radix(23, [2, 3, 4]) == (1, 2, 3)
```

Mixed-radix codes make dense lookup arrays possible because every valid product cell has a
unique consecutive integer. They also make silent metadata errors dangerous. A code has no
meaning without the ordered level counts and the mapping from level names to integers.

## 6. Factor subsets and the power set

Sometimes a query reveals only a subset of factors. If the full factor index set is
$U=\{0,1,\ldots,F-1\}$, a known subset $K$ is any subset of $U$. Its complement contains
the unknown factors:

$$
K^{c}=U\setminus K.
$$

The superscript $c$ means complement relative to $U$. If $U=\{0,1,2,3\}$ and
$K=\{0,2\}$, then $K^c=\{1,3\}$.

The set of all subsets is called the power set. An $F$-element set has $2^F$ subsets.
This includes the empty set and the complete set.

```python
from itertools import combinations

def all_subsets(items):
    items = tuple(items)
    for size in range(len(items) + 1):
        yield from combinations(items, size)

subsets = list(all_subsets(range(3)))
assert len(subsets) == 8
```

### Misconception: a subset is not a partial row

The subset $K=\{0,2\}$ names which factors are known. It does not specify their values.
A partial assignment needs both the subset and values, such as "factor 0 is 1 and factor
2 is 0."

## 7. Complementary binary assignments

For a binary vector $b$, its bitwise complement flips every factor:

$$
\bar b_j=1-b_j.
$$

The bar identifies the complementary assignment. For $b=(1,0,1)$, the complement is
$(0,1,0)$. Applying the operation twice returns the original vector.

```python
def complement(bits):
    return tuple(1 - int(bit) for bit in bits)

bits = (1, 0, 1)
assert complement(bits) == (0, 1, 0)
assert complement(complement(bits)) == bits
```

Complement is meaningful only after a binary coding is defined. For a three-level factor,
there is no unique opposite level unless the scientific design supplies one.

## 8. Query enumeration

A query can be represented as `(known_indices, known_values)`. To find compatible cells,
compare only the known columns.

```python
known_indices = np.array([0, 2])
known_values = np.array([1, 0])
compatible = np.all(states[:, known_indices] == known_values, axis=1)
matches = states[compatible]

assert matches.shape == (2, 3)
```

The comparison first creates a Boolean array with shape `(M, len(K))`. `np.all(...,
axis=1)` reduces each row to one compatibility decision. This vectorized operation is
usually faster and clearer than looping through every cell in Python.

For a large state space, do not materialize every query-state pair. Encode queries as
integer masks and values, or generate compatible cells lazily.

## 9. Constrained state spaces and structural zeros

A Cartesian product assumes every combination is possible. Scientific designs often have
**structural zeros**, which are cells that cannot occur by definition. For example, a
sensor factor may have levels `RGB` and `thermal`, while a color-temperature factor is
defined only for RGB. Blindly crossing both factors invents thermal cells with meaningless
color-temperature labels.

Start with the product space $S$ and define a feasibility rule. The constrained space is

$$
S_{\mathrm{valid}}=\{s\in S:\text{the feasibility rule accepts }s\}.
$$

$s$ denotes one complete cell. The text after the colon is a condition that decides
whether that cell is allowed. Counts, priors, and partial-information ceilings must use
$S_{\mathrm{valid}}$, not the unfiltered product.

Structural zeros differ from missing observations. A valid cell with no collected sample
is empty because of sampling or data loss. An invalid cell does not belong to the target
state space at all. Combining the two can make a coverage report look better or worse than
the scientific design warrants.

For small spaces, enumerate then filter with a clearly tested predicate. For large spaces,
generate only feasible branches. Constraint-aware generation avoids spending exponential
work on cells that will be discarded.

## 10. Exact probabilities with rational arithmetic

Finite combinatorial probabilities are often exact fractions. Floating-point `1/3` is
only an approximation, which can make equality checks and accumulated counts awkward.

```python
from fractions import Fraction

probability = Fraction(1, 3) + Fraction(1, 6)
assert probability == Fraction(1, 2)
```

`Fraction` stores an integer numerator and denominator and automatically reduces them.
Use it for small exact enumerations. Convert to `float` only for plotting or APIs that
require floating-point values. For millions of operations, integer counts followed by one
final division are usually faster.

## 11. Partial-information ceilings

Suppose the true cell is one of eight binary cells, but a query reveals only the first
factor. Four cells remain compatible. If those four cells are equally likely conditional
on the revealed value, and the decision rule receives no other information, any exact-cell
guess succeeds with probability at most $1/4$.

![One known bit leaves four compatible cells](../images/11_partial_information_ceiling.svg)

For a full Cartesian product of $F$ binary factors and any consistent assignment to $k$
known factors, the number of compatible cells is

$$
C=2^{F-k}.
$$

Here $C$ counts completions and $F-k$ counts unknown bits. This count assumes there are no
structural zeros and that the known values do not impose further constraints. Under a
uniform conditional distribution over those completions, the exact-identification ceiling
for that partial query is

$$
p_{\max}=\frac{1}{C}=2^{-(F-k)}.
$$

This is an information ceiling relative to the stated input, not a statement about a weak
model. The partial query does not distinguish the compatible cells.

For nonbinary unknown factors in an unconstrained Cartesian product, replace the power of
two with their level-count product:

$$
C=\prod_{f\in K^c}L_f.
$$

For a constrained state space, the product can be wrong. Let $x_K$ denote the revealed
values on known factor set $K$. The compatible set is

$$
\mathcal{C}(x_K)=\{s\in S_{\mathrm{valid}}:s_K=x_K\},
$$

and its size is $C(x_K)=|\mathcal{C}(x_K)|$. The subscript $s_K$ means the entries of cell
$s$ at the known factor positions. Structural constraints can make $C(x_K)$ depend on the
particular known values, not only on how many factors are unknown.

If compatible cells are not equally likely, the best guess chooses the most probable one.
If random variable $Z$ is the true cell, the conditional ceiling for revealed assignment
$x_K$ is

$$
p_{\max}(x_K)=\max_{s\in\mathcal{C}(x_K)}P(Z=s\mid X_K=x_K).
$$

The variable $X_K$ contains the revealed factor values. The maximum selects the most
probable compatible true cell. The ceiling is therefore not necessarily $1/C(x_K)$,
especially when the target population has strong cell imbalance.

For example, suppose four cells are compatible but their conditional probabilities are
0.7, 0.1, 0.1, and 0.1. A classifier that sees only the partial query should always choose
the first cell and will succeed 70 percent of the time. The uniform formula would predict
25 percent and would be wrong because the prior is not uniform.

This example also reveals why a ceiling must state its information set. If the classifier
receives a feature correlated with the missing factors, then those four states are no
longer indistinguishable. A valid ceiling lists exactly which variables the decision rule
may use and computes conditional probabilities given those variables.

If evaluation averages over several partial queries, one fixed $1/C$ is generally not the
overall ceiling. Average the query-specific Bayes success probabilities using the same
query distribution and weights as the evaluation. This distinction matters when constraints
or class imbalance make some revealed assignments easier than others.

### Worked numerical example

There are four factors with level counts 2, 3, 2, and 4. A query reveals the first and
third factors. The unknown factors have 3 and 4 levels, so $C=3\times4=12$ completions.
Under a uniform conditional distribution, exact identification cannot exceed $1/12$.
Knowing one additional value from the 4-level factor reduces the candidates to 3 and
raises the ceiling to $1/3$.

## 12. Failure modes and efficiency notes

1. **Changing factor order:** integer and bit encodings silently change meaning.
2. **Ignoring impossible cells:** a Cartesian product may include combinations ruled out
   by physics or protocol. Filter them and call the result a constrained state space.
3. **Assuming a uniform prior:** $1/C$ is valid only under equal conditional probability.
4. **Confusing factor subsets with values:** store both known indices and assignments.
5. **Materializing huge products:** iterate lazily with `itertools.product` or use masks.
6. **Using floats for exact counts:** keep integer counts or `Fraction` until the end.

The total number of cells grows exponentially for binary factors. Enumeration is excellent
for small spaces and for validating formulas. For large $F$, reason symbolically or sample
states rather than allocating an array with $2^F$ rows.

## Exercises

### Exercise 1

Factors have 2, 2, 3, and 5 levels. How many complete cells exist?

**Brief solution:** $2\times2\times3\times5=60$ cells.

### Exercise 2

Encode the binary vector `(1, 1, 0, 1)` as an integer using the convention above.

**Brief solution:** $8+4+0+1=13$.

### Exercise 3

Six binary factors are present and two are known. What is the uniform exact-cell ceiling?

**Brief solution:** Four bits remain unknown, so there are $2^4=16$ completions and the
ceiling is $1/16$.

### Exercise 4

Why might a measured accuracy exceed the uniform ceiling without violating mathematics?

**Brief solution:** Compatible cells may have unequal conditional probabilities, or the
model may receive additional information not included in the ceiling calculation.

## Recap

A factorial state space is an address book of complete assignments. Cartesian products
multiply level choices, binary cells admit useful bit encodings, and subsets describe which
parts of an address are known. Exact arithmetic keeps finite probabilities honest. A
partial-information ceiling follows from how many states remain indistinguishable and from
their conditional probabilities.

## Continue

- Previous: [10. Regularized linear estimation and calibration](10_regularized_linear_estimation.md)
- Notebook: [11. Factorial state spaces](../implementations/11_factorial_state_spaces.ipynb)
- Next: [12. Blockwise distances and ranking](12_blockwise_distances_and_ranking.md)
- Curriculum: [Tutorial README](../README.md)
