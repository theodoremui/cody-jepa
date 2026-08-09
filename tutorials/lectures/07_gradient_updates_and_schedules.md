# 07. Gradient updates and parameter schedules

![Overview of a stable parameter update](../images/07_gradient_updates_and_schedules.svg)

## Why this lesson matters

Suppose a model fits one batch well, but its loss becomes unstable after you increase the batch size, enable mixed precision, or add gradient accumulation. Each technique seems small, yet their order and scaling determine the actual parameter update.

This lesson follows one update from a scalar loss to new parameter values. We first build backpropagation from the chain rule. We then add microbatch accumulation, adaptive moments, decoupled weight decay, parameter groups, clipping, a learning-rate schedule, and mixed precision.

## Prerequisites

You should know algebra and the idea of a derivative as a local slope. [Lesson 06](06_representation_collapse.md) provides losses that an optimizer can minimize.

## Learning goals

By the end of this lesson, you will be able to:

1. Compute a small gradient by hand and verify it with autograd.
2. Explain why gradient accumulation requires loss scaling.
3. Derive Adam moment estimates and bias correction.
4. Distinguish L2 regularization from AdamW weight decay.
5. Use parameter groups and global norm clipping.
6. Build a linear-warmup and cosine-decay schedule.
7. Place mixed-precision scaling and clipping in the correct order.

## 1. A gradient is a local sensitivity

Consider a one-parameter model:

$$
\widehat y=wx.
$$

The input $x$ and parameter $w$ are scalars. The prediction is $\widehat y$. For target $y$, use squared-error loss:

$$
L=\frac{1}{2}(\widehat y-y)^2.
$$

The factor $1/2$ simplifies the derivative. Define the prediction error $e=\widehat y-y$. The loss derivative with respect to the prediction is

$$
\frac{\partial L}{\partial \widehat y}=e.
$$

The prediction derivative with respect to the parameter is

$$
\frac{\partial \widehat y}{\partial w}=x.
$$

The chain rule multiplies these local sensitivities:

$$
\frac{\partial L}{\partial w}
{}={}
\frac{\partial L}{\partial \widehat y}
\frac{\partial \widehat y}{\partial w}
{}={}
ex.
$$

If $w=2$, $x=3$, and $y=5$, then $\widehat y=6$, $e=1$, and the gradient is $3$. A small increase in $w$ increases the loss, so an optimizer should move $w$ downward.

~~~python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = torch.tensor(5.0)
loss = 0.5 * (w * x - y).square()
loss.backward()
assert w.grad.item() == 3.0
~~~

Autograd records the forward operations and applies the chain rule in reverse. It computes vector-Jacobian products without constructing a huge Jacobian matrix.

### From one scalar to a layer

For a linear layer, an input batch $X$ has shape $(B,D_{\mathrm{in}})$. A weight matrix $W$ has shape $(D_{\mathrm{in}},D_{\mathrm{out}})$, and a bias $b$ has shape $(D_{\mathrm{out}},)$. The layer output is

$$
H=XW+b.
$$

The output $H$ has shape $(B,D_{\mathrm{out}})$. Broadcasting adds the same bias row to all $B$ observations.

Backpropagation returns a gradient for every trainable object with the same shape as that object. The gradient $\nabla_W L$ has shape $(D_{\mathrm{in}},D_{\mathrm{out}})$, and $\nabla_b L$ has shape $(D_{\mathrm{out}},)$. Shape agreement is a valuable debugging invariant.

The gradient with respect to $W$ combines input values with upstream sensitivities. A feature that is always zero produces no data gradient for its corresponding weight row. A saturated activation can similarly make upstream sensitivity very small.

### Computation graphs and saved values

Reverse-mode autodifferentiation needs selected forward values during backward. For example, the derivative of a matrix product needs its inputs. This is why training uses more memory than inference.

Gradient checkpointing saves fewer intermediate activations and recomputes them during backward. It trades extra arithmetic for lower memory. This technique is different from saving a training checkpoint to disk.

## 2. The optimizer transforms a gradient into an update

The gradient describes the direction of fastest local increase. It is not itself the parameter update.

Plain gradient descent uses

$$
\theta_{t+1}
{}={}
\theta_t-\eta_t g_t.
$$

The vector $\theta_t$ contains all parameters at step $t$. The vector $g_t$ is the gradient at that step. The positive scalar $\eta_t$ is the learning rate. Subtraction moves against local increase.

With $\theta_t=2$, $g_t=3$, and $\eta_t=0.1$, the next value is $1.7$.

### Mental model

The gradient is a compass direction and the learning rate is the step length. Adaptive optimizers modify both using recent gradient history.

## 3. Gradients accumulate unless you clear them

In PyTorch, each call to <code>backward()</code> adds into a parameter's <code>.grad</code> buffer. This is intentional. It supports sums of several loss terms and several microbatches.

It also creates a common bug. If you forget to clear gradients between effective batches, the next update includes old gradients.

~~~python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
~~~

The option <code>set_to_none=True</code> avoids writing explicit zeros into every gradient tensor. It can reduce memory traffic and makes a missing gradient path visible as <code>None</code>.

## 4. Accumulation should reproduce a larger batch mean

For $N$ examples, let $\ell_i$ be the loss of example $i$. The batch mean is

$$
L_N=\frac{1}{N}\sum_{i=1}^{N}\ell_i.
$$

Differentiation is linear, so

$$
\nabla_\theta L_N
{}={}
\frac{1}{N}
\sum_{i=1}^{N}
\nabla_\theta \ell_i.
$$

Suppose memory holds only $K$ equal-sized microbatches. Each microbatch loss is already a mean. Dividing each microbatch mean by $K$ makes their accumulated gradients equal the full-batch mean gradient.

![Microbatch means must be scaled before accumulation](../images/07_gradient_accumulation.svg)

~~~python
optimizer.zero_grad(set_to_none=True)
for micro_x, micro_y in microbatches:
    loss = criterion(model(micro_x), micro_y) / len(microbatches)
    loss.backward()
optimizer.step()
~~~

Without division by $K$, the gradient is $K$ times larger. That silently changes the effective learning rate.

If microbatches have unequal sizes, do not divide every mean equally. Weight microbatch $k$ by $n_k/N$, where $n_k$ is its number of examples and $N=\sum_k n_k$.

### Conceptual checkpoint

Two microbatches contain 8 and 2 examples. Their mean gradients are $g_1$ and $g_2$. The full-batch mean gradient is

$$
\frac{8}{10}g_1+\frac{2}{10}g_2,
$$

not $(g_1+g_2)/2$.

### When accumulation does not exactly match one large forward pass

Even with correct loss scaling, microbatch accumulation can differ from a single large-batch forward pass.

Batch normalization computes means and variances separately for each microbatch. A large batch would use different statistics. Dropout draws different masks, although both procedures remain stochastic estimators. A loss that couples observations, such as covariance regularization, also changes when computed separately on microbatches.

Therefore accumulation exactly reproduces a large-batch mean only when each example's loss and forward computation are independent of other examples in the batch. Test equivalence on a deterministic model before assuming it.

## 5. Adam remembers direction and scale

Stochastic gradients vary from batch to batch and from coordinate to coordinate. Adam tracks two exponential moving averages.

The first moment is

$$
m_t
{}={}
\beta_1 m_{t-1}
+(1-\beta_1)g_t.
$$

The vector $m_t$ is a smoothed gradient. The scalar $\beta_1$ is close to one, commonly 0.9.

The second raw moment is

$$
v_t
{}={}
\beta_2 v_{t-1}
+(1-\beta_2)g_t^2.
$$

The square is elementwise. The vector $v_t$ tracks recent squared gradient scale. The scalar $\beta_2$ is often 0.999.

Both moments start at zero, so early values are biased toward zero. Bias correction divides out the missing exponential mass:

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
$$

Adam's normalized direction is

$$
u_t
{}={}
\frac{\widehat m_t}
{\sqrt{\widehat v_t}+\varepsilon}.
$$

All operations are coordinatewise. The small positive $\varepsilon$ prevents division by zero. Coordinates with consistently large gradients receive a larger denominator.

### First-step numerical example

Let $g_1=0.5$, $\beta_1=0.9$, and $\beta_2=0.99$. Then

$$
m_1=0.05,
\qquad
v_1=0.0025.
$$

Bias correction gives $\widehat m_1=0.5$ and $\widehat v_1=0.25$. Ignoring $\varepsilon$, the normalized direction is $0.5/\sqrt{0.25}=1$.

### What the moving averages remember

The first moment reduces rapid sign changes. If several gradients point in the same direction, $m_t$ builds momentum. If directions alternate, positive and negative contributions partially cancel.

The second moment reacts to magnitude regardless of sign. A coordinate with repeated large gradients develops a large $v_t$ and therefore a smaller normalized step. A coordinate with small but consistent gradients can receive a relatively larger step.

Adam is not completely scale-free because $\varepsilon$, finite history, clipping, and weight decay matter. Its adaptive behavior can also make training less sensitive to raw feature scales without removing the need for sensible initialization and normalization.

## 6. AdamW decouples weight decay

L2 regularization adds a penalty to the objective:

$$
L_{\mathrm{reg}}
{}={}
L+\frac{\lambda}{2}\lVert\theta\rVert_2^2.
$$

The positive scalar $\lambda$ controls penalty strength. Its gradient adds $\lambda\theta$ to the data gradient.

For plain stochastic gradient descent, this is equivalent to shrinking the parameter and applying the data gradient. In Adam, putting $\lambda\theta$ inside $g_t$ sends regularization through adaptive moment normalization. Different coordinates then receive history-dependent shrinkage.

AdamW applies decay directly:

$$
\theta_{t+1}
{}={}
(1-\eta_t\lambda)\theta_t
-\eta_t u_t.
$$

The first term shrinks the current parameter. The second applies the Adam direction. This separation makes the meaning of weight decay clearer.

![AdamW separates adaptive direction from parameter shrinkage](../images/07_adamw_and_schedule.svg)

## 7. Parameter groups encode deliberate exceptions

Not every parameter should use the same learning rate or decay. Bias vectors and normalization scale parameters are often excluded from weight decay. A newly initialized head can also use a larger learning rate than a pretrained encoder.

~~~python
decay, no_decay = [], []
for name, parameter in model.named_parameters():
    if parameter.ndim == 1 or name.endswith("bias"):
        no_decay.append(parameter)
    else:
        decay.append(parameter)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.05},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
~~~

Audit two invariants:

1. Every trainable parameter appears in exactly one group.
2. No frozen parameter appears by accident.

## 8. Global norm clipping limits rare spikes

Let $g_k$ be the gradient tensor for parameter tensor $k$. The global Euclidean norm is

$$
\lVert g\rVert_2
{}={}
\sqrt{\sum_k \lVert g_k\rVert_2^2}.
$$

Choose a threshold $c>0$. If the norm exceeds $c$, multiply every gradient by the same factor:

$$
g_k
\gets
g_k
\frac{c}{\lVert g\rVert_2}.
$$

Otherwise, leave every gradient unchanged. The division is evaluated only in the branch $\lVert g\rVert_2>c$, so the zero-gradient case never divides by zero. The shared multiplier preserves the gradient direction. Elementwise clipping does not.

~~~python
norm_before = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
~~~

Log the returned pre-clipping norm. If clipping happens almost every step, the learning rate or model dynamics may be unstable. Clipping should not hide a systematic problem.

## 9. Warmup and cosine decay control step length

Early optimizer moments are uncertain, and model activations can change rapidly. Linear warmup grows the learning rate over $W$ updates:

$$
\eta_t
{}={}
\eta_{\max}\frac{t+1}{W},
\qquad
0\le t<W.
$$

The integer $t$ is the update index. The scalar $\eta_{\max}$ is the peak learning rate.

There are $T-W$ decay updates with indices $W,W+1,\ldots,T-1$. One endpoint convention is

$$
p=\frac{t-W+1}{T-W},
\qquad
W\le t<T.
$$

The integer $T>W$ is the planned total number of updates. Under this convention, the final planned update has $p=1$. Cosine decay uses

$$
\eta_t
{}={}
\eta_{\min}
+\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(1+\cos(\pi p)\right).
$$

The first decay update is just below $\eta_{\max}$ when the decay phase contains many updates, and the final planned update is exactly $\eta_{\min}$. Clamp $p$ to $[0,1]$ if training might exceed $T$.

Test schedule endpoints explicitly. Off-by-one choices determine whether the first update uses zero, a warmup fraction, or the full base rate.

### Schedule units must match update units

A scheduler indexed by optimizer updates should advance only when an optimizer update succeeds. With $K$ accumulated microbatches, it advances once per $K$ microbatches, not once per microbatch.

If mixed-precision overflow causes a scaler to skip an optimizer update, the scheduler should also pause when its index is defined as the count of completed parameter updates. Advancing it despite a skipped update shortens the effective schedule.

### Numerical schedule example

Let $\eta_{\max}=0.001$ and $W=4$. The four warmup learning rates under the formula above are $0.00025$, $0.00050$, $0.00075$, and $0.00100$.

If training then decays to $\eta_{\min}=0.00005$, the final learning rate remains positive. A nonzero floor can support continued adaptation, while a zero floor brings updates toward a stop.

## 10. Mixed precision is an arithmetic policy

Float32 gives a wide numeric range and good precision. Lower-precision formats reduce memory traffic and can accelerate matrix multiplication on supported hardware.

- Float16 has a narrow exponent range, so small gradients can underflow.
- Bfloat16 keeps the float32 exponent range but has fewer fraction bits.
- Optimizer states and sensitive reductions are usually kept in float32.

Automatic mixed precision selects a suitable type for each operation:

~~~python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = criterion(model(x), y)
~~~

For float16, a gradient scaler multiplies the loss before backpropagation so tiny gradients become representable. Before clipping, gradients must be unscaled.

### Why loss scaling does not change the intended gradient

Let the scale be $s>0$. Backpropagation through $sL$ produces gradient $s\nabla_\theta L$. Dividing the stored gradients by $s$ before the optimizer step recovers the original gradient.

The benefit is numerical. Intermediate float16 gradient values are larger and less likely to underflow to zero. If overflow occurs, the scaler skips the update and lowers $s$.

Autocast and gradient scaling solve different problems. Autocast chooses arithmetic types. Gradient scaling protects small float16 gradients. Bfloat16 often does not need scaling because its exponent range is wider.

## 11. The complete update order

For accumulated float16 training:

1. Clear gradients.
2. Run each microbatch forward under autocast.
3. Divide or weight the microbatch loss correctly.
4. Scale the loss and call backward.
5. After all microbatches, unscale gradients once.
6. Clip the unscaled global gradient norm.
7. Take the optimizer step.
8. Update the scaler.
9. Advance the learning-rate scheduler only if the optimizer update succeeded.

Clipping scaled gradients applies the threshold in the wrong units. Advancing a step-based scheduler once per epoch makes the schedule far slower than intended.

### Fixed exposure makes model comparisons interpretable

The number of examples processed is part of an experiment, not merely a runtime detail.
If the effective batch size is $B_{mathrm{eff}}$ and training completes $U$ updates,
then sampled-example exposure is

$$
C=B_{mathrm{eff}}U.
$$

Two conditions can use different data support while receiving the same $C$, $U$, batch
size, optimizer, and schedule. Repeated examples still count toward exposure. They do
not become new support simply because the optimizer sees them again. Stopping every run
at the same planned update also makes the final checkpoint rule concrete.

Hardware speed should change wall-clock time, not scientific exposure. If cost requires
a smaller exposure, choose one common tier with an outcome-blind eight-job concurrent
throughput probe before training outcomes are viewed. The frozen rule selects 8,192,000
examples only when all eight jobs sustain at least 60 examples per second per GPU. It
selects 4,096,000 when all eight sustain at least 30 but at least one is below 60. It
cancels below 30 or when shared-storage performance is unstable. The same selected tier
then applies to every model. Choosing a tier separately for each condition would confound
the condition with exposure.

### Resume provenance is part of optimizer state

An exact resume needs more than model weights and Adam moments. Before loading, require
the saved and requested metadata to contain the same frozen field set. Compare the
manifest digest, sequence-support and window-policy labels, total exposure, effective
batch size, completed update, optimization seed, and versions of the sequence, temporal,
spatial, and mask streams. Also restore the scheduler and mixed-precision scaler. Stop on
any missing, extra, or changed field. Continuing with different provenance creates a new
training trajectory while retaining stale optimizer history.

## 12. Verify the update pipeline in layers

Before long training, test small invariants:

1. Compare a hand-derived scalar gradient with autograd.
2. Compare one deterministic full batch with correctly scaled microbatch accumulation.
3. Confirm every trainable parameter has a finite gradient or an intentional <code>None</code>.
4. Confirm parameter groups cover each trainable tensor once.
5. Evaluate schedule values at the first step, warmup boundary, and final step.
6. Force a large gradient and confirm clipping limits the post-clipping norm.
7. Save and reload a checkpoint, then compare the next update.

These checks isolate errors before model stochasticity makes them difficult to diagnose.

## 13. Efficiency notes

- Use vectorized model operations instead of loops over examples.
- Keep <code>.item()</code> calls out of accelerator hot loops because they can synchronize execution.
- Build the optimizer and scheduler once.
- Use <code>set_to_none=True</code> when compatible with the training loop.
- Save model, optimizer, scheduler, scaler, and update count for exact resumption.
- Profile before enabling compilation or fused kernels.

## 14. Common failure modes

1. **Forgotten clearing:** old gradients leak into new effective batches.
2. **Unscaled accumulation:** update magnitude grows with the number of microbatches.
3. **Unequal microbatches weighted equally:** the gradient no longer represents an example mean.
4. **Decay on every tensor:** biases and normalization parameters can be harmed.
5. **Clipping before unscaling:** the threshold has the wrong meaning.
6. **Scheduler at the wrong frequency:** an update schedule becomes an epoch schedule.
7. **Only weights in checkpoints:** Adam moment history and schedule position are lost.
8. **Unequal exposure across conditions:** compute and data support change together.
9. **Unchecked resume metadata:** a changed manifest or seed version enters an old trajectory.

## 15. Exercises

1. Three equal microbatches produce mean gradients $g_1$, $g_2$, and $g_3$. What gradient matches the combined mean?
2. Why does global norm clipping preserve direction?
3. At Adam's first step, what is $\widehat m_1/\sqrt{\widehat v_1}$ for a nonzero scalar gradient when $\varepsilon$ is ignored?
4. Why must the optimizer step precede the usual scheduler step in PyTorch?

### Brief solutions

1. Accumulate $(g_1+g_2+g_3)/3$.
2. Every coordinate is multiplied by the same positive scalar.
3. It is $g_1/|g_1|$, the sign of the gradient.
4. Standard schedulers are defined around completed optimizer updates. Reversing the order can skip or shift the first scheduled value.

## Recap

Backpropagation computes sensitivities by reverse chain rule. An optimizer transforms those sensitivities into parameter changes. Accumulation, adaptive moments, weight decay, clipping, scheduling, and mixed precision must share consistent units and update boundaries.

## Next lesson

[08: Group-aware sampling and shortcut learning](08_group_aware_sampling.md) turns from the update rule to the data units that supply each update.

## Continue in the notebook

[Open the executable lesson 07 notebook](../implementations/07_gradient_updates_and_schedules.ipynb) to verify accumulation, inspect AdamW groups, and run a complete scheduled update loop.
