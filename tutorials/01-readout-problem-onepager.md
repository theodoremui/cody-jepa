# The Readout Problem, on one page

**Proposal 1 for ICLR 2027.** Abstract due September 18, paper due September 25, 2026. Full version: [01-readout-problem.md](01-readout-problem.md).

---

## Why this matters

Self-supervised training saves hundreds of copies of a model and somebody has to keep one. There are no labels to choose with, so the field picks using a single number computed from the model alone.

Across our eleven runs, that does not work. Training loss predicts which model is actually useful at rank correlation **0.187** (p = 0.58). Token effective rank, the standard geometric score, gives **minus 0.100**. Both are noise, on a decision nobody can avoid.

## What the question is

A model describes one clip with 2,048 vectors. Nothing downstream uses 2,048 vectors, so every use first applies a map, usually an average.

Averaging is linear, so it has a kernel. With $x_t = \mu + d_t$ and $\sum_t d_t = 0$, everything about how patches differ *within* a clip is annihilated; only differences *between* clips survive. So the variance splits exactly in two. A per-patch classifier never averages, so it lives on the half that averaging destroys.

> ### Do the two halves grow independently while the model trains?
>
> If they do, no single number can rank models, because a model can improve for one use and get worse for another at once. **Every score the field uses is a single number.**

![One encoder, two readouts, two quality axes](images/readout-problem.svg)

## Why it is novel

RankMe, LiDAR, LDReg and the rest disagree about which spectral quantity to compute. They agree, without saying so, that one number per model, computed in isolation, is the right shape of answer. An adversarial prior-art search found nothing that splits a model's output by what a given use can reach.

## Why the results make it significant

Recomputed from checkpoints already on disk, no new training.

| | position | within-clip, $\gamma$ | between-clip, $\beta$ |
|---|---:|---:|---:|
| random weights | 96.4% | 3.58% | 0.025% |
| trained, no regulariser | **91.7%** | 8.27% | **0.063%** |
| trained, clip-axis variance | 83.1% | 6.44% | 10.4% |

**The standard score measures the discarded half.** For the trained model, 91.7 percent of everything it computes is a fixed function of position, identical in every clip and contributing exactly zero to any average. What a pooled probe can reach is 0.063 percent.

**The halves appear to move independently.** Between two checkpoints $\beta$ rose by a factor of **196** while $\gamma$ *fell* 8 percent. Not a trade-off, and the question's answer in miniature.

## Testable now, and killable now

If the halves are independent, **the model you pick by averaging is the wrong model for a per-patch task**, wrong in ranking with no warning. The labels are already on disk as per-pixel body-part maps aligned to the inputs, so the second readout costs no new annotation.

If instead $\beta$ and $\gamma$ are tightly coupled, one number suffices and the thesis is false. **That check needs no labels and runs in week one.**

## Honest state

Independence rests on two checkpoints and the margin is lopsided, since $\beta$ moves 196-fold while $\gamma$ moves 8 percent, so the dense half of the prediction could vanish. The supporting mechanism is prior art. Novelty is roughly 7 out of 10 today, 8.5 if the crossover appears.
