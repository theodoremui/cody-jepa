# CoDy-JEPA Proposal, v3

## Learning Motion Without Memorizing Who Moves

Tutorials:

- [CoDy-JEPA 10-Week Execution Plan](tutorials/cody-jepa-10-week-plan.md)
- [Week 2 HAIC + GaitLU-1M Tutorial Track](tutorials/week2-haic-gaitlu-index.md)
- [Health&Gait Dataset Handling Tutorial](tutorials/healthxgait-guide.md)

CoDy-JEPA is a self-supervised video method for articulated systems: people walking, hands gesturing, robot arms reaching, or assistive devices moving with a user. Its main claim is simple. A useful motion model should learn **how the system moves** without hiding too much information about **who or what is moving** inside the motion representation.

This matters because video datasets often contain shortcuts: decision rules that work on familiar benchmark splits but fail when users, scenes, or cameras change [6]. In a gait dataset, one person may always appear in one room and walk at one speed. A standard model can then learn "this room means slow walking" instead of learning gait phase or cadence. CoDy-JEPA attacks that failure mode by splitting each clip into two latent summaries and training them with a counterfactual prediction task.

![Stable and dynamic factorization](images/05-v3-factorization.svg)

The two summaries have different jobs. `S_attr` is the **stable attribute summary**: body proportions, clothing, camera view, background, robot morphology, or tool shape. `S_dyn` is the **dynamics summary**: pose transition, gait phase, cadence, velocity, contact state, or gesture phase. These factors are not perfectly independent in the real world. For example, body shape constrains walking style. The proposal treats the split as a useful inductive bias, not as a claim that nature gives clean labels for structure and motion.

## Method

Let a video clip be $x_{1:T}$, where $T$ is the number of frames. A video encoder $f_\theta$ maps it to spatiotemporal tokens:

$$
z_{t,p}=f_\theta(x_{1:T})_{t,p},
$$

where $t$ indexes time and $p$ indexes the token slot being tracked within each frame. In a patch-based video transformer, $p$ can mean a spatial image patch, such as the lower-left region containing a foot. In a pose-based model, $p$ can mean a body part or joint, such as the left knee. In a learned-token model, $p$ can mean a slot that the network learns to attach to a recurring visual factor. CoDy-JEPA estimates how much each token slot changes over time:

$$
\bar z_p = \frac{1}{T}\sum_{t=1}^{T} z_{t,p}, \qquad
v_p = \frac{1}{T}\sum_{t=1}^{T}\lVert z_{t,p}-\bar z_p\rVert_2^2 .
$$

Low-variation tokens are routed toward the attribute encoder $g_a$. High-variation tokens are routed toward the dynamics encoder $g_d$:

$$
S_{\text{attr}} = g_a(\left\lbrace z_{t,p}: v_p \le \tau_a \right\rbrace), \qquad
S_{\text{dyn}}(1:t) = g_d(\left\lbrace z_{s,p}: s \le t,\ v_p \ge \tau_d \right\rbrace).
$$

In a walking clip, the stable stream may receive torso shape, clothing, and camera view. The dynamics stream may receive alternating foot positions and knee motion. In a robot clip, stable tokens may describe link lengths and the gripper, while dynamic tokens describe joint phase and contact.

The core intervention is **Cross-Instance Token-Swapping Intervention**, or CI-TSI. Choose two clips, A and B, from the same broad domain. CoDy-JEPA builds a counterfactual context by combining stable attributes from A with motion history from B:

$$
C_{A \leftarrow B} = [S_{\text{attr}}(A), S_{\text{dyn}}(B,1:t)].
$$

The predictor $q_\psi$ must forecast the future dynamics of B, not future pixels:

$$
\hat{S}_{\text{dyn}}(B,t+k)=q_\psi(C_{A \leftarrow B}, k).
$$

The target comes from a slowly updated target encoder, as in JEPA-style learning:

$$
\mathcal{L}_{\text{pred}} =
\left\lVert \mathrm{norm}(\hat{S}_{\text{dyn}}(B,t+k)) -
\mathrm{sg}\left(\mathrm{norm}(\bar{S}_{\text{dyn}}(B,t+k))\right)\right\rVert_2^2 .
$$

Here `sg` means stop-gradient. The model predicts a latent future summary rather than reconstructing RGB frames. That is the JEPA idea: learn by predicting representations, so the model can focus on predictable semantic and physical structure instead of every pixel detail [1, 2]. This differs from masked reconstruction methods such as MAE and VideoMAE, which learn strong features but still train against image or video reconstruction targets [4, 5].

![Counterfactual objective](images/06-v3-objective.svg)

CI-TSI makes identity and motion less reliable as a paired shortcut. If A gives the body and B gives the stride rhythm, then "Person A usually walks slowly" no longer solves the task. The model must preserve the motion information from B while treating A's stable context as a condition, not as the answer.

Prediction alone is not enough, because both streams could still duplicate the same information. CoDy-JEPA therefore uses HSIC, the Hilbert-Schmidt Independence Criterion, to penalize dependence between minibatch summaries [7]. For a batch of $n$ clips, define Gram matrices $K_{ij}=k(S_{\text{attr}}^i,S_{\text{attr}}^j)$ and $L_{ij}=l(S_{\text{dyn}}^i,S_{\text{dyn}}^j)$. With $H=I_n-\frac{1}{n}\mathbf{1}\mathbf{1}^\top$,

$$
\mathrm{HSIC}(S_{\text{attr}},S_{\text{dyn}})=
\frac{1}{(n-1)^2}\mathrm{tr}(KHLH).
$$

If two examples are close in `S_attr` exactly when they are close in `S_dyn`, HSIC is large. Minimizing it discourages systematic overlap, such as identity leaking into the motion stream. To avoid trivial constant summaries, CoDy-JEPA also uses VICReg-style variance and covariance safeguards, following the broader redundancy-reduction lesson behind VICReg and Barlow Twins [8, 9]:

$$
\begin{gathered}
\mathcal{L} = \mathcal{L}_{\text{pred}} \\
{}+ \lambda_h \mathrm{HSIC}(S_{\text{attr}},S_{\text{dyn}}) \\
{}+ \lambda_v \mathcal{L}_{\text{var}} \\
{}+ \lambda_c \mathcal{L}_{\text{cov}} .
\end{gathered}
$$

The variance term keeps each latent dimension active across the batch. The covariance term reduces redundant dimensions within a stream. Together, these terms express the desired behavior: predict future motion, separate stable and dynamic information, and keep both summaries noncollapsed.

## Experiments

The main experiment should use unlabeled articulated video for pretraining and labels only for evaluation. Human gait is the clean first domain because it has clear stable factors, clear dynamic factors, and real HCI relevance. Silhouette data can be used first to reduce privacy and background leakage. RGB, pose, depth, and robot video can be added after the basic factorization is stable. MC-JEPA is the closest JEPA-family comparison because it explicitly studies motion and content features, while CoDy-JEPA makes the separation and leakage tests the main claim [3].

Evaluation should not ask only whether the training loss decreases. It should ask what information each stream contains after the encoders are frozen.

![Evaluation matrix](images/07-v3-evaluation.svg)

Use linear or shallow probes:

| Probe target | `S_attr` expected result | `S_dyn` expected result |
| --- | --- | --- |
| Subject identity, body shape, robot morphology | High | Low |
| Gait phase, cadence, speed, action state | Low | High |
| Camera, room, dataset source | Low or controlled | Low or controlled |

The central metric is the **leakage gap**. For example, if `S_dyn` predicts gait phase well but predicts identity poorly, that supports the claim. If `S_dyn` predicts both gait phase and identity, the model may still be using shortcuts. Transfer tests should then train probes on one set of users, views, or robot embodiments and evaluate on held-out ones. CoDy-JEPA should be compared with four baselines: a single-stream V-JEPA-style predictor, a dual-stream model without CI-TSI, a model without HSIC, and a model without variance or covariance safeguards.

The strongest result would show three things at once: competitive future-dynamics prediction, lower wrong-stream leakage than baselines, and better low-label transfer to new users or contexts. A weaker but still useful result would be diagnostic: if CI-TSI lowers identity leakage but hurts dynamics prediction, the method has found the tradeoff that future work must solve.

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas. [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243). arXiv, 2023.

[2] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas. [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). arXiv, 2024.

[3] Adrien Bardes, Jean Ponce, Yann LeCun. [MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698). arXiv, 2023.

[4] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). arXiv, 2021.

[5] Zhan Tong, Yibing Song, Jue Wang, Limin Wang. [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602). arXiv, 2022.

[6] Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780). arXiv, 2020.

[7] Arthur Gretton, Ralf Herbrich, Alexander Smola, Olivier Bousquet, Bernhard Scholkopf. [Kernel Methods for Measuring Independence](https://www.jmlr.org/papers/v6/gretton05a.html). Journal of Machine Learning Research, 2005.

[8] Adrien Bardes, Jean Ponce, Yann LeCun. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). arXiv, 2021.

[9] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stephane Deny. [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230). arXiv, 2021.

[10] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Ratsch, Sylvain Gelly, Bernhard Scholkopf, Olivier Bachem. [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359). arXiv, 2018.

[11] Bernhard Scholkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, Yoshua Bengio. [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107). arXiv, 2021.
