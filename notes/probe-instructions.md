**Role** You are an expert programmer and AI research scientist.

**Task** You are to carefully and systematically implement a probe to check whether the current single-stream JEPA representations carry task-relevant information. Use the retained local artifacts in `outputs/jepa-v4`; use `haic-results/job_91108.ipynb` only as execution provenance. The failed 90881 and 91023 notebooks and `outputs/jepa-v3` were intentionally deleted and are not comparison inputs.

A good first probe runner should answer one narrow question:
Given frozen baseline features, what information is linearly recoverable from them?

Ultrathink on how to implement a clean version of the probe containing 3 main stages:


1. Export Frozen Features
Start by turning a checkpoint into a table of examples. Each row should represent one clip window.

Required columns:
sequence_id
split
subject_id
gait_system
trial
window_start
feature_0 ... feature_D
For the current baseline, use one feature vector per clip. The simplest good default is:
features = target_encoder(video, return_pre_norm=True)[1].mean(dim=1)

Important: export features under torch.inference_mode(), with all models in .eval(), and never backprop through the encoder. The probe must measure the representation, not fine-tune it.


2. Define Probe Tasks
Implement two first tasks.

Subject Identity Probe
Question: can a shallow classifier recover subject_id from the baseline feature?
This measures identity leakage or identity content. In the final CoDy-JEPA world, high subject identity in S_attr may be acceptable, but high subject identity in S_dyn would be bad. For the current single-stream baseline, it simply tells you how much identity is present in the shared representation.
One subtle point: a normal closed-set identity classifier cannot be trained on train subjects and evaluated on validation subjects if the subjects are disjoint. The classes do not overlap. So use two identity protocols:

identity_closed_set:
  Train and validate on different clips/windows from the same subject set.
  Measures whether identity is linearly recoverable.

identity_heldout_retrieval:
  For held-out validation subjects, build one or more labeled centroids per subject,
  then classify remaining held-out clips by nearest centroid.
  Measures whether unseen-subject clips cluster by identity.

Do not pretend closed-set accuracy across disjoint subjects is meaningful. It is not.

Gait-Speed Or Motion Proxy Probe
Question: can a shallow classifier recover gait_system, such as FGS versus UGS, or another motion proxy?
This is the first dynamics-oriented probe. It should use true subject-held-out validation:
Train probe on baseline features from train subjects.
Evaluate probe on baseline features from validation subjects.
This is meaningful because FGS and UGS labels exist in both splits, while subjects do not overlap. Good held-out performance suggests the representation contains motion-speed information that transfers across people instead of just memorizing identities.
If gait_system is too noisy or imbalanced, add a simple computed proxy from the clip, such as mean absolute frame difference. Bucket it into low/medium/high motion. But prefer gait_system first because it is already in the manifest and has a clear interpretation.


3. Train Simple Probe Models
Keep the first runner boring on purpose.

Use:
StandardScaler
LogisticRegression(max_iter=...)
or a tiny PyTorch linear layer. Scikit-learn is easiest if available. The probe should report:
task
feature_source
train_examples
val_examples
num_classes
majority_baseline
accuracy
balanced_accuracy
macro_f1
confusion_matrix (in the canonical JSON artifact; the CSV is a compact scalar summary)

For gait_system, prioritize balanced_accuracy because class balance may vary. For subject identity, report top-1 accuracy plus the majority baseline, and later add retrieval accuracy.


**Output** 

Suggested File Shape
Add a script like:
scripts/export_single_stream_features.py
scripts/eval_probes.py
First script: checkpoint plus loaders in, CSV/NPZ out.
Second script: feature table in, metrics JSON/CSV out.
That separation is useful because feature export is GPU/model work, while probe evaluation is fast CPU work.
