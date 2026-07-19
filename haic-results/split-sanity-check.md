# HAIC Split Sanity Check

Source notebook: `haic-results/01-cody-jepa-haic.executed.ipynb`

## Verdict

The Health&Gait train and validation splits look truly separate at the subject,
path, and exact selected-frame-content levels. I found no evidence that exact
clips or exact frames leak from train into validation.

The lower validation loss in the executed notebook is better explained by the
measurement protocol than by split leakage:

- training loss is averaged online while weights are changing during the epoch;
- validation loss is measured after the epoch with the final epoch weights;
- training runs with dropout enabled, validation runs with dropout disabled;
- training uses random clip windows by epoch, validation uses center windows;
- train and validation use different mask RNG streams.

## Notebook Results Checked

The executed notebook reported:

- rows: 3,130
- train clips: 2,506
- validation clips: 624
- train subjects: 318
- validation subjects: 80
- subject overlap: `[]`
- corrupt frames: 0
- missing frames: 0
- short clips dropped: 0

The training history showed validation loss below training loss on all 25
epochs. Final printed values were train loss `0.0042`, validation loss `0.0037`.

## Structural Split Checks

Local manifest checks on
`data/healthgait/manifests/silhouette_subject_split_seed0.csv`:

- subjects appearing in both splits: 0
- manifest `subject_id` mismatches against `frame_dir` subject: 0
- duplicate frame directories: 0
- duplicate sequence keys: 0
- train-only subjects: 318
- val-only subjects: 80

Trial and gait-system counts are balanced:

- FGS: train 1,254, val 312
- UGS: train 1,252, val 312
- WJ_1: train 617, val 152
- WJ_2: train 617, val 152
- WoJ_1: train 636, val 160
- WoJ_2: train 636, val 160

## Exact Content Checks

The notebook trains for 25 epochs with 2,506 train sequences and one random
16-frame window per train sequence per epoch. Validation uses one centered
16-frame window for each of the 624 validation sequences.

Exact content audit:

- train windows checked: 62,650
- validation windows checked: 624
- selected unique frame files hashed: 250,490
- exact train/validation 16-frame clip hash matches: 0
- exact selected frame byte-hash overlap between train and validation: 0

This is the strongest leakage check: if the same files, copied frames, or copied
16-frame clips were crossing the split, they should appear here.

## Near-Duplicate Checks

I compared every validation center clip against the actual train windows used by
the 25 notebook epochs using compact 8x8x16 silhouette thumbnails, then refined
the top 12 candidates at the model input resolution, 72x72x16.

Low-resolution nearest-neighbor pass:

- nearest RMSE median: `0.0128`
- nearest RMSE min: `0.00664`
- nearest cosine median: `0.9935`
- nearest cosine max: `0.9980`
- nearest cosine > `0.999`: 0 validation clips

The low-resolution top candidates are phase-aligned walking silhouettes, which
is expected in this dataset. They are not exact copies, and all top candidates
use different subject IDs.

Full-resolution refinement for the top 12 candidates:

- 72x72 RMSE min: `0.0418`
- 72x72 RMSE median: `0.0537`
- 72x72 RMSE max: `0.0592`
- foreground IoU median: `0.8335`

For calibration, same-subject repeated-trial center clips inside a split can be
even closer than the top train/validation candidates:

- closest same-subject repeated-trial 72x72 RMSE: `0.0286`
- closest same-subject repeated-trial foreground IoU: `0.9170`

No exact duplicate patient-measure or gait-parameter rows were found across
train and validation subject IDs.

Visual contact sheet for the top candidates:

- `haic-results/split_nearest_pairs_top12.png`

## Why Validation Loss Is Lower

The notebook's `train_loss` and `val_loss` are not apples-to-apples metrics.

In `train_jepa`, the training metric is accumulated while `ctx_enc.train()` and
`predictor.train()` are active. That means dropout is active, and the average
mixes losses from earlier and later model states inside the epoch. The
validation metric is computed after the epoch with `ctx_enc.eval()`,
`predictor.eval()`, and `tgt_enc.eval()`, so dropout is disabled and the weights
are the final weights for the epoch.

The data path also differs. Training windows are randomized by epoch through
`train_dataset.set_epoch(epoch)` and `random_windows=True`; validation clips are
deterministic center windows. A local controlled run using the local checkpoint
from `outputs/cody-jepa-haic/01-cody-jepa.pt` supports this directionally:

- train random windows, eval mode: `0.007995`
- train center windows, eval mode: `0.006906`
- validation center windows, eval mode: `0.007105`
- train random windows, train mode: `0.008174`

That checkpoint is not the HAIC executed-notebook checkpoint: its stored history
has 10 epochs and 1,500 steps, while the HAIC notebook output has 25 epochs and
1,000 steps. Treat the local controlled run as a protocol sanity check, not as a
reproduction of the HAIC numbers.

## Recommended Follow-Up

Add an apples-to-apples metric to the notebook:

- `train_eval_center_loss`: final epoch weights, eval mode, train split, center windows
- `train_eval_random_loss`: final epoch weights, eval mode, train split, random windows
- `val_loss`: final epoch weights, eval mode, validation split, center windows

Also keep the exact split/content audit as a cheap guardrail before training:

- assert zero subject overlap;
- assert zero frame-directory overlap;
- assert zero exact selected-frame hash overlap;
- write a top-nearest visual contact sheet for perceptual inspection.
