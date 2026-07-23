# Health&Gait Dataset Handling Tutorial

This tutorial explains how to download the full Health&Gait dataset and prepare it for CoDy-JEPA experiments without mixing self-supervised training data with evaluation metadata.

Use uv exclusively for this workflow: install with `uv sync --frozen`, launch
all Python and Jupyter commands through `uv run`, and change dependencies only with `uv add`,
`uv remove`, or `uv lock`.

The guiding rule is simple:

$$
\mathcal{M}_{\text{ssl}} \cap \mathcal{M}_{\text{eval}} = \emptyset
$$

The self-supervised loader should see video-like frame sequences and only the minimal routing information needed to find those frames. Subject identity, anthropometrics, gait labels, speed labels, and other evaluation targets should be used only for split construction, diagnostics, and frozen-representation probes.

## 1. What The Dataset Contains

Health&Gait is hosted on Zenodo:

- Zenodo record: <https://zenodo.org/records/14039922>
- Upstream reference code: [AVAuco/healthgait](https://github.com/AVAuco/healthgait)
- Dataset-use agreement: [upstream `DUA.txt`](https://github.com/AVAuco/healthgait/blob/main/DUA.txt)
- Upstream README: [Health&Gait README](https://github.com/AVAuco/healthgait/blob/main/README.md)

The upstream README describes 1,564 videos from 398 participants, with these available modalities:

- `pose`: AlphaPose 2D pose JSON files.
- `semantic_segmentation`: DensePose PNG images.
- `optical_flow`: TVL1 and GMFlow PNG images.
- `silhouette`: YOLOv8 silhouette JPEG images.
- `participants_measures.csv`: subject-level anthropometric metadata.
- `gait_parameters.csv`: gait parameters from measurement systems.
- `gait_parameters_estimation.csv`: gait parameters estimated from pose.

For the first CoDy-JEPA implementation, use `silhouette` first. It has less background leakage than RGB-like data and is easier to inspect than optical flow. Add `semantic_segmentation` next, then pose and optical flow after the loader and diagnostics are stable.

## 2. Keep Files Cleanly Organized

Create one local data root. Do not put raw archives or extracted data inside `healthgait/`; that directory is upstream reference code.

```text
data/
  healthgait/
    archives/       # downloaded Zenodo archives, ignored by git
    raw/            # extracted Health&Gait dataset, ignored by git
    manifests/      # generated clip and subject indexes
    processed/      # optional caches or converted tensors
    diagnostics/    # sampled windows, frame diffs, motion maps, reports
    probe_exports/  # latent vectors and probe-ready labels
```

The repo already ignores `data/`, `*.zip`, and split archive files matching `*.z[0-9][0-9]` in [`.gitignore`](../.gitignore). Keep it that way. Do not commit raw data, frame grids that reveal participants, extracted archives, or latent exports that include subject-identifying metadata.

## 3. Review Data-Use Constraints

Before downloading the full dataset, read the upstream [Health&Gait data-use agreement](https://github.com/AVAuco/healthgait/blob/main/DUA.txt). In practical terms:

1. Use the dataset only for academic or research work unless the provider gives separate written permission.
2. Do not attempt to identify or contact participants.
3. Do not redistribute raw data, extracted frames, archives, or derived media.
4. Keep the dataset under reasonable access controls.
5. Attribute the dataset in publications and reports.

The code in this repo can be committed. The downloaded dataset and generated participant media should stay local.

## 4. Download The Full Dataset

The full Zenodo download is a multipart archive:

- `Health_Gait.z01` through `Health_Gait.z25`
- `Health_Gait.zip`

The small `dataset_samples.zip` is useful for smoke tests, but it is not the full dataset.

From the repo root:

```bash
mkdir -p data/healthgait/{archives,raw,manifests,processed,diagnostics,probe_exports}
cd data/healthgait/archives

for n in $(seq -w 1 25); do
  f="Health_Gait.z${n}"
  curl -L -C - -o "$f" "https://zenodo.org/records/14039922/files/$f?download=1"
done

curl -L -C - -o Health_Gait.zip "https://zenodo.org/records/14039922/files/Health_Gait.zip?download=1"
curl -L -C - -o dataset_samples.zip "https://zenodo.org/records/14039922/files/dataset_samples.zip?download=1"
```

Notes:

- `-L` follows Zenodo redirects.
- `-C -` resumes partially downloaded files.
- Keep all parts in the same directory before extracting.
- Extract from `Health_Gait.zip`, not from `Health_Gait.z01`.

## 5. Verify The Download

Save Zenodo's file checksums locally:

```bash
cd data/healthgait/archives

curl -s https://zenodo.org/api/records/14039922 \
  | jq -r '.files[] | [.key, .checksum] | @tsv' \
  > ../manifests/zenodo_checksums.tsv
```

Generate local checksums:

```bash
md5 Health_Gait.z* Health_Gait.zip dataset_samples.zip \
  > ../manifests/local_md5.txt
```

Before continuing, confirm:

1. Every part from `Health_Gait.z01` through `Health_Gait.z25` exists.
2. `Health_Gait.zip` exists.
3. Local checksums match the Zenodo metadata.
4. File sizes are nonzero and no download was interrupted.

If any part is missing or corrupt, the multipart archive may fail late during extraction.

## 6. Extract The Dataset

Use the final `.zip` file as the entry point:

```bash
cd data/healthgait/archives
unzip Health_Gait.zip -d ../raw
```

If the system `unzip` fails on the split archive, use 7-Zip:

```bash
7z x Health_Gait.zip -o../raw
```

After extraction, locate the dataset root:

```bash
cd ../../..

find data/healthgait/raw -maxdepth 4 \
  \( -name participants_measures.csv -o -name gait_parameters.csv -o -name gait_parameters_estimation.csv \)

find data/healthgait/raw -maxdepth 4 -type d \
  \( -name silhouette -o -name semantic_segmentation -o -name optical_flow -o -name pose \)
```

Record the resolved root path in local notes or a local config. For example:

```text
HEALTHGAIT_ROOT=data/healthgait/raw/Health_Gait
```

Do not hard-code a machine-specific absolute path in committed code.

## 7. Expected Dataset Shape

The upstream repository's [loading recommendations](https://github.com/AVAuco/healthgait#recommendations-to-load-and-use-the-dataset) expect modality-specific frame directories. Their path logic is approximately:

```text
DATA_ROOT/
  silhouette/
    PA000/
      UGS/
        NW-WJ_1_.../
        NW-WJ_2_.../
        NW-WoJ_1_.../
        NW-WoJ_2_.../
      FGS/
        ...
  semantic_segmentation/
    PA000/
      UGS/
        ...
  optical_flow/
    PA000/
      UGS/
        TVL1/
          ...
        GMFLOW/
          ...
```

The exact extracted root may differ slightly. The manifest builder should discover paths by scanning rather than assuming one fixed nesting forever.

## 8. Check The Extraction Before Training

Before creating train and validation splits, do one quick sanity check. This saves time because a partially extracted multipart archive can still leave many folders on disk.

From the repo root:

```bash
find data/healthgait/raw/Health_Gait -maxdepth 2 -type f -name '*.csv'

find data/healthgait/raw/Health_Gait/silhouette -mindepth 1 -maxdepth 1 -type d | wc -l
find data/healthgait/raw/Health_Gait/semantic_segmentation -mindepth 1 -maxdepth 1 -type d | wc -l
find data/healthgait/raw/Health_Gait/optical_flow -mindepth 1 -maxdepth 1 -type d | wc -l
```

You should see metadata CSV files such as `gait_parameters.csv` and `gait_parameters_estimation.csv`. You should also see a reasonable number of participant folders for the modality you plan to use.

If the CSVs are missing, or if a modality has far fewer participants than expected, test and re-extract the archive with 7-Zip:

```bash
cd data/healthgait/archives
7z t Health_Gait.zip
7z x Health_Gait.zip -o../raw
```

Do this before building final splits. It is fine to smoke-test code on a partial extraction, but do not report results from it.

## 9. Start With A Clip Manifest

The first useful artifact is a clip manifest. A manifest is a small CSV file where each row points to one trial directory. It does not copy images. It only records enough information to find frames later.

For example, a silhouette trial may look like this:

```text
data/healthgait/raw/Health_Gait/
  silhouette/
    PA045/
      UGS/
        WoJ_1_YOLOV8/
          001.jpg
          002.jpg
          ...
```

That trial should become one manifest row:

```text
subject_id,modality,gait_system,trial,frame_dir,num_frames
PA045,silhouette,UGS,WoJ_1_YOLOV8,data/healthgait/raw/Health_Gait/silhouette/PA045/UGS/WoJ_1_YOLOV8,120
```

Keep the manifest simple. For self-supervised training, it should include paths and routing fields, not labels that the model could accidentally learn from.

## 10. Create Subject-Level Train/Val Splits

Split by participant ID, not by frame and not by random windows.

This is the most important rule:

```text
No subject should appear in both train and validation.
```

The reason is simple. Adjacent clips from the same person share body shape, clothing, capture setup, and gait style. If one person's frames appear in both train and validation, the validation loss can look good even when the model is mostly recognizing the person.

Use an 80/20 subject split for the first run:

```text
train subjects: 80% of participants
val subjects:   20% of participants
```

Use a fixed random seed and write the chosen split into the manifest. Do not regenerate it differently for every run.

Here is a minimal manifest builder for one modality:

```python
from pathlib import Path
import csv
import random

root = Path("data/healthgait/raw/Health_Gait")
modality = "silhouette"
min_frames = 16
seed = 0
val_fraction = 0.2

rows = []

for trial_dir in sorted((root / modality).glob("PA*/**/*")):
    if not trial_dir.is_dir():
        continue

    frames = sorted(
        list(trial_dir.glob("*.jpg")) +
        list(trial_dir.glob("*.png"))
    )
    if len(frames) < min_frames:
        continue

    rel = trial_dir.relative_to(root / modality)
    if len(rel.parts) < 3:
        continue

    subject_id = rel.parts[0]
    gait_system = rel.parts[1]
    trial = rel.parts[-1]

    rows.append({
        "subject_id": subject_id,
        "modality": modality,
        "gait_system": gait_system,
        "trial": trial,
        "frame_dir": str(trial_dir),
        "num_frames": len(frames),
    })

subjects = sorted({row["subject_id"] for row in rows})
rng = random.Random(seed)
rng.shuffle(subjects)

n_val = max(1, round(val_fraction * len(subjects)))
val_subjects = set(subjects[:n_val])

for row in rows:
    row["split"] = "val" if row["subject_id"] in val_subjects else "train"

out = Path(f"data/healthgait/manifests/{modality}_subject_split_seed{seed}.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "subject_id",
        "modality",
        "gait_system",
        "trial",
        "frame_dir",
        "num_frames",
        "split",
    ])
    writer.writeheader()
    writer.writerows(rows)

train_subjects = {row["subject_id"] for row in rows if row["split"] == "train"}
val_subjects = {row["subject_id"] for row in rows if row["split"] == "val"}

print(f"wrote {len(rows)} clips to {out}")
print(f"train subjects: {len(train_subjects)}")
print(f"val subjects: {len(val_subjects)}")
print(f"overlap: {train_subjects & val_subjects}")
```

The final `overlap` line should print an empty set:

```text
overlap: set()
```

If it does not, stop and fix the split before training.

## 11. Use The Manifest In A Dataset Class

After you have a split manifest, the PyTorch dataset should read rows from the CSV and sample fixed-length windows from each trial directory.

The loader should return video tensors in this shape:

```text
[T, C, H, W]
```

Then the PyTorch `DataLoader` will batch them into:

```text
[B, T, C, H, W]
```

For self-supervised CoDy-JEPA training, the returned sample can stay minimal:

```python
{
    "video": video_tensor,
    "subject_id": subject_id,
    "trial": trial,
}
```

Use `subject_id` for diagnostics and checks, not as a training label. If you want the strictest self-supervised boundary, return `subject_id` only in validation or debug modes.

A simple loading policy is enough at first:

1. Sort frame filenames numerically.
2. Pick a contiguous window of `T` frames.
3. Resize each frame to a fixed size.
4. Convert grayscale or RGB images to a consistent channel format.
5. Normalize pixel values.

For training, sample random windows. For validation, sample deterministic windows. Deterministic validation makes loss curves easier to compare across runs.

## 12. Recommended First Split Setup

For the first CoDy-JEPA experiment, keep the setup narrow:

```text
modality:       silhouette, if extraction is complete enough
clip length:    16 or 32 frames
split unit:     subject_id
train fraction: 80% of subjects
val fraction:   20% of subjects
seed:           0
```

If `silhouette` is incomplete locally, use `semantic_segmentation` for the first full split and come back to silhouette after fixing extraction.

Do not mix these concepts:

```text
self-supervised train split:
  frame windows used to train CoDy-JEPA

self-supervised val split:
  held-out subject windows used to monitor representation learning

probe split:
  frozen embeddings plus metadata labels used after pretraining
```

The probe split can reuse the same subject-held-out discipline, but it has a different purpose. It measures what information the learned representation contains. It should not feed labels back into the self-supervised model.

## 13. First Run Checklist

Before launching a real training run, confirm:

1. The archive passes `7z t Health_Gait.zip`.
2. The extracted dataset contains the expected CSV metadata files.
3. The chosen modality has enough participant folders.
4. The manifest has one row per usable trial directory.
5. Train and validation subject sets have no overlap.
6. A batch from the dataset has shape `[B, T, C, H, W]`.
7. Validation sampling is deterministic.
8. No raw data, generated media, or participant-level exports are committed.

Once all eight checks pass, the dataset is ready for the first baseline training run.
