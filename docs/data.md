# Data

## Access and permitted use

Health&Gait is human-participant data and is not redistributed with this repository.

- Dataset release: [Zenodo record 14039922](https://zenodo.org/records/14039922)
- Provider code and file descriptions: [AVAuco/healthgait](https://github.com/AVAuco/healthgait)
- Data-use agreement: [DUA.txt](https://github.com/AVAuco/healthgait/blob/main/DUA.txt)
- Dataset paper: [Health & Gait: A Dataset for Gait-Based Analysis](https://www.nature.com/articles/s41597-024-04327-4)

Read the provider's agreement before downloading. Use the data only under its permitted research terms. Do not redistribute archives, frames, participant tables, derived media, or participant-level embeddings. Do not attempt re-identification or participant contact.

Store the extracted release under the ignored local data tree:

```text
data/
  healthgait/
    raw/Health_Gait/
    manifests/
    processed/
    diagnostics/
    probe_exports/
```

The project keeps `data/`, checkpoints, and generated participant-level outputs outside Git.

## Dataset structure

Health&Gait follows participants as they walk at usual and fast instructed speeds, with and without a jacket when available. Each back-and-forth recording is separated into right-to-left and left-to-right direction clips. The release provides silhouettes, body-part segmentation, two-dimensional pose, and optical flow derived from the source videos.

The levels are:

```text
participant
  -> speed and jacket condition
      -> source back-and-forth recording
          -> direction clip
              -> ordered frames
```

Direction clips from the same source recording are related observations. Frames and sampled windows are parts of a clip, not independent trials or participants.

The primary model input is the silhouette sequence. Silhouettes remove most scene texture but retain body shape, jacket outline, frame count, position, foreground area, and motion. These remaining cues motivate the shortcut comparison.

## Factor labels

A path such as

```text
silhouette/PA000/FGS/WJ_2_YOLOV8/027.jpg
```

encodes participant `PA000`, fast instructed speed (`FGS`), jacket (`WJ`), left-to-right direction (`2` or `L2R`), and one ordered frame. `UGS` denotes usual speed, `WoJ` no jacket, and direction `1` right to left (`R2L`).

Code must read these values from a checked manifest rather than infer a missing condition from a nearby directory. Jacket trials were collected when possible, so not every participant is guaranteed to have all eight cells.

## Manifest and subject split

After placing the extracted data at the path above, run:

```bash
uv run python scripts/build_healthgait_manifest.py --fps 30
```

The example uses 30 frames per second. Confirm the frame rate for the local release and supply it explicitly: the builder does not infer timing from filenames, and duration is one of the shortcut controls.

With the default 16-frame model clips, a direction recording needs at least 18 contiguous frames so feature export can produce three distinct deterministic windows. The builder excludes shorter recordings; it never repeats a window to reach the configured count.

The training manifest is written to `data/healthgait/manifests/silhouette_subject_split_seed0.csv`. A GFC-ready recording table contains at least:

```text
subject_id,recording_id,source_video_id,direction_clip_id,speed,clothing,direction,frame_dir,num_frames,fps,split
```

It also records `source_video_id` for the shared back-and-forth recording and `direction_clip_id` for the direction-specific frame folder. The two direction clips from one back-and-forth recording share a `source_video_id`; `direction_clip_id` equals the direction-level `recording_id`.

Shortcut values are computed from every decoded silhouette frame in a direction clip. They are log frame count, duration (`num_frames / fps`), signed endpoint horizontal-centroid displacement (last frame minus first, using horizontal coordinates normalized by `width - 1`), its absolute magnitude, and the mean, population standard deviation, 25th percentile, median, and 75th percentile of foreground-area fraction. The foreground threshold defaults to `0.5` on grayscale values scaled to `[0, 1]`.

Paths should be relative to the repository or configured data root, not machine-specific absolute paths.

The split unit is the participant. No participant may appear in both training and held-out subsets. The baseline uses a deterministic seed-0 80/20 subject split. The GFC configuration maps `train` to development fitting, `val` to development evaluation, and `test` to confirmation. If no confirmation subset is available, only the development analysis can be run.

The manifest builder and loader check metadata columns, minimum clip length, frame counts, filename order, and image readability. Deterministic validation windows allow comparisons across runs; training can use seeded random windows.

## Model and evaluation boundaries

The self-supervised encoder consumes only frame sequences. Participant and factor labels are used outside the encoder for grouping, subject-safe splits, query construction, adapters, probes, and evaluation.

Three deterministic feature windows from one recording are averaged before GFC. Direction clips from one back-and-forth walk remain related observations and must not be described as independent capture sessions.

Adapters, coordinate selection, standardization, principal components, imputation, and probe fitting use training participants only. Held-out values cannot guide feature selection or preprocessing. A complete Health&Gait GFC participant must have one valid recording in each of the eight speed-by-clothing-by-direction cells.

## Participant and gait tables

The participant table contains a replacement participant key and variables such as age, recorded sex, height, weight, body-mass index, waist, hip, and neck circumference. These values can be sensitive and never enter the self-supervised encoder. Any adjustment model fits them within training folds only.

`gait_parameters.csv` contains six instrumented measurements for usual and fast instructed speed:

- step length;
- stride length;
- cadence;
- single-limb support time;
- bipedal support time;
- velocity.

These measurements were collected with OptoGait and MuscleLAB. They are participant-by-speed summaries and were not synchronized frame by frame with the videos. Missingness must be recomputed for the local release and reported for each analysis.

`gait_parameters_estimation.csv` contains camera-derived estimates. Because those estimates come from the same visual source as the model input, they are not an independent external criterion.

## Privacy and release rules

- Limit access to authorized researchers.
- Keep archives, extracted frames, participant tables, and participant-level feature exports out of Git.
- Publish aggregate results rather than participant rows.
- Review any proposed embedding release because representations can retain identity and body information.
- Preserve the provider's required attribution.
- Do not describe a result as clinical or population-general without evidence designed for that claim.

## Dataset limitations

- One direction clip per factor cell does not measure same-cell repeatability.
- A within-participant gallery does not test identity invariance across people.
- Silhouettes retain body-shape, duration, framing, and motion shortcuts.
- Instrumented gait values are not synchronized to each video pass.
- The controlled side-view cohort does not establish performance in other populations or cameras.
- A Health&Gait result alone cannot establish that GFC generalizes to other domains.
