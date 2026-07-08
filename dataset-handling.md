# Health&Gait Dataset Handling Tutorial

This tutorial explains how to download the full Health&Gait dataset and prepare it for CoDy-JEPA experiments without mixing self-supervised training data with evaluation metadata.

The guiding rule is simple:

$$
\mathcal{M}_{\text{ssl}} \cap \mathcal{M}_{\text{eval}} = \emptyset
$$

The self-supervised loader should see video-like frame sequences and only the minimal routing information needed to find those frames. Subject identity, anthropometrics, gait labels, speed labels, and other evaluation targets should be used only for split construction, diagnostics, and frozen-representation probes.

## 1. What The Dataset Contains

Health&Gait is hosted on Zenodo:

- Zenodo record: <https://zenodo.org/records/14039922>
- Local upstream reference code: [`healthgait/`](healthgait/)
- Dataset-use agreement: [`healthgait/DUA.txt`](healthgait/DUA.txt)
- Upstream README: [`healthgait/README.md`](healthgait/README.md)

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

The repo already ignores `data/`, `*.zip`, and split archive files matching `*.z[0-9][0-9]` in [`.gitignore`](.gitignore). Keep it that way. Do not commit raw data, frame grids that reveal participants, extracted archives, or latent exports that include subject-identifying metadata.

## 3. Review Data-Use Constraints

Before downloading the full dataset, read [`healthgait/DUA.txt`](healthgait/DUA.txt). In practical terms:

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

The upstream loader in [`healthgait/code/train/train_MoviNet_classification.py`](healthgait/code/train/train_MoviNet_classification.py) expects modality-specific frame directories. Its path logic is approximately:

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
