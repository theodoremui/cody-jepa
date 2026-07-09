# Week 2.1: GaitLU-1M Storage And Extraction

GaitLU-1M is a large encrypted silhouette-sequence archive. The upstream README describes more than one million walking sequences, averaging 92 frames, for more than 92 million silhouette images. It is distributed as multipart zip data that is repaired into `GaitLU_Anno.zip` and decompressed only after obtaining the official password through the dataset agreement process.

The laptop path `../../Downloads/gaitlu-1m` is only an archive staging source. Do not fully extract GaitLU-1M on the laptop. Use HAIC storage and SLURM jobs for full extraction.

## 1. Keep The Directory Contract Simple

Use this layout on HAIC:

```text
$CODY_JEPA_DATA/
  gaitlu-1m/
    archives/       # encrypted multipart archive files
    raw/            # extracted .pkl sequence tree
    manifests/      # index and split CSV files
    diagnostics/    # contact sheets, summary tables, plots
    probe_exports/  # dummy or real frozen latent exports
```

In this repo, the matching local ignored path is:

```text
data/gaitlu-1m/
  archives/
  raw/
  manifests/
  diagnostics/
  probe_exports/
```

Everything under `data/` is ignored by git. Keep raw archives, extracted `.pkl` files, generated participant media, large notebook outputs, and latent exports there.

## 2. Inspect The Laptop Staging Folder

On the laptop, inspect names and sizes only:

```bash
cd ../../Downloads/gaitlu-1m
find . -maxdepth 2 -type f | sort
du -sh .
```

If the archive already contains `GaitLU_Anno.zip`, record that. If it contains multipart names such as `GaitLU_Anno_part.zip` plus split parts, repair on HAIC after transfer.

You can list zip members for a small check:

```bash
unzip -l GaitLU_Anno.zip | head
```

Do not run full extraction on the laptop.

## 3. Transfer Archives To HAIC

From the laptop, transfer the staging directory into the HAIC archive directory:

```bash
rsync -av --progress ../../Downloads/gaitlu-1m/ \
  <sunetid>@<haic-login-host>:"$GAITLU_ARCHIVE_DIR/"
```

If your shell does not expand `$GAITLU_ARCHIVE_DIR` on the remote side, replace it with the concrete HAIC path you chose after checking quota. Do not commit that machine-specific path.

After transfer, log into HAIC and verify:

```bash
cd "$GAITLU_ARCHIVE_DIR"
find . -maxdepth 2 -type f | sort
du -sh .
```

## 4. Repair The Multipart Zip If Needed

The official README shows this repair command:

```bash
zip -F GaitLU_Anno_part.zip --out GaitLU_Anno.zip
```

Use it only if your archive files match that naming pattern and `GaitLU_Anno.zip` is not already present.

Run repair inside a small SLURM job if the archive is large:

```bash
#!/bin/bash
#SBATCH --job-name=gaitlu_zip_repair
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${GAITLU_ARCHIVE_DIR:?Set GAITLU_ARCHIVE_DIR before sbatch}"

test -f GaitLU_Anno_part.zip
zip -F GaitLU_Anno_part.zip --out GaitLU_Anno.zip
ls -lh GaitLU_Anno.zip
```

Submit and monitor:

```bash
sbatch scripts/gaitlu_zip_repair.sbatch
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

## 5. Extract Only After Legal Password Access

The upstream README says the password is obtained by signing the release agreement and ethical requirement and sending them to the dataset administrator. Do not share, commit, echo, or save the password in a script.

Avoid command-line password flags such as `unzip -P` for the real archive because process arguments can be visible to other local inspection tools while the command runs. The safest portable pattern is to request a SLURM compute allocation, run `unzip` there, and type the password at the archive tool's prompt.

Example interactive compute allocation:

```bash
salloc \
  --job-name=gaitlu_extract \
  --time=08:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G

cd "${GAITLU_ARCHIVE_DIR:?Set GAITLU_ARCHIVE_DIR before sbatch}"
mkdir -p "${GAITLU_EXTRACTED_DIR:?Set GAITLU_EXTRACTED_DIR before sbatch}"
unzip GaitLU_Anno.zip -d "$GAITLU_EXTRACTED_DIR"
exit
```

If HAIC provides a managed secret mechanism that is approved for batch jobs, use it with a local `sbatch` extraction script. Keep the secret outside the repo, outside shell history, and outside logs. The script should still keep directives at the top:

```bash
#!/bin/bash
#SBATCH --job-name=gaitlu_extract
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${GAITLU_ARCHIVE_DIR:?Set GAITLU_ARCHIVE_DIR before sbatch}"
mkdir -p "${GAITLU_EXTRACTED_DIR:?Set GAITLU_EXTRACTED_DIR before sbatch}"

# Replace this placeholder with HAIC-approved secret retrieval.
# Do not put the password directly in this file or on a command line.
echo "Run extraction with HAIC-approved secret handling, or use the interactive allocation above."
exit 2
```

The invariant is that the password does not enter git, notebooks, shell history, process arguments, or logs.

## 6. Discover The Extracted Pickle Root

The official README shows the extracted root as:

```text
silhouette_cut_pkl/
  000/
    000/
      000/
        000_000_000.pkl
```

Local staging archives may show a different top-level name, such as:

```text
anonymized_sil/
  ...
    sequence.pkl
```

Do not hardcode either name in later code. Discover the root by scanning for `.pkl` files:

```bash
find "$GAITLU_EXTRACTED_DIR" -maxdepth 6 -type f -name '*.pkl' | head
```

Then use Python root discovery in the notebooks:

```python
from pathlib import Path
import os

def discover_pkl_root(base):
    base = Path(base)
    for name in ("silhouette_cut_pkl", "anonymized_sil"):
        candidate = base / name
        if candidate.exists() and any(candidate.rglob("*.pkl")):
            return candidate

    pkls = sorted(base.rglob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files under {base}")

    common = Path(os.path.commonpath([str(p.parent) for p in pkls[:1000]]))
    return common
```

## 7. First Extraction Checks

After extraction, run these as a small SLURM job:

```bash
find "$GAITLU_EXTRACTED_DIR" -type f -name '*.pkl' | wc -l
find "$GAITLU_EXTRACTED_DIR" -maxdepth 4 -type d | head
du -sh "$GAITLU_EXTRACTED_DIR"
```

A partial extraction can still leave many directories, so do not trust folder presence alone. The next tutorial creates an index with `read_ok` and `error` columns so failed reads are visible.

## Week 2 Gate For This Lesson

This lesson passes when:

1. The archive files live under `GAITLU_ARCHIVE_DIR` on HAIC.
2. Full extraction is planned as a SLURM job, not a login-node command.
3. The pickle root is discovered by scanning, not by assuming one fixed top-level name.
4. The companion notebook proves the synthetic Week 2 data gate: reproducible splits, reproducible validation batches, and `[B, T, C, H, W]` batch shape.
