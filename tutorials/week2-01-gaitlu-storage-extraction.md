# Week 2.1: GaitLU-1M Storage And Extraction

GaitLU-1M is a large encrypted silhouette-sequence archive. The upstream README describes more than one million walking sequences, averaging 92 frames, for more than 92 million silhouette images. It is distributed as multipart zip data that is repaired into `GaitLU_Anno.zip` and decompressed only after you obtain the official password through the dataset agreement process.

The laptop path `../../Downloads/gaitlu-1m` is only a staging source for archive files. Do not fully extract GaitLU-1M on your laptop. Use HAIC or Sherlock storage and SLURM compute jobs for full extraction.

This lesson has one safety theme: know which machine owns each path.

```text
laptop download folder -> cluster archive directory -> cluster extracted-data directory
```

## 1. Keep The Directory Contract Simple

Use this layout on HAIC or Sherlock:

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

On Sherlock, Stanford's resources overview points users to shared compute through SLURM and to larger storage options such as scratch and Oak. For this tutorial, the practical rule is simple:

| Storage location | Use it for | Do not use it for |
| --- | --- | --- |
| `$HOME` | Code, shell configuration, small notes, and small smoke-test outputs. | Full GaitLU archives or extracted data. |
| `$SCRATCH` | Large temporary GaitLU archives, extraction, manifests, diagnostics, and job I/O. | Permanent archive storage. |
| Oak or project storage | Longer-lived shared research data, if your PI or group has access. | Data your group has not approved for that location. |

If your administrator says to use `$SCRATCH`, set `CODY_JEPA_DATA="$SCRATCH/cody-jepa-data"` and keep the GaitLU directories under that path.

Before you continue on HAIC or Sherlock, confirm that the variables from Lesson 0 are set. Run this from any directory on the login node:

```bash
haic-login$ echo "$GAITLU_ARCHIVE_DIR"
haic-login$ echo "$GAITLU_EXTRACTED_DIR"
haic-login$ echo "$GAITLU_MANIFEST_DIR"
```

Each `echo` prints one cluster path. If a line prints nothing, go back to Lesson 0 and set the variables before transferring data. If a path starts with `/cody-jepa-data`, stop and fix `CODY_JEPA_DATA` before creating directories.

## 2. Inspect The Laptop Staging Folder

This step runs on your laptop, not on HAIC or Sherlock. It only checks file names and sizes. It does not extract the dataset.

Run this from the repo root on your laptop, assuming the download folder is two directories above the repo:

```bash
laptop$ cd ../../Downloads/gaitlu-1m
laptop$ find . -maxdepth 2 -type f | sort
laptop$ du -sh .
```

Read each line:

| Line | Meaning |
| --- | --- |
| `cd ../../Downloads/gaitlu-1m` | Move your laptop shell into the local staging folder. If your download is elsewhere, use your actual path. |
| `find . -maxdepth 2 -type f | sort` | List files up to two directory levels below the current folder, then sort the names. |
| `du -sh .` | Print the total size of the current folder. `-s` means summary, and `-h` means human-readable units. |

If the archive already contains `GaitLU_Anno.zip`, record that. If it contains multipart names such as `GaitLU_Anno_part.zip` plus split parts, repair on the cluster after transfer.

You can list zip members for a small check. Run this from the laptop staging folder only if `GaitLU_Anno.zip` exists:

```bash
laptop$ unzip -l GaitLU_Anno.zip | head
```

`unzip -l` lists archive contents without extracting them. `head` shows only the first few lines. Do not run full extraction on the laptop.

## 3. Transfer Archives To HAIC

This command starts on your laptop. It copies files to cluster storage through SSH, but it does not open an interactive cluster shell.

Before running `rsync`, log into HAIC or Sherlock once and copy the exact path printed by `echo "$GAITLU_ARCHIVE_DIR"`. Then return to your laptop shell.

Run this from the repo root on your laptop if `../../Downloads/gaitlu-1m/` is the correct local source path:

```bash
laptop$ rsync -av --progress ../../Downloads/gaitlu-1m/ \
  <sunetid>@<haic-login-host>:/absolute/haic/path/to/gaitlu-1m/archives/
```

Read it piece by piece:

| Piece | Meaning |
| --- | --- |
| `laptop$` | The command starts on your laptop. Do not type this label. |
| `rsync` | A file-copy tool that can resume transfers and skip files that already match. |
| `-a` | Archive mode. Preserve directory structure, file times, and permissions where possible. |
| `-v` | Verbose mode. Print what is happening. |
| `--progress` | Show transfer progress for large files. |
| `../../Downloads/gaitlu-1m/` | The local source folder on your laptop. The trailing slash means copy the contents of the folder. |
| `<sunetid>@<haic-login-host>:` | The remote account and cluster login server. The colon separates the remote machine from the remote path. |
| `/absolute/haic/path/to/gaitlu-1m/archives/` | The destination folder on the cluster. Replace this with the exact path printed by `echo "$GAITLU_ARCHIVE_DIR"` on the login node. |

Using the concrete cluster path avoids a common quoting mistake where your laptop shell expands `$GAITLU_ARCHIVE_DIR` before `rsync` connects. Do not commit that machine-specific path.

After transfer, log into HAIC or Sherlock and verify the files. Start from any directory on your laptop:

```bash
laptop$ ssh <sunetid>@<haic-login-host>
```

Then run these on the login node from the archive directory:

```bash
haic-login$ cd "$GAITLU_ARCHIVE_DIR"
haic-login$ pwd
haic-login$ find . -maxdepth 2 -type f | sort
haic-login$ du -sh .
```

The verification commands mean:

| Line | Meaning |
| --- | --- |
| `cd "$GAITLU_ARCHIVE_DIR"` | Move into the cluster directory that should contain the encrypted archives. |
| `pwd` | Print the current cluster directory so you can confirm the location before inspecting files. |
| `find . -maxdepth 2 -type f | sort` | List the transferred files under the archive directory. |
| `du -sh .` | Print the total size of the transferred archive directory. |

## 4. Repair The Multipart Zip If Needed

Skip this section if `GaitLU_Anno.zip` is already present and valid.

The official README shows this repair command:

```bash
zip -F GaitLU_Anno_part.zip --out GaitLU_Anno.zip
```

Use it only if your archive files match that naming pattern and `GaitLU_Anno.zip` is not already present.

Because the archive is large, run repair as a SLURM job. Create a local script named `scripts/gaitlu_zip_repair.sbatch` in your cluster repo checkout.

First make sure the repo has local `scripts/` and `logs/` directories. Run this from the repo root on the HAIC or Sherlock login node:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ mkdir -p scripts logs
```

`scripts/` will hold your site-specific batch scripts. `logs/` will hold SLURM output and error files.

Save this content in `scripts/gaitlu_zip_repair.sbatch`:

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

The script lines mean:

| Line | Meaning |
| --- | --- |
| `#!/bin/bash` | Run the script with Bash. |
| `#SBATCH --job-name=gaitlu_zip_repair` | Give the job a clear name. |
| `#SBATCH --time=02:00:00` | Request two hours. |
| `#SBATCH --ntasks=1` | Request one task. |
| `#SBATCH --cpus-per-task=2` | Request two CPU cores. |
| `#SBATCH --mem=16G` | Request 16 GB of memory. |
| `#SBATCH --output=logs/%x-%j.out` | Write normal output to a job log under `logs/`. |
| `#SBATCH --error=logs/%x-%j.err` | Write errors to a matching log under `logs/`. |
| `set -euo pipefail` | Stop on common shell errors. |
| `cd "${GAITLU_ARCHIVE_DIR:?Set GAITLU_ARCHIVE_DIR before sbatch}"` | Enter the cluster archive directory and stop with a clear error if the variable is unset. |
| `test -f GaitLU_Anno_part.zip` | Fail early if the expected multipart source file is missing. |
| `zip -F GaitLU_Anno_part.zip --out GaitLU_Anno.zip` | Repair the multipart archive into `GaitLU_Anno.zip`. |
| `ls -lh GaitLU_Anno.zip` | Print the repaired file's size and permissions. |

Submit and monitor from the repo root on the HAIC or Sherlock login node:

```bash
haic-login$ cd "$CODY_JEPA_ROOT"
haic-login$ sbatch scripts/gaitlu_zip_repair.sbatch
haic-login$ squeue -u "$USER"
haic-login$ sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

`sbatch` submits the repair script to SLURM. The repair runs on a compute node when scheduled. `squeue` shows whether the job is pending or running. `sacct` shows what happened after the job finishes. Replace `<job_id>` with the numeric job ID printed by `sbatch`.

## 5. Extract Only After Legal Password Access

The upstream README says the password is obtained by signing the release agreement and ethical requirement and sending them to the dataset administrator. Do not share, commit, echo, or save the password in a script.

Avoid command-line password flags such as `unzip -P` for the real archive because process arguments can be visible to other local inspection tools while the command runs. The safest portable pattern is to request a SLURM compute allocation, run `unzip` there, and type the password at the archive tool's prompt.

Start from the HAIC or Sherlock login node. You can request the allocation from any directory:

```bash
haic-login$ salloc \
  --job-name=gaitlu_extract \
  --time=08:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G
```

This asks SLURM for an interactive compute shell. After the allocation starts, the prompt is labeled `haic-compute$` in this tutorial.

Run the extraction commands from the cluster archive directory on the compute node:

```bash
haic-compute$ cd "${GAITLU_ARCHIVE_DIR:?Set GAITLU_ARCHIVE_DIR before extraction}"
haic-compute$ mkdir -p "${GAITLU_EXTRACTED_DIR:?Set GAITLU_EXTRACTED_DIR before extraction}"
haic-compute$ unzip GaitLU_Anno.zip -d "$GAITLU_EXTRACTED_DIR"
haic-compute$ exit
```

Read the extraction lines:

| Line | Meaning |
| --- | --- |
| `cd "${GAITLU_ARCHIVE_DIR:?...}"` | Enter the cluster directory containing `GaitLU_Anno.zip`. Stop clearly if the variable is unset. |
| `mkdir -p "${GAITLU_EXTRACTED_DIR:?...}"` | Create the cluster extracted-data directory if it does not already exist. |
| `unzip GaitLU_Anno.zip -d "$GAITLU_EXTRACTED_DIR"` | Extract the archive into the extracted-data directory. Let `unzip` prompt for the password. |
| `exit` | End the compute allocation and return to the HAIC login shell. |

If HAIC or Sherlock provides a managed secret mechanism that is approved for batch jobs, use it with a local `sbatch` extraction script. Keep the secret outside the repo, outside shell history, and outside logs. The script should still keep directives at the top:

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

# Replace this placeholder with cluster-approved secret retrieval.
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

Do not hardcode either name in later code. Discover the root by scanning for `.pkl` files.

Run this on the HAIC or Sherlock login node for a light listing, or inside a compute allocation if the filesystem scan is slow. You can run it from any directory because the command uses the full extracted-data path:

```bash
haic-login$ find "$GAITLU_EXTRACTED_DIR" -maxdepth 6 -type f -name '*.pkl' | head
```

This finds pickle files under the extracted directory, limits the search depth to avoid a huge first scan, and prints only the first few matches.

The notebooks use Python root discovery instead of assuming one folder name:

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

The function first checks the two known top-level names. If neither works, it scans for `.pkl` files and returns a common parent directory. If no pickle files exist, it raises a clear error instead of silently indexing the wrong location.

## 7. First Extraction Checks

After extraction, run these checks inside a compute allocation if the dataset is large. Start from any directory because each command uses `$GAITLU_EXTRACTED_DIR`:

```bash
haic-compute$ find "$GAITLU_EXTRACTED_DIR" -type f -name '*.pkl' | wc -l
haic-compute$ find "$GAITLU_EXTRACTED_DIR" -maxdepth 4 -type d | head
haic-compute$ du -sh "$GAITLU_EXTRACTED_DIR"
```

The checks mean:

| Line | Meaning |
| --- | --- |
| `find "$GAITLU_EXTRACTED_DIR" -type f -name '*.pkl' | wc -l` | Count extracted pickle files. |
| `find "$GAITLU_EXTRACTED_DIR" -maxdepth 4 -type d | head` | Print a few extracted directories so you can see the nesting. |
| `du -sh "$GAITLU_EXTRACTED_DIR"` | Print the extracted data size. |

A partial extraction can still leave many directories, so do not trust folder presence alone. The next tutorial creates an index with `read_ok` and `error` columns so failed reads are visible.

## Week 2 Gate For This Lesson

This lesson passes when:

1. The archive files live under `GAITLU_ARCHIVE_DIR` on HAIC, Sherlock, or approved cluster storage.
2. Full extraction is planned as a SLURM job or compute allocation, not a login-node command.
3. The pickle root is discovered by scanning, not by assuming one fixed top-level name.
4. The companion notebook proves the synthetic Week 2 data gate: reproducible splits, reproducible validation batches, and `[B, T, C, H, W]` batch shape.
