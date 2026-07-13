# Week 2 HAIC + GaitLU-1M Tutorial Track

Week 2 is the data-contract week for CoDy-JEPA. The goal is not to train a model yet. The goal is to prove that the data path is safe and repeatable before Week 3 adds a JEPA baseline.

By the end of these tutorials, you should know where the GaitLU-1M files live, how the files are extracted, how sequence manifests and splits are created, how the loader shapes video tensors, and how diagnostics and probe-export tables are written.

The full GaitLU-1M archive is too large for laptop-only work. Use your laptop only for archive inspection, file transfer, and tiny synthetic smoke tests. Use HAIC or Sherlock storage and SLURM jobs for extraction, indexing, diagnostics, and any full-dataset pass.

Stanford's SDSS-CC resources overview describes Sherlock as Stanford's shared HPC cluster. It also notes that public Sherlock partitions include `normal`, `gpu`, `dev`, `bigmem`, and `owners`, that some SDSS users may also use the `serc` partition, and that Oak storage is available at paths like `/oak/stanford/schools/ees/{PI SUNetID}` when your group has access. Treat those names as starting points, not guesses. Always discover what your account can actually use before submitting jobs.

## Read This Before Typing Commands

The tutorials use prompt labels to show where a command runs. Do not type the prompt label itself.

| Prompt label | Machine | Directory expectation | What belongs there |
| --- | --- | --- | --- |
| `laptop$` | Your own computer | Usually the repo checkout or your local download folder | Inspect local downloads, start file transfers, and open SSH connections. |
| `haic-login$` | A HAIC or Sherlock login node reached through SSH | Usually `$CODY_JEPA_ROOT` for repo commands, or a specific cluster storage directory for file checks | Edit small files, check storage, submit SLURM jobs, and monitor jobs. Do not run heavy computation here. |
| `haic-compute$` | A compute node allocated by SLURM | Usually `$CODY_JEPA_ROOT` for Python work, or `$GAITLU_ARCHIVE_DIR` for archive extraction | Run dependency sync, notebook execution, extraction, indexing, diagnostics, and training. |

Always answer two questions before pressing Enter:

1. Which machine am I controlling: laptop, cluster login node, or cluster compute node?
2. Which directory am I in?

Use these two commands whenever you are unsure:

```bash
hostname
pwd
```

`hostname` prints the machine name. `pwd` prints the current directory. These commands are safe and light enough to run on login nodes.

## SSH, File Transfer, And SLURM

SSH means Secure Shell. It opens a terminal on another machine.

Run this from any directory on your laptop:

```bash
laptop$ ssh <sunetid>@<haic-login-host>
```

Read the command line by line:

| Piece | Meaning |
| --- | --- |
| `laptop$` | This tells you the command starts on your laptop. Do not type it. |
| `ssh` | Start a secure terminal connection. |
| `<sunetid>` | Replace this placeholder with your Stanford account name. |
| `@` | Separates the username from the remote computer name. |
| `<haic-login-host>` | Replace this placeholder with the login hostname from your instructions. If you are using Sherlock directly, the Stanford overview shows `sherlock.stanford.edu`. |

After login, commands run on the cluster until you type `exit`.

If your prompt looks like `tedmui@haic:~$`, you are already on the login node. The `~` means your home directory. It is a directory, while `haic-login$` is only a tutorial label for the machine and shell you are using.

File transfer is separate from login. Run this from the laptop directory that makes the local source path correct:

```bash
laptop$ rsync -av --progress local_folder/ <sunetid>@<haic-login-host>:/remote/folder/
```

This copies files from your laptop to HAIC through SSH. `rsync` is usually safer than drag-and-drop for large datasets because it can resume a transfer and skip files that already match.

SLURM is also separate from SSH. SSH gets you to a login node. SLURM gets work onto compute nodes.

| Command | Where to run it | What it does |
| --- | --- | --- |
| `sbatch script.sbatch` | Cluster login node, usually from `$CODY_JEPA_ROOT` | Submit a batch script. The work runs later on a compute node. |
| `salloc ...` | Cluster login node | Request an interactive compute allocation. After allocation, heavy commands can run on the compute node. |
| `squeue -u "$USER"` | Cluster login node | Show your pending and running jobs. |
| `sacct -j <job_id> ...` | Cluster login node | Inspect a completed job. Replace `<job_id>` with the number printed by `sbatch`. |

## Lessons

Read the lessons in order. Each one builds on the previous directory and environment-variable setup.

| Order | Tutorial | Notebook | What it proves |
| --- | --- | --- | --- |
| 0 | [HAIC, uv, and SLURM basics](week2-00-haic-uv-slurm.md) | [week2_00_haic_uv_slurm.ipynb](notebooks/week2_00_haic_uv_slurm.ipynb) | You can set up the repo, discover cluster resources, and submit small jobs without computing on login nodes. |
| 1 | [GaitLU-1M storage and extraction](week2-01-gaitlu-storage-extraction.md) | [week2_01_gaitlu_storage_extraction.ipynb](notebooks/week2_01_gaitlu_storage_extraction.ipynb) | You can stage the encrypted archive legally, extract it on cluster storage, and discover the real pickle root. |
| 2 | [Index, splits, and loader](week2-02-index-splits-loader.md) | [week2_02_index_splits_loader.ipynb](notebooks/week2_02_index_splits_loader.ipynb) | You can create reproducible manifests and a `DataLoader` that returns `[B, T, C, H, W]`. |
| 3 | [Diagnostics and probe exports](week2-03-diagnostics-probe-exports.md) | [week2_03_diagnostics_probe_exports.ipynb](notebooks/week2_03_diagnostics_probe_exports.ipynb) | You can inspect batches, measure motion, and export dummy probe tables without leaking raw data. |

## Ground Rules

1. Do not run extraction, indexing, model training, or large diagnostics on a login node.
2. Put SLURM `#SBATCH` directives at the top of each batch script, before the first executable line.
3. Submit jobs with `sbatch`, check live status with `squeue`, and inspect finished jobs with `sacct`.
4. For password-protected extraction, prefer a compute allocation that lets the archive tool prompt securely, or use a HAIC-managed secret mechanism.
5. Manage Python dependencies with `uv` only. Do not mix other package managers into this repo.
6. Keep raw archives, extracted `.pkl` files, generated participant media, notebook outputs, and latent exports under ignored `data/gaitlu-1m/` paths locally, or under `$SCRATCH/cody-jepa-data` or approved project storage on Sherlock.
7. Do not load or extract real GaitLU-1M data until the dataset agreement is complete and the official archive password has been obtained.

## Week 2 Gate

Week 2 passes only when all three checks are true:

1. Split manifests reproduce exactly from the same inputs and seed.
2. Validation batches reproduce exactly from the same manifest, seed, and clip length.
3. A batch from the loader has shape `[B, T, C, H, W]`.

The notebooks prove the gate on synthetic `.pkl` fixtures. On HAIC, repeat the same checks after legal extraction of the full archive.

## Sources To Read First

- Stanford Research Computing SLURM basics: <https://stanford-rc.github.io/docs-earth/docs/slurm-basics>
- Stanford SDSS-CC resources overview: <https://stanford-rc.github.io/docs-earth/docs/resources_overview>
- SLURM `sbatch` manual: <https://slurm.schedmd.com/sbatch.html>
- GaitLU-1M upstream README: <https://github.com/ShiqiYu/OpenGait/blob/master/datasets/GaitLU-1M/README.md>
