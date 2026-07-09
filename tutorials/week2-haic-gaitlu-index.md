# Week 2 HAIC + GaitLU-1M Tutorial Track

Week 2 is the data-contract week for CoDy-JEPA. The goal is not to train a model yet. The goal is to prove that the GaitLU-1M data path, split policy, loader, diagnostics, and probe export format are trustworthy before Week 3 adds a JEPA baseline.

The full GaitLU-1M archive is too large for laptop-only work. Use the laptop only for archive inspection and tiny synthetic smoke tests. Use HAIC storage and SLURM jobs for extraction, indexing, diagnostics, and any full-dataset pass.

## Lessons

| Order | Tutorial | Notebook | What it proves |
| --- | --- | --- | --- |
| 0 | [HAIC, uv, and SLURM basics](week2-00-haic-uv-slurm.md) | [week2_00_haic_uv_slurm.ipynb](notebooks/week2_00_haic_uv_slurm.ipynb) | You can set up the repo, discover cluster resources, and submit small jobs without computing on login nodes. |
| 1 | [GaitLU-1M storage and extraction](week2-01-gaitlu-storage-extraction.md) | [week2_01_gaitlu_storage_extraction.ipynb](notebooks/week2_01_gaitlu_storage_extraction.ipynb) | You can stage the encrypted archive legally, extract it on HAIC, and discover the real pickle root. |
| 2 | [Index, splits, and loader](week2-02-index-splits-loader.md) | [week2_02_index_splits_loader.ipynb](notebooks/week2_02_index_splits_loader.ipynb) | You can create reproducible manifests and a `DataLoader` that returns `[B, T, C, H, W]`. |
| 3 | [Diagnostics and probe exports](week2-03-diagnostics-probe-exports.md) | [week2_03_diagnostics_probe_exports.ipynb](notebooks/week2_03_diagnostics_probe_exports.ipynb) | You can inspect batches, measure motion, and export dummy probe tables without leaking raw data. |

## Ground Rules

1. Do not run extraction, indexing, model training, or large diagnostics on a login node.
2. Put SLURM `#SBATCH` directives at the top of each batch script, before the first executable line.
3. Submit jobs with `sbatch`, check live status with `squeue`, and inspect finished jobs with `sacct`.
4. For password-protected extraction, prefer a compute allocation that lets the archive tool prompt securely, or use a HAIC-managed secret mechanism.
5. Manage Python dependencies with `uv` only. Do not mix other package-manager instructions into this repo.
6. Keep raw archives, extracted `.pkl` files, generated participant media, notebook outputs, and latent exports under ignored `data/gaitlu-1m/` paths.
7. Do not load or extract real GaitLU-1M data until the dataset agreement is complete and the official archive password has been obtained.

## Week 2 Gate

Week 2 passes only when all three checks are true:

1. Split manifests reproduce exactly from the same inputs and seed.
2. Validation batches reproduce exactly from the same manifest, seed, and clip length.
3. A batch from the loader has shape `[B, T, C, H, W]`.

The notebooks prove the gate on synthetic `.pkl` fixtures. On HAIC, repeat the same checks after legal extraction of the full archive.

## Sources To Read First

- Stanford Research Computing SLURM basics: <https://stanford-rc.github.io/docs-earth/docs/slurm-basics>
- SLURM `sbatch` manual: <https://slurm.schedmd.com/sbatch.html>
- GaitLU-1M upstream README: <https://github.com/ShiqiYu/OpenGait/blob/master/datasets/GaitLU-1M/README.md>
