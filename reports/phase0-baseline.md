# Phase 0 baseline report

**Status:** Locked. Job 91108 and `outputs/jepa-v4/` remain read-only evidence.

**Canonical checkpoint:** `best_loss.pt` (`sha256:ab1e24043b2ba453e03fa427b0e845b74b2771682220732267d966be360097a5`).
It is locked by the training-time subject-balanced validation-loss rule, independent of probe scores.

## Frozen evaluation protocol

- Manifest SHA-256: `3074603602400af0b639c9569d69f5da99f43daebf3980b480154adddfdfb10e`
- Manifest schema: `subject_id, modality, gait_system, trial, frame_dir, num_frames, split`
- Split counts: train 2506 sequences / 318 subjects; val 624 sequences / 80 subjects
- Feature formula: `target_encoder(video, return_pre_norm=True)[1].mean(dim=1)`
- Reference export runtime: device `mps`, batch size 4, loader workers 4
- Feature rows: train 7518; val 1872 (3 deterministic windows per sequence)
- Probe seed: 0; max iterations: 2000; closed-set validation fraction: 0.25; retrieval enrollment sequences: 1
- Sampled inventory SHA-256: `b46c484e8af0a4e31f0669aa436a92936e1725ee0082d8615065c617bb29576d`. Scope: All frame names and sizes plus first, middle, and last image bytes per sequence; this is not a full byte hash of every frame.

## Checkpoint comparison

| Checkpoint | Epoch | Checkpoint SHA-256 | Feature SHA-256 | Effective rank | Rank ratio | Wrong-context gap | Closed-set identity | Held-out retrieval | Gait balanced accuracy |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_loss.pt` (canonical) | 80 | `ab1e24043b2ba453e03fa427b0e845b74b2771682220732267d966be360097a5` | `db3e57a09b33bf9a3f4e32ab71880f97d1a056574e7c8c3ca276a00c6727da53` | 10.45 | 0.0272 | 0.000154 | 0.0925 | 0.0245 | 0.9257 |
| `latest.pt` | 100 | `5571a59c045dab3d4fd87d57e0baa296ad13f28992ab6d32f425f9340a848dad` | `191dfad0cb49cbd1636496794ee539769b3bb36133dd39b31d37bd2b9743fb65` | 10.17 | 0.0265 | 0.000160 | 0.0941 | 0.0270 | 0.9247 |

Probe metrics are computed per exported clip window. Gait balanced accuracy is class-balanced, not subject-balanced. All three probes were rerun from distinct clean feature exports under current code. Full metrics, artifact paths, hashes, and the machine-readable frozen contract are in the adjacent JSON report.
