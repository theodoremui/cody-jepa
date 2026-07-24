# Phase 0 evidence recovery incident

Date: 2026-07-23  
Scope: local read-only evidence for HAIC job 91108

## Outcome

Three retained evidence files had been rewritten after the original baseline was captured. Their drifted bytes were archived before recovery, and the original bytes were restored and verified. Both checkpoint files were unchanged and already matched their locked hashes.

The drift archive is intentionally ignored by Git and is outside the protected baseline directory at `outputs/evidence-recovery/g0-2026-07-23/local-drift/`.

| Evidence | Original/locked SHA-256 | Altered SHA-256 | Archived altered SHA-256 | Restored SHA-256 |
| --- | --- | --- | --- | --- |
| `haic-results/job_91108.ipynb` | `11c62c4f8a5d365e97a2e111cc2375993481a4331cd2a05175e561a35f11cb93` | `70157ee7029d48912d7590ed4dae7752b771c0978ef69b20805d92b4eaba25d4` | `70157ee7029d48912d7590ed4dae7752b771c0978ef69b20805d92b4eaba25d4` | `11c62c4f8a5d365e97a2e111cc2375993481a4331cd2a05175e561a35f11cb93` |
| `outputs/jepa-v4/probe_metrics.csv` | `787a47fee5e1db880ed6b2effc7d73d77c82bcc7100c6c2821fe9256cb512bde` | `449e39e27d3a7e50a320dc49b0980d4f32037970514676e22e6d8b4e909e1fc2` | `449e39e27d3a7e50a320dc49b0980d4f32037970514676e22e6d8b4e909e1fc2` | `787a47fee5e1db880ed6b2effc7d73d77c82bcc7100c6c2821fe9256cb512bde` |
| `outputs/jepa-v4/probe_metrics.json` | `8c1c7ed83b90f791cf7ce9881633fe0ff53c911224a2d65eca96526de5f5fc4c` | `1329e7c4c3eb539bdf8efe7c75d920d1b616bf5f7d85062e39d51160be6d3f0b` | `1329e7c4c3eb539bdf8efe7c75d920d1b616bf5f7d85062e39d51160be6d3f0b` | `8c1c7ed83b90f791cf7ce9881633fe0ff53c911224a2d65eca96526de5f5fc4c` |
| `outputs/jepa-v4/best_loss.pt` | `ab1e24043b2ba453e03fa427b0e845b74b2771682220732267d966be360097a5` | unchanged | not applicable | `ab1e24043b2ba453e03fa427b0e845b74b2771682220732267d966be360097a5` |
| `outputs/jepa-v4/latest.pt` | `5571a59c045dab3d4fd87d57e0baa296ad13f28992ab6d32f425f9340a848dad` | unchanged | not applicable | `5571a59c045dab3d4fd87d57e0baa296ad13f28992ab6d32f425f9340a848dad` |

## Recovery sources and path mapping

- The notebook was recovered from its pre-rewrite Git object.
- The legacy CSV was recovered from a surviving byte-identical local upload cache.
- The JSON was recovered by restoring its sole rewritten provenance path. Reinstating `/hai/scratch/tedmui/cody-jepa/outputs/single-stream-jepa-h100-v4/frozen_features.npz` reproduced the original locked digest exactly.
- Historical HAIC paths in the notebook and JSON remain unchanged. The HAIC checkpoint directory `/hai/scratch/tedmui/cody-jepa/outputs/single-stream-jepa-h100-v4/` maps to the retained local evidence directory `outputs/jepa-v4/`; this mapping is documentation only and is not written into the historical evidence.

The legacy probe feature table `frozen_features.npz` remains unavailable locally. The legacy probe files are retained as historical evidence only and are not used to select the canonical checkpoint. Canonical selection remains the training-time `subject_balanced_loss` rule recorded in `best_loss.pt`; current features and probes are independently regenerated from the locked checkpoints.

## Remote status

No HAIC files were read or modified during this local recovery. Two read-only SSH connectivity probes to `haic.stanford.edu` timed out on port 22, including an out-of-sandbox retry, so the remote HAIC bytes were not independently verified or restored. Any future remote drift recovery must first archive the remote bytes outside the HAIC `jepa-v4` evidence directory, then restore atomically and verify against the hashes above.
