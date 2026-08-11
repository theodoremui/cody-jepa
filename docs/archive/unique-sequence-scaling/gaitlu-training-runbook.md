# GaitLU-1M preparation and training runbook

This runbook is the executable path from the official `gaitlu-000.tar.gz` through
`gaitlu-099.tar.gz` files to the twenty fixed-exposure JEPA runs. Raw and prepared data
stay on HAIC. Git transfers code; it does not transfer the dataset.

## Current repository status

As of August 6, 2026, the repository contains the v2 preparation code, indexed loader,
fixed-exposure sampler, smoke and primary configurations, twenty-row training launcher,
and synthetic tests. The private HAIC shards have not been converted with this code,
the throughput tier has not been selected, no eligible GaitLU checkpoint has been
produced, and the locked Health&Gait outcome cohort remains closed.

The next research operation is therefore **preparation and systems qualification**, not
outcome evaluation. Do not launch the twenty primary jobs until all of the following are
true:

1. these changes are committed and available on HAIC at an immutable commit or annotated
   training tag;
2. all 100 shards finish conversion and their exclusions are reviewed;
3. finalization reports plausible eligible, duplicate, holdout, and rung counts;
4. the 25k and full-pool probes select one exposure tier; and
5. the selected configuration, registry, and code revision are frozen together.

## 1. Install the exact code on HAIC

First commit and push the intended code from the local repository. The current
uncommitted worktree cannot be reproduced by `git pull` on HAIC. Record the exact commit
or annotated training tag that will be used, then connect:

```bash
ssh YOUR_SUNET@haic.stanford.edu

export CODY_JEPA_ROOT="/hai/scratch/$USER/cody-jepa"
export GAITLU_DATA_ROOT="$CODY_JEPA_ROOT/data/gaitlu-1m"
export GAITLU_RAW_ROOT="$GAITLU_DATA_ROOT/shards"
export GAITLU_SOURCE_ROOT="$GAITLU_DATA_ROOT/source"
export GAITLU_STAGING_ROOT="$GAITLU_DATA_ROOT/staging"
export GAITLU_PREPARED_ROOT="$GAITLU_DATA_ROOT/prepared"
export CODY_JEPA_RUN_ROOT="/hai/scratch/$USER/cody-jepa-runs/gaitlu-scaling"
export GAITLU_TRAINING_REF="REPLACE_WITH_IMMUTABLE_COMMIT_OR_ANNOTATED_TAG"

cd "$CODY_JEPA_ROOT"
git fetch --tags origin
git switch --detach "$GAITLU_TRAINING_REF"
git status --short
git rev-parse HEAD
git describe --tags --exact-match 2>/dev/null || true
uv sync --frozen
```

`git status --short` must be empty. Save the printed commit in the private study log.
Preparation outputs are tied to this code revision even though they remain outside Git.

These variables match the current HAIC layout:

```text
data/gaitlu-1m/
  shards/gaitlu-000.tar.gz ... gaitlu-099.tar.gz
  source/GaitLU_Anno.zip
  staging/anonymized_sil/
  prepared/                         # created by this runbook
```

`GAITLU_RAW_ROOT` is named for its role in the converter; it points to the existing
`shards/` directory, so the files do not need to be renamed or moved. The current
training path reads the 100 `.tar.gz` files from `shards/`. `source/GaitLU_Anno.zip` and
`staging/anonymized_sil/` are retained source/staging artifacts and are not inputs to the
indexed-shard conversion commands below. The repository-wide `/data/` ignore rule keeps
all four data directories out of Git.

Adjust `#SBATCH --account=mind` in the two GaitLU Slurm scripts if `sacctmgr` reports a
different account. Do not run conversion or training on the HAIC login node.

```bash
sacctmgr show user "$USER" withassoc format=User,Account,Partition
sinfo -o "%P %l %G %D"
```

## 2. Confirm the raw release

```bash
find "$GAITLU_RAW_ROOT" -maxdepth 1 -name 'gaitlu-*.tar.gz' | sort | wc -l
tar -tzf "$GAITLU_RAW_ROOT/gaitlu-000.tar.gz" | sed -n '1,40p'
```

The count must be exactly 100, with no missing or additional shard number. Paths such as
`000/030/001/001.pkl` are interpreted using the
OpenGait convention: the sequence key is the complete containing path
`000/030/001`. Neither `000` nor `001` is treated as a verified person identity.

Pickle loading can execute code. The converter requires `--trust-pickles` so that this
decision is explicit. Use it only for the official, trusted GaitLU release.

## 3. Convert all shards

The converter streams each gzip tar without extracting tiny files. It validates every
sequence, threshold-packs binary-looking silhouettes, and writes a seekable uncompressed
tar plus a CSV index. Invalid sequences remain in the audit inventory with an exclusion
reason. Preparation computes a SHA-256 fingerprint over each packed record and its shape.
Those fingerprints name packed records, allow records to be reused within a shard, and
drive exact deduplication across shards; they remain in the inventories but are not
copied into the finalized training or holdout manifests.

```bash
mkdir -p "$GAITLU_PREPARED_ROOT"

cd "$CODY_JEPA_ROOT"
bash slurm/prepare-gaitlu-shards.sbatch
```

Run the launcher on the HAIC login node, preferably inside `tmux` so that a disconnected
SSH session does not stop later groups from being submitted. Do not pass the launcher to
`sbatch`: it submits at most eight shard workers, waits for that group to finish, and then
submits the next group. Worker logs go to
`$PREP_LOG_ROOT/slurm-prepare-gaitlu-<array-job-id>_<shard>.out`. If
`PREP_LOG_ROOT` is unset, it defaults to `logs/prepare-shards` beside the prepared-data
directory. Monitor the array jobs printed by the launcher:

```bash
squeue -u "$USER"
sacct -j ARRAY_JOB_ID --format=JobID,State,Elapsed,ExitCode
```

Every one of the 100 tasks must finish successfully. Then verify the sidecars:

```bash
find "$GAITLU_PREPARED_ROOT/inventories" -name 'gaitlu-*.csv' | wc -l
find "$GAITLU_PREPARED_ROOT/shards" -name 'gaitlu-*.tar' | wc -l
```

Both counts must be 100. A successful Slurm task means the shard was processed; it does
not mean that every sequence passed validation. Sum and review the per-shard summaries:

```bash
uv run --frozen --no-sync python - "$GAITLU_PREPARED_ROOT" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1]) / "inventories"
paths = sorted(root.glob("gaitlu-*.summary.json"))
if len(paths) != 100:
    raise SystemExit(f"expected 100 summaries, found {len(paths)}")
summaries = [json.loads(path.read_text()) for path in paths]
print("sequences:", sum(row["sequences"] for row in summaries))
print("valid:", sum(row["valid_sequences"] for row in summaries))
print("excluded:", sum(row["excluded_sequences"] for row in summaries))
print("packed records:", sum(row["unique_records_in_shard"] for row in summaries))
PY
```

Inspect every nonzero exclusion class in the CSV inventories before finalization. Do not
weaken validation thresholds merely to recover a nominal one-million count.

Preparation outputs use the v2 manifest and study-summary
schemas. Regenerate them into a clean prepared directory: v1 outputs are not upgraded
in place, and there is no v1 adapter or migration command.

For a one-shard pilot before launching the array:

```bash
uv run --frozen --no-sync cody-jepa-prepare-gaitlu pack-shard \
  --input "$GAITLU_RAW_ROOT/gaitlu-000.tar.gz" \
  --prepared-root "$GAITLU_DATA_ROOT/pilot-prepared" \
  --trust-pickles
```

Use a separate pilot directory because the converter refuses to overwrite existing
outputs.

## 4. Deduplicate and freeze pool membership

```bash
cd "$CODY_JEPA_ROOT"

uv run --frozen --no-sync cody-jepa-prepare-gaitlu finalize \
  --prepared-root "$GAITLU_PREPARED_ROOT" \
  --holdout-size 10000 \
  --holdout-seed 20260806 \
  --pool-seeds 0 1 2 3 4 \
  --pool-sizes 2500 25000 250000 \
  --training-exposure 8192000 \
  --expected-shards 100
```

When no verified source-group metadata are supplied, the command uses the protocol's
declared singleton grouping policy after exact-content deduplication. If the distributor
provides verified grouping, pass a complete CSV with exactly `sequence_id,source_group`
using `--source-groups`. “Complete” means every sequence in the shard inventories,
including sequences later excluded as invalid, has exactly one mapping row.

Inspect these outputs before training:

```text
prepared/
  inventory.csv
  study_pools.json
  training_registry.csv
  inventories/gaitlu-000.csv ... gaitlu-099.csv
  shards/gaitlu-000.tar ... gaitlu-099.tar
  manifests/common-holdout.csv
  manifests/smoke-holdout.csv
  manifests/ladder-0-small.csv ... ladder-4-full.csv
```

`study_pools.json` records actual counts, exact duplicates, exclusions, grouping policy,
holdout seed, pool seeds, and the relative manifest paths. The v2 finalized manifests do
not contain per-record hashes. Review the summary rather than assuming the nominal
million-example count. Confirm in particular that it reports schema
`gaitlu-scaling-pools-v2`, 100 source inventories, 20 pools, a 10,000-sequence holdout,
strictly increasing rung sizes within every ladder, and the same full-pool count across
all five ladders.

`training_registry.csv` is the launcher input with this exact schema:

```text
model_label,ladder,rung,train_manifest,val_manifest,manifest_sha256,
pool_seed,optimization_seed,unique_sequences,training_exposure
```

Finalization computes `manifest_sha256` from the exact bytes of the ordered train and
common-holdout manifests. The value is role-sensitive: swapping the two files changes
the digest. A legacy nine-column registry must be regenerated by rerunning finalization
in a clean prepared directory.

## 5. Run loader and model smoke tests

Request one interactive H100 allocation:

```bash
srun --account=mind --partition=hai-interactive --gres=gpu:h100:1 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

Inside the allocation, run one optimizer update and a 64-sequence holdout evaluation:

```bash
cd "$CODY_JEPA_ROOT"

uv run --frozen --no-sync cody-jepa-train \
  --config configs/train/gaitlu_smoke.json \
  --dataset gaitlu \
  --train-manifest "$GAITLU_PREPARED_ROOT/manifests/ladder-0-small.csv" \
  --val-manifest "$GAITLU_PREPARED_ROOT/manifests/smoke-holdout.csv" \
  --data-root "$GAITLU_PREPARED_ROOT" \
  --output-dir "/hai/scratch/$USER/cody-jepa-runs/gaitlu-smoke-small" \
  --repo-root "$CODY_JEPA_ROOT" \
  --device cuda \
  --num-workers 8
```

This one-update command validates archive reads, collation, tensor shape, one backward
pass, checkpoint writing, and a 64-sequence validation pass. It is **not** long enough to
select the exposure tier. Smoke checkpoints are not eligible study models.

### Measure the 25k and full-pool paths

Create a temporary one-virtual-epoch probe configuration outside the repository. It
processes 65,536 training examples, or 1,024 optimizer updates, before evaluating the
64-sequence smoke holdout:

```bash
export GAITLU_PROBE_CONFIG="/hai/scratch/$USER/cody-jepa-runs/gaitlu-probe.json"

uv run --frozen --no-sync python - \
  "$CODY_JEPA_ROOT/configs/train/gaitlu_scaling.json" \
  "$GAITLU_PROBE_CONFIG" <<'PY'
from pathlib import Path
import json
import sys

source, output = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
config.update({
    "run_id": "gaitlu-throughput-probe",
    "steps": 1024,
    "num_epochs": 1,
    "warmup_steps": 200,
    "eval_every_epochs": 1,
    "checkpoint_every_epochs": 1,
})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(config, indent=2) + "\n")
PY
```

Run the same command twice, first with `ladder-0-medium.csv` and then with
`ladder-0-full.csv`, always using a new empty output directory:

```bash
uv run --frozen --no-sync cody-jepa-train \
  --config "$GAITLU_PROBE_CONFIG" \
  --dataset gaitlu \
  --train-manifest "$GAITLU_PREPARED_ROOT/manifests/ladder-0-medium.csv" \
  --val-manifest "$GAITLU_PREPARED_ROOT/manifests/smoke-holdout.csv" \
  --data-root "$GAITLU_PREPARED_ROOT" \
  --output-dir "/hai/scratch/$USER/cody-jepa-runs/gaitlu-probe-medium" \
  --repo-root "$CODY_JEPA_ROOT" \
  --device cuda \
  --num-workers 8
```

Repeat with the full manifest and `gaitlu-probe-full`. Record both reported
`examples_per_second` values and use the slower result:

- at least 60 examples/s/GPU: retain 8,192,000 examples per run;
- 30–59 examples/s/GPU: use 4,096,000 examples for **every** run; or
- below 30 examples/s/GPU, failed full-pool streaming, or insufficient storage: cancel
  the scaling claim.

The checked-in `gaitlu_scaling.json` and generated registry currently implement only the
8,192,000-example tier. The 4,096,000 fallback must be added and frozen before use. Its
compatible schedule is `batch_size=16`, `accumulation_steps=4`,
`loader_epoch_examples=32768`, `steps=64000`, and `num_epochs=125`; finalization must be
rerun with `--training-exposure 4096000`. Do not edit a registry after training starts.

## 6. Freeze and launch the twenty runs

Before launch, commit the loader, the selected exposure configuration, and scripts;
create an annotated training freeze tag; then check out that tag in a clean HAIC
worktree. Keep manifests, `prepared/`, raw data, and checkpoints outside the repository.
Keep `study_pools.json` and `training_registry.csv` with the prepared dataset.

The primary configuration uses:

- microbatch 16 and accumulation 4, for effective batch 64;
- 65,536 sampled examples per virtual epoch;
- 1,024 optimizer updates per virtual epoch;
- 125 virtual epochs and 128,000 optimizer updates;
- exactly 8,192,000 sampled examples per model.

Launch at most eight independent single-GPU runs concurrently:

```bash
cd "$CODY_JEPA_ROOT"
bash slurm/train-gaitlu-study.sbatch
```

Run the launcher on the HAIC login node inside `tmux`; do not pass it to `sbatch`. It
submits runs 0–7, waits for the entire group, then submits 8–15 and finally 16–19. The
launcher stops without submitting a later group if any current-group run fails. The
Slurm array index selects one of the twenty rows in `training_registry.csv`. Each
checkpoint embeds the required final-step study metadata. An existing output directory
is rejected. To resume an
interrupted run manually, invoke
`cody-jepa-train-gaitlu-study` with the same row and add `--resume-existing`.

The primary configuration checkpoints every five virtual epochs. Resume is exact only
from the saved epoch boundary: model, optimizer, scaler, mask RNG, data contract, epoch,
and loader state are restored, while the fixed-exposure sampler reconstructs its draw
stream from the frozen seed scheme.

Each checkpoint stores one role-sensitive SHA-256 digest over the exact bytes of the
complete train and holdout manifest files. Resume requires the same digest, manifest
roles, loader options, and `splitmix64-v1` seed scheme. A checkpoint created with the v1
prototype data contract cannot be resumed with the v2 loader.

## 7. What the loader returns

Each training batch satisfies the existing trainer contract:

```python
{
    "video": FloatTensor[B, 16, 1, 112, 112],  # range [0, 1]
    "split": ["train", ...],
    "sequence_id": [...],
    "subject_id": [...],       # source group, not an inferred identity
    "source_group": [...],
}
```

The fixed-exposure sampler draws with replacement. Its draw number participates in the
stateless crop seed, so repeated short-pool sequences do not receive one identical clip
throughout a virtual epoch. Validation uses deterministic center windows from the common
holdout.

Loading still checks the exact manifest schema, safe relative shard paths, dimensions,
record sizes and bounds, missing shards, and short reads. It does not recompute packed
content hashes. Consequently, same-length corruption inside an otherwise readable
record is not detected during loading; retain and audit the preparation inventories and
recreate suspect prepared data from the trusted raw shards.

## 8. Handoff to feature export and GFC-v2

`training_registry.csv` is a private **training launcher registry**. It is not the GFC
study registry accepted by `cody-jepa-gfc-study`. After all twenty final-step checkpoints
exist, feature export and the evaluation registry still need to be assembled. The latter
has this separate schema:

```text
model_label,ladder,rung,checkpoint_id,checkpoint_path,feature_path,
pool_seed,optimization_seed,unique_sequences,training_exposure
```

Do not open locked outcomes merely because training finished. First export all twenty
Health&Gait feature archives, build the evaluation registry, and run the frozen
`cody-jepa-gfc-study preflight` from the analysis tag with
both `--registry` and the original `--training-registry`. Preflight requires the same 20
model labels, validates the final step and public `study_metadata`, and compares every
checkpoint's manifest digest with the corresponding training-registry entry. All 20
checkpoint contracts are checked before any Health&Gait feature archive is parsed.

The current repository does not yet include a twenty-checkpoint feature-export batch
launcher. That orchestration remains a required step between training completion and
the GFC-v2 preflight described in [the documentation map](../../README.md).
