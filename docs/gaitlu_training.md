# GaitLU-1M preparation runbook

## Purpose

This runbook covers the shared, experiment-independent path from the private GaitLU release to validated prepared data. It is used by the active hierarchical-diversity study and by the legacy unique-sequence fallback.

It does **not** define a training registry, pool sizes, phase-origin policy, model count, or final training launch. Those are study-specific decisions:

- Active phase-allocation study: [execution plan](hierarchical-diversity/execution-plan.md)
- Legacy unique-sequence runbook: [archive](archive/unique-sequence-scaling/gaitlu-training-runbook.md)

## 1. Install an immutable code revision on HAIC

Training data and prepared artifacts stay on HAIC. Git transfers code, not data. Before conversion, commit the intended code and record an immutable commit or annotated tag.

~~~bash
ssh YOUR_SUNET@haic.stanford.edu

export CODY_JEPA_ROOT="/hai/scratch/$USER/cody-jepa"
export GAITLU_DATA_ROOT="$CODY_JEPA_ROOT/data/gaitlu-1m"
export GAITLU_RAW_ROOT="$GAITLU_DATA_ROOT/shards"
export GAITLU_PREPARED_ROOT="$GAITLU_DATA_ROOT/prepared"
export GAITLU_PREPARATION_REF="REPLACE_WITH_IMMUTABLE_COMMIT_OR_TAG"

cd "$CODY_JEPA_ROOT"
git fetch --tags origin
git switch --detach "$GAITLU_PREPARATION_REF"
git status --short
git rev-parse HEAD
uv sync --frozen
~~~

The working tree must be clean. Keep raw data, prepared data, manifests, and checkpoints outside Git.

## 2. Verify the raw release

~~~bash
find "$GAITLU_RAW_ROOT" -maxdepth 1 -name 'gaitlu-*.tar.gz' | sort | wc -l
tar -tzf "$GAITLU_RAW_ROOT/gaitlu-000.tar.gz" | sed -n '1,40p'
~~~

There must be exactly 100 shards. The complete containing archive path is a sequence key. It is not verified person identity. Pickle loading can execute code, so use --trust-pickles only for the official trusted release.

## 3. Convert and audit all shards

The converter streams each gzip tar, validates sequences, threshold-packs binary-looking silhouettes, and writes a seekable tar plus inventory. Invalid sequences remain visible with an exclusion reason. Exact packed-record fingerprints support deduplication but do not belong in public manifests.

~~~bash
mkdir -p "$GAITLU_PREPARED_ROOT"
cd "$CODY_JEPA_ROOT"
bash slurm/prepare-gaitlu-shards.sbatch
~~~

Run the launcher from a HAIC login node, preferably in tmux. It submits bounded worker groups. Do not run conversion workers on the login node.

For a one-shard smoke conversion, use a separate output directory because preparation refuses to overwrite artifacts:

~~~bash
uv run --frozen --no-sync cody-jepa-prepare-gaitlu pack-shard \
  --input "$GAITLU_RAW_ROOT/gaitlu-000.tar.gz" \
  --prepared-root "$GAITLU_DATA_ROOT/pilot-prepared" \
  --trust-pickles
~~~

Inspect every nonzero exclusion class before moving on. Do not weaken validation thresholds simply to recover a nominal record count.

## 4. Freeze the common prepared-data inventory

Run the appropriate study finalizer only after all 100 shard inventories are complete and reviewed. Every finalizer must produce a versioned manifest schema, source-group policy, exact-duplicate summary, exclusion summary, and content digests.

The active phase-allocation study additionally requires an outcome-blind phase catalog, common k=4 eligibility, source-group and near-duplicate cluster summaries, and the frozen common-pool rule. See [the execution plan](hierarchical-diversity/execution-plan.md).

## 5. Smoke-test the loader

Use one interactive H100 allocation:

~~~bash
srun --account=mind --partition=hai-interactive --gres=gpu:h100:1 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
~~~

Run the relevant study smoke configuration with a small prepared manifest. A smoke run validates archive reads, collation, tensor shape, one backward pass, checkpoint writing, and holdout evaluation. It never selects the exposure tier and is not an eligible study model.

## Data and privacy boundary

Prepared GaitLU data train encoders only. Health&Gait belongs to frozen downstream evaluation and must never update a primary encoder. Retain inventories and audits privately. Do not publish raw recordings, participant tables, embeddings, identity-capable checkpoints, or private paths.
