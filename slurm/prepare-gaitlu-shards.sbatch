#!/bin/bash
#SBATCH --job-name=prepare-gaitlu
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --array=0-99%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

PROJECT_ROOT="${CODY_JEPA_ROOT:-/hai/scratch/$USER/cody-jepa}"
RAW_ROOT="${GAITLU_RAW_ROOT:?set GAITLU_RAW_ROOT to the directory containing gaitlu-000.tar.gz}"
PREPARED_ROOT="${GAITLU_PREPARED_ROOT:?set GAITLU_PREPARED_ROOT to the output directory}"
SHARD_NUMBER="$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"

cd "$PROJECT_ROOT"

uv run --frozen --no-sync cody-jepa-prepare-gaitlu pack-shard \
  --input "$RAW_ROOT/gaitlu-$SHARD_NUMBER.tar.gz" \
  --prepared-root "$PREPARED_ROOT" \
  --trust-pickles
