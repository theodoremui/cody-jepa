#!/bin/bash
#SBATCH --job-name=prepare-gaitlu
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

PROJECT_ROOT="${CODY_JEPA_ROOT:-/hai/scratch/$USER/cody-jepa}"
RAW_ROOT="${GAITLU_RAW_ROOT:?set GAITLU_RAW_ROOT to the directory containing gaitlu-000.tar.gz}"
PREPARED_ROOT="${GAITLU_PREPARED_ROOT:?set GAITLU_PREPARED_ROOT to the output directory}"
PREP_LOG_ROOT="${PREP_LOG_ROOT:-$(dirname "$PREPARED_ROOT")/logs/prepare-shards}"

cd "$PROJECT_ROOT"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this launcher directly on the login node; do not submit it with sbatch." >&2
    exit 2
  fi

  mkdir -p "$PREP_LOG_ROOT"
  for START in $(seq 0 8 96); do
    END=$((START + 7))
    if ((END > 99)); then
      END=99
    fi
    echo "Submitting GaitLU shards $START-$END"
    sbatch --wait \
      --array="$START-$END%8" \
      --export=ALL \
      --output="$PREP_LOG_ROOT/slurm-%x-%A_%a.out" \
      "$PROJECT_ROOT/slurm/prepare-gaitlu-shards.sbatch"
  done
  exit 0
fi

SHARD_NUMBER="$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"

uv run --frozen --no-sync cody-jepa-prepare-gaitlu pack-shard \
  --input "$RAW_ROOT/gaitlu-$SHARD_NUMBER.tar.gz" \
  --prepared-root "$PREPARED_ROOT" \
  --trust-pickles
