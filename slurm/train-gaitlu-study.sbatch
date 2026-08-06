#!/bin/bash
#SBATCH --job-name=gaitlu-jepa
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --array=0-19%8
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

PROJECT_ROOT="${CODY_JEPA_ROOT:-/hai/scratch/$USER/cody-jepa}"
PREPARED_ROOT="${GAITLU_PREPARED_ROOT:?set GAITLU_PREPARED_ROOT to prepared GaitLU root}"
RUN_ROOT="${CODY_JEPA_RUN_ROOT:?set CODY_JEPA_RUN_ROOT to the checkpoint output root}"
CONFIG="${CODY_JEPA_CONFIG:-configs/train/gaitlu_scaling.json}"
NUM_WORKERS="${CODY_JEPA_NUM_WORKERS:-8}"

cd "$PROJECT_ROOT"
nvidia-smi

uv run --frozen --no-sync cody-jepa-train-gaitlu-study \
  --registry "$PREPARED_ROOT/training_registry.csv" \
  --run-index "$SLURM_ARRAY_TASK_ID" \
  --config "$CONFIG" \
  --data-root "$PREPARED_ROOT" \
  --output-root "$RUN_ROOT" \
  --repo-root "$PROJECT_ROOT" \
  --num-workers "$NUM_WORKERS" \
  --device cuda
