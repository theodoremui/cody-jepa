#!/bin/bash
#SBATCH --job-name=cody-jepa-train
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "/hai/scratch/$USER/cody-jepa"

CONFIG="${CODY_JEPA_CONFIG:-configs/train/healthgait_baseline.json}"
MANIFEST="${CODY_JEPA_MANIFEST:-data/healthgait/manifests/silhouette_subject_split_seed0.csv}"
OUTPUT_DIR="${CODY_JEPA_OUTPUT_DIR:-outputs/training-baseline-${SLURM_JOB_ID}}"
NUM_WORKERS="${CODY_JEPA_NUM_WORKERS:-4}"

nvidia-smi

uv run --frozen --no-sync cody-jepa-train \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --num-workers "$NUM_WORKERS"
