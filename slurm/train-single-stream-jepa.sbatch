#!/bin/bash
#SBATCH --job-name=single-stream-jepa
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/single-stream-jepa-%j.out

set -euo pipefail

cd "/hai/scratch/$USER/cody-jepa"

# New-run configuration. Use a different directory name for another fresh run.
export CODY_JEPA_RUN_FULL_TRAINING=1
export CODY_JEPA_RUN_DATA_AUDIT=0
export CODY_JEPA_RUN_EXHAUSTIVE_DATA_AUDIT=0
export CODY_JEPA_OUTPUT_DIR=outputs/single-stream-jepa-h100-v2
unset CODY_JEPA_RESUME_CHECKPOINT
export MPLCONFIGDIR=/tmp/mpl

# Refuse to overwrite or accidentally resume an existing run.
if [[ -e "$CODY_JEPA_OUTPUT_DIR/latest.pt" ]]; then
  echo "Refusing to overwrite $CODY_JEPA_OUTPUT_DIR/latest.pt" >&2
  echo "Choose a new CODY_JEPA_OUTPUT_DIR or configure an explicit resume." >&2
  exit 2
fi

mkdir -p logs notebook-runs "$CODY_JEPA_OUTPUT_DIR"

echo "Training environment:"
env | sort | grep -E '^(CODY_JEPA_|MPLCONFIGDIR)'

# Fail before loading the dataset if Slurm supplied the wrong GPU model.
nvidia-smi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$gpu_name" in
  *H100*) ;;
  *)
    echo "Expected an NVIDIA H100, got: $gpu_name" >&2
    exit 2
    ;;
esac

# Record utilization independently because nbconvert captures cell output in the
# executed notebook instead of streaming all training metrics to the Slurm log.
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=60 > "logs/gpu-${SLURM_JOB_ID}.csv" &
gpu_monitor_pid=$!
cleanup() {
  kill "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute notebooks/single-stream-jepa.ipynb \
  --output-dir notebook-runs \
  --output "single-stream-jepa-${SLURM_JOB_ID}.executed.ipynb" \
  --ExecutePreprocessor.timeout=-1
