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

# The documented pipeline CLI must export every destination and the predeclared
# criterion. jepa-v4 is retained evidence and is rejected by the CLI.
: "${CODY_JEPA_OUTPUT_DIR:?submit through scripts/run_phase0_pipeline.py}"
: "${CODY_JEPA_ARTIFACT_DIR:?submit through scripts/run_phase0_pipeline.py}"
: "${CODY_JEPA_REPORT_PATH:?submit through scripts/run_phase0_pipeline.py}"
: "${CODY_JEPA_CHECKPOINT_NAME:?submit through scripts/run_phase0_pipeline.py}"
: "${CODY_JEPA_SUCCESS_CRITERION:?submit through scripts/run_phase0_pipeline.py}"
export MPLCONFIGDIR=/tmp/mpl

# Claiming and all writes happen only inside the allocation. Reject existing
# destinations before expensive setup; the Python entry point claims atomically.
if [[ -e "$CODY_JEPA_OUTPUT_DIR" || -e "$CODY_JEPA_ARTIFACT_DIR" ]]; then
  echo "Refusing to overwrite an existing run or artifact directory" >&2
  exit 2
fi

mkdir -p logs notebook-runs

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

uv run --frozen --no-sync python scripts/run_phase0_pipeline.py run \
  --run-dir "$CODY_JEPA_OUTPUT_DIR" \
  --artifact-dir "$CODY_JEPA_ARTIFACT_DIR" \
  --report "$CODY_JEPA_REPORT_PATH" \
  --checkpoint-name "$CODY_JEPA_CHECKPOINT_NAME" \
  --success-criterion "$CODY_JEPA_SUCCESS_CRITERION" \
  --notebook-dir notebook-runs \
  --device cuda \
  --batch-size 16 \
  --num-workers 4 \
  --windows-per-sequence 3 \
  --seed 0 \
  --max-iter 2000 \
  --identity-validation-fraction 0.25 \
  --retrieval-enrollment-sequences 1
