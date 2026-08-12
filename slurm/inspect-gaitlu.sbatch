#!/bin/bash
#SBATCH --job-name=inspect-gaitlu
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%A_%a.out

# Execute notebooks/gaitlu_inspection.ipynb headlessly and write a self-contained HTML
# report. No GPU: this only reads prepared shards and renders figures.
#
#   sbatch --export=ALL slurm/inspect-gaitlu.sbatch                  # gaitlu-000
#   sbatch --export=ALL --array=0-9 slurm/inspect-gaitlu.sbatch      # gaitlu-000..009

set -euo pipefail

PROJECT_ROOT="${CODY_JEPA_ROOT:-/hai/scratch/$USER/cody-jepa}"
PREPARED_ROOT="${GAITLU_PREPARED_ROOT:?set GAITLU_PREPARED_ROOT to the prepared GaitLU root}"
REPORT_ROOT="${GAITLU_REPORT_ROOT:-$(dirname "$PREPARED_ROOT")/reports/gaitlu-inspection}"

# An array task picks its own shard; a plain submission honors GAITLU_SHARD.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD="gaitlu-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
else
  SHARD="${GAITLU_SHARD:-gaitlu-000}"
fi
export GAITLU_SHARD="$SHARD"
export GAITLU_PREPARED_ROOT

cd "$PROJECT_ROOT"
mkdir -p "$REPORT_ROOT"

echo "shard=$SHARD prepared=$PREPARED_ROOT report=$REPORT_ROOT"

# Section 3 unpickles the raw release, so it stays off unless GAITLU_RAW_ROOT is exported.
if [[ -n "${GAITLU_RAW_ROOT:-}" ]]; then
  export GAITLU_RAW_ROOT
  echo "raw root=$GAITLU_RAW_ROOT (section 3 enabled)"
else
  echo "GAITLU_RAW_ROOT unset; section 3 cells will error (use --allow-errors below)"
fi

# --allow-errors keeps a missing raw root or manifest from discarding the whole report;
# failed cells appear as tracebacks in the HTML. Drop it to make any failure fatal.
uv run --frozen --no-sync jupyter nbconvert \
  --to html \
  --execute \
  --allow-errors \
  --ExecutePreprocessor.timeout=1200 \
  --output-dir "$REPORT_ROOT" \
  --output "$SHARD.html" \
  notebooks/gaitlu_inspection.ipynb

echo "wrote $REPORT_ROOT/$SHARD.html"
