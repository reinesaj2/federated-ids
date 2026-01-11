#!/bin/bash
#SBATCH --job-name=unsw-simple-cic-iiot-mirror-dedup
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/results/unsw_simple_cic_iiot_mirror_dedup/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/unsw_simple_cic_iiot_mirror_dedup/%x-%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-1000%17

set -euo pipefail

source /usr/share/Modules/init/bash 2>/dev/null || true
source "/scratch/${USER}/venvs/fedids-py311/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export MANIFEST="/scratch/${USER}/results/unsw_simple_cic_iiot_mirror_dedup/manifest.jsonl"
RUNNER="/scratch/${USER}/results/manifest_runners/run_manifest_experiment.py"

TOTAL=$(python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MANIFEST"])
with path.open() as handle:
    print(sum(1 for _ in handle))
PY
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OFFSET="${OFFSET:-0}"
INDEX=$((TASK_ID + OFFSET))

if [ "${INDEX}" -ge "${TOTAL}" ]; then
  echo "Task ${INDEX} out of range (total=${TOTAL})"
  exit 0
fi

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/unsw_simple_cic_iiot_mirror_dedup"
cd "/scratch/${USER}/federated-ids"

python "$RUNNER" --manifest "$MANIFEST" --index "$INDEX" --timeout 10800
