#!/bin/bash
#SBATCH --job-name=unsw-bulyan-adv20-n12
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/unsw_bulyan_adv20_n12/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/unsw_bulyan_adv20_n12/%x-%A_%a.err
#SBATCH --exclusive
#SBATCH --array=0-194%20

set -euo pipefail

source /usr/share/Modules/init/bash 2>/dev/null || true
source "/scratch/${USER}/venvs/fedids-py311/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

RESULTS_DIR="/scratch/${USER}/results/unsw_bulyan_adv20_n12"
mkdir -p "$MPLCONFIGDIR" "$RESULTS_DIR"
cd "/scratch/${USER}/federated-ids"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/unsw_bulyan_adv20_n12_manifest.jsonl}"
RUNNER="${RUNNER:-/scratch/${USER}/results/manifest_runners/run_manifest_experiment.py}"
OFFSET="${OFFSET:-0}"
INDEX="$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))"

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

if [ ! -f "$RUNNER" ]; then
  echo "Runner not found: $RUNNER"
  exit 1
fi

python "$RUNNER" --manifest "$MANIFEST" --index "$INDEX" --timeout 10800
