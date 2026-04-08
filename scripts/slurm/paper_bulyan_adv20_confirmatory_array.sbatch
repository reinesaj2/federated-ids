#!/bin/bash
#SBATCH --job-name=paper-bulyan20
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/results/paper_bulyan20/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/paper_bulyan20/%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /usr/share/Modules/init/bash 2>/dev/null || true
source "/scratch/${USER}/venvs/fedids-py311/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/paper_bulyan20"
cd "/scratch/${USER}/federated-ids"

DATASET="${DATASET:?DATASET is required (cic, unsw, or edge-iiotset-full)}"
DATA_PATH="${DATA_PATH:-}"
NUM_CLIENTS="${NUM_CLIENTS:-15}"
NUM_ROUNDS="${NUM_ROUNDS:-20}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61}"
SPLIT_TOTAL="${SPLIT_TOTAL:-20}"
SPLIT_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-28800}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-28800}"

COMMAND=(
  python
  scripts/comparative_analysis.py
  --dimension
  full
  --dataset
  "${DATASET}"
  --aggregation-methods
  bulyan
  --alpha-values
  0.5
  --adversary-fractions
  0.2
  --dp-noise-multipliers
  0.0
  --personalization-epochs
  0
  --seeds
  "${SEEDS}"
  --num_clients
  "${NUM_CLIENTS}"
  --num_rounds
  "${NUM_ROUNDS}"
  --server_timeout
  "${SERVER_TIMEOUT}"
  --client_timeout
  "${CLIENT_TIMEOUT}"
  --split-total
  "${SPLIT_TOTAL}"
  --split-index
  "${SPLIT_INDEX}"
)

if [[ -n "${DATA_PATH}" ]]; then
  COMMAND+=(--data_path "${DATA_PATH}")
fi

echo "================================================================================"
echo "Paper Bulyan 20% Confirmatory Campaign"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array: ${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} (${SPLIT_INDEX}/${SPLIT_TOTAL})"
echo "Node: ${SLURMD_NODENAME}"
echo "Dataset: ${DATASET}"
echo "Clients: ${NUM_CLIENTS}"
echo "Rounds: ${NUM_ROUNDS}"
echo "Seeds: ${SEEDS}"
echo "================================================================================"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
