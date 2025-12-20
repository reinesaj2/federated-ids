#!/bin/bash
#SBATCH --job-name=hybrid-cross-source
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/results/hybrid/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/hybrid/%x-%A_%a.err
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

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/hybrid"
cd "/scratch/${USER}/federated-ids"

NUM_CLIENTS=9
NUM_ROUNDS=20
ALPHA_VALUES="0.5,1.0"
FEDPROX_MU_VALUES="0.0,0.01,0.05,0.1"
SEEDS="42,43,44,45,46"

SERVER_TIMEOUT=7200
CLIENT_TIMEOUT=7200

SPLIT_TOTAL=40
SPLIT_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

echo "================================================================================"
echo "Hybrid Cross-Source Federated Learning"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array: ${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} (${SPLIT_INDEX}/${SPLIT_TOTAL})"
echo "Node: ${SLURMD_NODENAME}"
echo "Clients: ${NUM_CLIENTS} (3 per source: CIC, UNSW, IIoT)"
echo "Rounds: ${NUM_ROUNDS}"
echo "Alphas (within-source): ${ALPHA_VALUES}"
echo "FedProx Mus: ${FEDPROX_MU_VALUES}"
echo "Seeds: ${SEEDS}"
echo "Dataset: hybrid (CIC-IDS2017 + UNSW-NB15 + Edge-IIoTset)"
echo "================================================================================"

python scripts/comparative_analysis.py \
  --dimension hybrid \
  --dataset hybrid \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --alpha-values "${ALPHA_VALUES}" \
  --fedprox-mu-values "${FEDPROX_MU_VALUES}" \
  --seeds "${SEEDS}" \
  --server_timeout "${SERVER_TIMEOUT}" \
  --client_timeout "${CLIENT_TIMEOUT}" \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
