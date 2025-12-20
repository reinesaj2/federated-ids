#!/bin/bash
#SBATCH --job-name=iiot-full-hetero-fedavg
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/results/iiot_full/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/iiot_full/%x-%A_%a.err
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

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/iiot_full"
cd "/scratch/${USER}/federated-ids"

NUM_CLIENTS=10
NUM_ROUNDS=20
ALPHA_VALUES="0.02,0.05,0.1,0.2,0.5,1.0,inf"
SEEDS="42,43,44,45,46"

SERVER_TIMEOUT=28800
CLIENT_TIMEOUT=28800

SPLIT_TOTAL=35
SPLIT_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

echo "================================================================================"
echo "IIoT Full Heterogeneity (FedAvg baseline)"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array: ${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} (${SPLIT_INDEX}/${SPLIT_TOTAL})"
echo "Node: ${SLURMD_NODENAME}"
echo "Clients: ${NUM_CLIENTS}"
echo "Rounds: ${NUM_ROUNDS}"
echo "Alphas: ${ALPHA_VALUES}"
echo "Seeds: ${SEEDS}"
echo "Dataset: edge-iiotset-full (data/edge-iiotset/edge_iiotset_full.csv)"
echo "================================================================================"

python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --alpha-values "${ALPHA_VALUES}" \
  --seeds "${SEEDS}" \
  --server_timeout "${SERVER_TIMEOUT}" \
  --client_timeout "${CLIENT_TIMEOUT}" \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
