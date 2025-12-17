#!/bin/bash
#SBATCH --job-name=iiot-full-smoke-10c-20r
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/results/iiot_full/%x-%j.out
#SBATCH --error=/scratch/%u/results/iiot_full/%x-%j.err
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
ALPHA_VALUES="0.5"
SEEDS="42"
FEDPROX_MU_VALUES="0.01"

SERVER_TIMEOUT=43200
CLIENT_TIMEOUT=43200

echo "================================================================================"
echo "IIoT Full Smoke (10 clients, 20 rounds)"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Clients: ${NUM_CLIENTS}"
echo "Rounds: ${NUM_ROUNDS}"
echo "Alpha: ${ALPHA_VALUES}"
echo "Seed: ${SEEDS}"
echo "FedProx mu: ${FEDPROX_MU_VALUES}"
echo "Dataset: edge-iiotset-full (data/edge-iiotset/edge_iiotset_full.csv)"
echo "================================================================================"

echo "[1/2] FedAvg baseline (heterogeneity)"
python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --alpha-values "${ALPHA_VALUES}" \
  --seeds "${SEEDS}" \
  --server_timeout "${SERVER_TIMEOUT}" \
  --client_timeout "${CLIENT_TIMEOUT}"

echo "[2/2] FedProx (heterogeneity_fedprox)"
python scripts/comparative_analysis.py \
  --dimension heterogeneity_fedprox \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --alpha-values "${ALPHA_VALUES}" \
  --fedprox-mu-values "${FEDPROX_MU_VALUES}" \
  --seeds "${SEEDS}" \
  --server_timeout "${SERVER_TIMEOUT}" \
  --client_timeout "${CLIENT_TIMEOUT}"

echo "SMOKE COMPLETED SUCCESSFULLY"
