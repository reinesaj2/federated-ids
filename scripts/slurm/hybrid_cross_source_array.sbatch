#!/bin/bash
#SBATCH --job-name=hybrid-cross-source
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/results/hybrid_3dataset_1500/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/hybrid_3dataset_1500/%x-%A_%a.err
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

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/hybrid_3dataset_1500"
cd "/scratch/${USER}/federated-ids-hybrid-1500"

NUM_CLIENTS=12
NUM_ROUNDS=30
AGGREGATION_METHODS="fedavg,krum,bulyan,median"
ALPHA_VALUES="0.02,0.05,0.1,0.5,1.0"
FEDPROX_MU_VALUES="0.01,0.05,0.2,1.0,5.0"
ADVERSARY_FRACTIONS="0.0,0.05,0.1,0.15,0.2"
SEEDS="60,61,62"

SERVER_TIMEOUT=10800
CLIENT_TIMEOUT=10800

SPLIT_TOTAL=1500
SPLIT_OFFSET="${SPLIT_OFFSET:-0}"
SPLIT_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SPLIT_INDEX="$((SPLIT_INDEX + SPLIT_OFFSET))"

echo "================================================================================"
echo "Hybrid Cross-Source Federated Learning"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array: ${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} (${SPLIT_INDEX}/${SPLIT_TOTAL})"
echo "Node: ${SLURMD_NODENAME}"
echo "Clients: ${NUM_CLIENTS} (4 per source: CIC, UNSW, IIoT)"
echo "Rounds: ${NUM_ROUNDS}"
echo "Aggregations: ${AGGREGATION_METHODS}"
echo "Alphas (within-source): ${ALPHA_VALUES}"
echo "FedProx Mus: ${FEDPROX_MU_VALUES}"
echo "Adversary Fractions: ${ADVERSARY_FRACTIONS}"
echo "Seeds: ${SEEDS}"
echo "Dataset: hybrid (CIC-IDS2017 + UNSW-NB15 + Edge-IIoTset)"
echo "================================================================================"

python scripts/comparative_analysis.py \
  --dimension hybrid \
  --dataset hybrid \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --aggregation-methods "${AGGREGATION_METHODS}" \
  --alpha-values "${ALPHA_VALUES}" \
  --fedprox-mu-values "${FEDPROX_MU_VALUES}" \
  --adversary-fractions "${ADVERSARY_FRACTIONS}" \
  --seeds "${SEEDS}" \
  --server_timeout "${SERVER_TIMEOUT}" \
  --client_timeout "${CLIENT_TIMEOUT}" \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
