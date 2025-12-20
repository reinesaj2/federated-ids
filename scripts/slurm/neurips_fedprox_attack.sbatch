#!/bin/bash
#SBATCH --job-name=neurips-fedprox-attack
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/neurips_fedprox_attack/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/neurips_fedprox_attack/%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /usr/share/Modules/init/bash 2> /dev/null || true
source "/scratch/${USER}/venvs/fedids-py311/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/neurips_fedprox_attack"
cd "/scratch/${USER}/federated-ids"

NUM_CLIENTS=10
NUM_ROUNDS=20
SEEDS="42,43,44,45,46"

SPLIT_TOTAL=180
OFFSET="${OFFSET:-0}"
SPLIT_INDEX="$((${SLURM_ARRAY_TASK_ID:-0} + OFFSET))"

python -u scripts/comparative_analysis.py \
  --dimension attack_fedprox \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --seeds "${SEEDS}" \
  --server_timeout 21600 \
  --client_timeout 21600 \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
