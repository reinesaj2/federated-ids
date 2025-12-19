#!/bin/bash
#SBATCH --job-name=iiot-agg-highadv
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/results/iiot_attack/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/iiot_attack/%x-%A_%a.err
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

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/iiot_attack"
cd "/scratch/${USER}/federated-ids"

NUM_CLIENTS=10
NUM_ROUNDS=20
ALPHA_VALUES="0.02,0.05,0.1,0.2,0.5,1.0,inf"
AGGREGATION_METHODS="fedavg,krum,median"
ADVERSARY_FRACTIONS="0.2,0.3"
SEEDS="42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61"

SPLIT_TOTAL=840
OFFSET="${OFFSET:-0}"
SPLIT_INDEX="$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))"

python -u scripts/comparative_analysis.py \
  --dimension full \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --alpha-values "${ALPHA_VALUES}" \
  --aggregation-methods "${AGGREGATION_METHODS}" \
  --adversary-fractions "${ADVERSARY_FRACTIONS}" \
  --dp-noise-multipliers "0.0" \
  --personalization-epochs "0" \
  --seeds "${SEEDS}" \
  --server_timeout 21600 \
  --client_timeout 21600 \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
