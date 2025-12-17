#!/bin/bash
#SBATCH --job-name=tvp-baseline
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source ~/federated-ids/venv/bin/activate
cd ~/federated-ids

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="${HOME}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR"

TUNING_SEEDS="42,43,44"
ALPHA_VALUES="0.02,0.05,0.1,0.2,0.5,1.0,inf"

python scripts/comparative_analysis.py \
    --dimension heterogeneity \
    --dataset edge-iiotset-full \
    --num_clients 10 \
    --num_rounds 20 \
    --alpha-values "$ALPHA_VALUES" \
    --seeds "$TUNING_SEEDS" \
    --server_timeout 21600 \
    --client_timeout 21600 \
    --temporal_validation \
    --split-total "${SLURM_ARRAY_TASK_COUNT:-21}" \
    --split-index "${SLURM_ARRAY_TASK_ID:-0}"

echo "Task ${SLURM_ARRAY_TASK_ID:-0} completed at $(date)"
