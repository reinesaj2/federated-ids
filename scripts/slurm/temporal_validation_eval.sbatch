#!/bin/bash
#SBATCH --job-name=tvp-eval
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /scratch/reinesaj/venvs/fedids-py311/bin/activate
cd /scratch/reinesaj/federated-ids

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/reinesaj/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR"

EVAL_SEEDS="45,46,47,48,49"
ALPHA_VALUES="0.02,0.05,0.1,0.2,0.5,1.0,inf"

declare -A MU_STAR
MU_STAR["0.02"]=0.02
MU_STAR["0.05"]=1.0
MU_STAR["0.1"]=1.0
MU_STAR["0.2"]=0.5
MU_STAR["0.5"]=0.5
MU_STAR["1.0"]=1.0
MU_STAR["inf"]=0.5

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

IFS=',' read -ra ALPHAS <<< "$ALPHA_VALUES"
IFS=',' read -ra SEEDS <<< "$EVAL_SEEDS"

N_ALPHAS=${#ALPHAS[@]}
N_SEEDS=${#SEEDS[@]}

ALPHA_IDX=$((TASK_ID / N_SEEDS))
SEED_IDX=$((TASK_ID % N_SEEDS))

if [ "$ALPHA_IDX" -ge "$N_ALPHAS" ]; then
    echo "Task $TASK_ID: Index out of range (alpha_idx=$ALPHA_IDX >= $N_ALPHAS)"
    exit 0
fi

ALPHA="${ALPHAS[$ALPHA_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
MU="${MU_STAR[$ALPHA]}"

echo "Task $TASK_ID: alpha=$ALPHA, mu*=$MU, seed=$SEED"

python -u scripts/comparative_analysis.py \
    --dimension heterogeneity_fedprox \
    --dataset edge-iiotset-full \
    --num_clients 10 \
    --num_rounds 20 \
    --alpha-values "$ALPHA" \
    --fedprox-mu-values "$MU" \
    --seeds "$SEED" \
    --server_timeout 21600 \
    --client_timeout 21600 \
    --temporal_validation \
    --split-total 1 \
    --split-index 0

echo "Task $TASK_ID completed at $(date)"
