#!/bin/bash
#SBATCH --job-name=centralized-baseline
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/centralized_baseline/%x-%j.out
#SBATCH --error=/scratch/%u/results/centralized_baseline/%x-%j.err
#SBATCH --exclusive

set -euo pipefail

DATASET_NAME="edge-iiotset-full"
DATASET_PATH="/scratch/${USER}/datasets/edge-iiotset/edge_iiotset_full.csv"
REPO_DIR="/scratch/${USER}/federated-ids"
VENV_DIR="/scratch/${USER}/venvs/fedids-py311"
LOGDIR="/scratch/${USER}/results/centralized_baseline/${DATASET_NAME}_${SLURM_JOB_ID}"

EPOCHS=20
BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=1e-4
SEED=42
MODEL="encoder"

source /usr/share/Modules/init/bash 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"

mkdir -p "$LOGDIR" "$MPLCONFIGDIR"

cd "$REPO_DIR"

python scripts/train_centralized.py \
  --dataset "$DATASET_NAME" \
  --data_path "$DATASET_PATH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --model "$MODEL" \
  --logdir "$LOGDIR"
