#!/bin/bash
#SBATCH --job-name=fedids-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/results/smoke_test_%j.out
#SBATCH --error=/scratch/%u/results/smoke_test_%j.err

set -euo pipefail

echo "=== FedIDS Cluster Smoke Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo ""

# Use Python 3.12 directly from venv (bypass all PATH issues)
PYTHON_BIN="/scratch/$USER/venvs/fedids/bin/python"

echo "Python: $($PYTHON_BIN --version)"
echo "Python path: $PYTHON_BIN"
echo ""

cd /scratch/$USER/federated-ids

$PYTHON_BIN scripts/comparative_analysis.py \
  --dimension heterogeneity_fedprox \
  --dataset edge-iiotset-full \
  --data_path /scratch/$USER/datasets/edge-iiotset/edge_iiotset_full.csv \
  --output_dir /scratch/$USER/results/smoke_test \
  --num_clients 20 \
  --num_rounds 30 \
  --fedprox-mu-values "0.01" \
  --alpha-values "0.5" \
  --seeds "42"

echo ""
echo "Completed: $(date)"
echo "=== Smoke Test Finished ==="
