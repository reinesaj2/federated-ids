#\!/bin/bash
#SBATCH --job-name=fedids-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Memory: using node default
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

# CRITICAL: Deactivate pyenv first, then activate Python 3.12 venv
export PATH=$(echo "$PATH" | tr : n | grep -v pyenv | tr n : | sed s/:$//)
unset PYENV_ROOT
unset PYENV_VERSION

# Activate Python 3.12 venv
source /scratch/$USER/venvs/fedids/bin/activate

echo "Python: $(python --version)"
echo "Python path: $(which python)"
echo "Pip list (first 5):"
pip list | head -5
echo ""

# Run smoke test
cd /scratch/$USER/federated-ids

python scripts/comparative_analysis.py \
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
