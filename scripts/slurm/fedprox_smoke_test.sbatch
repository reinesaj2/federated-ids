#!/bin/bash
#SBATCH --job-name=fedids-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/results/smoke_test/slurm-%j.out
#SBATCH --error=/scratch/%u/results/smoke_test/slurm-%j.err

# FedProx Smoke Test - Slurm Batch Script
# Reduced scale: 10 clients, 15 rounds for single-node memory constraints
# Uses Edge-IIoTset full dataset on compute node

set -euo pipefail

# Configuration - reduced for single-node smoke test
NUM_CLIENTS=10
NUM_ROUNDS=15
SERVER_PORT=8099
DATASET_PATH="/scratch/${USER}/datasets/edge-iiotset/edge_iiotset_full.csv"
REPO_DIR="/scratch/${USER}/federated-ids"
VENV_DIR="/scratch/${USER}/venvs/fedids-py311"
LOGDIR="/scratch/${USER}/results/smoke_test/slurm_${SLURM_JOB_ID}"

# Initialize modules if needed
source /usr/share/Modules/init/bash 2>/dev/null || true

# Activate virtual environment FIRST
source "${VENV_DIR}/bin/activate"

# Environment variables - set AFTER venv activation to ensure they persist
export SEED=42
export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"

# Create directories
mkdir -p "$LOGDIR" "$MPLCONFIGDIR"

echo "================================================================================"
echo "SLURM JOB: FedProx Smoke Test"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE:-unlimited}"
echo "Dataset: $DATASET_PATH"
echo "Log directory: $LOGDIR"
echo "Configuration: ${NUM_CLIENTS} clients, ${NUM_ROUNDS} rounds, FedProx mu=0.01, alpha=0.5"
echo "FEDIDS_USE_OPACUS=${FEDIDS_USE_OPACUS}"
echo "================================================================================"
echo ""

cd "$REPO_DIR"

# Record start time
START_TIME=$(date +%s)

# Start Flower server in background
echo "[$(date)] Starting Flower server..."
python server.py \
    --rounds $NUM_ROUNDS \
    --aggregation fedavg \
    --fedprox_mu 0.01 \
    --server_address "0.0.0.0:$SERVER_PORT" \
    --logdir "$LOGDIR/server" \
    --min_fit_clients $NUM_CLIENTS \
    --min_eval_clients $NUM_CLIENTS \
    --min_available_clients $NUM_CLIENTS \
    > "$LOGDIR/server.log" 2>&1 &

SERVER_PID=$!
echo "[$(date)] Server started (PID: $SERVER_PID)"

# Wait for server to initialize
sleep 10

# Start clients in parallel
echo "[$(date)] Starting $NUM_CLIENTS clients..."
declare -a CLIENT_PIDS
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    python client.py \
        --server_address "127.0.0.1:$SERVER_PORT" \
        --client_id $i \
        --alpha 0.5 \
        --local_epochs 1 \
        --data_path "$DATASET_PATH" \
        --adversary_mode none \
        --fedprox_mu 0.01 \
        --logdir "$LOGDIR/client_$i" \
        > "$LOGDIR/client_$i.log" 2>&1 &

    CLIENT_PIDS[$i]=$!

    # Small stagger to avoid thundering herd
    sleep 0.3
done

echo "[$(date)] All clients started"
echo "[$(date)] Waiting for training to complete..."

# Wait for server to finish (it exits after all rounds)
wait $SERVER_PID
SERVER_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
if [ $SERVER_EXIT -eq 0 ]; then
    echo "SMOKE TEST COMPLETED SUCCESSFULLY"
else
    echo "SMOKE TEST FAILED (server exit code: $SERVER_EXIT)"
fi
echo "================================================================================"
echo "Total time: ${ELAPSED}s ($((ELAPSED / 60)) minutes)"
echo "Results: $LOGDIR"
echo ""

# Display metrics summary if available
if [ -f "$LOGDIR/server/metrics.csv" ]; then
    echo "Server metrics (first 15 lines):"
    head -15 "$LOGDIR/server/metrics.csv"
    echo ""
    echo "Final round metrics:"
    tail -3 "$LOGDIR/server/metrics.csv"
fi

echo ""
echo "================================================================================"
echo "Job completed at $(date)"
echo "================================================================================"

exit $SERVER_EXIT
