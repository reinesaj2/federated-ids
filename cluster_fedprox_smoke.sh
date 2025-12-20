#!/bin/bash
# Cluster FedProx Smoke Test - Final version with all env vars
# 20 clients, 30 rounds, FedProx μ=0.01, Edge-IIoTset full

set -euo pipefail

# Configuration
NUM_CLIENTS=20
NUM_ROUNDS=30
SERVER_PORT=8099
DATASET_PATH="/scratch/reinesaj/datasets/edge-iiotset/edge_iiotset_full.csv"
LOGDIR="/scratch/reinesaj/results/smoke_test/logs_$(date +%Y%m%d_%H%M%S)"
SEED=42

# Required environment variables
export SEED=$SEED
export FEDIDS_USE_OPACUS=1

# Activate venv
source /scratch/reinesaj/venvs/fedids-py311/bin/activate

# Create log directory
mkdir -p "$LOGDIR"

echo "================================================================================"
echo "CLUSTER SMOKE TEST: FedProx with ${NUM_CLIENTS} clients, ${NUM_ROUNDS} rounds"
echo "================================================================================"
echo "Dataset: $DATASET_PATH"
echo "Log directory: $LOGDIR"
echo "Server port: $SERVER_PORT"
echo "FedProx μ: 0.01"
echo "Heterogeneity α: 0.5"
echo "Environment: FEDIDS_USE_OPACUS=$FEDIDS_USE_OPACUS, SEED=$SEED"
echo "================================================================================"

cd /scratch/reinesaj/federated-ids

# Start server in background
echo "Starting Flower server..."
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
echo "Server started (PID: $SERVER_PID)"

# Wait for server to start
sleep 10

# Start clients in parallel
echo "Starting $NUM_CLIENTS clients in parallel..."
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

  # Stagger client starts slightly
  sleep 0.5
done

echo "All clients started"
echo "Waiting for server to complete..."

# Wait for server (it will exit when done)
wait $SERVER_PID
SERVER_EXIT=$?

echo ""
echo "================================================================================"
if [ $SERVER_EXIT -eq 0 ]; then
  echo "SMOKE TEST COMPLETED SUCCESSFULLY"
else
  echo "SMOKE TEST FAILED (server exit code: $SERVER_EXIT)"
fi
echo "================================================================================"
echo "Logs: $LOGDIR"
echo "Server log: $LOGDIR/server.log"
echo "Client logs: $LOGDIR/client_*.log"
echo "================================================================================"

# Show summary
if [ -f "$LOGDIR/server/metrics.csv" ]; then
  echo ""
  echo "Server metrics summary:"
  head -15 "$LOGDIR/server/metrics.csv"
fi

exit $SERVER_EXIT
