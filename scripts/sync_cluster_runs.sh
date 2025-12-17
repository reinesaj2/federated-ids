#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cluster}"
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR:-/scratch/reinesaj/federated-ids/runs}"
LOCAL_DIR="${LOCAL_DIR:-../cluster-experiments/cluster-runs}"
INTERVAL="${INTERVAL:-60}"

mkdir -p "$LOCAL_DIR"

echo "=== Cluster Runs Sync ==="
echo "Remote: $REMOTE_HOST:$REMOTE_RUNS_DIR"
echo "Local:  $LOCAL_DIR"
echo "Interval: ${INTERVAL}s"
echo ""

sync_once() {
    echo "[$(date '+%H:%M:%S')] Syncing..."
    rsync -avz --progress \
        --include='*/' \
        --include='config.json' \
        --include='metrics.csv' \
        --include='server_metrics.csv' \
        --include='*.log' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='__pycache__' \
        "$REMOTE_HOST:$REMOTE_RUNS_DIR/" "$LOCAL_DIR/" 2>&1 | tail -5

    local run_count=$(find "$LOCAL_DIR" -name "config.json" 2>/dev/null | wc -l | tr -d ' ')
    local metrics_count=$(find "$LOCAL_DIR" -name "metrics.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] Synced: $run_count runs, $metrics_count with metrics"
}

if [[ "${1:-}" == "--once" ]]; then
    sync_once
    exit 0
fi

echo "Starting continuous sync (Ctrl+C to stop)..."
echo ""

while true; do
    sync_once
    echo ""
    sleep "$INTERVAL"
done
