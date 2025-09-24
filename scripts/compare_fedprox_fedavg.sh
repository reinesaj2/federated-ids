#!/usr/bin/env bash
set -euo pipefail

# FedProx vs FedAvg Comparison Script
# Compares federated learning performance on non-IID data (alpha=0.05)
# following the verify_readme.sh pattern for consistency

PORT_BASE=${PORT_BASE:-9000}
ROUNDS=${ROUNDS:-5}
TIMEOUT_SECS=${TIMEOUT_SECS:-60}
LOGDIR=${LOGDIR:-"./comparison_logs"}
SEED=${SEED:-42}
ALPHA=${ALPHA:-0.05}  # Non-IID Dirichlet alpha as per issue #12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$LOGDIR"

info() { echo "[compare] $*"; }

wait_for_port() {
  local host=$1
  local port=$2
  local max_wait=${3:-10}
  local waited=0
  while ! nc -z "$host" "$port" >/dev/null 2>&1; do
    sleep 0.5
    waited=$((waited + 1))
    if [ $waited -ge $((max_wait * 2)) ]; then
      return 1
    fi
  done
  return 0
}

run_algorithm() {
  local algorithm=$1
  local port=$2
  local fedprox_mu=${3:-0.0}
  local exp_dir="$LOGDIR/${algorithm}"

  mkdir -p "$exp_dir"

  local server_log="$exp_dir/server.log"
  local c1_log="$exp_dir/client1.log"
  local c2_log="$exp_dir/client2.log"

  info "Starting ${algorithm} experiment (port=${port}, mu=${fedprox_mu}, alpha=${ALPHA})"

  # Start server
  SEED=$SEED python server.py \
    --rounds "$ROUNDS" \
    --aggregation fedavg \
    --server_address 127.0.0.1:"$port" \
    --logdir "$exp_dir" \
    >"$server_log" 2>&1 &
  local server_pid=$!

  # Ensure cleanup
  trap 'kill $server_pid 2>/dev/null || true' EXIT

  if ! wait_for_port 127.0.0.1 "$port" 10; then
    info "${algorithm} server failed to open port ${port}"
    kill $server_pid 2>/dev/null || true
    return 1
  fi

  info "Starting ${algorithm} clients (non-IID, alpha=${ALPHA})"

  # Start clients with non-IID synthetic data
  python client.py \
    --server_address 127.0.0.1:"$port" \
    --client_id 0 --num_clients 2 \
    --dataset synthetic --samples 1000 --features 20 \
    --partition_strategy dirichlet --alpha "$ALPHA" \
    --fedprox_mu "$fedprox_mu" \
    --local_epochs 3 \
    --logdir "$exp_dir" \
    --seed "$SEED" \
    >"$c1_log" 2>&1 &
  local c1=$!

  python client.py \
    --server_address 127.0.0.1:"$port" \
    --client_id 1 --num_clients 2 \
    --dataset synthetic --samples 1000 --features 20 \
    --partition_strategy dirichlet --alpha "$ALPHA" \
    --fedprox_mu "$fedprox_mu" \
    --local_epochs 3 \
    --logdir "$exp_dir" \
    --seed "$SEED" \
    >"$c2_log" 2>&1 &
  local c2=$!

  # Wait for clients with timeout
  local waited=0
  while kill -0 $c1 2>/dev/null || kill -0 $c2 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ $waited -ge $TIMEOUT_SECS ]; then
      info "${algorithm} timeout waiting for clients; killing"
      kill $c1 $c2 2>/dev/null || true
      kill $server_pid 2>/dev/null || true
      return 124
    fi
  done

  # Check completion
  sleep 2
  if ! grep -q "Run finished ${ROUNDS} round(s)" "$server_log"; then
    info "${algorithm} server did not complete ${ROUNDS} rounds"
    tail -n 10 "$server_log" "$c1_log" "$c2_log" || true
    kill $server_pid 2>/dev/null || true
    return 2
  fi

  kill $server_pid 2>/dev/null || true
  trap - EXIT
  info "${algorithm} experiment completed successfully"
  return 0
}

extract_final_metrics() {
  local algorithm=$1
  local metrics_file="$LOGDIR/${algorithm}/metrics.csv"

  if [ ! -f "$metrics_file" ]; then
    echo "No metrics file for $algorithm"
    return
  fi

  # Extract final round metrics
  local final_round=$(tail -n 1 "$metrics_file" | cut -d',' -f1)
  local l2_distance=$(tail -n 1 "$metrics_file" | cut -d',' -f5)
  local cos_similarity=$(tail -n 1 "$metrics_file" | cut -d',' -f6)
  local update_norm=$(tail -n 1 "$metrics_file" | cut -d',' -f8)

  echo "$algorithm: Round=$final_round, L2_dist=$l2_distance, Cos_sim=$cos_similarity, Norm=$update_norm"
}

ensure_env() {
  info "Checking Python and venv"
  if [ ! -d .venv ]; then
    info ".venv not found; creating"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -c "import flwr, torch, pandas, sklearn" >/dev/null 2>&1 || {
    info "Installing requirements"
    python -m pip install -U pip setuptools wheel >/dev/null
    pip install -r requirements.txt >/dev/null
  }
}

main() {
  cd "$REPO_ROOT"
  ensure_env

  # Clean previous comparison logs
  rm -rf "$LOGDIR"
  mkdir -p "$LOGDIR"

  info "Starting FedProx vs FedAvg comparison (alpha=${ALPHA}, rounds=${ROUNDS})"

  # Run FedAvg (mu=0.0)
  if ! run_algorithm "fedavg" $((PORT_BASE)) 0.0; then
    info "FedAvg experiment failed"
    exit 1
  fi

  # Run FedProx (mu=0.01 as default)
  if ! run_algorithm "fedprox" $((PORT_BASE + 1)) 0.01; then
    info "FedProx experiment failed"
    exit 2
  fi

  # Display comparison summary
  info "Comparison Results:"
  extract_final_metrics "fedavg"
  extract_final_metrics "fedprox"

  info "Detailed logs and metrics saved to: $LOGDIR"
  info "Generate plots with: python scripts/plot_metrics.py --fedprox_comparison --logdir $LOGDIR"

  info "Comparison completed successfully"
}

main "$@"