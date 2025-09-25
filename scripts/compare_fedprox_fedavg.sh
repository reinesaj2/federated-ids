#!/usr/bin/env bash
set -euo pipefail

# FedProx vs FedAvg Matrix Comparison Script
# Compares federated learning performance across multiple alpha/mu combinations
# Supports both single values and comma-separated matrix parameters

PORT_BASE=${PORT_BASE:-9000}
ROUNDS=${ROUNDS:-5}
TIMEOUT_SECS=${TIMEOUT_SECS:-60}
LOGDIR=${LOGDIR:-"./comparison_logs"}
SEED=${SEED:-42}
ALPHA_VALUES=${ALPHA_VALUES:-"0.05"}  # Comma-separated alpha values (e.g., "0.05,0.1,0.5")
MU_VALUES=${MU_VALUES:-"0.0,0.01"}    # Comma-separated mu values (e.g., "0.0,0.01,0.1")

# Legacy support for single ALPHA parameter
if [ -n "${ALPHA:-}" ]; then
  ALPHA_VALUES="$ALPHA"
fi

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
  local alpha=${4:-0.05}
  local exp_dir="$LOGDIR/${algorithm}_alpha${alpha}_mu${fedprox_mu}"

  mkdir -p "$exp_dir"

  local server_log="$exp_dir/server.log"
  local c1_log="$exp_dir/client1.log"
  local c2_log="$exp_dir/client2.log"

  info "Starting ${algorithm} experiment (port=${port}, mu=${fedprox_mu}, alpha=${alpha})"

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

  info "Starting ${algorithm} clients (non-IID, alpha=${alpha})"

  # Start clients with non-IID synthetic data
  python client.py \
    --server_address 127.0.0.1:"$port" \
    --client_id 0 --num_clients 2 \
    --dataset synthetic --samples 1000 --features 20 \
    --partition_strategy dirichlet --alpha "$alpha" \
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
    --partition_strategy dirichlet --alpha "$alpha" \
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
  local experiment_name=$1
  local metrics_file="$LOGDIR/${experiment_name}/metrics.csv"

  if [ ! -f "$metrics_file" ]; then
    echo "No metrics file for $experiment_name"
    return
  fi

  # Extract final round metrics
  local final_round=$(tail -n 1 "$metrics_file" | cut -d',' -f1)
  local l2_distance=$(tail -n 1 "$metrics_file" | cut -d',' -f5)
  local cos_similarity=$(tail -n 1 "$metrics_file" | cut -d',' -f6)
  local update_norm=$(tail -n 1 "$metrics_file" | cut -d',' -f8)

  echo "$experiment_name: Round=$final_round, L2_dist=$l2_distance, Cos_sim=$cos_similarity, Norm=$update_norm"
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

  # Parse comma-separated values into arrays
  IFS=',' read -ra ALPHA_ARRAY <<< "$ALPHA_VALUES"
  IFS=',' read -ra MU_ARRAY <<< "$MU_VALUES"

  # Clean previous comparison logs
  rm -rf "$LOGDIR"
  mkdir -p "$LOGDIR"

  info "Starting FedProx vs FedAvg matrix comparison"
  info "Alpha values: ${ALPHA_ARRAY[*]}"
  info "Mu values: ${MU_ARRAY[*]}"
  info "Total combinations: $((${#ALPHA_ARRAY[@]} * ${#MU_ARRAY[@]}))"
  info "Rounds per experiment: ${ROUNDS}"

  local port_counter=0
  local failed_experiments=0
  local completed_experiments=()

  # Run all combinations of alpha and mu
  for alpha in "${ALPHA_ARRAY[@]}"; do
    alpha=$(echo "$alpha" | xargs)  # trim whitespace
    for mu in "${MU_ARRAY[@]}"; do
      mu=$(echo "$mu" | xargs)  # trim whitespace

      # Determine algorithm name based on mu value
      local algorithm
      if [[ "$mu" == "0.0" ]] || [[ "$mu" == "0" ]]; then
        algorithm="fedavg"
      else
        algorithm="fedprox"
      fi

      local port=$((PORT_BASE + port_counter))
      port_counter=$((port_counter + 1))

      info "Running $algorithm with alpha=$alpha, mu=$mu (port=$port)"

      if run_algorithm "$algorithm" "$port" "$mu" "$alpha"; then
        local exp_name="${algorithm}_alpha${alpha}_mu${mu}"
        completed_experiments+=("$exp_name")
        info "✓ Completed: $exp_name"
      else
        failed_experiments=$((failed_experiments + 1))
        info "✗ Failed: ${algorithm}_alpha${alpha}_mu${mu}"
      fi

      # Small delay between experiments to avoid port conflicts
      sleep 1
    done
  done

  # Display comparison summary
  info ""
  info "Matrix Comparison Results:"
  info "========================="
  for exp_name in "${completed_experiments[@]}"; do
    extract_final_metrics "$exp_name"
  done

  info ""
  info "Summary:"
  info "  Completed: ${#completed_experiments[@]} experiments"
  info "  Failed: $failed_experiments experiments"
  info "  Detailed logs and metrics saved to: $LOGDIR"

  if [ ${#completed_experiments[@]} -gt 0 ]; then
    info "  Generate analysis with: python scripts/analyze_fedprox_comparison.py --artifacts_dir $LOGDIR --output_dir analysis"
  fi

  if [ $failed_experiments -gt 0 ]; then
    info "Matrix comparison completed with $failed_experiments failures"
    exit 1
  else
    info "Matrix comparison completed successfully"
  fi
}

main "$@"