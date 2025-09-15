#!/usr/bin/env bash
set -euo pipefail

PORT_MAIN=${PORT_MAIN:-8099}
PORT_ALT=${PORT_ALT:-8100}
ROUNDS=${ROUNDS:-2}
TIMEOUT_SECS=${TIMEOUT_SECS:-30}
LOGDIR=${LOGDIR:-".verify_logs"}
SEED=${SEED:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$LOGDIR"

info() { echo "[verify] $*"; }

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

run_once() {
  local port=$1
  local name=$2
  local server_log="$LOGDIR/server_${name}.log"
  local c1_log="$LOGDIR/client1_${name}.log"
  local c2_log="$LOGDIR/client2_${name}.log"

  info "Starting server on 127.0.0.1:${port} (rounds=${ROUNDS})"
  SEED=$SEED python server.py --rounds "$ROUNDS" --aggregation fedavg --server_address 127.0.0.1:"$port" >"$server_log" 2>&1 &
  local server_pid=$!

  # Ensure we always clean up server
  trap 'kill $server_pid 2>/dev/null || true' EXIT

  if ! wait_for_port 127.0.0.1 "$port" 10; then
    info "Server failed to open port ${port} in time"
    kill $server_pid 2>/dev/null || true
    return 1
  fi

  info "Starting clients"
  python client.py --server_address 127.0.0.1:"$port" --samples 1000 --features 10 --seed "$SEED" >"$c1_log" 2>&1 &
  local c1=$!
  python client.py --server_address 127.0.0.1:"$port" --samples 1000 --features 10 --seed "$SEED" >"$c2_log" 2>&1 &
  local c2=$!

  # Wait for clients with timeout
  local waited=0
  while kill -0 $c1 2>/dev/null || kill -0 $c2 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ $waited -ge $TIMEOUT_SECS ]; then
      info "Timeout waiting for clients; killing"
      kill $c1 $c2 2>/dev/null || true
      kill $server_pid 2>/dev/null || true
      return 124
    fi
  done

  # Give server a moment to wrap up and then check rounds
  sleep 1
  if ! grep -q "Run finished ${ROUNDS} round(s)" "$server_log"; then
    info "Server did not report ${ROUNDS} completed rounds"
    tail -n +1 "$server_log" "$c1_log" "$c2_log" || true
    kill $server_pid 2>/dev/null || true
    return 2
  fi

  kill $server_pid 2>/dev/null || true
  trap - EXIT
  info "Run '${name}' completed successfully"
}

ensure_env() {
  info "Checking Python and venv"
  if [ ! -d .venv ]; then
    info ".venv not found; creating"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -c "import flwr, torch, pandas, sklearn, scapy, Crypto" >/dev/null 2>&1 || {
    info "Installing requirements"
    python -m pip install -U pip setuptools wheel >/dev/null
    pip install -r requirements.txt >/dev/null
  }
}

extract_history() {
  local server_log=$1
  grep -E "^INFO :\s+\s*round [12]:|^INFO :\s+History \(loss, distributed\):|^INFO :\s+round [0-9]+:" "$server_log" || true
}

main() {
  cd "$REPO_ROOT"
  ensure_env

  # A) Main run on PORT_MAIN
  run_once "$PORT_MAIN" main

  # B) Alternate port check
  run_once "$PORT_ALT" alt

  # C) Reproducibility: two runs on same port and compare histories
  run_once "$PORT_MAIN" repro1
  run_once "$PORT_MAIN" repro2
  local h1; h1=$(extract_history "$LOGDIR/server_repro1.log")
  local h2; h2=$(extract_history "$LOGDIR/server_repro2.log")
  if [ "$h1" != "$h2" ]; then
    info "Reproducibility check FAILED"
    diff -u <(echo "$h1") <(echo "$h2") || true
    exit 3
  fi
  info "Reproducibility check passed"

  info "All checks passed"
}

main "$@"
