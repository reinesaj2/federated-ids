#!/usr/bin/env python3
"""
Direct experiment runner for Objective 2 robust aggregation.
Bypasses comparative_analysis.py and runs server/client directly.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path("obj2_direct_queue.json")
PROGRESS_FILE = Path("obj2_direct_progress.json")
LOG_FILE = Path("obj2_direct.log")
BASE_DIR = Path(__file__).parent.parent


def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")
        f.flush()


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "current_index": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def find_free_port(start=8080):
    """Find a free port starting from start."""
    import socket
    for port in range(start, start + 1000):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free port found")


def run_single_experiment(agg: str, alpha: float, seed: int) -> bool:
    """Run a single experiment directly with server.py and client.py."""
    
    alpha_str = str(alpha) if alpha != float("inf") else "inf"
    preset_name = f"dsedge-iiotset-nightly_comp_{agg}_alpha{alpha}_adv0_dp0_pers0_mu0.0_seed{seed}_datasetedge-iiotset-nightly"
    run_dir = BASE_DIR / "runs" / preset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        "aggregation": agg,
        "alpha": alpha,
        "adversary_fraction": 0.0,
        "dp_enabled": False,
        "dp_noise_multiplier": 0.0,
        "personalization_epochs": 0,
        "num_clients": 6,
        "num_rounds": 20,
        "seed": seed,
        "dataset": "edge-iiotset-nightly",
        "data_path": "data/edge-iiotset/edge_iiotset_nightly.csv",
        "fedprox_mu": 0.0,
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    port = find_free_port()
    num_clients = 6
    num_rounds = 20
    
    log_message(f"  Run dir: {run_dir}")
    log_message(f"  Port: {port}")
    
    # Start server
    server_cmd = [
        sys.executable, "server.py",
        "--rounds", str(num_rounds),
        "--aggregation", agg,
        "--server_address", f"localhost:{port}",
        "--logdir", str(run_dir),
        "--min_fit_clients", str(num_clients),
        "--min_eval_clients", str(num_clients),
        "--min_available_clients", str(num_clients),
        "--fraction_fit", "1.0",
        "--fraction_eval", "1.0",
    ]
    
    server_log = open(run_dir / "server.log", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
    )
    
    time.sleep(3)  # Wait for server to start
    
    # Start clients
    client_procs = []
    client_logs = []
    
    for client_id in range(num_clients):
        client_log_file = open(run_dir / f"client_{client_id}.log", "w")
        client_logs.append(client_log_file)
        
        client_cmd = [
            sys.executable, "client.py",
            "--server_address", f"localhost:{port}",
            "--data_path", "data/edge-iiotset/edge_iiotset_nightly.csv",
            "--num_clients", str(num_clients),
            "--client_id", str(client_id),
            "--seed", str(seed),
            "--alpha", alpha_str,
            "--adversary_mode", "none",
            "--personalization_epochs", "0",
            "--logdir", str(run_dir),
        ]
        
        client_proc = subprocess.Popen(
            client_cmd,
            stdout=client_log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
        client_procs.append(client_proc)
        time.sleep(0.5)  # Stagger client starts
    
    # Wait for completion
    try:
        server_exit = server_proc.wait(timeout=1800)  # 30 min timeout
    except subprocess.TimeoutExpired:
        log_message("  TIMEOUT waiting for server")
        server_proc.kill()
        for p in client_procs:
            p.kill()
        server_log.close()
        for cl in client_logs:
            cl.close()
        return False
    
    # Clean up
    for p in client_procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    
    server_log.close()
    for cl in client_logs:
        cl.close()
    
    # Check for metrics.csv
    metrics_file = run_dir / "metrics.csv"
    if metrics_file.exists():
        log_message(f"  SUCCESS: metrics.csv exists")
        return True
    else:
        log_message(f"  FAILED: no metrics.csv generated")
        return False


def main():
    queue_path = BASE_DIR / QUEUE_FILE
    if not queue_path.exists():
        print(f"ERROR: Queue file not found: {queue_path}")
        sys.exit(1)

    with open(queue_path) as f:
        queue = json.load(f)

    progress = load_progress()
    start_index = progress["current_index"]

    log_message("=" * 80)
    log_message("OBJECTIVE 2 DIRECT EXPERIMENT RUNNER")
    log_message("Running server.py + client.py directly")
    log_message("=" * 80)
    log_message(f"Total experiments: {len(queue)}")
    log_message(f"Starting from index: {start_index}")
    log_message(f"Remaining: {len(queue) - start_index}")
    log_message("=" * 80)

    for i in range(start_index, len(queue)):
        exp = queue[i]
        agg = exp["aggregation"]
        alpha = exp["alpha"]
        seed = exp["seed"]
        
        log_message("")
        log_message(f"[{i+1}/{len(queue)}] {agg} alpha={alpha} seed={seed}")

        success = run_single_experiment(agg, alpha, seed)

        if success:
            progress["completed"].append(exp)
        else:
            progress["failed"].append(exp)

        progress["current_index"] = i + 1
        save_progress(progress)

        if i < len(queue) - 1:
            log_message("Waiting 10 seconds...")
            time.sleep(10)

    log_message("")
    log_message("=" * 80)
    log_message("QUEUE COMPLETE")
    log_message("=" * 80)
    log_message(f"Completed: {len(progress['completed'])}/{len(queue)}")
    log_message(f"Failed: {len(progress['failed'])}/{len(queue)}")
    log_message("=" * 80)


if __name__ == "__main__":
    main()
