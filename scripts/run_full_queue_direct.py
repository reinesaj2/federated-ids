#!/usr/bin/env python3
"""
Direct experiment runner for Objectives 2-5.
Runs server.py + client.py directly, bypassing comparative_analysis.py.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import math

QUEUE_FILE = Path("experiment_queue_alpha_fix.json")
PROGRESS_FILE = Path("experiment_progress_alpha_fix.json")
LOG_FILE = Path("experiment_run_alpha_fix.log")
BASE_DIR = Path(__file__).parent.parent

DATASET_PATHS = {
    "iiot": "data/edge-iiotset/edge_iiotset_nightly.csv",
    "edge-iiotset-nightly": "data/edge-iiotset/edge_iiotset_nightly.csv",
    "cic": "data/cic/combined_cic_ids2017.csv",
    "unsw": "data/unsw/UNSW_NB15_training-set.csv",
    "curated-500k": "datasets/edge-iiotset/processed/edge_iiotset_500k_curated.csv",
}

# Map friendly names to client.py expected names
DATASET_NAMES = {
    "iiot": "edge-iiotset-nightly",
    "edge-iiotset-nightly": "edge-iiotset-nightly",
    "cic": "cic",
    "unsw": "unsw",
    "curated-500k": "edge-iiotset-full",
}


def log_message(message: str):
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


def find_free_port(start: int = 8080) -> int:
    """Return an available localhost port.

    Prefer OS-assigned ephemeral port to avoid exhaustion; fall back to
    sequential probing if that fails.
    """
    import socket

    # Try OS-assigned ephemeral port first
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]
    except OSError:
        pass

    # Fallback to sequential probe
    for port in range(start, start + 2000):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError("No free port found")


def run_experiment(exp: dict, index: int) -> bool:
    """Run a single experiment directly with server.py and client.py."""

    dataset_key = exp.get("dataset", "iiot")
    dataset = DATASET_NAMES.get(dataset_key, "edge-iiotset-nightly")
    data_path = exp.get("dataset_path", DATASET_PATHS.get(dataset_key, DATASET_PATHS["iiot"]))
    agg = exp["aggregation"]
    alpha = exp["alpha"]
    seed = exp["seed"]
    adv_frac = exp.get("adversary_fraction", 0.0)
    pers_epochs = exp.get("personalization_epochs", 0)
    dp_enabled = exp.get("dp_enabled", False)
    dp_noise = exp.get("dp_noise_multiplier", 1.0)
    fedprox_mu = exp.get("fedprox_mu", 0.0)
    num_clients = exp.get("num_clients", 12)
    num_rounds = exp.get("num_rounds", 15)
    local_epochs = exp.get("local_epochs", 1)
    client_fraction = float(exp.get("client_fraction", 1.0))
    fraction_eval = float(exp.get("fraction_eval", client_fraction))
    min_fit_clients = max(1, math.ceil(num_clients * client_fraction))
    min_eval_clients = max(1, math.ceil(num_clients * fraction_eval))
    
    # Build preset name
    alpha_str = str(alpha) if alpha != float("inf") else "inf"
    adv_pct = int(adv_frac * 100)
    dp_flag = 1 if dp_enabled else 0
    sigma_str = str(dp_noise) if dp_enabled else "0.0"
    preset_name = (
        f"ds{dataset}_comp_{agg}"
        f"_alpha{alpha_str}"
        f"_adv{adv_pct}"
        f"_dp{dp_flag}"
        f"_sigma{sigma_str}"
        f"_pers{pers_epochs}"
        f"_mu{fedprox_mu}"
        f"_cf{client_fraction}"
        f"_le{local_epochs}"
        f"_idx{index}"
        f"_seed{seed}"
    )
    run_dir = BASE_DIR / "runs" / preset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        "aggregation": agg,
        "alpha": alpha,
        "adversary_fraction": adv_frac,
        "dp_enabled": dp_enabled,
        "dp_noise_multiplier": dp_noise if dp_enabled else 0.0,
        "personalization_epochs": pers_epochs,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "seed": seed,
        "dataset": dataset,
        "data_path": data_path,
        "fedprox_mu": fedprox_mu,
        "client_fraction": client_fraction,
        "fraction_eval": fraction_eval,
        "local_epochs": local_epochs,
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    port = find_free_port()
    num_adversaries = int(adv_frac * num_clients)
    
    log_message(f"  Run dir: {preset_name}")
    log_message(f"  Port: {port}, Adversaries: {num_adversaries}")
    
    # Server aggregation (fedprox uses fedavg on server side)
    server_agg = "fedavg" if agg == "fedprox" else agg
    
    # Start server
    server_cmd = [
        sys.executable, "server.py",
        "--rounds", str(num_rounds),
        "--aggregation", server_agg,
        "--server_address", f"localhost:{port}",
        "--logdir", str(run_dir),
        "--min_fit_clients", str(min_fit_clients),
        "--min_eval_clients", str(min_eval_clients),
        "--min_available_clients", str(num_clients),
        "--fraction_fit", str(client_fraction),
        "--fraction_eval", str(fraction_eval),
    ]
    
    server_log = open(run_dir / "server.log", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
    )
    
    time.sleep(3)
    
    # Start clients
    client_procs = []
    client_logs = []
    
    for client_id in range(num_clients):
        client_log_file = open(run_dir / f"client_{client_id}.log", "w")
        client_logs.append(client_log_file)
        
        # Determine adversary mode
        is_adversary = client_id < num_adversaries
        adv_mode = "grad_ascent" if is_adversary else "none"
        
        client_cmd = [
            sys.executable, "client.py",
            "--server_address", f"localhost:{port}",
            "--dataset", dataset,
            "--data_path", data_path,
            "--partition_strategy", "dirichlet",  # Always use Dirichlet; alpha controls heterogeneity
            "--num_clients", str(num_clients),
            "--client_id", str(client_id),
            "--seed", str(seed),
            "--alpha", alpha_str,
            "--adversary_mode", adv_mode,
            "--personalization_epochs", str(pers_epochs),
            "--logdir", str(run_dir),
            "--local_epochs", str(local_epochs),
        ]
        
        # Add FedProx mu if applicable
        if fedprox_mu > 0:
            client_cmd.extend(["--fedprox_mu", str(fedprox_mu)])
        
        # Add DP if applicable
        if dp_enabled:
            client_cmd.extend([
                "--dp_enabled",
                "--dp_noise_multiplier", str(dp_noise),
            ])
        
        client_proc = subprocess.Popen(
            client_cmd,
            stdout=client_log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
        client_procs.append(client_proc)
        time.sleep(0.5)
    
    # Wait for completion
    try:
        server_proc.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        log_message("  TIMEOUT")
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
    
    # Check success
    metrics_file = run_dir / "client_0_metrics.csv"
    if metrics_file.exists():
        log_message("  SUCCESS")
        return True
    else:
        log_message("  FAILED: no metrics")
        return False


def main():
    queue_path = BASE_DIR / QUEUE_FILE
    if not queue_path.exists():
        log_message(f"ERROR: Queue file not found: {queue_path}")
        sys.exit(1)

    with open(queue_path) as f:
        queue = json.load(f)

    progress = load_progress()
    start_index = progress["current_index"]

    log_message("=" * 70)
    log_message("FULL EXPERIMENT QUEUE: Objectives 2-5")
    log_message("=" * 70)
    log_message(f"Total: {len(queue)} | Starting from: {start_index} | Remaining: {len(queue) - start_index}")
    log_message("=" * 70)

    for i in range(start_index, len(queue)):
        exp = queue[i]
        
        log_message("")
        log_message(f"[{i+1}/{len(queue)}] {exp['description']}")

        success = run_experiment(exp, i)

        if success:
            progress["completed"].append(i)
        else:
            progress["failed"].append(i)

        progress["current_index"] = i + 1
        save_progress(progress)

        if i < len(queue) - 1:
            time.sleep(5)

    log_message("")
    log_message("=" * 70)
    log_message("QUEUE COMPLETE")
    log_message(f"Completed: {len(progress['completed'])} | Failed: {len(progress['failed'])}")
    log_message("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Direct experiment runner")
    parser.add_argument("--queue_file", type=Path, default=QUEUE_FILE, help="Path to queue JSON")
    parser.add_argument("--progress_file", type=Path, default=PROGRESS_FILE, help="Path to progress JSON")
    parser.add_argument("--log_file", type=Path, default=LOG_FILE, help="Path to log file")
    args = parser.parse_args()

    # Override globals so helpers use the requested files
    QUEUE_FILE = Path(args.queue_file)
    PROGRESS_FILE = Path(args.progress_file)
    LOG_FILE = Path(args.log_file)

    main()
