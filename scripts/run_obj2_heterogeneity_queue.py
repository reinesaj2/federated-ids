#!/usr/bin/env python3
"""
Run Objective 2 heterogeneity experiments sequentially.
Designed to run in background with nohup without crashing the machine.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path("obj2_heterogeneity_queue_full.json")
PROGRESS_FILE = Path("obj2_heterogeneity_progress_full.json")
LOG_FILE = Path("obj2_heterogeneity_full.log")


def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")
        f.flush()


def load_progress():
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "current_index": 0}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_single_experiment(exp: dict) -> bool:
    """Run a single heterogeneity experiment."""
    alpha = exp["alpha"]
    seed = exp["seed"]
    fedprox_mu = exp.get("fedprox_mu", 0.0)
    dimension = exp.get("dimension", "heterogeneity")

    alpha_str = str(alpha)

    cmd = [
        "python",
        "scripts/comparative_analysis.py",
        "--dimension", dimension,
        "--dataset", "edge-iiotset-nightly",
        "--aggregation-methods", "fedavg",
        "--alpha-values", alpha_str,
        "--adversary-fractions", "0.0",
        "--seeds", str(seed),
        "--num_clients", "6",
        "--num_rounds", "20",
    ]

    if fedprox_mu > 0:
        cmd.extend(["--fedprox-mu-values", str(fedprox_mu)])

    log_message(f"Running: alpha={alpha_str} mu={fedprox_mu} seed={seed}")
    log_message(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(Path(__file__).parent.parent),
        )

        if result.returncode == 0:
            log_message(f"SUCCESS: alpha={alpha_str} mu={fedprox_mu} seed={seed}")
            return True
        else:
            log_message(f"FAILED: alpha={alpha_str} mu={fedprox_mu} seed={seed}")
            log_message(f"stdout (last 1000): {result.stdout[-1000:]}")
            log_message(f"stderr (last 1000): {result.stderr[-1000:]}")
            return False

    except subprocess.TimeoutExpired:
        log_message(f"TIMEOUT: alpha={alpha_str} mu={fedprox_mu} seed={seed}")
        return False
    except Exception as e:
        log_message(f"ERROR: alpha={alpha_str} mu={fedprox_mu} seed={seed} - {e}")
        return False


def main():
    queue_path = Path(__file__).parent.parent / QUEUE_FILE
    if not queue_path.exists():
        print(f"ERROR: Queue file not found: {queue_path}")
        sys.exit(1)

    with open(queue_path) as f:
        queue = json.load(f)

    progress = load_progress()
    start_index = progress["current_index"]

    log_message("=" * 80)
    log_message("OBJECTIVE 2 HETEROGENEITY EXPERIMENT QUEUE")
    log_message("=" * 80)
    log_message(f"Total experiments: {len(queue)}")
    log_message(f"Starting from index: {start_index}")
    log_message(f"Remaining: {len(queue) - start_index}")
    log_message("=" * 80)

    for i in range(start_index, len(queue)):
        exp = queue[i]
        exp_str = f"alpha={exp['alpha']} mu={exp.get('fedprox_mu', 0)} seed={exp['seed']}"

        log_message("")
        log_message(f"[{i+1}/{len(queue)}] Starting: {exp_str}")

        success = run_single_experiment(exp)

        if success:
            progress["completed"].append(exp)
        else:
            progress["failed"].append(exp)

        progress["current_index"] = i + 1
        save_progress(progress)

        if i < len(queue) - 1:
            log_message("Waiting 15 seconds before next experiment...")
            time.sleep(15)

    log_message("")
    log_message("=" * 80)
    log_message("QUEUE COMPLETE")
    log_message("=" * 80)
    log_message(f"Completed: {len(progress['completed'])}/{len(queue)}")
    log_message(f"Failed: {len(progress['failed'])}/{len(queue)}")

    if progress["failed"]:
        log_message("")
        log_message("Failed experiments:")
        for exp in progress["failed"]:
            exp_str = f"alpha={exp['alpha']} mu={exp.get('fedprox_mu', 0)} seed={exp['seed']}"
            log_message(f"  - {exp_str}")

    log_message("=" * 80)


if __name__ == "__main__":
    main()
