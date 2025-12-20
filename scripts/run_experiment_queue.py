#!/usr/bin/env python3
"""
Run experiments from queue sequentially to avoid machine crashes.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path("obj2_heterogeneity_queue.json")
PROGRESS_FILE = Path("obj2_heterogeneity_progress.json")
LOG_FILE = Path("obj2_heterogeneity.log")


def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


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
    """
    Run a single experiment using comparative_analysis.py.

    Args:
        exp: Experiment parameters dict

    Returns:
        True if successful, False otherwise
    """
    agg = exp["aggregation"]
    alpha = exp["alpha"]
    adv_pct = exp["adv_pct"]
    seed = exp["seed"]
    dimension = exp.get("dimension", "attack")
    pers_epochs = exp.get("pers_epochs", 0)

    # Determine adversary fraction
    adv_frac = adv_pct / 100.0

    # Build command based on dimension
    alpha_str = "inf" if alpha == float('inf') else str(alpha)

    cmd = [
        "python",
        "scripts/comparative_analysis.py",
        "--dimension",
        dimension,
        "--dataset",
        "edge-iiotset-nightly",
        "--aggregation-methods",
        agg,
        "--alpha-values",
        alpha_str,
        "--adversary-fractions",
        str(adv_frac),
        "--seeds",
        str(seed),
        "--num_clients",
        "11" if adv_frac > 0 else "6",  # 11 for attacks, 6 for benign
        "--num_rounds",
        "20",
    ]

    # Add personalization epochs if needed
    if pers_epochs > 0:
        cmd.extend(["--personalization-epochs", str(pers_epochs)])

    log_message(f"Running: {agg} alpha={alpha_str} adv={adv_pct}% pers={pers_epochs} seed={seed} [{dimension}]")
    log_message(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 60 minute timeout per experiment
        )

        if result.returncode == 0:
            log_message(f"SUCCESS: {agg} adv={adv_pct}% seed={seed}")
            return True
        else:
            log_message(f"FAILED: {agg} adv={adv_pct}% seed={seed}")
            log_message(f"stdout: {result.stdout[-500:]}")  # Last 500 chars
            log_message(f"stderr: {result.stderr[-500:]}")
            return False

    except subprocess.TimeoutExpired:
        log_message(f"TIMEOUT: {agg} adv={adv_pct}% seed={seed}")
        return False
    except Exception as e:
        log_message(f"ERROR: {agg} adv={adv_pct}% seed={seed} - {e}")
        return False


def main():
    if not QUEUE_FILE.exists():
        print(f"ERROR: Queue file not found: {QUEUE_FILE}")
        print("Run: python scripts/analyze_missing_experiments.py")
        sys.exit(1)

    # Load queue
    with open(QUEUE_FILE) as f:
        queue = json.load(f)

    # Load progress
    progress = load_progress()
    start_index = progress["current_index"]

    log_message("=" * 80)
    log_message("EXPERIMENT QUEUE RUNNER")
    log_message("=" * 80)
    log_message(f"Total experiments: {len(queue)}")
    log_message(f"Starting from index: {start_index}")
    log_message(f"Remaining: {len(queue) - start_index}")
    log_message("=" * 80)

    # Process queue
    for i in range(start_index, len(queue)):
        exp = queue[i]
        exp_str = f"{exp['aggregation']} alpha={exp['alpha']} adv={exp['adv_pct']}% seed={exp['seed']}"

        log_message("")
        log_message(f"[{i+1}/{len(queue)}] Starting: {exp_str}")

        success = run_single_experiment(exp)

        if success:
            progress["completed"].append(exp)
        else:
            progress["failed"].append(exp)

        progress["current_index"] = i + 1
        save_progress(progress)

        # Wait between experiments to let system stabilize
        if i < len(queue) - 1:
            log_message("Waiting 10 seconds before next experiment...")
            time.sleep(10)

    # Final summary
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
            exp_str = f"{exp['aggregation']} alpha={exp['alpha']} adv={exp['adv_pct']}% seed={exp['seed']}"
            log_message(f"  - {exp_str}")

    log_message("=" * 80)


if __name__ == "__main__":
    main()
