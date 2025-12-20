#!/usr/bin/env python3
"""
Sequential experiment runner for Objectives 2-5.
Runs experiments one at a time to avoid crashing the machine.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path("experiment_queue_obj2_to_5.json")
PROGRESS_FILE = Path("experiment_progress.json")
LOG_FILE = Path("experiment_run.log")

# Dataset paths
DATASET_PATHS = {
    "iiot": "data/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv",
    "cic": "data/cic/combined_cic_ids2017.csv",
    "unsw": "data/unsw/UNSW_NB15_training-set.csv",
}


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "current_index": 0}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_experiment(exp: dict, index: int) -> bool:
    """Run a single experiment using comparative_analysis.py or direct server/client."""
    dataset = exp.get("dataset", "iiot")
    data_path = DATASET_PATHS.get(dataset, DATASET_PATHS["iiot"])

    log(f"[{index}] Starting: {exp['description']}")

    # Build command for comparative_analysis.py
    cmd = [
        sys.executable,
        "scripts/comparative_analysis.py",
        "--dimension",
        "single",
        "--aggregation-methods",
        exp["aggregation"],
        "--alpha-values",
        str(exp["alpha"]),
        "--adversary-fractions",
        str(exp["adversary_fraction"]),
        "--seeds",
        str(exp["seed"]),
        "--num-clients",
        "6",
        "--num-rounds",
        "10",
        "--data-path",
        data_path,
        "--dataset",
        dataset,
    ]

    # Add FedProx mu if applicable
    if exp.get("fedprox_mu", 0) > 0:
        cmd.extend(["--fedprox-mu-values", str(exp["fedprox_mu"])])

    # Add personalization if applicable
    if exp.get("personalization_epochs", 0) > 0:
        cmd.extend(["--personalization-epochs", str(exp["personalization_epochs"])])

    # Add DP if applicable
    if exp.get("dp_enabled", False):
        cmd.extend(["--dp-enabled"])
        if "dp_noise_multiplier" in exp:
            cmd.extend(["--dp-noise-multiplier", str(exp["dp_noise_multiplier"])])

    log(f"[{index}] Command: {' '.join(cmd[:10])}...")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per experiment
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log(f"[{index}] SUCCESS in {elapsed:.1f}s")
            return True
        else:
            log(f"[{index}] FAILED (exit {result.returncode}) in {elapsed:.1f}s")
            log(f"[{index}] stderr: {result.stderr[:500] if result.stderr else 'none'}")
            return False

    except subprocess.TimeoutExpired:
        log(f"[{index}] TIMEOUT after 30 min")
        return False
    except Exception as e:
        log(f"[{index}] ERROR: {e}")
        return False


def main():
    log("=" * 60)
    log("STARTING EXPERIMENT QUEUE: Objectives 2-5")
    log("=" * 60)

    # Load queue
    if not QUEUE_FILE.exists():
        log(f"ERROR: Queue file not found: {QUEUE_FILE}")
        sys.exit(1)

    with open(QUEUE_FILE) as f:
        experiments = json.load(f)

    log(f"Loaded {len(experiments)} experiments")

    # Load progress
    progress = load_progress()
    start_index = progress["current_index"]

    if start_index > 0:
        log(f"Resuming from experiment {start_index}")

    # Run experiments
    for i, exp in enumerate(experiments[start_index:], start=start_index):
        progress["current_index"] = i
        save_progress(progress)

        success = run_experiment(exp, i)

        if success:
            progress["completed"].append(i)
        else:
            progress["failed"].append(i)

        save_progress(progress)

        # Progress report every 10 experiments
        if (i + 1) % 10 == 0:
            completed = len(progress["completed"])
            failed = len(progress["failed"])
            remaining = len(experiments) - i - 1
            log(f"PROGRESS: {completed} done, {failed} failed, {remaining} remaining")

    # Final report
    log("=" * 60)
    log("EXPERIMENT QUEUE COMPLETE")
    log(f"  Completed: {len(progress['completed'])}")
    log(f"  Failed: {len(progress['failed'])}")
    if progress["failed"]:
        log(f"  Failed indices: {progress['failed']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
