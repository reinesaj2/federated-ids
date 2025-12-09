#!/usr/bin/env python3
"""
Run 3 experiments to validate macro-F1 improvements for Edge-IIoTset.

Experiment 1: Analyze existing SimpleNet baseline (pre-improvement)
Experiment 2: PerDatasetEncoderNet only
Experiment 3: PerDatasetEncoderNet + FocalLoss

Uses edge-iiotset-nightly (500k samples) for full validation.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_experiment(exp_name: str, cmd: list[str]) -> tuple[bool, str]:
    """Run a single experiment and return (success, run_dir)."""
    log(f"Starting {exp_name}...")
    log(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 60 min timeout for 500k dataset
            cwd=str(ROOT),
        )

        if result.returncode == 0:
            log(f"SUCCESS: {exp_name}")
            # Extract run directory from output
            for line in result.stdout.split("\n"):
                if "Saving config to" in line or "runs/" in line:
                    # Try to extract run directory
                    pass
            return True, ""
        else:
            log(f"FAILED: {exp_name}")
            log(f"stderr: {result.stderr[-500:]}")
            return False, ""

    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {exp_name}")
        return False, ""
    except Exception as e:
        log(f"ERROR: {exp_name} - {e}")
        return False, ""


def analyze_existing_baseline():
    """Find and analyze existing SimpleNet baseline."""
    log("=" * 80)
    log("EXPERIMENT 1: Analyzing existing SimpleNet baseline")
    log("=" * 80)

    # Find most recent FedAvg experiment with seed42
    pattern = "dsedge-iiotset-*_comp_fedavg_*_seed42*"
    matches = sorted(RUNS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not matches:
        log("WARNING: No existing baseline found, will run new SimpleNet experiment")
        return None

    baseline_dir = matches[0]
    log(f"Found baseline: {baseline_dir.name}")

    # Read final macro-f1
    client_metrics = baseline_dir / "client_0_metrics.csv"
    if client_metrics.exists():
        import pandas as pd
        df = pd.read_csv(client_metrics)
        if "macro_f1_after" in df.columns:
            final_f1 = df["macro_f1_after"].iloc[-1]
            mean_f1 = df["macro_f1_after"].mean()
            log(f"Baseline macro-F1: final={final_f1:.4f}, mean={mean_f1:.4f}")
            return {"dir": baseline_dir, "final_f1": final_f1, "mean_f1": mean_f1}

    return None


def main():
    log("Macro-F1 Improvement Validation Experiments")
    log("=" * 80)

    # Experiment 1: Analyze existing baseline
    baseline = analyze_existing_baseline()

    # Experiment 2: PerDatasetEncoderNet only (model_arch=auto uses PerDatasetEncoderNet for edge)
    log("")
    log("=" * 80)
    log("EXPERIMENT 2: PerDatasetEncoderNet (auto architecture)")
    log("=" * 80)

    cmd_exp2 = [
        "python",
        "scripts/comparative_analysis.py",
        "--dimension", "heterogeneity",
        "--dataset", "edge-iiotset-nightly",
        "--aggregation-methods", "fedavg",
        "--alpha-values", "0.5",
        "--adversary-fractions", "0.0",
        "--seeds", "42",
        "--num_clients", "6",
        "--num_rounds", "10",
    ]

    success_exp2, _ = run_experiment("Experiment 2 (PerDatasetEncoderNet)", cmd_exp2)

    # Experiment 3: PerDatasetEncoderNet + FocalLoss
    # Note: comparative_analysis.py doesn't support focal_loss flags yet
    # We'll need to run this via environment variables or modify the script
    log("")
    log("=" * 80)
    log("EXPERIMENT 3: PerDatasetEncoderNet + FocalLoss")
    log("=" * 80)
    log("SKIPPED: comparative_analysis.py doesn't support --use-focal-loss yet")
    log("To run manually:")
    log("  1. Start server with edge-iiotset-quick")
    log("  2. Start 6 clients with --use_focal_loss --focal_gamma 2.0")

    # Summary
    log("")
    log("=" * 80)
    log("EXPERIMENT SUMMARY")
    log("=" * 80)

    if baseline:
        log(f"Exp 1 (SimpleNet baseline): macro-F1 = {baseline['final_f1']:.4f}")
    if success_exp2:
        log("Exp 2 (PerDatasetEncoderNet): COMPLETED - check latest runs/")
    log("Exp 3 (+ FocalLoss): MANUAL RUN NEEDED")

    log("")
    log("Next steps:")
    log("1. Analyze results in runs/ directory")
    log("2. Compare macro-F1 scores across experiments")
    log("3. Check per-class F1 improvements")


if __name__ == "__main__":
    main()
