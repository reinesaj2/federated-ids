#!/usr/bin/env python3
"""
Sequential runner for 7 failed experiments with extended timeout.

Runs experiments one at a time to avoid resource contention and ensure
completion of edge cases (Bulyan+30% adversary, IID baseline, DP seed43).
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Import from comparative_analysis
from comparative_analysis import ExperimentConfig, run_federated_experiment

# 7 failed experiments that need re-running with extended timeout
FAILED_EXPERIMENTS = [
    "comp_bulyan_alpha0.5_adv30_dp0_pers0_seed42",
    "comp_bulyan_alpha0.5_adv30_dp0_pers0_seed43",
    "comp_bulyan_alpha0.5_adv30_dp0_pers0_seed44",
    "comp_fedavg_alpha1.0_adv0_dp0_pers0_seed42",
    "comp_fedavg_alpha1.0_adv0_dp0_pers0_seed43",
    "comp_fedavg_alpha1.0_adv0_dp0_pers0_seed44",
    "comp_fedavg_alpha0.5_adv0_dp1_pers0_seed43",
]


def log_message(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def parse_preset_name(preset: str) -> ExperimentConfig:
    """Parse preset name into ExperimentConfig.

    Format: comp_{aggregation}_alpha{alpha}_adv{adv_pct}_dp{dp}_pers{pers}_seed{seed}
    """
    pattern = r"comp_(\w+)_alpha([\d.]+)_adv(\d+)_dp(\d)_pers(\d+)_seed(\d+)"
    match = re.match(pattern, preset)

    if not match:
        raise ValueError(f"Invalid preset format: {preset}")

    aggregation, alpha, adv_pct, dp, pers, seed = match.groups()

    return ExperimentConfig(
        aggregation=aggregation,
        alpha=float(alpha),
        adversary_fraction=int(adv_pct) / 100.0,
        dp_enabled=bool(int(dp)),
        dp_noise_multiplier=1.0 if int(dp) else 0.0,
        personalization_epochs=int(pers),
        num_clients=11,
        num_rounds=20,
        seed=int(seed),
        dataset="unsw",
        data_path="data/unsw/UNSW_NB15_training-set.csv",
    )


def run_single_experiment(preset: str, base_dir: Path, port_offset: int) -> bool:
    """Run a single experiment and return success status."""
    log_message(f"Starting experiment: {preset}")

    try:
        config = parse_preset_name(preset)
        port_start = 8080 + (port_offset * 200)

        start_time = time.time()
        result = run_federated_experiment(config, base_dir, port_start=port_start)
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60

        if result.get("server_exit_code") == 0:
            log_message(f"SUCCESS: {preset} completed in {elapsed_min:.1f} minutes")
            return True
        else:
            exit_code = result.get("server_exit_code", "unknown")
            log_message(f"FAILED: {preset} exited with code {exit_code} after {elapsed_min:.1f} minutes")
            return False

    except Exception as e:
        log_message(f"ERROR: {preset} failed with exception: {e}")
        return False


def verify_completion(preset: str, base_dir: Path) -> bool:
    """Verify experiment completed all 20 rounds."""
    metrics_file = base_dir / "runs" / preset / "metrics.csv"

    if not metrics_file.exists():
        return False

    try:
        with open(metrics_file) as f:
            lines = f.readlines()
            # Header + 20 data rows = 21 lines
            if len(lines) >= 21:
                return True
    except Exception:
        pass

    return False


def main() -> int:
    """Main execution function."""
    base_dir = Path("/Users/abrahamreines/Documents/Thesis/federated-ids")

    if not base_dir.exists():
        log_message(f"ERROR: Base directory not found: {base_dir}")
        return 1

    log_message("=" * 80)
    log_message("Sequential Experiment Runner - 7 Failed Experiments")
    log_message("Extended timeout: 3 hours per experiment")
    log_message(f"Total experiments: {len(FAILED_EXPERIMENTS)}")
    log_message("Estimated completion: 12-15 hours")
    log_message("=" * 80)

    results = {}
    total_start = time.time()

    for i, preset in enumerate(FAILED_EXPERIMENTS):
        log_message(f"\n{'=' * 80}")
        log_message(f"Experiment {i + 1}/{len(FAILED_EXPERIMENTS)}: {preset}")
        log_message(f"{'=' * 80}")

        # Use unique port offset for each experiment to avoid conflicts
        success = run_single_experiment(preset, base_dir, port_offset=i)

        # Verify completion
        if success:
            completed = verify_completion(preset, base_dir)
            if not completed:
                log_message(f"WARNING: {preset} reported success but didn't complete 20 rounds")
                success = False

        results[preset] = success

        # Progress summary
        successful = sum(1 for v in results.values() if v)
        log_message(f"\nProgress: {successful}/{i + 1} experiments successful")

        # Brief pause between experiments
        if i < len(FAILED_EXPERIMENTS) - 1:
            log_message("Pausing 10 seconds before next experiment...")
            time.sleep(10)

    # Final summary
    total_elapsed = time.time() - total_start
    total_hours = total_elapsed / 3600

    log_message("\n" + "=" * 80)
    log_message("FINAL SUMMARY")
    log_message("=" * 80)
    log_message(f"Total runtime: {total_hours:.2f} hours")

    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    log_message(f"\nSuccessful ({len(successful)}/{len(FAILED_EXPERIMENTS)}):")
    for preset in successful:
        log_message(f"  SUCCESS: {preset}")

    if failed:
        log_message(f"\nFailed ({len(failed)}/{len(FAILED_EXPERIMENTS)}):")
        for preset in failed:
            log_message(f"  FAILED: {preset}")

    log_message("\n" + "=" * 80)

    if len(successful) == len(FAILED_EXPERIMENTS):
        log_message("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        log_message(f"WARNING: {len(failed)} experiments failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
