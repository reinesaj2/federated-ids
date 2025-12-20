#!/usr/bin/env python3
"""
Cluster Smoke Test - FedProx with 20 clients, 30 rounds

Tests cluster setup and computational performance with an intense workload.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run cluster smoke test")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to edge_iiotset_full.csv")
    parser.add_argument("--output-dir", type=str, default="/scratch/$USER/results/smoke_test", help="Output directory for results")
    args = parser.parse_args()

    # Validate dataset exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    # Prepare output directory
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment command
    preset_name = "smoke-test_comp_fedprox_alpha0.5_adv0_dp0_pers0_mu0.01_seed42"

    cmd = [
        sys.executable,  # Use same python
        "experiment.py",
        preset_name,
        "--num_clients",
        "20",
        "--num_rounds",
        "30",
        "--aggregation",
        "fedprox",
        "--alpha",
        "0.5",
        "--adversary_fraction",
        "0",
        "--dp_enabled",
        "0",
        "--personalization_epochs",
        "0",
        "--fedprox_mu",
        "0.01",
        "--seed",
        "42",
        "--data_path",
        str(dataset_path),
    ]

    print("=" * 80)
    print("CLUSTER SMOKE TEST")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Clients: 20")
    print(f"  - Rounds: 30")
    print(f"  - Aggregation: FedProx (μ=0.01)")
    print(f"  - Heterogeneity: α=0.5 (moderate non-IID)")
    print(f"  - Output: {output_dir}")
    print("=" * 80)
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nStarting experiment...\n")

    # Record start time
    start_time = time.time()

    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)

        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print("SMOKE TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"\nResults should be in: runs/{preset_name}/")

        # Save timing info
        timing_file = output_dir / "smoke_test_timing.json"
        with open(timing_file, "w") as f:
            json.dump(
                {
                    "elapsed_seconds": elapsed,
                    "elapsed_minutes": elapsed / 60,
                    "num_clients": 20,
                    "num_rounds": 30,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )
        print(f"Timing saved to: {timing_file}")

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n\nSMOKE TEST FAILED after {elapsed:.1f}s")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
