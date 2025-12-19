#!/usr/bin/env python3
"""
Script to run thesis experiments 10-12 (FedProx) using the Nightly Edge-IIoTset dataset.
Worktree: iiot-experiments
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure we can import from current directory
sys.path.insert(0, str(Path(__file__).parent))

from comparative_analysis import ExperimentConfig
from run_experiments_optimized import run_experiment_with_state, print_resource_status


def main():
    print(f"\n{'='*70}")
    print("THESIS EXPERIMENTS 10-12: FEDPROX (NIGHTLY DATASET)")
    print(f"{'='*70}")

    base_dir = Path.cwd()

    # Experiments 10, 11, 12 from checklist
    # Fixed parameters
    dataset = "edge-iiotset-nightly"  # User requested NIGHTLY
    alpha = 0.02  # Highly Non-IID
    seed = 42
    num_clients = 6
    num_rounds = 15  # Matching checklist status/manual runs

    # Varying mu
    mu_values = [0.01, 0.1, 1.0]

    configs = []
    for mu in mu_values:
        config = ExperimentConfig.with_dataset(
            dataset,
            aggregation="fedavg",  # MUST be fedavg for server.py to accept it. mu>0 enables FedProx logic.
            alpha=alpha,
            adversary_fraction=0.0,
            dp_enabled=False,
            dp_noise_multiplier=0.0,
            personalization_epochs=0,
            num_clients=num_clients,
            num_rounds=num_rounds,
            seed=seed,
            fedprox_mu=mu,
        )
        configs.append(config)

    workers = 2
    total = len(configs)
    print(f"Queuing {total} experiments with {workers} workers...")
    print(f"Dataset: {dataset}")

    completed_tasks = 0
    failed_tasks = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_experiment_with_state,
                config,
                base_dir,
                100 + (idx * 20),  # Safe port offset: 8080 + 20000 + ...
                idx % workers,
                "full",  # dataset_type (affects timeout calculation)
                2,  # max retries
            ): config
            for idx, config in enumerate(configs)
        }

        for future in as_completed(futures):
            result = future.result()

            completed_tasks += 1
            duration = result.get("duration", 0)

            if result["status"] == "success":
                print(f"[COMPLETE] {result['preset']} ({duration/60:.1f}m)")
            elif result["status"] == "skipped":
                print(f"[SKIP] {result['preset']}: {result['message']}")
            else:
                failed_tasks += 1
                print(f"[FAILED] {result['preset']}: {result.get('error')}")

            print_resource_status()

    print(f"\nFinished. Success: {completed_tasks-failed_tasks}, Failed: {failed_tasks}")


if __name__ == "__main__":
    main()
