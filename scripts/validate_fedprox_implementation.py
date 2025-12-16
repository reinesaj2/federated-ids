#!/usr/bin/env python3
"""
FedProx Implementation Validation Experiments

Validates that our AdamW-based FedProx implementation produces correct results by testing:
1. FedProx improves over FedAvg baseline
2. FedProx helps with heterogeneous data (varying alpha)
3. Results are consistent across multiple seeds

Dataset: Edge-IIoTset 500k curated
Experiments: 16 total (12 core + 4 replications)
Execution: Sequential (one at a time)
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from comparative_analysis import ExperimentConfig
from run_experiments_optimized import run_experiment_with_state, print_resource_status


# Experimental Design
# Based on Li et al. (2020) and our research in docs/FEDPROX_OPTIMIZER_RESEARCH.md
EXPERIMENTS = [
    # Phase 1: IID baseline validation (alpha=1.0)
    # Tests: Does FedProx work correctly with IID data?
    {"phase": 1, "desc": "IID Baseline (FedAvg)", "alpha": 1.0, "mu": 0.0, "seed": 42},
    {"phase": 1, "desc": "IID + Weak FedProx", "alpha": 1.0, "mu": 0.01, "seed": 42},
    {"phase": 1, "desc": "IID + Moderate FedProx", "alpha": 1.0, "mu": 0.1, "seed": 42},
    {"phase": 1, "desc": "IID + Strong FedProx", "alpha": 1.0, "mu": 1.0, "seed": 42},

    # Phase 2: High non-IID validation (alpha=0.1)
    # Tests: Does FedProx improve heterogeneous training? (Primary thesis question)
    {"phase": 2, "desc": "High Non-IID Baseline", "alpha": 0.1, "mu": 0.0, "seed": 42},
    {"phase": 2, "desc": "High Non-IID + Weak FedProx", "alpha": 0.1, "mu": 0.01, "seed": 42},
    {"phase": 2, "desc": "High Non-IID + Moderate FedProx", "alpha": 0.1, "mu": 0.1, "seed": 42},
    {"phase": 2, "desc": "High Non-IID + Strong FedProx", "alpha": 0.1, "mu": 1.0, "seed": 42},

    # Phase 3: Moderate non-IID validation (alpha=0.5)
    # Tests: FedProx performance across heterogeneity spectrum
    {"phase": 3, "desc": "Moderate Non-IID Baseline", "alpha": 0.5, "mu": 0.0, "seed": 42},
    {"phase": 3, "desc": "Moderate Non-IID + Weak FedProx", "alpha": 0.5, "mu": 0.01, "seed": 42},
    {"phase": 3, "desc": "Moderate Non-IID + Moderate FedProx", "alpha": 0.5, "mu": 0.1, "seed": 42},
    {"phase": 3, "desc": "Moderate Non-IID + Strong FedProx", "alpha": 0.5, "mu": 1.0, "seed": 42},

    # Phase 4: Statistical significance (replications with different seeds)
    # Tests: Consistency of results across random seeds
    {"phase": 4, "desc": "High Non-IID Baseline (seed 43)", "alpha": 0.1, "mu": 0.0, "seed": 43},
    {"phase": 4, "desc": "High Non-IID + Moderate FedProx (seed 43)", "alpha": 0.1, "mu": 0.1, "seed": 43},
    {"phase": 4, "desc": "High Non-IID Baseline (seed 44)", "alpha": 0.1, "mu": 0.0, "seed": 44},
    {"phase": 4, "desc": "High Non-IID + Moderate FedProx (seed 44)", "alpha": 0.1, "mu": 0.1, "seed": 44},
]

# Fixed experiment parameters
DATASET = "edge-iiotset-full"
DATASET_PATH = "data/edge-iiotset/edge_iiotset_fedprox_validation.csv"
NUM_CLIENTS = 5
NUM_ROUNDS = 10
AGGREGATION = "fedavg"  # FedProx is enabled via fedprox_mu parameter


def create_experiment_config(exp):
    """Create ExperimentConfig from experiment specification."""
    return ExperimentConfig(
        dataset=DATASET,
        data_path=DATASET_PATH,
        aggregation=AGGREGATION,
        alpha=exp["alpha"],
        adversary_fraction=0.0,
        dp_enabled=False,
        dp_noise_multiplier=0.0,
        personalization_epochs=0,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        seed=exp["seed"],
        fedprox_mu=exp["mu"]
    )


def print_experiment_header(idx, total, exp):
    """Print formatted experiment header."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {idx+1}/{total}")
    print(f"Phase {exp['phase']}: {exp['desc']}")
    print(f"Parameters: alpha={exp['alpha']}, mu={exp['mu']}, seed={exp['seed']}")
    print(f"{'='*80}\n")


def print_phase_summary(phase_results):
    """Print summary of completed phase."""
    print(f"\n{'-'*80}")
    print(f"PHASE {phase_results[0]['phase']} COMPLETE")
    print(f"{'-'*80}")
    for result in phase_results:
        status = "SUCCESS" if result['status'] == 'success' else "FAILED"
        duration = result.get('duration', 0) / 60
        print(f"  [{status}] {result['desc']}: {duration:.1f}m")
    print(f"{'-'*80}\n")


def main():
    print(f"\n{'='*80}")
    print("FEDPROX IMPLEMENTATION VALIDATION")
    print(f"{'='*80}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total Experiments: {len(EXPERIMENTS)}")
    print(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")
    print(f"{'='*80}\n")

    print("EXPERIMENTAL DESIGN:")
    print("  Phase 1 (4 exps): IID data validation")
    print("  Phase 2 (4 exps): High non-IID validation (primary)")
    print("  Phase 3 (4 exps): Moderate non-IID validation")
    print("  Phase 4 (4 exps): Statistical significance replications")
    print()
    print("Starting experiments...")
    print()

    base_dir = Path.cwd()
    total = len(EXPERIMENTS)
    results = []
    phase_results = []
    current_phase = 1

    for idx, exp in enumerate(EXPERIMENTS):
        print_experiment_header(idx, total, exp)

        # Track phase transitions
        if exp["phase"] != current_phase:
            print_phase_summary(phase_results)
            phase_results = []
            current_phase = exp["phase"]

        config = create_experiment_config(exp)

        # Run experiment sequentially (max_workers=1 equivalent)
        start_time = time.time()
        result = run_experiment_with_state(
            config,
            base_dir,
            port_offset=0,
            worker_id=0,
            dataset_type="full",
            max_retries=2
        )
        duration = time.time() - start_time

        result["phase"] = exp["phase"]
        result["desc"] = exp["desc"]
        result["alpha"] = exp["alpha"]
        result["mu"] = exp["mu"]
        result["seed"] = exp["seed"]
        result["duration"] = duration

        results.append(result)
        phase_results.append(result)

        # Print result
        if result["status"] == "success":
            print(f"\n[SUCCESS] Completed in {duration/60:.1f} minutes")
        elif result["status"] == "skipped":
            print(f"\n[SKIPPED] {result['message']}")
        else:
            print(f"\n[FAILED] {result.get('error', 'Unknown error')}")

        print_resource_status()

        # Small delay between experiments
        if idx < total - 1:
            print("\nWaiting 10 seconds before next experiment...")
            time.sleep(10)

    # Print final phase summary
    if phase_results:
        print_phase_summary(phase_results)

    # Print overall summary
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] != "success" and r["status"] != "skipped")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")

    print(f"Results: {success_count} success, {failed_count} failed, {skipped_count} skipped")
    print(f"{'='*80}\n")

    # Print results by phase
    for phase in [1, 2, 3, 4]:
        phase_results = [r for r in results if r["phase"] == phase]
        if phase_results:
            print(f"Phase {phase} Results:")
            for r in phase_results:
                status_symbol = "✓" if r["status"] == "success" else "✗"
                print(f"  {status_symbol} {r['desc']}")
            print()

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(130)
