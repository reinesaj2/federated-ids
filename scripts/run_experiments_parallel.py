#!/usr/bin/env python3
"""
Parallel experiment launcher with resume capability.

Runs multiple federated learning experiments concurrently using ProcessPoolExecutor.
Skips already-completed experiments based on existence of metrics.csv.
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path to import comparative_analysis
sys.path.insert(0, str(Path(__file__).parent))

from comparative_analysis import (
    ComparisonMatrix,
    run_federated_experiment,
)


def is_experiment_complete(run_dir: Path) -> bool:
    """Check if experiment already completed successfully."""
    return (run_dir / "metrics.csv").exists() and (run_dir / "config.json").exists()


def run_experiment_safe(config, base_dir, port_offset=0):
    """Wrapper to run experiment with exception handling.

    Args:
        config: Experiment configuration
        base_dir: Project base directory
        port_offset: Port offset to avoid conflicts (each experiment uses base + offset * 100)
    """
    try:
        port_start = 8080 + (port_offset * 100)
        result = run_federated_experiment(config, base_dir, port_start=port_start)
        return {"status": "success", "preset": config.to_preset_name(), "result": result}
    except Exception as e:
        return {"status": "error", "preset": config.to_preset_name(), "error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument(
        "--dimension",
        type=str,
        default="attack",
        choices=["aggregation", "heterogeneity", "heterogeneity_fedprox", "attack", "privacy", "personalization"],
        help="Experiment dimension",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (experiments to run concurrently)",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        default=True,
        help="Skip experiments that already have metrics.csv",
    )

    args = parser.parse_args()

    base_dir = Path.cwd()

    # Generate configs
    matrix = ComparisonMatrix()
    all_configs = matrix.generate_configs(filter_dimension=args.dimension)

    # Filter out completed experiments
    pending_configs = []
    completed_count = 0

    for config in all_configs:
        run_dir = base_dir / "runs" / config.to_preset_name()
        if args.skip_completed and is_experiment_complete(run_dir):
            completed_count += 1
            print(f"[SKIP] {config.to_preset_name()} (already completed)")
        else:
            pending_configs.append(config)

    total = len(all_configs)
    pending = len(pending_configs)

    print(f"\n{'='*60}")
    print(f"Total experiments: {total}")
    print(f"Already completed: {completed_count}")
    print(f"Pending: {pending}")
    print(f"Parallel workers: {args.workers}")
    print(f"{'='*60}\n")

    if pending == 0:
        print("All experiments already completed!")
        return

    # Run experiments in parallel
    completed_tasks = 0
    failed_tasks = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks with unique port offsets to avoid conflicts
        futures = {
            executor.submit(run_experiment_safe, config, base_dir, idx): config
            for idx, config in enumerate(pending_configs)
        }

        # Process as they complete
        for future in as_completed(futures):
            config = futures[future]
            result = future.result()

            completed_tasks += 1
            progress_pct = (completed_count + completed_tasks) / total * 100

            if result["status"] == "success":
                print(
                    f"[{completed_count + completed_tasks}/{total}] "
                    f"({progress_pct:.1f}%) [PASS] {result['preset']}"
                )
            else:
                failed_tasks += 1
                print(
                    f"[{completed_count + completed_tasks}/{total}] "
                    f"({progress_pct:.1f}%) [FAIL] {result['preset']}: {result['error']}"
                )

    # Summary
    print(f"\n{'='*60}")
    print(f"Experiments completed: {completed_tasks}")
    print(f"Experiments failed: {failed_tasks}")
    print(f"Total now complete: {completed_count + completed_tasks}/{total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
