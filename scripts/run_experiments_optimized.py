#!/usr/bin/env python3
"""
Optimized parallel experiment launcher for full dataset experiments.

Key improvements over original:
- Memory-aware worker calculation based on available RAM
- Adaptive timeouts based on dataset size
- Experiment state tracking with JSON files
- Resource monitoring and health checks
- Better port management with wider spacing
- Failed experiment retry capability
- ETA calculation and progress tracking
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from comparative_analysis import (
    ComparisonMatrix,
    ExperimentConfig,
    run_federated_experiment,
)


@dataclass
class ExperimentState:
    """Track state of each experiment for recovery."""

    preset: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    duration_sec: float | None = None
    retry_count: int = 0
    worker_id: int | None = None


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback if psutil not available
        import subprocess

        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024**3)
        return 8.0  # Conservative fallback


def calculate_optimal_workers(dataset_type: str = "full") -> int:
    """
    Calculate optimal worker count based on available RAM and dataset size.

    Memory requirements (conservative estimates):
    - Full UNSW (82k samples): ~4.5GB per experiment
    - Full CIC (10k samples): ~2GB per experiment
    - Sampled UNSW (8k): ~1.5GB per experiment
    """
    available_gb = get_available_memory_gb()

    # Reserve 4GB for OS and other processes
    usable_gb = max(1, available_gb - 4)

    if dataset_type == "full":
        gb_per_experiment = 4.5  # Full UNSW dataset
    elif dataset_type == "cic":
        gb_per_experiment = 2.0  # Full CIC dataset
    else:
        gb_per_experiment = 1.5  # Sampled datasets

    optimal = int(usable_gb / gb_per_experiment)

    # Apply sensible limits
    optimal = max(1, optimal)  # At least 1 worker
    optimal = min(optimal, 4)  # Cap at 4 to avoid thrashing

    print(f"System RAM: {available_gb:.1f}GB available")
    print(f"Calculated optimal workers: {optimal} (using {gb_per_experiment:.1f}GB per experiment)")

    return optimal


def estimate_experiment_timeout(config: ExperimentConfig, dataset_type: str = "full") -> int:
    """
    Estimate timeout based on dataset size and configuration.

    Formula:
    - Base overhead: 60 seconds
    - Per round with full UNSW (82k): ~180 seconds
    - Per round with sampled (8k): ~20 seconds
    - Add buffer for setup/teardown
    """
    if dataset_type == "full":
        if config.dataset == "unsw":
            # Full UNSW: 82k samples
            per_round = 180  # 3 minutes per round
        elif config.dataset == "cic":
            # Full CIC: 10k samples
            per_round = 60  # 1 minute per round
        else:
            per_round = 180  # Conservative default
    else:
        # Sampled datasets
        per_round = 20  # 20 seconds per round

    base_timeout = 60  # Base overhead
    rounds_timeout = per_round * config.num_rounds
    clients_overhead = config.num_clients * 5  # 5 sec per client

    # Add personalization overhead if enabled
    if config.personalization_epochs > 0:
        rounds_timeout += config.personalization_epochs * per_round

    # Total with 50% buffer
    total = int((base_timeout + rounds_timeout + clients_overhead) * 1.5)

    # Minimum 10 minutes, maximum 2 hours
    return max(600, min(7200, total))


def load_experiment_state(run_dir: Path) -> ExperimentState | None:
    """Load experiment state from JSON file."""
    state_file = run_dir / ".experiment_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                data = json.load(f)
                return ExperimentState(**data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def save_experiment_state(state: ExperimentState, run_dir: Path):
    """Save experiment state to JSON file for recovery."""
    state_file = run_dir / ".experiment_state.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(asdict(state), f, indent=2)


def is_experiment_complete(run_dir: Path) -> bool:
    """Check if experiment completed successfully."""
    # Check state file first
    state = load_experiment_state(run_dir)
    if state and state.status == "completed":
        return True

    # Fallback to checking for metrics file
    return (run_dir / "metrics.csv").exists() and (run_dir / "config.json").exists()


def is_experiment_stuck(run_dir: Path, timeout_minutes: int = 90) -> bool:
    """Detect experiments that are stuck or hanging."""
    state = load_experiment_state(run_dir)
    if state and state.status == "running":
        if state.start_time:
            elapsed = time.time() - state.start_time
            if elapsed > timeout_minutes * 60:
                return True
    return False


def run_experiment_with_state(
    config: ExperimentConfig, base_dir: Path, port_offset: int = 0, worker_id: int = 0, dataset_type: str = "full", max_retries: int = 2
):
    """
    Run experiment with state tracking and retry capability.
    """
    preset = config.to_preset_name()
    run_dir = base_dir / "runs" / preset

    # Check if already complete
    if is_experiment_complete(run_dir):
        return {"status": "skipped", "preset": preset, "message": "Already completed"}

    # Load existing state or create new
    state = load_experiment_state(run_dir)
    if state is None:
        state = ExperimentState(preset=preset, status="pending", worker_id=worker_id)

    # Check retry limit
    if state.retry_count >= max_retries:
        return {"status": "failed", "preset": preset, "error": f"Max retries ({max_retries}) exceeded"}

    # Update state to running
    state.status = "running"
    state.start_time = time.time()
    state.worker_id = worker_id
    save_experiment_state(state, run_dir)

    try:
        # Use wider port spacing to avoid conflicts
        port_start = 8080 + (port_offset * 200)  # Wider spacing

        # Calculate adaptive timeout
        timeout = estimate_experiment_timeout(config, dataset_type)

        # Run the experiment with timeout
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_context(seconds):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Experiment timeout after {seconds} seconds")

            # Set the timeout handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        # Run with timeout
        start = time.time()
        try:
            with timeout_context(timeout):
                result = run_federated_experiment(config, base_dir, port_start=port_start)
        except TimeoutError as err:
            raise TimeoutError(f"Experiment {preset} timed out after {timeout}s") from err

        duration = time.time() - start

        # Update state to completed
        state.status = "completed"
        state.end_time = time.time()
        state.duration_sec = duration
        state.error = None
        save_experiment_state(state, run_dir)

        return {"status": "success", "preset": preset, "result": result, "duration": duration, "worker": worker_id}

    except Exception as e:
        # Update state to failed
        state.status = "failed"
        state.end_time = time.time()
        state.error = str(e)
        state.retry_count += 1
        save_experiment_state(state, run_dir)

        return {"status": "error", "preset": preset, "error": str(e), "retry_count": state.retry_count, "worker": worker_id}


def print_resource_status():
    """Print current resource usage."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        print(f"\nResources: CPU {cpu:.1f}% | RAM {mem.percent:.1f}% ({mem.available/(1024**3):.1f}GB free)")
    except Exception:
        pass  # Silent fail if psutil not available


def main():
    parser = argparse.ArgumentParser(
        description="Optimized parallel experiment runner for full datasets", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="all",
        choices=["all", "aggregation", "heterogeneity", "heterogeneity_fedprox", "attack", "privacy", "personalization"],
        help="Experiment dimension(s) to run",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (auto-calculated if not specified)")
    parser.add_argument(
        "--dataset-type", type=str, default="full", choices=["full", "sample", "cic"], help="Dataset type (affects memory calculation)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="edge-iiotset-full",
        choices=["unsw", "cic", "edge-iiotset-quick", "edge-iiotset-nightly", "edge-iiotset-full"],
        help="Dataset to use for experiments",
    )
    parser.add_argument("--skip-completed", action="store_true", default=True, help="Skip already completed experiments")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retries for failed experiments")
    parser.add_argument("--monitor-interval", type=int, default=30, help="Resource monitoring interval in seconds")

    args = parser.parse_args()
    base_dir = Path.cwd()

    # Auto-calculate workers if not specified
    if args.workers is None:
        args.workers = calculate_optimal_workers(args.dataset_type)

    print(f"\n{'='*70}")
    print("OPTIMIZED EXPERIMENT RUNNER FOR FULL DATASETS")
    print(f"{'='*70}")

    # Generate configs
    dataset_paths = {
        "unsw": "data/unsw/UNSW_NB15_training-set.csv",
        "cic": "data/cic/cic_ids2017_multiclass.csv",
        "edge-iiotset-quick": "data/edge-iiotset/edge_iiotset_quick.csv",
        "edge-iiotset-nightly": "data/edge-iiotset/edge_iiotset_nightly.csv",
        "edge-iiotset-full": "data/edge-iiotset/edge_iiotset_full.csv",
    }
    data_path = dataset_paths[args.dataset]
    matrix = ComparisonMatrix(dataset=args.dataset, data_path=data_path)

    if args.dimension == "all":
        dimensions = ["aggregation", "heterogeneity", "attack", "privacy", "personalization"]
        all_configs = []
        for dim in dimensions:
            all_configs.extend(matrix.generate_configs(filter_dimension=dim))
    else:
        all_configs = matrix.generate_configs(filter_dimension=args.dimension)

    # Filter completed and stuck experiments
    pending_configs = []
    completed_count = 0
    stuck_count = 0

    for config in all_configs:
        run_dir = base_dir / "runs" / config.to_preset_name()

        if args.skip_completed and is_experiment_complete(run_dir):
            completed_count += 1
            print(f"[COMPLETE] {config.to_preset_name()}")
        elif is_experiment_stuck(run_dir):
            stuck_count += 1
            print(f"??  [STUCK] {config.to_preset_name()} - will retry")
            # Reset state for retry
            state = load_experiment_state(run_dir)
            if state:
                state.status = "pending"
                save_experiment_state(state, run_dir)
            pending_configs.append(config)
        else:
            pending_configs.append(config)

    total = len(all_configs)
    pending = len(pending_configs)

    # Calculate ETA
    if args.dataset_type == "full":
        avg_minutes_per_exp = 60  # Full dataset
    else:
        avg_minutes_per_exp = 10  # Sampled

    eta_hours = (pending * avg_minutes_per_exp) / (60 * args.workers)

    print("\nEXPERIMENT SUMMARY:")
    print(f"  Total configurations: {total}")
    print(f"  Already completed: {completed_count}")
    print(f"  Stuck/retry: {stuck_count}")
    print(f"  Pending: {pending}")
    print(f"  Parallel workers: {args.workers}")
    print(f"  Estimated time: {eta_hours:.1f} hours")
    print(f"  Dataset type: {args.dataset_type}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*70}\n")

    if pending == 0:
        print("All experiments already completed!")
        return

    # Run experiments with progress tracking
    completed_tasks = 0
    failed_tasks = 0
    start_time = time.time()
    last_monitor = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks with worker IDs for tracking
        futures = {
            executor.submit(
                run_experiment_with_state,
                config,
                base_dir,
                idx % 50,  # Port offset (50 ports per range)
                idx % args.workers,  # Worker ID
                args.dataset_type,
                args.max_retries,
            ): config
            for idx, config in enumerate(pending_configs)
        }

        # Process as they complete
        for future in as_completed(futures):
            config = futures[future]
            result = future.result()

            completed_tasks += 1
            progress_pct = (completed_count + completed_tasks) / total * 100
            elapsed = time.time() - start_time

            # Calculate ETA
            if completed_tasks > 0:
                avg_time = elapsed / completed_tasks
                remaining = pending - completed_tasks
                eta_sec = remaining * avg_time
                eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.0f}m"
            else:
                eta_str = "calculating..."

            # Print result
            if result["status"] == "success":
                duration = result.get("duration", 0)
                print(
                    f"[{completed_count + completed_tasks}/{total}] "
                    f"({progress_pct:.1f}%) [COMPLETE] {result['preset']} "
                    f"[{duration/60:.1f}m on worker {result.get('worker', '?')}] "
                    f"ETA: {eta_str}"
                )
            elif result["status"] == "skipped":
                print(f"[SKIP] {result['preset']}: {result['message']}")
            else:
                failed_tasks += 1
                retry_info = f"retry {result.get('retry_count', 0)}/{args.max_retries}"
                print(
                    f"[{completed_count + completed_tasks}/{total}] "
                    f"({progress_pct:.1f}%) FAILED {result['preset']}: "
                    f"{result['error'][:50]}... [{retry_info}]"
                )

            # Periodic resource monitoring
            if time.time() - last_monitor > args.monitor_interval:
                print_resource_status()
                last_monitor = time.time()

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {completed_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Avg time per experiment: {total_time/max(1,completed_tasks)/60:.1f} minutes")
    print(f"Total now complete: {completed_count + completed_tasks}/{total}")

    if failed_tasks > 0:
        print(f"\n{failed_tasks} experiments failed. Run again to retry.")
    else:
        print("\nAll experiments completed successfully!")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
