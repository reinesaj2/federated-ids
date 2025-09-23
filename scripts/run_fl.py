#!/usr/bin/env python3
"""
Federated Learning orchestration script with model persistence.
Coordinates FL training runs and generates metrics and model artifacts.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def wait_for_port(host: str, port: int, max_wait: int = 10) -> bool:
    """Wait for a port to become available."""
    import socket

    waited = 0
    while waited < max_wait:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex((host, port))
                if result == 0:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
        waited += 0.5
    return False


def create_run_directory(preset: str) -> Path:
    """Create and return the run directory for this experiment."""
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    run_dir = runs_dir / preset
    run_dir.mkdir(exist_ok=True)

    return run_dir


def start_server(
    run_dir: Path,
    rounds: int,
    aggregation: str,
    server_address: str,
    save_models: str,
    num_features: int,
    num_classes: int,
    logdir: str,
) -> subprocess.Popen:
    """Start the FL server process."""
    cmd = [
        "python",
        "server.py",
        "--rounds",
        str(rounds),
        "--aggregation",
        aggregation,
        "--server_address",
        server_address,
        "--save_models",
        save_models,
        "--num_features",
        str(num_features),
        "--num_classes",
        str(num_classes),
        "--logdir",
        logdir,
    ]

    print(f"Starting server: {' '.join(cmd)}")
    server_log = run_dir / "server.log"

    with open(server_log, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, SEED=str(os.environ.get("SEED", "42"))),
        )

    return process


def start_clients(
    run_dir: Path,
    clients: int,
    server_address: str,
    partition_strategy: str,
    alpha: float,
    adversary_mode: Optional[str],
    leakage_safe: bool,
    save_models: str,
    logdir: str,
) -> List[subprocess.Popen]:
    """Start the FL client processes."""
    processes = []

    for client_id in range(clients):
        cmd = [
            "python",
            "client.py",
            "--server_address",
            server_address,
            "--num_clients",
            str(clients),
            "--client_id",
            str(client_id),
            "--partition_strategy",
            partition_strategy,
            "--alpha",
            str(alpha),
            "--samples",
            "1000",
            "--features",
            "20",
            "--save_models",
            save_models,
            "--logdir",
            logdir,
        ]

        if adversary_mode:
            cmd.extend(["--adversary_mode", adversary_mode])

        if leakage_safe:
            cmd.append("--leakage_safe")

        print(f"Starting client {client_id}: {' '.join(cmd)}")
        client_log = run_dir / f"client_{client_id}.log"

        with open(client_log, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, SEED=str(os.environ.get("SEED", "42"))),
            )

        processes.append(process)

    return processes


def wait_for_completion(
    server_process: subprocess.Popen,
    client_processes: List[subprocess.Popen],
    timeout: int = 60,
) -> bool:
    """Wait for FL training to complete with timeout."""
    start_time = time.time()

    while True:
        # Check if timeout exceeded
        if time.time() - start_time > timeout:
            print(f"Timeout after {timeout}s, killing processes")
            server_process.terminate()
            for client in client_processes:
                client.terminate()
            return False

        # Check if all clients finished
        all_clients_done = all(client.poll() is not None for client in client_processes)

        if all_clients_done:
            # Give server a moment to finish
            time.sleep(2)
            server_process.terminate()
            return True

        time.sleep(1)


def run_fl_experiment(
    preset: str,
    clients: int,
    rounds: int,
    alpha: float,
    partition_strategy: str,
    aggregation: str = "fedavg",
    adversary_mode: Optional[str] = None,
    leakage_safe: bool = False,
    save_models: str = "final",
    server_port: int = 8080,
) -> bool:
    """Run a complete FL experiment with model persistence."""
    # Create output directory
    run_dir = create_run_directory(preset)
    logdir = str(run_dir)

    server_address = f"127.0.0.1:{server_port}"
    host, port = server_address.split(":")
    port = int(port)

    print(f"Running FL experiment: {preset}")
    print(f"  Clients: {clients}, Rounds: {rounds}")
    print(f"  Alpha: {alpha}, Strategy: {partition_strategy}")
    print(f"  Aggregation: {aggregation}, Save models: {save_models}")
    if adversary_mode:
        print(f"  Adversary mode: {adversary_mode}")
    print(f"  Output directory: {run_dir}")

    try:
        # Start server
        server_process = start_server(
            run_dir=run_dir,
            rounds=rounds,
            aggregation=aggregation,
            server_address=server_address,
            save_models=save_models,
            num_features=20,  # Fixed for synthetic data
            num_classes=2,  # Fixed for binary classification
            logdir=logdir,
        )

        # Wait for server to be ready
        if not wait_for_port(host, port, max_wait=10):
            print("Server failed to start in time")
            server_process.terminate()
            return False

        print("Server is ready, starting clients")

        # Start clients
        client_processes = start_clients(
            run_dir=run_dir,
            clients=clients,
            server_address=server_address,
            partition_strategy=partition_strategy,
            alpha=alpha,
            adversary_mode=adversary_mode,
            leakage_safe=leakage_safe,
            save_models=save_models,
            logdir=logdir,
        )

        # Wait for completion
        success = wait_for_completion(server_process, client_processes, timeout=120)

        if success:
            print("✓ FL experiment completed successfully")

            # Check for expected artifacts
            expected_files = [
                run_dir / "metrics.csv",
            ]

            if save_models == "final":
                expected_files.append(run_dir / "final_global_model.pth")
            elif save_models == "all":
                expected_files.extend(
                    [
                        run_dir / f"global_model_round_{rounds}.pth",
                        run_dir / "final_global_model.pth",
                    ]
                )

            missing_files = [f for f in expected_files if not f.exists()]
            if missing_files:
                print(f"Warning: Missing expected files: {missing_files}")
                return False

            return True
        else:
            print("✗ FL experiment failed or timed out")
            return False

    except Exception as e:
        print(f"Error running FL experiment: {e}")
        return False


def main() -> None:
    """Main federated learning orchestration entry point."""
    parser = argparse.ArgumentParser(
        description="Run federated learning experiments with model persistence"
    )

    # Required arguments from CI
    parser.add_argument(
        "--clients", type=int, required=True, help="Number of federated clients"
    )
    parser.add_argument(
        "--rounds", type=int, required=True, help="Number of training rounds"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Dirichlet alpha parameter for data partitioning",
    )
    parser.add_argument(
        "--preset", type=str, required=True, help="Experiment preset name"
    )
    parser.add_argument(
        "--partition_strategy",
        type=str,
        choices=["iid", "dirichlet"],
        required=True,
        help="Data partitioning strategy",
    )

    # Optional arguments
    parser.add_argument(
        "--aggregation",
        type=str,
        default="fedavg",
        choices=["fedavg", "median", "krum", "bulyan"],
        help="Aggregation algorithm",
    )
    parser.add_argument(
        "--adversary_mode",
        type=str,
        choices=["label_flip", "grad_ascent"],
        help="Adversarial attack mode",
    )
    parser.add_argument(
        "--leakage_safe", action="store_true", help="Enable leakage-safe mode"
    )
    parser.add_argument(
        "--save_models",
        type=str,
        default="final",
        choices=["none", "final", "all"],
        help="Model saving strategy",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8080,
        help="Server port number",
    )

    args = parser.parse_args()

    success = run_fl_experiment(
        preset=args.preset,
        clients=args.clients,
        rounds=args.rounds,
        alpha=args.alpha,
        partition_strategy=args.partition_strategy,
        aggregation=args.aggregation,
        adversary_mode=args.adversary_mode,
        leakage_safe=args.leakage_safe,
        save_models=args.save_models,
        server_port=args.server_port,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
