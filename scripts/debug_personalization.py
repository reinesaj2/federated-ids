#!/usr/bin/env python3
"""
Diagnostic script for investigating personalization zero-gain results.

Runs controlled experiments with debug logging to identify why personalization
shows negligible improvements on real IDS datasets.

Usage:
    python scripts/debug_personalization.py --dataset unsw \\
        --data_path data/unsw/unsw_nb15_sample.csv
"""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


def summarize_client_metrics(
    row: dict,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract global/personalized F1 and gain from a metrics CSV row."""

    def to_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        value_str = str(value).strip()
        if value_str == "" or value_str.lower() in {"none", "nan"}:
            return None
        try:
            return float(value_str)
        except ValueError:
            return None

    global_f1 = to_float(row.get("macro_f1_global"))
    pers_f1 = to_float(row.get("macro_f1_personalized"))
    gain = to_float(row.get("personalization_gain"))

    # Fall back to post-evaluation metrics when personalization columns are empty
    if global_f1 is None:
        global_f1 = to_float(row.get("macro_f1_after")) or to_float(
            row.get("macro_f1_before")
        )
    if pers_f1 is None:
        pers_f1 = to_float(row.get("macro_f1_after")) or global_f1

    if gain is None and global_f1 is not None and pers_f1 is not None:
        gain = pers_f1 - global_f1

    return global_f1, pers_f1, gain


def run_experiment(
    dataset: str,
    data_path: str,
    num_clients: int,
    num_rounds: int,
    personalization_epochs: int,
    alpha: float,
    lr: float,
    port: int,
) -> None:
    """Run single experiment with debug logging enabled."""
    print(f"\n{'=' * 80}")
    print(
        f"Experiment: dataset={dataset}, alpha={alpha}, "
        f"pers_epochs={personalization_epochs}, lr={lr}"
    )
    print(f"{'=' * 80}\n")

    # Set environment variables for debug output
    env = os.environ.copy()
    env["DEBUG_PERSONALIZATION"] = "1"
    env["D2_EXTENDED_METRICS"] = "1"
    env["SEED"] = "42"

    # Clean up logs directory for this experiment only
    exp_name = f"{dataset}_alpha{alpha}_pers{personalization_epochs}_lr{lr}".replace(
        ".", "p"
    )
    logs_dir = Path("logs_debug") / exp_name
    if logs_dir.exists():
        import shutil

        shutil.rmtree(logs_dir)
    logs_dir.mkdir(parents=True)

    # Start server
    server_cmd = [
        "python",
        "server.py",
        "--rounds",
        str(num_rounds),
        "--aggregation",
        "fedavg",
        "--server_address",
        f"127.0.0.1:{port}",
    ]
    print(f"Starting server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Wait for server to start
    time.sleep(3)

    # Start clients
    client_procs = []
    for client_id in range(num_clients):
        client_cmd = [
            "python",
            "client.py",
            "--server_address",
            f"127.0.0.1:{port}",
            "--dataset",
            dataset,
            "--data_path",
            data_path,
            "--client_id",
            str(client_id),
            "--num_clients",
            str(num_clients),
            "--local_epochs",
            "2",
            "--lr",
            str(lr),
            "--personalization_epochs",
            str(personalization_epochs),
            "--partition_strategy",
            "dirichlet",
            "--alpha",
            str(alpha),
            "--logdir",
            str(logs_dir),
        ]
        print(
            f"Starting client {client_id}: "
            f"personalization_epochs={personalization_epochs}"
        )
        proc = subprocess.Popen(
            client_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        client_procs.append(proc)

    # Wait for all clients
    for idx, proc in enumerate(client_procs):
        stdout, _ = proc.communicate()
        print(f"\n--- Client {idx} output ---")
        print(stdout.decode("utf-8"))

    # Wait for server
    stdout, _ = server_proc.communicate()
    print("\n--- Server output ---")
    print(stdout.decode("utf-8"))

    # Analyze results
    print(f"\n{'=' * 80}")
    print("Results Summary")
    print(f"{'=' * 80}\n")
    for client_id in range(num_clients):
        csv_path = logs_dir / f"client_{client_id}_metrics.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
                if rows:
                    last_row = rows[-1]
                    global_f1, pers_f1, gain = summarize_client_metrics(last_row)
                    print(
                        f"Client {client_id}: "
                        f"global_F1={global_f1}, "
                        f"pers_F1={pers_f1}, "
                        f"gain={gain}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug personalization zero-gain")
    parser.add_argument("--dataset", type=str, default="unsw", choices=["unsw", "cic"])
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/unsw/unsw_nb15_sample.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 80)
    print("Personalization Zero-Gain Diagnostic")
    print("=" * 80)
    print("\nThis script runs controlled experiments to investigate why")
    print("personalization shows zero or negligible gains on real IDS data.")
    print("\nTests will vary:")
    print("  - Personalization epochs: 0, 3, 5")
    print("  - Learning rates: 0.001, 0.01")
    print("  - Data heterogeneity (alpha): 0.1, 1.0")
    print("\n" + "=" * 80 + "\n")

    base_port = 8100
    experiment_id = 0

    # Test 1: Baseline (no personalization)
    print("\n### Test 1: Baseline (no personalization) ###")
    run_experiment(
        dataset=args.dataset,
        data_path=args.data_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        personalization_epochs=0,
        alpha=0.1,
        lr=0.01,
        port=base_port + experiment_id,
    )
    experiment_id += 1
    time.sleep(2)

    # Test 2: Moderate personalization, non-IID data
    print("\n### Test 2: 3 personalization epochs, non-IID (alpha=0.1) ###")
    run_experiment(
        dataset=args.dataset,
        data_path=args.data_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        personalization_epochs=3,
        alpha=0.1,
        lr=0.01,
        port=base_port + experiment_id,
    )
    experiment_id += 1
    time.sleep(2)

    # Test 3: More personalization epochs
    print("\n### Test 3: 5 personalization epochs, non-IID (alpha=0.1) ###")
    run_experiment(
        dataset=args.dataset,
        data_path=args.data_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        personalization_epochs=5,
        alpha=0.1,
        lr=0.01,
        port=base_port + experiment_id,
    )
    experiment_id += 1
    time.sleep(2)

    # Test 4: Higher learning rate
    print("\n### Test 4: 5 epochs, higher LR (0.001 -> 0.01) ###")
    run_experiment(
        dataset=args.dataset,
        data_path=args.data_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        personalization_epochs=5,
        alpha=0.1,
        lr=0.001,
        port=base_port + experiment_id,
    )
    experiment_id += 1
    time.sleep(2)

    # Test 5: IID data (should show minimal gain)
    print("\n### Test 5: 5 epochs, IID data (alpha=1.0) - expect low gain ###")
    run_experiment(
        dataset=args.dataset,
        data_path=args.data_path,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        personalization_epochs=5,
        alpha=1.0,
        lr=0.01,
        port=base_port + experiment_id,
    )

    print("\n" + "=" * 80)
    print("All diagnostic tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
