#!/usr/bin/env python3
import argparse
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small FL experiment (server + clients)")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--preset", type=str, default="smoke")
    parser.add_argument("--partition_strategy", type=str, default="iid", choices=["iid","dirichlet","protocol"]) 
    parser.add_argument("--adversary_mode", type=str, default="none", choices=["none","label_flip","grad_ascent"]) 
    parser.add_argument("--leakage_safe", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.environ.setdefault("SEED", str(args.seed))
    os.environ.setdefault("D2_EXTENDED_METRICS", "1")

    run_dir = Path("runs") / args.preset
    run_dir.mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    server_cmd = [
        sys.executable, "server.py",
        "--rounds", str(args.rounds),
        "--aggregation", "fedavg",
        "--server_address", f"127.0.0.1:{port}",
        "--logdir", str(run_dir),
    ]
    server = subprocess.Popen(server_cmd, cwd=str(Path(__file__).resolve().parents[1]))
    time.sleep(1.5)

    procs = []
    for i in range(args.clients):
        adv = args.adversary_mode if (args.adversary_mode != "none" and i == 0) else "none"
        client_cmd = [
            sys.executable, "client.py",
            "--server_address", f"127.0.0.1:{port}",
            "--dataset", "synthetic",
            "--samples", "1000",
            "--features", "20",
            "--partition_strategy", args.partition_strategy,
            "--num_clients", str(args.clients),
            "--alpha", str(args.alpha),
            "--client_id", str(i),
            "--logdir", str(run_dir),
            "--seed", str(args.seed),
            "--adversary_mode", adv,
        ]
        proc = subprocess.Popen(client_cmd, cwd=str(Path(__file__).resolve().parents[1]))
        procs.append(proc)

    exit_code = 0
    try:
        for p in procs:
            p.wait()
    finally:
        server.wait(timeout=30)

    # Generate plots
    plot_cmd = [sys.executable, "scripts/plot_metrics.py", "--run_dir", str(run_dir), "--output_dir", str(run_dir)]
    subprocess.run(plot_cmd, cwd=str(Path(__file__).resolve().parents[1]), check=False)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


