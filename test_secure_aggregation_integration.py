import csv
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _find_free_port() -> int:
    for _ in range(64):
        candidate = random.randint(20000, 60000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("Unable to locate an available TCP port for integration test")


def _wait_for_server(port: int, proc: subprocess.Popen[str], timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"Server exited early.\nstdout:\n{stdout}\nstderr:\n{stderr}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.1)
    raise TimeoutError(f"Server did not open port {port} within {timeout} seconds")


def _read_last_row(csv_path: Path) -> dict[str, str]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise AssertionError(f"No rows found in {csv_path}")
        return rows[-1]


@pytest.mark.integration
def test_secure_aggregation_round_completes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent
    try:
        port = _find_free_port()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    logdir = tmp_path / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    server_cmd = [
        sys.executable,
        "server.py",
        "--rounds",
        "1",
        "--secure_aggregation",
        "--min_fit_clients",
        "2",
        "--min_eval_clients",
        "0",
        "--fraction_eval",
        "0.0",
        "--server_address",
        f"127.0.0.1:{port}",
        "--logdir",
        str(logdir),
    ]

    env = os.environ.copy()
    env.setdefault("SEED", "123")

    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_server(port, server_proc)

        client_processes: list[subprocess.Popen[str]] = []
        client_env = env.copy()
        client_env["D2_EXTENDED_METRICS"] = "1"

        base_client_cmd = [
            sys.executable,
            "client.py",
            "--server_address",
            f"127.0.0.1:{port}",
            "--secure_aggregation",
            "--dataset",
            "synthetic",
            "--samples",
            "200",
            "--features",
            "4",
            "--num_classes",
            "2",
            "--num_clients",
            "2",
            "--local_epochs",
            "1",
            "--logdir",
            str(logdir),
        ]

        for client_id in (0, 1):
            cmd = base_client_cmd + ["--client_id", str(client_id)]
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=client_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            client_processes.append(proc)

        for proc in client_processes:
            stdout, stderr = proc.communicate(timeout=120)
            assert proc.returncode == 0, f"Client failed\nstdout:\n{stdout}\nstderr:\n{stderr}"

        stdout, stderr = server_proc.communicate(timeout=120)
        assert server_proc.returncode == 0, f"Server failed\nstdout:\n{stdout}\nstderr:\n{stderr}"

    finally:
        if server_proc.poll() is None:
            server_proc.kill()
            server_proc.wait()

    server_metrics = logdir / "metrics.csv"
    assert server_metrics.exists(), "Server metrics CSV missing"
    assert server_metrics.stat().st_size > 0, "Server metrics CSV empty"

    for client_id in (0, 1):
        client_csv = logdir / f"client_{client_id}_metrics.csv"
        assert client_csv.exists(), f"Client {client_id} metrics CSV missing"
        last_row = _read_last_row(client_csv)
        assert last_row.get("secure_aggregation") == "True"
        checksum = last_row.get("secure_aggregation_mask_checksum")
        assert checksum is not None and checksum != "", "Mask checksum should be recorded"
        seed_value = last_row.get("secure_aggregation_seed")
        assert seed_value is not None and seed_value != "", "Secure aggregation seed should be recorded"
