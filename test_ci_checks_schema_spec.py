"""
Schema validation tests for scripts/ci_checks.py to prevent regressions.
Tests artifact validation functionality and CLI interface.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def create_valid_run_directory(
    base_dir: Path, run_name: str, clients: int = 2, rounds: int = 3
) -> Path:
    """Create a valid FL run directory with all required artifacts."""
    run_dir = base_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create valid server metrics
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("round,accuracy,loss\n")
        for r in range(rounds):
            f.write(f"{r},{0.5 + r * 0.1},{1.0 - r * 0.1}\n")

    # Create valid client metrics
    for client_id in range(clients):
        client_metrics_path = run_dir / f"client_{client_id}_metrics.csv"
        with open(client_metrics_path, "w") as f:
            f.write("round,local_accuracy,local_loss\n")
            for r in range(rounds):
                f.write(f"{r},{0.4 + r * 0.12},{1.2 - r * 0.08}\n")

    # Create plot files
    (run_dir / "client_metrics_plot.png").touch()
    (run_dir / "server_metrics_plot.png").touch()

    return run_dir


def test_ci_checks_script_importable():
    """Test that scripts/ci_checks.py can be imported without errors."""
    scripts_dir = Path(__file__).parent / "scripts"
    ci_checks_path = scripts_dir / "ci_checks.py"
    assert (
        ci_checks_path.exists()
    ), f"scripts/ci_checks.py not found at {ci_checks_path}"

    # Test that the script can be imported as a module
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); import ci_checks",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to import ci_checks: {result.stderr}"


def test_ci_checks_cli_help():
    """Test that ci_checks.py shows help without errors."""
    result = subprocess.run(
        [sys.executable, "scripts/ci_checks.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "validate" in result.stdout.lower()
    assert "--runs_dir" in result.stdout


def test_ci_checks_valid_artifacts():
    """Test that ci_checks.py validates correct artifacts successfully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a valid run directory
        create_valid_run_directory(temp_path, "test_valid")

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        assert "validation passed" in result.stdout.lower()


def test_ci_checks_missing_runs_directory():
    """Test that ci_checks.py fails when runs directory doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--runs_dir",
                str(Path(temp_dir) / "nonexistent"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail when runs directory doesn't exist"
        assert "does not exist" in result.stderr.lower()


def test_ci_checks_missing_server_metrics():
    """Test that ci_checks.py fails when server metrics.csv is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_dir = create_valid_run_directory(temp_path, "test_missing_server")

        # Remove server metrics file
        (run_dir / "metrics.csv").unlink()

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail when server metrics are missing"
        assert "metrics.csv" in result.stderr.lower()


def test_ci_checks_invalid_server_schema():
    """Test that ci_checks.py fails when server metrics has wrong schema."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_dir = create_valid_run_directory(temp_path, "test_invalid_schema")

        # Create invalid server metrics (missing required 'round' column)
        metrics_path = run_dir / "metrics.csv"
        with open(metrics_path, "w") as f:
            f.write("accuracy,loss\n")  # Missing 'round' column
            f.write("0.5,0.3\n")

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail when schema is invalid"
        assert "missing required columns" in result.stderr.lower()


def test_ci_checks_missing_client_metrics():
    """Test that ci_checks.py fails when client metrics are missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_dir = create_valid_run_directory(temp_path, "test_missing_clients")

        # Remove all client metrics files
        for client_file in run_dir.glob("client_*_metrics.csv"):
            client_file.unlink()

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail when client metrics are missing"
        assert "no client metrics files" in result.stderr.lower()


def test_ci_checks_missing_plot_files():
    """Test that ci_checks.py fails when plot files are missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_dir = create_valid_run_directory(temp_path, "test_missing_plots")

        # Remove plot files
        (run_dir / "client_metrics_plot.png").unlink()
        (run_dir / "server_metrics_plot.png").unlink()

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail when plot files are missing"
        assert "plot file missing" in result.stderr.lower()


def test_ci_checks_multiple_run_directories():
    """Test that ci_checks.py validates multiple run directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple valid run directories
        create_valid_run_directory(temp_path, "run1")
        create_valid_run_directory(temp_path, "run2")
        create_valid_run_directory(temp_path, "run3")

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Multi-run validation failed: {result.stderr}"
        assert "found 3 run directories" in result.stdout.lower()
        assert "all 3 run directories passed" in result.stdout.lower()


def test_ci_checks_empty_csv_files():
    """Test that ci_checks.py fails when CSV files have headers but no data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_dir = create_valid_run_directory(temp_path, "test_empty_data")

        # Create CSV with headers but no data
        metrics_path = run_dir / "metrics.csv"
        with open(metrics_path, "w") as f:
            f.write("round,accuracy,loss\n")  # Headers only, no data

        script_path = Path("scripts/ci_checks.py").resolve()
        result = subprocess.run(
            [sys.executable, str(script_path), "--runs_dir", str(temp_path / "runs")],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail when CSV has no data rows"
        assert "no data rows" in result.stderr.lower()
