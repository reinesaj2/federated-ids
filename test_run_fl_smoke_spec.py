"""
Smoke tests for scripts/run_fl.py to prevent regressions.
Tests basic import functionality and CLI interface.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_run_fl_script_importable():
    """Test that scripts/run_fl.py can be imported without errors."""
    scripts_dir = Path(__file__).parent / "scripts"

    # Test that the script file exists
    run_fl_path = scripts_dir / "run_fl.py"
    assert run_fl_path.exists(), f"scripts/run_fl.py not found at {run_fl_path}"

    # Test that the script can be imported as a module
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); import run_fl",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to import run_fl: {result.stderr}"


def test_run_fl_cli_help():
    """Test that run_fl.py shows help without errors."""
    result = subprocess.run(
        [sys.executable, "scripts/run_fl.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "federated learning" in result.stdout.lower()
    assert "--clients" in result.stdout
    assert "--rounds" in result.stdout
    assert "--alpha" in result.stdout


def test_run_fl_required_arguments():
    """Test that run_fl.py requires essential arguments."""
    # Test missing required arguments
    result = subprocess.run(
        [sys.executable, "scripts/run_fl.py"], capture_output=True, text=True
    )
    assert result.returncode != 0, "Should fail when required arguments are missing"
    assert "required" in result.stderr.lower()


def test_run_fl_smoke_execution():
    """Test that run_fl.py can execute a minimal smoke test without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory to avoid polluting the real runs/
        original_cwd = Path.cwd()
        try:
            Path(temp_dir).resolve()
            temp_path = Path(temp_dir)

            # Copy script to temp directory for isolated execution
            script_content = (Path("scripts/run_fl.py")).read_text()
            temp_script = temp_path / "run_fl.py"
            temp_script.write_text(script_content)

            result = subprocess.run(
                [
                    sys.executable,
                    str(temp_script),
                    "--clients",
                    "2",
                    "--rounds",
                    "3",
                    "--alpha",
                    "0.1",
                    "--preset",
                    "test_smoke",
                    "--partition_strategy",
                    "iid",
                    "--leakage_safe",
                ],
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Smoke execution failed: {result.stderr}"
            assert "completed successfully" in result.stdout.lower()

            # Verify expected artifacts were created
            runs_dir = temp_path / "runs" / "test_smoke"
            assert runs_dir.exists(), "Run directory not created"

            expected_files = [
                "metrics.csv",
                "client_0_metrics.csv",
                "client_1_metrics.csv",
                "client_metrics_plot.png",
                "server_metrics_plot.png",
            ]

            for expected_file in expected_files:
                file_path = runs_dir / expected_file
                assert (
                    file_path.exists()
                ), f"Expected artifact not found: {expected_file}"

        finally:
            # Restore original working directory
            original_cwd


def test_run_fl_adversarial_mode():
    """Test that run_fl.py accepts adversarial mode arguments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        script_content = (Path("scripts/run_fl.py")).read_text()
        temp_script = temp_path / "run_fl.py"
        temp_script.write_text(script_content)

        result = subprocess.run(
            [
                sys.executable,
                str(temp_script),
                "--clients",
                "2",
                "--rounds",
                "2",
                "--alpha",
                "0.1",
                "--preset",
                "test_adversarial",
                "--partition_strategy",
                "dirichlet",
                "--adversary_mode",
                "label_flip",
                "--leakage_safe",
            ],
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        assert (
            result.returncode == 0
        ), f"Adversarial mode execution failed: {result.stderr}"
        assert "adversary mode: label_flip" in result.stdout.lower()


def test_run_fl_invalid_arguments():
    """Test that run_fl.py rejects invalid argument values."""
    invalid_test_cases = [
        # Invalid partition strategy
        [
            "--clients",
            "2",
            "--rounds",
            "2",
            "--alpha",
            "0.1",
            "--preset",
            "test",
            "--partition_strategy",
            "invalid",
        ],
        # Invalid adversary mode
        [
            "--clients",
            "2",
            "--rounds",
            "2",
            "--alpha",
            "0.1",
            "--preset",
            "test",
            "--partition_strategy",
            "iid",
            "--adversary_mode",
            "invalid",
        ],
    ]

    for invalid_args in invalid_test_cases:
        result = subprocess.run(
            [sys.executable, "scripts/run_fl.py"] + invalid_args,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"Should reject invalid args: {invalid_args}"
