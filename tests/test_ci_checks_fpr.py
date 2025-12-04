#!/usr/bin/env python3
"""
Tests for FPR tolerance validation in CI checks.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from scripts.ci_checks import ArtifactValidationError, validate_fpr_tolerance


def create_client_metrics_csv(directory: Path, client_id: int, fpr_values: list):
    """Helper to create client metrics CSV with specified FPR values."""
    csv_path = directory / f"client_{client_id}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "benign_fpr_bin_tau"])
        writer.writeheader()
        for round_num, fpr in enumerate(fpr_values, start=1):
            writer.writerow({"round": round_num, "benign_fpr_bin_tau": fpr})
    return csv_path


def test_fpr_tolerance_strict_mode_passes():
    """Test that FPR within tolerance passes in strict mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create client with FPR exactly at target
        create_client_metrics_csv(run_dir, 0, [0.09, 0.10, 0.10])

        # Should not raise in strict mode
        validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=True)


def test_fpr_tolerance_strict_mode_fails():
    """Test that FPR outside tolerance fails in strict mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create client with FPR well outside tolerance
        create_client_metrics_csv(run_dir, 0, [0.10, 0.20, 0.30])

        # Should raise in strict mode
        with pytest.raises(ArtifactValidationError, match="FPR tolerance check failed"):
            validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=True)


def test_fpr_tolerance_non_strict_mode_warns():
    """Test that FPR outside tolerance only warns in non-strict mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create client with FPR well outside tolerance (like UNSW data)
        create_client_metrics_csv(run_dir, 0, [0.10, 0.50, 0.923])

        # Should NOT raise in non-strict mode (warnings only)
        validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=False)


def test_fpr_tolerance_multiple_clients():
    """Test FPR validation with multiple clients."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Client 0: within tolerance
        create_client_metrics_csv(run_dir, 0, [0.08, 0.09, 0.10])

        # Client 1: outside tolerance
        create_client_metrics_csv(run_dir, 1, [0.10, 0.15, 0.20])

        # Strict mode should fail on client 1
        with pytest.raises(ArtifactValidationError, match="client_1_metrics.csv"):
            validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=True)

        # Non-strict mode should warn but not fail
        validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=False)


def test_fpr_tolerance_missing_column():
    """Test that missing FPR column is gracefully skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create CSV without benign_fpr_bin_tau column
        csv_path = run_dir / "client_0_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "acc_after"])
            writer.writeheader()
            writer.writerow({"round": 1, "acc_after": 0.8})

        # Should skip validation without error
        validate_fpr_tolerance(run_dir, strict=True)


def test_fpr_tolerance_no_clients():
    """Test that no client files is gracefully handled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Empty directory - should skip validation without error
        validate_fpr_tolerance(run_dir, strict=True)


def test_fpr_tolerance_edge_case_boundary():
    """Test FPR exactly at tolerance boundary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # FPR exactly at upper boundary (0.10 + 0.02 = 0.12)
        create_client_metrics_csv(run_dir, 0, [0.12])

        # Should pass (<=, not <)
        validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=True)


def test_fpr_tolerance_custom_target_and_tolerance():
    """Test validation with custom FPR target and tolerance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Test with target=0.05, tolerance=0.10
        create_client_metrics_csv(run_dir, 0, [0.05, 0.10, 0.15])

        # Should pass (0.15 is within 0.05 ± 0.10)
        validate_fpr_tolerance(run_dir, target_fpr=0.05, tolerance=0.10, strict=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
