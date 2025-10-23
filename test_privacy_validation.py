#!/usr/bin/env python3
"""
Unit tests for privacy experiment validation functions.
Tests validation of DP parameters, epsilon computation, and privacy-utility curve data.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import patch, mock_open
import sys

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent / "scripts"))

from ci_checks import (
    validate_privacy_experiments,
    check_privacy_regressions,
    validate_privacy_utility_curve_data,
    ArtifactValidationError,
)


class TestValidatePrivacyExperiments:
    """Test validation of privacy experiment data."""

    def test_validates_dp_parameters_logged(self):
        """Should validate that DP parameters are properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock experiment directory with DP parameters
            exp_dir = temp_path / "comparative-analysis-privacy-1"
            exp_dir.mkdir()

            # Create client metrics with DP parameters
            client_metrics = exp_dir / "client_0_metrics.csv"
            with open(client_metrics, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                writer.writerow([0, "True", "0.5", "1.2", "1e-5", "0.85"])
                writer.writerow([1, "True", "0.5", "1.3", "1e-5", "0.87"])

            # Should not raise exception for valid DP parameters
            try:
                validate_privacy_experiments(temp_path)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for valid DP parameters")

    def test_fails_when_dp_parameters_missing(self):
        """Should fail when DP parameters are missing from experiment data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock experiment directory without DP parameters
            exp_dir = temp_path / "comparative-analysis-privacy-1"
            exp_dir.mkdir()

            # Create client metrics without DP parameters
            client_metrics = exp_dir / "client_0_metrics.csv"
            with open(client_metrics, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "macro_f1"])
                writer.writerow([0, "0.85"])
                writer.writerow([1, "0.87"])

            # Should raise exception for missing DP parameters
            with pytest.raises(ArtifactValidationError, match="Missing DP parameters"):
                validate_privacy_experiments(temp_path)

    def test_validates_epsilon_values_computed(self):
        """Should validate that epsilon values are computed and reasonable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock experiment directory with epsilon values
            exp_dir = temp_path / "comparative-analysis-privacy-1"
            exp_dir.mkdir()

            # Create client metrics with epsilon values
            client_metrics = exp_dir / "client_0_metrics.csv"
            with open(client_metrics, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                writer.writerow([0, "True", "0.5", "1.2", "1e-5", "0.85"])
                writer.writerow([1, "True", "0.5", "1.3", "1e-5", "0.87"])

            # Should not raise exception for valid epsilon values
            try:
                validate_privacy_experiments(temp_path)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for valid epsilon values")

    def test_fails_when_epsilon_values_invalid(self):
        """Should fail when epsilon values are invalid or missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock experiment directory with invalid epsilon values
            exp_dir = temp_path / "comparative-analysis-privacy-1"
            exp_dir.mkdir()

            # Create client metrics with invalid epsilon values
            client_metrics = exp_dir / "client_0_metrics.csv"
            with open(client_metrics, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                writer.writerow([0, "True", "0.5", "", "1e-5", "0.85"])  # Empty epsilon
                writer.writerow([1, "True", "0.5", "invalid", "1e-5", "0.87"])  # Invalid epsilon

            # Should raise exception for invalid epsilon values
            with pytest.raises(ArtifactValidationError, match="No valid epsilon values"):
                validate_privacy_experiments(temp_path)

    def test_validates_noise_multiplier_consistency(self):
        """Should validate that noise multiplier is consistent across rounds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock experiment directory with consistent noise multiplier
            exp_dir = temp_path / "comparative-analysis-privacy-1"
            exp_dir.mkdir()

            # Create client metrics with consistent noise multiplier
            client_metrics = exp_dir / "client_0_metrics.csv"
            with open(client_metrics, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                writer.writerow([0, "True", "0.5", "1.2", "1e-5", "0.85"])
                writer.writerow([1, "True", "0.5", "1.3", "1e-5", "0.87"])

            # Should not raise exception for consistent noise multiplier
            try:
                validate_privacy_experiments(temp_path)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for consistent noise multiplier")

    def test_fails_when_noise_multiplier_inconsistent(self):
        """Should fail when noise multiplier is inconsistent across rounds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock experiment directory with inconsistent noise multiplier
            exp_dir = temp_path / "comparative-analysis-privacy-1"
            exp_dir.mkdir()

            # Create client metrics with inconsistent noise multiplier
            client_metrics = exp_dir / "client_0_metrics.csv"
            with open(client_metrics, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                writer.writerow([0, "True", "0.5", "1.2", "1e-5", "0.85"])
                writer.writerow([1, "True", "1.0", "1.3", "1e-5", "0.87"])  # Different noise multiplier

            # Should raise exception for inconsistent noise multiplier
            with pytest.raises(ArtifactValidationError, match="Inconsistent noise multiplier"):
                validate_privacy_experiments(temp_path)


class TestCheckPrivacyRegressions:
    """Test privacy regression detection."""

    def test_detects_epsilon_computation_changes(self):
        """Should detect changes in epsilon computation between runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create baseline experiment
            baseline_dir = temp_path / "baseline"
            baseline_dir.mkdir()
            baseline_exp = baseline_dir / "comparative-analysis-privacy-1"
            baseline_exp.mkdir()

            # Create current experiment with different epsilon values
            current_dir = temp_path / "current"
            current_dir.mkdir()
            current_exp = current_dir / "comparative-analysis-privacy-1"
            current_exp.mkdir()

            # Create client metrics with different epsilon values
            for exp_dir, epsilon_val in [(baseline_exp, "1.2"), (current_exp, "2.0")]:
                client_metrics = exp_dir / "client_0_metrics.csv"
                with open(client_metrics, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                    writer.writerow([0, "True", "0.5", epsilon_val, "1e-5", "0.85"])

            # Should detect regression in epsilon computation
            with pytest.raises(ArtifactValidationError, match="Privacy regression detected"):
                check_privacy_regressions(baseline_dir, current_dir)

    def test_passes_when_epsilon_values_similar(self):
        """Should pass when epsilon values are similar between runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create baseline experiment
            baseline_dir = temp_path / "baseline"
            baseline_dir.mkdir()
            baseline_exp = baseline_dir / "comparative-analysis-privacy-1"
            baseline_exp.mkdir()

            # Create current experiment with similar epsilon values
            current_dir = temp_path / "current"
            current_dir.mkdir()
            current_exp = current_dir / "comparative-analysis-privacy-1"
            current_exp.mkdir()

            # Create client metrics with similar epsilon values
            for exp_dir, epsilon_val in [(baseline_exp, "1.2"), (current_exp, "1.25")]:
                client_metrics = exp_dir / "client_0_metrics.csv"
                with open(client_metrics, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["round", "dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta", "macro_f1"])
                    writer.writerow([0, "True", "0.5", epsilon_val, "1e-5", "0.85"])

            # Should not raise exception for similar epsilon values
            try:
                check_privacy_regressions(baseline_dir, current_dir)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for similar epsilon values")


class TestValidatePrivacyUtilityCurveData:
    """Test validation of privacy-utility curve data."""

    def test_validates_csv_structure(self):
        """Should validate that privacy-utility curve CSV has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid privacy-utility curve CSV
            curve_csv = temp_path / "privacy_utility_curve.csv"
            with open(curve_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"])
                writer.writerow([1.5, 0.79, 0.54, 1.04, 5, 0.7, 0])
                writer.writerow([None, 0.89, 0.89, 0.89, 5, None, 1])

            # Should not raise exception for valid CSV structure
            try:
                validate_privacy_utility_curve_data(curve_csv)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for valid CSV structure")

    def test_fails_when_csv_missing_required_columns(self):
        """Should fail when CSV is missing required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid privacy-utility curve CSV missing required columns
            curve_csv = temp_path / "privacy_utility_curve.csv"
            with open(curve_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epsilon", "macro_f1_mean"])  # Missing required columns
                writer.writerow([1.5, 0.79])

            # Should raise exception for missing required columns
            with pytest.raises(ArtifactValidationError, match="Missing required columns"):
                validate_privacy_utility_curve_data(curve_csv)

    def test_validates_epsilon_values_in_range(self):
        """Should validate that epsilon values are in reasonable range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create privacy-utility curve CSV with epsilon values in range
            curve_csv = temp_path / "privacy_utility_curve.csv"
            with open(curve_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"])
                writer.writerow([1.5, 0.79, 0.54, 1.04, 5, 0.7, 0])
                writer.writerow([2.0, 0.75, 0.50, 1.00, 5, 1.0, 0])

            # Should not raise exception for epsilon values in range
            try:
                validate_privacy_utility_curve_data(curve_csv)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for epsilon values in range")

    def test_fails_when_epsilon_values_out_of_range(self):
        """Should fail when epsilon values are out of reasonable range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create privacy-utility curve CSV with epsilon values out of range
            curve_csv = temp_path / "privacy_utility_curve.csv"
            with open(curve_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"])
                writer.writerow([100.0, 0.79, 0.54, 1.04, 5, 0.7, 0])  # Epsilon too high
                writer.writerow([-1.0, 0.75, 0.50, 1.00, 5, 1.0, 0])  # Epsilon negative

            # Should raise exception for epsilon values out of range
            with pytest.raises(ArtifactValidationError, match="Epsilon values out of range"):
                validate_privacy_utility_curve_data(curve_csv)

    def test_validates_confidence_intervals(self):
        """Should validate that confidence intervals are properly formed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create privacy-utility curve CSV with valid confidence intervals
            curve_csv = temp_path / "privacy_utility_curve.csv"
            with open(curve_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"])
                writer.writerow([1.5, 0.79, 0.54, 1.04, 5, 0.7, 0])  # ci_lower < mean < ci_upper

            # Should not raise exception for valid confidence intervals
            try:
                validate_privacy_utility_curve_data(curve_csv)
            except ArtifactValidationError:
                pytest.fail("Should not raise ArtifactValidationError for valid confidence intervals")

    def test_fails_when_confidence_intervals_invalid(self):
        """Should fail when confidence intervals are invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create privacy-utility curve CSV with invalid confidence intervals
            curve_csv = temp_path / "privacy_utility_curve.csv"
            with open(curve_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"])
                writer.writerow([1.5, 0.79, 1.04, 0.54, 5, 0.7, 0])  # ci_lower > ci_upper

            # Should raise exception for invalid confidence intervals
            with pytest.raises(ArtifactValidationError, match="Invalid confidence intervals"):
                validate_privacy_utility_curve_data(curve_csv)
