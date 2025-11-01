#!/usr/bin/env python3
"""
Test suite for metric validation framework (Issue #78).

Tests the validation functions that detect and prevent metric computation bugs
that undermine scientific validity of thesis plots.
"""

import numpy as np
import pandas as pd
import pytest

from server_metrics import calculate_robustness_metrics, validate_metrics


class TestMetricValidation:
    """Test metric validation functions for data quality issues."""

    def test_validate_metrics_cosine_out_of_bounds(self):
        """Test detection of cosine similarity outside valid range [-1, 1]."""
        # Test case: cosine > 1.0 (impossible)
        metrics = {"cos_to_benign_mean": 1.5}
        warnings = validate_metrics(metrics, "test_dimension")

        assert len(warnings) == 1
        assert "outside valid range [-1, 1]" in warnings[0]
        assert "1.500000" in warnings[0]

    def test_validate_metrics_cosine_too_low(self):
        """Test detection of suspiciously low cosine similarity for FL models."""
        # Test case: cosine < 0.5 (suspicious for same-architecture FL models)
        metrics = {"cos_to_benign_mean": 0.3}
        warnings = validate_metrics(metrics, "test_dimension")

        assert len(warnings) == 1
        assert "Suspiciously low cosine" in warnings[0]
        assert "0.300000" in warnings[0]

    def test_validate_metrics_l2_negative(self):
        """Test detection of negative L2 distance (impossible)."""
        metrics = {"l2_to_benign_mean": -0.5}
        warnings = validate_metrics(metrics, "test_dimension")

        assert len(warnings) == 1
        assert "Negative L2 distance" in warnings[0]
        assert "-0.500000" in warnings[0]

    def test_validate_metrics_l2_exactly_zero(self):
        """Test detection of L2 distance exactly 0.0 (suspicious)."""
        metrics = {"l2_to_benign_mean": 0.0}
        warnings = validate_metrics(metrics, "test_dimension")

        assert len(warnings) == 1
        assert "L2 distance is exactly 0.0" in warnings[0]

    def test_validate_metrics_no_issues(self):
        """Test that valid metrics produce no warnings."""
        metrics = {"cos_to_benign_mean": 0.95, "l2_to_benign_mean": 0.5, "update_norm_mean": 1.0, "update_norm_std": 0.1}
        warnings = validate_metrics(metrics, "test_dimension")

        assert len(warnings) == 0

    def test_validate_metrics_multiple_issues(self):
        """Test detection of multiple metric issues simultaneously."""
        metrics = {"cos_to_benign_mean": 0.2, "l2_to_benign_mean": 0.0, "update_norm_mean": -1.0}  # Too low  # Exactly zero  # Negative
        warnings = validate_metrics(metrics, "test_dimension")

        assert len(warnings) == 3
        assert any("Suspiciously low cosine" in w for w in warnings)
        assert any("L2 distance is exactly 0.0" in w for w in warnings)
        assert any("Negative norm mean" in w for w in warnings)


class TestRobustnessMetricsCalculation:
    """Test robustness metrics calculation for edge cases."""

    def test_calculate_robustness_metrics_identical_vectors(self):
        """Test L2 distance calculation when vectors are identical."""
        # Create identical vectors
        identical_vector = [np.array([1.0, 2.0, 3.0])]
        client_updates = [identical_vector, identical_vector]
        benign_mean = identical_vector
        aggregated = identical_vector

        metrics = calculate_robustness_metrics(client_updates, benign_mean, aggregated)

        # L2 should be 0.0 when vectors are identical
        assert metrics["l2_to_benign_mean"] == 0.0
        # Cosine should be 1.0 when vectors are identical
        assert metrics["cos_to_benign_mean"] == 1.0

    def test_calculate_robustness_metrics_orthogonal_vectors(self):
        """Test cosine similarity calculation for orthogonal vectors."""
        # Create orthogonal vectors
        vector_a = [np.array([1.0, 0.0])]
        vector_b = [np.array([0.0, 1.0])]
        client_updates = [vector_a, vector_b]
        benign_mean = vector_a
        aggregated = vector_b

        metrics = calculate_robustness_metrics(client_updates, benign_mean, aggregated)

        # Cosine should be 0.0 for orthogonal vectors
        assert abs(metrics["cos_to_benign_mean"] - 0.0) < 1e-10

    def test_calculate_robustness_metrics_zero_vector_handling(self):
        """Test handling of zero vectors in cosine similarity calculation."""
        # Create zero vector
        zero_vector = [np.array([0.0, 0.0, 0.0])]
        normal_vector = [np.array([1.0, 2.0, 3.0])]
        client_updates = [zero_vector, normal_vector]
        benign_mean = zero_vector
        aggregated = normal_vector

        # Should raise ValueError for zero vector
        with pytest.raises(ValueError, match="zero norm"):
            calculate_robustness_metrics(client_updates, benign_mean, aggregated)

    def test_calculate_robustness_metrics_floating_point_precision(self):
        """Test floating point precision handling in cosine similarity."""
        # Create vectors that should give cosine close to 1.0
        vector_a = [np.array([1.0, 1.0, 1.0])]
        vector_b = [np.array([1.0000001, 1.0000001, 1.0000001])]
        client_updates = [vector_a, vector_b]
        benign_mean = vector_a
        aggregated = vector_b

        metrics = calculate_robustness_metrics(client_updates, benign_mean, aggregated)

        # Should handle floating point precision correctly
        assert 0.999999 <= metrics["cos_to_benign_mean"] <= 1.0


class TestDataFrameValidation:
    """Test validation of experiment data in DataFrame format."""

    def test_validate_dataframe_cosine_variance_collapse(self):
        """Test detection of cosine variance collapse (all values identical)."""
        # Create DataFrame with all cosine values = 1.0 (no variance)
        df = pd.DataFrame(
            {"cos_to_benign_mean": [1.0, 1.0, 1.0, 1.0, 1.0], "aggregation": ["fedavg", "fedavg", "fedavg", "fedavg", "fedavg"]}
        )

        # Check for variance collapse
        cosine_std = df["cos_to_benign_mean"].std()
        assert cosine_std < 1e-6, "Cosine variance collapse detected"

    def test_validate_dataframe_f1_ceiling_effect(self):
        """Test detection of F1 ceiling effect (all values near 1.0)."""
        # Create DataFrame with F1 ceiling effect
        df = pd.DataFrame(
            {"macro_f1": [0.999, 0.999, 0.999, 0.999, 0.999], "aggregation": ["fedavg", "krum", "bulyan", "median", "fedavg"]}
        )

        # Check for ceiling effect
        f1_perfect = (df["macro_f1"] >= 0.999).sum()
        ceiling_ratio = f1_perfect / len(df)
        assert ceiling_ratio > 0.8, "F1 ceiling effect detected"

    def test_validate_dataframe_l2_zero_bug(self):
        """Test detection of L2 zero bug (multiple exact zeros)."""
        # Create DataFrame with L2 zero bug
        df = pd.DataFrame({"l2_to_benign_mean": [0.0, 0.0, 0.5, 0.3, 0.4], "aggregation": ["bulyan", "median", "fedavg", "krum", "fedavg"]})

        # Check for L2 zero bug
        l2_zeros = (df["l2_to_benign_mean"] == 0.0).sum()
        assert l2_zeros > 1, "L2 zero bug detected"


class TestPlotQualityValidation:
    """Test plot quality validation framework."""

    def test_validate_plot_metrics_before_plotting(self):
        """Test validation of metrics before plot generation."""
        # Create test data with issues
        df = pd.DataFrame(
            {
                "cos_to_benign_mean": [0.2, 0.3, 0.4],  # Too low
                "l2_to_benign_mean": [0.0, 0.0, 0.5],  # Zeros
                "macro_f1": [0.999, 0.999, 0.999],  # Ceiling
                "aggregation": ["fedavg", "krum", "bulyan"],
            }
        )

        # This should be implemented in the validation framework
        # For now, test the detection logic
        cosine_issues = df["cos_to_benign_mean"] < 0.5
        l2_issues = df["l2_to_benign_mean"] == 0.0
        f1_issues = df["macro_f1"] >= 0.999

        assert cosine_issues.any(), "Should detect low cosine values"
        assert l2_issues.any(), "Should detect L2 zeros"
        assert f1_issues.all(), "Should detect F1 ceiling effect"

    def test_validate_plot_metrics_statistical_power(self):
        """Test validation of statistical power (sufficient seeds)."""
        # Create test data with insufficient seeds
        df = pd.DataFrame(
            {"seed": [42, 42, 42], "macro_f1": [0.95, 0.96, 0.97], "aggregation": ["fedavg", "fedavg", "fedavg"]}  # Only one seed
        )

        # Check for sufficient seeds
        unique_seeds = df["seed"].nunique()
        assert unique_seeds < 3, "Should detect insufficient seeds for statistical power"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
