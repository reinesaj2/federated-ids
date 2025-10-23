"""Unit tests for CI validation and statistical utility functions."""

import math

import pytest

from scripts.analyze_fedprox_comparison import cohens_d, _mean_ci
from scripts.ci_checks import (
    ArtifactValidationError,
    check_convergence_quality,
    check_no_nans_or_infs,
    check_seed_consistency,
)


class TestCohensd:
    """Test cohens_d effect size calculation."""

    def test_cohens_d_identical_groups(self):
        """Cohen's d should be 0 when groups are identical."""
        group = [1.0, 2.0, 3.0]
        result = cohens_d(group, group)
        assert result == 0.0

    def test_cohens_d_known_values(self):
        """Cohen's d should match hand-calculated value."""
        group1 = [1.0, 2.0, 3.0]
        group2 = [2.0, 3.0, 4.0]
        result = cohens_d(group1, group2)
        assert result < 0
        assert -2.0 < result < 0

    def test_cohens_d_empty_group(self):
        """Cohen's d should return NaN for empty groups."""
        result = cohens_d([], [1.0, 2.0])
        assert math.isnan(result)

    def test_cohens_d_nan_values(self):
        """Cohen's d should filter out NaN values."""
        group1 = [1.0, float("nan"), 3.0]
        group2 = [1.5, 2.5, 3.5]
        result = cohens_d(group1, group2)
        assert not math.isnan(result)

    def test_cohens_d_zero_variance(self):
        """Cohen's d should handle zero variance gracefully."""
        group1 = [2.0, 2.0, 2.0]
        group2 = [3.0, 3.0, 3.0]
        result = cohens_d(group1, group2)
        assert math.isnan(result)

    def test_cohens_d_single_sample(self):
        """Cohen's d with single sample should return mean difference."""
        result = cohens_d([1.0], [3.0])
        assert result == -2.0


class TestMeanCI:
    """Test confidence interval calculation."""

    def test_mean_ci_single_value(self):
        """CI should equal the value itself for single sample."""
        mean, ci_lower, ci_upper = _mean_ci([5.0])
        assert mean == 5.0
        assert ci_lower == 5.0
        assert ci_upper == 5.0

    def test_mean_ci_identical_values(self):
        """CI should equal the value for identical samples."""
        mean, ci_lower, ci_upper = _mean_ci([2.0, 2.0, 2.0])
        assert mean == 2.0
        assert ci_lower == 2.0
        assert ci_upper == 2.0

    def test_mean_ci_reasonable_range(self):
        """CI width should be reasonable for typical data."""
        values = [0.8, 0.85, 0.9, 0.88, 0.92]
        mean, ci_lower, ci_upper = _mean_ci(values)
        assert ci_lower < mean < ci_upper
        assert (ci_upper - ci_lower) < 0.2

    def test_mean_ci_filters_nans(self):
        """Mean CI should filter out NaN values."""
        values = [0.8, float("nan"), 0.9, float("nan")]
        mean, ci_lower, ci_upper = _mean_ci(values)
        assert 0.8 < mean < 0.95
        assert not math.isnan(mean)

    def test_mean_ci_all_nans(self):
        """Mean CI should return NaN for all-NaN input."""
        mean, ci_lower, ci_upper = _mean_ci([float("nan"), float("nan")])
        assert math.isnan(mean)
        assert math.isnan(ci_lower)
        assert math.isnan(ci_upper)

    def test_mean_ci_empty_list(self):
        """Mean CI should return NaN for empty list."""
        mean, ci_lower, ci_upper = _mean_ci([])
        assert math.isnan(mean)


class TestCheckConvergenceQuality:
    """Test convergence quality validation."""

    def test_check_convergence_quality_valid(self):
        """Valid convergence data should pass."""
        rows = [
            {"weighted_macro_f1": "0.8", "weighted_accuracy": "0.8"},
            {"weighted_macro_f1": "0.85", "weighted_accuracy": "0.85"},
            {"weighted_macro_f1": "0.9", "weighted_accuracy": "0.9"},
        ]
        check_convergence_quality(rows)

    def test_check_convergence_quality_empty(self):
        """Empty rows should raise error."""
        with pytest.raises(ArtifactValidationError):
            check_convergence_quality([])

    def test_check_convergence_quality_nan_value(self):
        """NaN values should raise error."""
        rows = [
            {"weighted_macro_f1": "0.8", "weighted_accuracy": "0.8"},
            {"weighted_macro_f1": "nan", "weighted_accuracy": "0.85"},
        ]
        with pytest.raises(ArtifactValidationError, match="nan"):
            check_convergence_quality(rows)

    def test_check_convergence_quality_inf_value(self):
        """Inf values should raise error."""
        rows = [
            {"weighted_macro_f1": "inf", "weighted_accuracy": "0.8"},
        ]
        with pytest.raises(ArtifactValidationError, match="inf"):
            check_convergence_quality(rows)

    def test_check_convergence_quality_low_final_accuracy(self):
        """Final accuracy below 0.70 should raise error."""
        rows = [
            {"weighted_macro_f1": "0.8", "weighted_accuracy": "0.8"},
            {"weighted_macro_f1": "0.85", "weighted_accuracy": "0.85"},
            {"weighted_macro_f1": "0.6", "weighted_accuracy": "0.6"},
        ]
        with pytest.raises(ArtifactValidationError, match="below minimum"):
            check_convergence_quality(rows)


class TestCheckNoNansOrInfs:
    """Test NaN/Inf detection."""

    def test_check_no_nans_or_infs_valid(self):
        """Valid rows should pass."""
        rows = [
            {"metric1": "0.8", "metric2": "0.9"},
            {"metric1": "0.85", "metric2": "0.95"},
        ]
        check_no_nans_or_infs(rows, ["metric1", "metric2"])

    def test_check_no_nans_or_infs_nan_detected(self):
        """NaN should raise error."""
        rows = [
            {"metric1": "0.8", "metric2": "nan"},
        ]
        with pytest.raises(ArtifactValidationError):
            check_no_nans_or_infs(rows, ["metric2"])

    def test_check_no_nans_or_infs_inf_detected(self):
        """Inf should raise error."""
        rows = [
            {"metric1": "inf", "metric2": "0.9"},
        ]
        with pytest.raises(ArtifactValidationError):
            check_no_nans_or_infs(rows, ["metric1"])

    def test_check_no_nans_or_infs_missing_column(self):
        """Missing column should not raise error."""
        rows = [
            {"metric1": "0.8"},
        ]
        check_no_nans_or_infs(rows, ["metric2"])


class TestCheckSeedConsistency:
    """Test seed consistency validation."""

    def test_check_seed_consistency_valid(self):
        """Valid seed counts should pass."""
        rows = [
            {"seed": "0", "metric": "0.8"},
            {"seed": "1", "metric": "0.85"},
            {"seed": "2", "metric": "0.9"},
            {"seed": "3", "metric": "0.88"},
            {"seed": "4", "metric": "0.92"},
        ]
        check_seed_consistency(rows, expected_seeds=5)

    def test_check_seed_consistency_insufficient_seeds(self):
        """Fewer seeds than required should raise error."""
        rows = [
            {"seed": "0", "metric": "0.8"},
            {"seed": "1", "metric": "0.85"},
        ]
        with pytest.raises(ArtifactValidationError, match="Only 2 seeds"):
            check_seed_consistency(rows, expected_seeds=5)

    def test_check_seed_consistency_no_seed_column(self):
        """Missing seed column should not raise error."""
        rows = [
            {"metric": "0.8"},
            {"metric": "0.85"},
        ]
        check_seed_consistency(rows, expected_seeds=5)

    def test_check_seed_consistency_duplicate_seeds(self):
        """Duplicate seeds should count only once."""
        rows = [
            {"seed": "0", "metric": "0.8"},
            {"seed": "0", "metric": "0.85"},
            {"seed": "1", "metric": "0.9"},
        ]
        with pytest.raises(ArtifactValidationError):
            check_seed_consistency(rows, expected_seeds=5)
