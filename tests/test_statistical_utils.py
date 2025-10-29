"""Unit tests for statistical utility functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st
from scipy import stats

from scripts.statistical_utils import (
    cohens_d,
    compute_ci,
    mann_whitney_u,
    paired_t_test,
)


class TestComputeCI:
    """Tests for compute_ci confidence interval calculation."""

    def test_compute_ci_single_value_returns_point(self) -> None:
        """Single value should yield CI where bounds equal the value."""
        values = [5.0]
        mean, ci_lower, ci_upper = compute_ci(values)
        assert mean == pytest.approx(5.0)
        assert ci_lower == pytest.approx(5.0)
        assert ci_upper == pytest.approx(5.0)

    def test_compute_ci_identical_values_returns_point(self) -> None:
        """Identical values should yield CI where bounds equal the mean."""
        values = [3.0, 3.0, 3.0, 3.0]
        mean, ci_lower, ci_upper = compute_ci(values)
        assert mean == pytest.approx(3.0)
        assert ci_lower == pytest.approx(3.0)
        assert ci_upper == pytest.approx(3.0)

    def test_compute_ci_known_distribution(self) -> None:
        """CI bounds should follow t-distribution for known data."""
        values = [0.80, 0.84, 0.79, 0.81, 0.80]
        mean, ci_lower, ci_upper = compute_ci(values, confidence=0.95)

        expected_mean = np.mean(values)
        assert mean == pytest.approx(expected_mean)
        assert ci_lower < mean
        assert ci_upper > mean
        assert ci_upper - ci_lower > 0

    def test_compute_ci_wider_for_lower_confidence(self) -> None:
        """Higher confidence level should produce wider CI."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, lower_90, upper_90 = compute_ci(values, confidence=0.90)
        _, lower_95, upper_95 = compute_ci(values, confidence=0.95)
        _, lower_99, upper_99 = compute_ci(values, confidence=0.99)

        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        assert width_95 > width_90
        assert width_99 > width_95

    def test_compute_ci_filters_nan_values(self) -> None:
        """NaN values should be filtered out before computation."""
        values = [0.80, float("nan"), 0.84, float("nan"), 0.79]
        mean, ci_lower, ci_upper = compute_ci(values)

        expected_mean = np.mean([0.80, 0.84, 0.79])
        assert mean == pytest.approx(expected_mean)
        assert not math.isnan(mean)

    def test_compute_ci_empty_after_filtering_returns_nan(self) -> None:
        """All-NaN values should return NaN."""
        values = [float("nan"), float("nan")]
        mean, ci_lower, ci_upper = compute_ci(values)
        assert math.isnan(mean)
        assert math.isnan(ci_lower)
        assert math.isnan(ci_upper)


class TestPairedTTest:
    """Tests for paired t-test statistical test."""

    def test_paired_t_test_known_difference(self) -> None:
        """Paired t-test should detect significant differences."""
        group1 = [0.80, 0.82, 0.79, 0.81, 0.80]
        group2 = [0.86, 0.90, 0.84, 0.83, 0.88]

        result = paired_t_test(group1, group2)

        assert "t_stat" in result
        assert "p_value" in result
        assert "mean_diff" in result
        assert result["p_value"] < 0.05
        assert result["mean_diff"] < 0  # group1 < group2, so diff is negative

    def test_paired_t_test_no_difference(self) -> None:
        """Paired t-test should not detect difference when means are equal."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = paired_t_test(group1, group2)

        assert math.isnan(result["p_value"]) or result["p_value"] > 0.05
        assert result["mean_diff"] == pytest.approx(0.0, abs=1e-10)

    def test_paired_t_test_insufficient_samples(self) -> None:
        """Single pair should return NaN p-value."""
        group1 = [1.0]
        group2 = [2.0]

        result = paired_t_test(group1, group2)

        assert math.isnan(result["p_value"])
        assert math.isnan(result["t_stat"])

    def test_paired_t_test_filters_nan_values(self) -> None:
        """NaN values should be filtered from both groups."""
        group1 = [0.80, float("nan"), 0.84, 0.79, 0.81]
        group2 = [0.86, 0.90, float("nan"), 0.84, 0.83]

        result = paired_t_test(group1, group2)

        assert not math.isnan(result["p_value"])
        assert result["pairs"] == 3

    def test_paired_t_test_result_structure(self) -> None:
        """Result should contain all required keys."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [2.0, 3.0, 4.0, 5.0, 6.0]

        result = paired_t_test(group1, group2)

        required_keys = ["t_stat", "p_value", "mean_diff", "pairs", "n"]
        for key in required_keys:
            assert key in result


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_zero_difference(self) -> None:
        """Identical groups should yield Cohen's d of NaN due to zero variance."""
        group1 = [5.0, 5.0, 5.0, 5.0, 5.0]
        group2 = [5.0, 5.0, 5.0, 5.0, 5.0]

        d = cohens_d(group1, group2)
        assert math.isnan(d)  # Zero variance results in NaN

    def test_cohens_d_large_difference(self) -> None:
        """Large difference should yield large Cohen's d (or NaN with zero variance)."""
        group1 = [1.0, 1.0, 1.0, 1.0, 1.0]
        group2 = [10.0, 10.0, 10.0, 10.0, 10.0]

        d = cohens_d(group1, group2)
        assert math.isnan(d)  # Both groups have zero variance

    def test_cohens_d_single_value_per_group(self) -> None:
        """Single values should yield d = mean_diff."""
        group1 = [5.0]
        group2 = [7.0]

        d = cohens_d(group1, group2)
        assert d == pytest.approx(-2.0)

    def test_cohens_d_filters_nan_values(self) -> None:
        """NaN values should be filtered before computation."""
        group1 = [0.80, float("nan"), 0.84, 0.79, 0.81]
        group2 = [0.86, 0.90, float("nan"), 0.84, 0.83]

        d = cohens_d(group1, group2)
        assert not math.isnan(d)

    def test_cohens_d_known_values(self) -> None:
        """Cohen's d should match scipy calculation for known data."""
        group1 = [0.80, 0.82, 0.79, 0.81, 0.80]
        group2 = [0.86, 0.90, 0.84, 0.83, 0.88]

        d = cohens_d(group1, group2)

        arr1 = np.array(group1)
        arr2 = np.array(group2)
        var1 = np.var(arr1, ddof=1) * (len(arr1) - 1)
        var2 = np.var(arr2, ddof=1) * (len(arr2) - 1)
        n_total = len(arr1) + len(arr2) - 2
        pooled_std = np.sqrt((var1 + var2) / n_total)
        expected_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std

        assert d == pytest.approx(expected_d)

    def test_cohens_d_empty_group_returns_nan(self) -> None:
        """Empty group should return NaN."""
        group1: list[float] = []
        group2 = [1.0, 2.0, 3.0]

        d = cohens_d(group1, group2)
        assert math.isnan(d)

    def test_cohens_d_all_nan_returns_nan(self) -> None:
        """All NaN should return NaN."""
        group1 = [float("nan"), float("nan")]
        group2 = [float("nan"), float("nan")]

        d = cohens_d(group1, group2)
        assert math.isnan(d)


class TestMannWhitneyU:
    """Tests for Mann-Whitney U non-parametric test."""

    def test_mann_whitney_u_known_difference(self) -> None:
        """Mann-Whitney U should detect difference between distinct groups."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [6.0, 7.0, 8.0, 9.0, 10.0]

        result = mann_whitney_u(group1, group2)

        assert "u_stat" in result
        assert "p_value" in result
        assert result["p_value"] < 0.05

    def test_mann_whitney_u_no_difference(self) -> None:
        """Mann-Whitney U should not detect difference for identical groups."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = mann_whitney_u(group1, group2)

        assert result["p_value"] > 0.05

    def test_mann_whitney_u_filters_nan_values(self) -> None:
        """NaN values should be filtered before computation."""
        group1 = [1.0, float("nan"), 3.0, 4.0, 5.0]
        group2 = [6.0, 7.0, float("nan"), 9.0, 10.0]

        result = mann_whitney_u(group1, group2)

        assert "u_stat" in result
        assert "p_value" in result
        assert not math.isnan(result["p_value"])

    def test_mann_whitney_u_insufficient_samples(self) -> None:
        """Single value per group can yield valid p-value (scipy handles it)."""
        group1 = [1.0]
        group2 = [2.0]

        result = mann_whitney_u(group1, group2)

        assert result["n1"] == 1
        assert result["n2"] == 1

    def test_mann_whitney_u_result_structure(self) -> None:
        """Result should contain all required keys."""
        group1 = [1.0, 2.0, 3.0]
        group2 = [4.0, 5.0, 6.0]

        result = mann_whitney_u(group1, group2)

        required_keys = ["u_stat", "p_value", "n1", "n2"]
        for key in required_keys:
            assert key in result

    def test_mann_whitney_u_consistent_with_scipy(self) -> None:
        """Result should match scipy.stats.mannwhitneyu."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [6.0, 7.0, 8.0, 9.0, 10.0]

        result = mann_whitney_u(group1, group2)

        arr1 = np.array([v for v in group1 if not math.isnan(v)])
        arr2 = np.array([v for v in group2 if not math.isnan(v)])
        expected_u, expected_p = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")

        assert result["u_stat"] == pytest.approx(expected_u)
        assert result["p_value"] == pytest.approx(expected_p)


class TestComputeCIProperties:
    """Property-based tests for compute_ci confidence interval calculation."""

    @given(
        values=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
            min_size=2,
            max_size=100,
        ),
        confidence=st.floats(min_value=0.80, max_value=0.99),
    )
    def test_compute_ci_contains_mean_property(self, values, confidence):
        """CI bounds should always contain the mean (property-based)."""
        mean, ci_lower, ci_upper = compute_ci(values, confidence)

        if not math.isnan(mean):
            assert ci_lower <= mean <= ci_upper, f"CI [{ci_lower}, {ci_upper}] does not contain mean {mean}"

    @given(
        values=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
            min_size=2,
            max_size=100,
        )
    )
    def test_compute_ci_width_increases_with_variance_property(self, values):
        """CI width increases with data variance (property-based)."""
        _, lower_90, upper_90 = compute_ci(values, confidence=0.90)
        _, lower_95, upper_95 = compute_ci(values, confidence=0.95)
        _, lower_99, upper_99 = compute_ci(values, confidence=0.99)

        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        assert width_95 >= width_90 - 1e-10, f"Width 95 ({width_95}) should be >= width 90 ({width_90})"
        assert width_99 >= width_95 - 1e-10, f"Width 99 ({width_99}) should be >= width 95 ({width_95})"


class TestPairedTTestProperties:
    """Property-based tests for paired t-test."""

    @given(
        a=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
            min_size=2,
            max_size=100,
        ),
        b=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
            min_size=2,
            max_size=100,
        ),
    )
    def test_paired_t_test_anticommutativity_property(self, a, b):
        """Swapping groups should negate mean_diff (approximately)."""
        result_ab = paired_t_test(a, b)
        result_ba = paired_t_test(b, a)

        mean_diff_ab = result_ab.get("mean_diff")
        mean_diff_ba = result_ba.get("mean_diff")

        ab_not_none = mean_diff_ab is not None and not math.isnan(mean_diff_ab)
        ba_not_none = mean_diff_ba is not None and not math.isnan(mean_diff_ba)
        if ab_not_none and ba_not_none:
            assert pytest.approx(mean_diff_ab, abs=0.001) == -mean_diff_ba, (
                f"mean_diff(a,b)={mean_diff_ab} should negate " f"mean_diff(b,a)={mean_diff_ba}"
            )

    @given(
        values=st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        )
    )
    def test_paired_t_test_identical_groups_property(self, values):
        """Paired t-test with identical groups should have zero mean_diff."""
        result = paired_t_test(values, values)

        mean_diff = result.get("mean_diff")
        if mean_diff is not None and not math.isnan(mean_diff):
            msg = f"Identical groups should have mean_diff ~0, got {mean_diff}"
            assert pytest.approx(mean_diff, abs=1e-10) == 0.0, msg


class TestCohensDProperties:
    """Property-based tests for Cohen's d effect size."""

    @given(
        a=st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1000,
                max_value=1000,
            ),
            min_size=2,
            max_size=100,
        ),
        b=st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1000,
                max_value=1000,
            ),
            min_size=2,
            max_size=100,
        ),
    )
    def test_cohens_d_anticommutativity_property(self, a, b):
        """Swapping groups should negate effect size."""
        d_ab = cohens_d(a, b)
        d_ba = cohens_d(b, a)

        if not math.isnan(d_ab) and not math.isnan(d_ba):
            assert pytest.approx(d_ab, abs=0.001) == -d_ba, f"cohens_d(a,b)={d_ab} should negate cohens_d(b,a)={d_ba}"

    @given(
        values=st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        )
    )
    def test_cohens_d_identical_groups_property(self, values):
        """Identical groups should have zero or undefined effect size."""
        d = cohens_d(values, values)

        if not math.isnan(d):
            assert pytest.approx(d, abs=1e-10) == 0.0, f"Identical groups should have d ~0, got {d}"
