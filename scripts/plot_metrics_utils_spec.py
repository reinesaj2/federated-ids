#!/usr/bin/env python3
"""Unit tests for plot_metrics_utils confidence interval computations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from plot_metrics_utils import compute_confidence_interval  # noqa: E402


def test_compute_ci_for_sample_data():
    """Compute correct CI for sample data."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, ci_lower, ci_upper = compute_confidence_interval(data)

    assert mean == 3.0
    assert ci_lower < mean
    assert ci_upper > mean
    assert ci_lower <= 3.0
    assert ci_upper >= 3.0


def test_returns_mean_for_single_value():
    """Return mean for single value without CI range."""
    data = np.array([5.0])
    mean, ci_lower, ci_upper = compute_confidence_interval(data)

    assert mean == 5.0
    assert ci_lower == 5.0
    assert ci_upper == 5.0


def test_uses_t_distribution_for_small_samples():
    """Use t-distribution for small samples (n<30)."""
    data = np.array([10.0, 12.0, 14.0])
    mean, ci_lower, ci_upper = compute_confidence_interval(data)

    expected_mean = 12.0
    expected_se = stats.sem(data)
    expected_margin = expected_se * stats.t.ppf(0.975, 2)

    assert mean == expected_mean
    assert abs(ci_lower - (expected_mean - expected_margin)) < 1e-10
    assert abs(ci_upper - (expected_mean + expected_margin)) < 1e-10


def test_handles_confidence_level_parameter():
    """Handle confidence level parameter - 90% CI narrower than 95% CI."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    mean_95, ci_lower_95, ci_upper_95 = compute_confidence_interval(data, confidence=0.95)
    mean_90, ci_lower_90, ci_upper_90 = compute_confidence_interval(data, confidence=0.90)

    assert mean_95 == mean_90
    assert (ci_upper_90 - ci_lower_90) < (ci_upper_95 - ci_lower_95)


def test_produces_symmetric_ci_for_symmetric_data():
    """Produce symmetric CI for symmetric data."""
    data = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    mean, ci_lower, ci_upper = compute_confidence_interval(data)

    lower_dist = mean - ci_lower
    upper_dist = ci_upper - mean

    assert abs(lower_dist - upper_dist) < 0.0001


def test_handles_larger_samples():
    """Handle larger samples correctly."""
    np.random.seed(42)
    data = np.random.normal(loc=100.0, scale=10.0, size=50)

    mean, ci_lower, ci_upper = compute_confidence_interval(data)

    assert 95.0 < mean < 105.0
    assert ci_lower < mean
    assert ci_upper > mean
    # CI should contain true mean (100) with high probability
    assert ci_lower < 100.0
    assert ci_upper > 100.0


def test_raises_for_empty_array():
    """Raise ValueError for empty array."""
    data = np.array([])
    try:
        compute_confidence_interval(data)
        assert False, "Expected ValueError for empty array"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_handles_array_with_nans_removed():
    """Handle arrays where NaNs have been removed."""
    data = np.array([1.0, 2.0, np.nan, 3.0])
    # Caller is responsible for removing NaNs
    clean_data = data[~np.isnan(data)]
    mean, ci_lower, ci_upper = compute_confidence_interval(clean_data)

    assert mean == 2.0
    assert ci_lower < mean
    assert ci_upper > mean
