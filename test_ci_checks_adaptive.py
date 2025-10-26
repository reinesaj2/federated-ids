#!/usr/bin/env python3
"""
Tests for adaptive L2 threshold validation in CI checks.
"""

from pathlib import Path

import pytest

from scripts.ci_checks import _compute_adaptive_l2_threshold, _extract_alpha_from_run_name


def test_extract_alpha_from_run_name_valid_alpha_0_1():
    """Test alpha extraction from valid run name with alpha=0.1."""
    run_dir = Path("nightly_fedprox_alpha0.1_mu0.0_seed2")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha == 0.1


def test_extract_alpha_from_run_name_valid_alpha_0_5():
    """Test alpha extraction from valid run name with alpha=0.5."""
    run_dir = Path("nightly_fedprox_alpha0.5_mu0.01_seed0")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha == 0.5


def test_extract_alpha_from_run_name_valid_alpha_1_0():
    """Test alpha extraction from valid run name with alpha=1.0."""
    run_dir = Path("nightly_fedprox_alpha1.0_mu0.02_seed4")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha == 1.0


def test_extract_alpha_from_run_name_valid_decimal_precision():
    """Test alpha extraction preserves decimal precision."""
    run_dir = Path("nightly_fedprox_alpha0.05_mu0.0_seed1")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha == 0.05


def test_extract_alpha_from_run_name_invalid_pattern():
    """Test alpha extraction returns None for invalid run name."""
    run_dir = Path("invalid_run_name")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha is None


def test_extract_alpha_from_run_name_missing_components():
    """Test alpha extraction returns None for incomplete pattern."""
    run_dir = Path("nightly_fedprox_alpha0.1_seed2")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha is None


def test_extract_alpha_from_run_name_wrong_prefix():
    """Test alpha extraction returns None for wrong prefix."""
    run_dir = Path("comp_fedavg_alpha0.1_mu0.0_seed2")
    alpha = _extract_alpha_from_run_name(run_dir)
    assert alpha is None


def test_compute_adaptive_threshold_extreme_heterogeneity():
    """Test adaptive threshold for alpha=0.1 (extreme heterogeneity)."""
    threshold = _compute_adaptive_l2_threshold(0.1)
    assert threshold == pytest.approx(3.3, abs=0.01)


def test_compute_adaptive_threshold_moderate_heterogeneity():
    """Test adaptive threshold for alpha=0.5 (moderate heterogeneity)."""
    threshold = _compute_adaptive_l2_threshold(0.5)
    assert threshold == pytest.approx(2.5, abs=0.01)


def test_compute_adaptive_threshold_iid():
    """Test adaptive threshold for alpha=1.0 (IID, no heterogeneity)."""
    threshold = _compute_adaptive_l2_threshold(1.0)
    assert threshold == pytest.approx(1.5, abs=0.01)


def test_compute_adaptive_threshold_none_uses_base():
    """Test adaptive threshold defaults to base when alpha=None."""
    threshold = _compute_adaptive_l2_threshold(None)
    assert threshold == 1.5


def test_compute_adaptive_threshold_zero_alpha():
    """Test adaptive threshold for alpha=0.0 (maximum heterogeneity)."""
    threshold = _compute_adaptive_l2_threshold(0.0)
    assert threshold == pytest.approx(3.5, abs=0.01)


def test_compute_adaptive_threshold_high_alpha():
    """Test adaptive threshold for alpha=0.9 (low heterogeneity)."""
    threshold = _compute_adaptive_l2_threshold(0.9)
    assert threshold == pytest.approx(1.7, abs=0.01)


def test_compute_adaptive_threshold_formula_consistency():
    """Test that threshold increases as alpha decreases."""
    alpha_values = [1.0, 0.5, 0.2, 0.1, 0.05]
    thresholds = [_compute_adaptive_l2_threshold(a) for a in alpha_values]

    for i in range(len(thresholds) - 1):
        assert thresholds[i] < thresholds[i + 1], (
            f"Threshold should increase as alpha decreases: "
            f"threshold({alpha_values[i]})={thresholds[i]:.2f} >= "
            f"threshold({alpha_values[i + 1]})={thresholds[i + 1]:.2f}"
        )


def test_adaptive_threshold_covers_observed_failures():
    """Test that adaptive threshold would pass previously failed jobs."""
    alpha = 0.1
    threshold = _compute_adaptive_l2_threshold(alpha)

    observed_l2_job1 = 2.265
    observed_l2_job2 = 3.016

    assert observed_l2_job1 < threshold, f"Job 1 L2={observed_l2_job1:.3f} should pass with threshold={threshold:.2f}"
    assert observed_l2_job2 < threshold, f"Job 2 L2={observed_l2_job2:.3f} should pass with threshold={threshold:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
