#!/usr/bin/env python3
"""Integration tests for DP privacy accounting in thesis plots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from privacy_accounting import compute_epsilon
from scripts.generate_thesis_plots import (
    _compute_epsilon_fallback,
    _prepare_privacy_curve_data,
    _render_privacy_curve,
)


def test_compute_epsilon_basic():
    """Test epsilon computation with typical FL parameters."""
    noise = 1.0
    delta = 1e-5
    steps = 20
    sample_rate = 1.0

    epsilon = compute_epsilon(noise, delta, steps, sample_rate)

    assert isinstance(epsilon, float)
    assert epsilon > 0.0
    assert epsilon < 100.0
    assert np.isfinite(epsilon)


def test_compute_epsilon_zero_noise():
    """Test that zero noise returns infinity (no privacy)."""
    epsilon = compute_epsilon(noise_multiplier=0.0, delta=1e-5, num_steps=10, sample_rate=1.0)

    assert epsilon == float("inf")


def test_compute_epsilon_high_noise():
    """Test that higher noise reduces epsilon (stronger privacy)."""
    epsilon_low = compute_epsilon(noise_multiplier=0.5, delta=1e-5, num_steps=10, sample_rate=1.0)
    epsilon_high = compute_epsilon(noise_multiplier=2.0, delta=1e-5, num_steps=10, sample_rate=1.0)

    assert epsilon_low > epsilon_high


def test_compute_epsilon_more_steps_increases_epsilon():
    """Test that more training steps increase epsilon (more privacy loss)."""
    epsilon_few = compute_epsilon(noise_multiplier=1.0, delta=1e-5, num_steps=5, sample_rate=1.0)
    epsilon_many = compute_epsilon(noise_multiplier=1.0, delta=1e-5, num_steps=50, sample_rate=1.0)

    assert epsilon_many > epsilon_few


def test_epsilon_fallback_with_valid_params():
    """Test fallback epsilon computation from config dict."""
    row = {
        "dp_noise_multiplier": 1.0,
        "dp_delta": 1e-5,
        "round": 20,
        "dp_sample_rate": 1.0,
    }
    final_row = pd.Series({"dp_sigma": 1.0, "dp_delta": 1e-5, "round": 20})

    epsilon = _compute_epsilon_fallback(row, final_row)

    assert epsilon is not None
    assert epsilon > 0.0
    assert np.isfinite(epsilon)


def test_epsilon_fallback_missing_noise():
    """Test fallback returns None when noise is missing."""
    row = {"dp_delta": 1e-5, "round": 20}
    final_row = None

    epsilon = _compute_epsilon_fallback(row, final_row)

    assert epsilon is None


def test_epsilon_fallback_zero_steps():
    """Test fallback returns None when steps is zero."""
    row = {"dp_noise_multiplier": 1.0, "dp_delta": 1e-5, "round": 0}
    final_row = None

    epsilon = _compute_epsilon_fallback(row, final_row)

    assert epsilon is None


def test_prepare_privacy_curve_data_with_dp():
    """Test privacy curve data preparation with DP experiments."""
    with TemporaryDirectory() as tmpdir:
        runs_root = Path(tmpdir)

        dp_run = runs_root / "comp_fedavg_dp1_seed42"
        dp_run.mkdir()

        config = {
            "dp_enabled": True,
            "dp_noise_multiplier": 1.0,
            "dp_delta": 1e-5,
            "round": 10,
            "seed": 42,
        }
        (dp_run / "config.json").write_text(json.dumps(config))

        client_df = pd.DataFrame(
            {
                "round": [10],
                "macro_f1_after": [0.85],
                "dp_epsilon": [3.5],
                "dp_sigma": [1.0],
            }
        )
        client_df.to_csv(dp_run / "client_0_metrics.csv", index=False)

        final_rounds = pd.DataFrame(
            {
                "run_dir": [str(dp_run)],
                "dp_enabled": [True],
                "dp_noise_multiplier": [1.0],
                "seed": [42],
            }
        )

        dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)

        assert not dp_df.empty
        assert "epsilon" in dp_df.columns
        assert "macro_f1" in dp_df.columns
        assert dp_df["epsilon"].iloc[0] > 0.0
        assert 0.0 < dp_df["macro_f1"].iloc[0] < 1.0


def test_prepare_privacy_curve_data_with_baseline():
    """Test privacy curve data includes baseline (no DP) experiments."""
    with TemporaryDirectory() as tmpdir:
        runs_root = Path(tmpdir)

        baseline_run = runs_root / "comp_fedavg_dp0_seed42"
        baseline_run.mkdir()

        config = {
            "dp_enabled": False,
            "dp_noise_multiplier": 0.0,
            "seed": 42,
        }
        (baseline_run / "config.json").write_text(json.dumps(config))

        client_df = pd.DataFrame({"round": [10], "macro_f1_after": [0.90]})
        client_df.to_csv(baseline_run / "client_0_metrics.csv", index=False)

        final_rounds = pd.DataFrame(
            {
                "run_dir": [str(baseline_run)],
                "dp_enabled": [False],
                "dp_noise_multiplier": [0.0],
                "seed": [42],
            }
        )

        dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)

        assert not baseline_df.empty
        assert "macro_f1" in baseline_df.columns
        assert baseline_df["macro_f1"].iloc[0] > 0.0


def test_render_privacy_curve_creates_csv():
    """Test that privacy curve rendering creates summary CSV."""
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        dp_df = pd.DataFrame(
            {
                "epsilon": [1.0, 2.0, 5.0],
                "macro_f1": [0.85, 0.88, 0.90],
                "seed": [42, 42, 42],
                "dp_noise_multiplier": [2.0, 1.5, 0.8],
            }
        )

        baseline_df = pd.DataFrame(
            {
                "macro_f1": [0.92, 0.93],
                "seed": [42, 43],
                "dp_noise_multiplier": [0.0, 0.0],
            }
        )

        _render_privacy_curve(dp_df, baseline_df, output_dir)

        csv_path = output_dir / "privacy_utility_curve.csv"
        assert csv_path.exists()

        summary_df = pd.read_csv(csv_path)
        assert "epsilon" in summary_df.columns
        assert "macro_f1_mean" in summary_df.columns
        assert "ci_lower" in summary_df.columns
        assert "ci_upper" in summary_df.columns
        assert "n" in summary_df.columns
        assert "is_baseline" in summary_df.columns

        baseline_rows = summary_df[summary_df["is_baseline"] == 1]
        assert len(baseline_rows) == 1


def test_render_privacy_curve_creates_plot():
    """Test that privacy curve rendering creates PNG plot."""
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        dp_df = pd.DataFrame(
            {
                "epsilon": [1.0, 2.0],
                "macro_f1": [0.85, 0.88],
                "seed": [42, 42],
                "dp_noise_multiplier": [1.5, 1.0],
            }
        )

        baseline_df = pd.DataFrame()

        _render_privacy_curve(dp_df, baseline_df, output_dir)

        plot_path = output_dir / "privacy_utility_curve.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


def test_render_privacy_curve_empty_data():
    """Test that empty DP data does not crash renderer."""
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        dp_df = pd.DataFrame()
        baseline_df = pd.DataFrame()

        _render_privacy_curve(dp_df, baseline_df, output_dir)

        csv_path = output_dir / "privacy_utility_curve.csv"
        plot_path = output_dir / "privacy_utility_curve.png"

        assert not csv_path.exists()
        assert not plot_path.exists()


def test_privacy_curve_with_multiple_seeds():
    """Test privacy curve aggregates across multiple seeds with CIs."""
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        dp_df = pd.DataFrame(
            {
                "epsilon": [2.0, 2.0, 2.0],
                "macro_f1": [0.84, 0.86, 0.85],
                "seed": [42, 43, 44],
                "dp_noise_multiplier": [1.0, 1.0, 1.0],
            }
        )

        baseline_df = pd.DataFrame()

        _render_privacy_curve(dp_df, baseline_df, output_dir)

        csv_path = output_dir / "privacy_utility_curve.csv"
        assert csv_path.exists()

        summary_df = pd.read_csv(csv_path)
        epsilon_2_row = summary_df[summary_df["epsilon"] == 2.0]

        assert len(epsilon_2_row) == 1
        assert epsilon_2_row["n"].iloc[0] == 3
        assert epsilon_2_row["macro_f1_mean"].iloc[0] == pytest.approx(0.85, abs=0.01)
        assert epsilon_2_row["ci_lower"].iloc[0] < 0.85
        assert epsilon_2_row["ci_upper"].iloc[0] > 0.85


def test_privacy_utility_consistency_check():
    """Test that epsilon increases as noise decreases (inverse relationship)."""
    epsilons = []
    noise_values = [2.0, 1.5, 1.0, 0.5]

    for noise in noise_values:
        epsilon = compute_epsilon(noise_multiplier=noise, delta=1e-5, num_steps=10, sample_rate=1.0)
        epsilons.append(epsilon)

    for i in range(len(epsilons) - 1):
        assert epsilons[i] < epsilons[i + 1], f"Epsilon should increase as noise decreases: {epsilons}"
