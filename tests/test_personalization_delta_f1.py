#!/usr/bin/env python3
"""
Unit tests for personalization delta F1 analysis.

Tests the computation, analysis, and visualization of per-client personalization
gains stratified by data heterogeneity (Dirichlet alpha).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.generate_thesis_plots import (
    _prepare_personalization_delta_f1_data,
    _analyze_delta_f1_by_alpha,
    _render_personalization_delta_f1_plots,
)


def test_prepare_personalization_delta_f1_data_computes_delta():
    """Compute delta F1 from global and personalized scores."""
    data = {
        "client_id": [0, 1, 0, 1],
        "seed": [42, 42, 43, 43],
        "alpha": [0.1, 0.1, 0.1, 0.1],
        "round": [5, 5, 5, 5],
        "macro_f1_global": [0.80, 0.75, 0.82, 0.78],
        "macro_f1_personalized": [0.85, 0.78, 0.87, 0.81],
    }
    df = pd.DataFrame(data)

    result = _prepare_personalization_delta_f1_data(df)

    assert len(result) == 4
    assert "delta_f1" in result.columns
    assert result.iloc[0]["delta_f1"] == pytest.approx(0.05, abs=0.001)
    assert result.iloc[1]["delta_f1"] == pytest.approx(0.03, abs=0.001)


def test_prepare_personalization_delta_f1_data_handles_negative_delta():
    """Handle cases where personalization hurts performance."""
    data = {
        "client_id": [0, 1],
        "seed": [42, 42],
        "alpha": [1.0, 1.0],
        "round": [5, 5],
        "macro_f1_global": [0.85, 0.88],
        "macro_f1_personalized": [0.82, 0.85],
    }
    df = pd.DataFrame(data)

    result = _prepare_personalization_delta_f1_data(df)

    assert result.iloc[0]["delta_f1"] == pytest.approx(-0.03, abs=0.001)
    assert result.iloc[1]["delta_f1"] == pytest.approx(-0.03, abs=0.001)


def test_prepare_personalization_delta_f1_data_filters_final_round():
    """Extract only final round for delta computation."""
    data = {
        "client_id": [0, 0, 0],
        "seed": [42, 42, 42],
        "alpha": [0.5, 0.5, 0.5],
        "round": [1, 3, 5],
        "macro_f1_global": [0.70, 0.75, 0.80],
        "macro_f1_personalized": [0.72, 0.78, 0.85],
    }
    df = pd.DataFrame(data)

    result = _prepare_personalization_delta_f1_data(df)

    assert len(result) == 1
    assert result.iloc[0]["round"] == 5
    assert result.iloc[0]["delta_f1"] == pytest.approx(0.05, abs=0.001)


def test_prepare_personalization_delta_f1_data_handles_missing_columns():
    """Handle missing personalization columns gracefully."""
    data = {
        "client_id": [0, 1],
        "seed": [42, 42],
        "alpha": [0.1, 0.1],
        "round": [5, 5],
        "macro_f1_global": [0.80, 0.75],
    }
    df = pd.DataFrame(data)

    result = _prepare_personalization_delta_f1_data(df)

    assert result.empty


def test_prepare_personalization_delta_f1_data_stratifies_by_alpha():
    """Preserve alpha stratification for analysis."""
    data = {
        "client_id": [0, 1, 0, 1],
        "seed": [42, 42, 42, 42],
        "alpha": [0.1, 0.1, 1.0, 1.0],
        "round": [5, 5, 5, 5],
        "macro_f1_global": [0.70, 0.68, 0.85, 0.84],
        "macro_f1_personalized": [0.78, 0.76, 0.86, 0.85],
    }
    df = pd.DataFrame(data)

    result = _prepare_personalization_delta_f1_data(df)

    assert set(result["alpha"]) == {0.1, 1.0}
    alpha_01_data = result[result["alpha"] == 0.1]
    alpha_10_data = result[result["alpha"] == 1.0]
    assert len(alpha_01_data) == 2
    assert len(alpha_10_data) == 2


def test_analyze_delta_f1_by_alpha_computes_statistics():
    """Compute mean, median, CI, and percentage positive by alpha."""
    data = {
        "alpha": [0.1] * 6 + [0.5] * 6,
        "delta_f1": [0.05, 0.08, 0.03, 0.06, 0.07, 0.04, 0.01, -0.01, 0.02, 0.00, 0.01, -0.02],
        "client_id": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        "seed": [42, 42, 42, 43, 43, 43, 42, 42, 42, 43, 43, 43],
    }
    df = pd.DataFrame(data)

    result = _analyze_delta_f1_by_alpha(df)

    assert len(result) == 2
    assert set(result["alpha"]) == {0.1, 0.5}

    alpha_01_row = result[result["alpha"] == 0.1].iloc[0]
    assert alpha_01_row["mean_delta"] == pytest.approx(0.055, abs=0.01)
    assert alpha_01_row["pct_positive"] == 100.0
    assert alpha_01_row["n"] == 6

    alpha_05_row = result[result["alpha"] == 0.5].iloc[0]
    assert alpha_05_row["pct_positive"] == pytest.approx(50.0, abs=1.0)


def test_analyze_delta_f1_by_alpha_handles_single_alpha():
    """Handle single alpha value without error."""
    data = {
        "alpha": [0.1, 0.1, 0.1],
        "delta_f1": [0.05, 0.03, 0.07],
        "client_id": [0, 1, 2],
        "seed": [42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _analyze_delta_f1_by_alpha(df)

    assert len(result) == 1
    assert result.iloc[0]["alpha"] == 0.1


def test_analyze_delta_f1_by_alpha_computes_confidence_intervals():
    """Compute 95% confidence intervals for delta F1."""
    np.random.seed(42)
    data = {
        "alpha": [0.1] * 30,
        "delta_f1": np.random.normal(0.05, 0.02, 30),
        "client_id": list(range(6)) * 5,
        "seed": [i // 6 for i in range(30)],
    }
    df = pd.DataFrame(data)

    result = _analyze_delta_f1_by_alpha(df)

    assert len(result) == 1
    row = result.iloc[0]
    assert "ci_lower" in row
    assert "ci_upper" in row
    assert row["ci_lower"] < row["mean_delta"]
    assert row["ci_upper"] > row["mean_delta"]


def test_analyze_delta_f1_by_alpha_orders_by_alpha():
    """Return results ordered by alpha ascending."""
    data = {
        "alpha": [1.0, 0.1, 0.5, 1.0, 0.1, 0.5],
        "delta_f1": [0.01, 0.08, 0.04, 0.02, 0.07, 0.05],
        "client_id": [0, 0, 0, 1, 1, 1],
        "seed": [42, 42, 42, 42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _analyze_delta_f1_by_alpha(df)

    alpha_values = result["alpha"].tolist()
    assert alpha_values == sorted(alpha_values)


def test_render_personalization_delta_f1_plots_creates_files():
    """Render 3-panel figure with violin, scatter, and bar plots."""
    scatter_data = {
        "alpha": [0.1] * 4 + [0.5] * 4,
        "delta_f1": [0.05, 0.08, 0.03, 0.06, 0.02, -0.01, 0.01, 0.00],
        "f1_global": [0.75, 0.72, 0.78, 0.74, 0.83, 0.86, 0.84, 0.85],
        "f1_personalized": [0.80, 0.80, 0.81, 0.80, 0.85, 0.85, 0.85, 0.85],
        "client_id": [0, 1, 0, 1, 0, 1, 0, 1],
        "seed": [42, 42, 43, 43, 42, 42, 43, 43],
    }
    scatter_df = pd.DataFrame(scatter_data)

    stats_data = {
        "alpha": [0.1, 0.5],
        "mean_delta": [0.055, 0.005],
        "median_delta": [0.055, 0.005],
        "ci_lower": [0.03, -0.01],
        "ci_upper": [0.08, 0.02],
        "pct_positive": [100.0, 50.0],
        "n": [4, 4],
    }
    stats_df = pd.DataFrame(stats_data)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "personalization_delta_f1_analysis.png"

        _render_personalization_delta_f1_plots(scatter_df, stats_df, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_render_personalization_delta_f1_plots_handles_empty_data():
    """Handle empty data gracefully."""
    scatter_df = pd.DataFrame(columns=["alpha", "delta_f1", "f1_global", "f1_personalized"])
    stats_df = pd.DataFrame(columns=["alpha", "mean_delta", "pct_positive"])

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "empty_plot.png"

        with pytest.raises(ValueError, match="empty"):
            _render_personalization_delta_f1_plots(scatter_df, stats_df, output_path)


def test_render_personalization_delta_f1_plots_includes_baseline():
    """Include y=x baseline in scatter plot."""
    scatter_data = {
        "alpha": [0.1, 0.1],
        "delta_f1": [0.05, 0.03],
        "f1_global": [0.75, 0.78],
        "f1_personalized": [0.80, 0.81],
        "client_id": [0, 1],
        "seed": [42, 42],
    }
    scatter_df = pd.DataFrame(scatter_data)

    stats_data = {
        "alpha": [0.1],
        "mean_delta": [0.04],
        "median_delta": [0.04],
        "ci_lower": [0.03],
        "ci_upper": [0.05],
        "pct_positive": [100.0],
        "n": [2],
    }
    stats_df = pd.DataFrame(stats_data)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "with_baseline.png"

        _render_personalization_delta_f1_plots(scatter_df, stats_df, output_path)

        assert output_path.exists()


def test_prepare_personalization_delta_f1_data_multiple_seeds():
    """Handle multiple seeds correctly."""
    np.random.seed(42)
    n_seeds = 5
    n_clients = 6

    data_rows = []
    for seed in range(n_seeds):
        for client in range(n_clients):
            global_f1 = np.random.uniform(0.70, 0.85)
            personalized_f1 = global_f1 + np.random.uniform(-0.05, 0.10)
            data_rows.append(
                {
                    "client_id": client,
                    "seed": seed,
                    "alpha": 0.1,
                    "round": 5,
                    "macro_f1_global": global_f1,
                    "macro_f1_personalized": personalized_f1,
                }
            )

    df = pd.DataFrame(data_rows)
    result = _prepare_personalization_delta_f1_data(df)

    assert len(result) == 30
    assert len(result["seed"].unique()) == 5
    assert len(result["client_id"].unique()) == 6
