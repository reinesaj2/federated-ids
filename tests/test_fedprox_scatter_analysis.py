#!/usr/bin/env python3
"""
Unit tests for FedProx per-client scatter visualization.

Tests the preparation, aggregation, and rendering of per-client scatter plots
showing individual client drift at each FedProx mu value with global mean overlay.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.generate_thesis_plots import (
    _prepare_client_scatter_data,
    _compute_global_mean_by_mu,
    _render_client_scatter_mu_plot,
)


def test_prepare_client_scatter_data_extracts_final_round():
    """Extract per-client scatter points from final round only."""
    data = {
        "round": [1, 1, 2, 2, 3, 3],
        "client_id": [0, 1, 0, 1, 0, 1],
        "fedprox_mu": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "l2_to_benign_mean": [0.5, 0.6, 0.4, 0.5, 0.3, 0.4],
        "seed": [42, 42, 42, 42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _prepare_client_scatter_data(df, metric="l2_to_benign_mean")

    assert len(result) == 2
    assert result["round"].unique() == [3]
    assert set(result["client_id"]) == {0, 1}
    assert "jitter" in result.columns


def test_prepare_client_scatter_data_groups_by_mu():
    """Group scatter points by mu value across clients and seeds."""
    data = {
        "round": [5] * 8,
        "client_id": [0, 1, 0, 1, 0, 1, 0, 1],
        "fedprox_mu": [0.01, 0.01, 0.05, 0.05, 0.01, 0.01, 0.05, 0.05],
        "l2_to_benign_mean": [0.3, 0.4, 0.2, 0.3, 0.35, 0.45, 0.25, 0.35],
        "seed": [42, 42, 42, 42, 43, 43, 43, 43],
    }
    df = pd.DataFrame(data)

    result = _prepare_client_scatter_data(df, metric="l2_to_benign_mean")

    mu_values = sorted(result["fedprox_mu"].unique())
    assert mu_values == [0.01, 0.05]

    mu_01_points = result[result["fedprox_mu"] == 0.01]
    assert len(mu_01_points) == 4


def test_prepare_client_scatter_data_adds_jitter():
    """Add jitter to x-axis positions for visibility."""
    data = {
        "round": [5] * 4,
        "client_id": [0, 1, 2, 3],
        "fedprox_mu": [0.05, 0.05, 0.05, 0.05],
        "l2_to_benign_mean": [0.3, 0.4, 0.35, 0.38],
        "seed": [42, 42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _prepare_client_scatter_data(df, metric="l2_to_benign_mean")

    assert "jitter" in result.columns
    assert result["jitter"].abs().max() <= 0.02 * 0.05
    assert (result["fedprox_mu"] + result["jitter"]).min() > 0


def test_prepare_client_scatter_data_handles_missing_metric():
    """Handle missing metric values gracefully."""
    data = {
        "round": [5, 5, 5],
        "client_id": [0, 1, 2],
        "fedprox_mu": [0.01, 0.01, 0.01],
        "l2_to_benign_mean": [0.3, np.nan, 0.4],
        "seed": [42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _prepare_client_scatter_data(df, metric="l2_to_benign_mean")

    assert len(result) == 2
    assert not result["l2_to_benign_mean"].isna().any()


def test_compute_global_mean_by_mu_aggregates_across_clients():
    """Compute global mean and CI aggregated across clients and seeds."""
    data = {
        "fedprox_mu": [0.01] * 4 + [0.05] * 4,
        "l2_to_benign_mean": [0.3, 0.4, 0.35, 0.38, 0.2, 0.25, 0.22, 0.24],
        "client_id": [0, 1, 0, 1, 0, 1, 0, 1],
        "seed": [42, 42, 43, 43, 42, 42, 43, 43],
    }
    df = pd.DataFrame(data)

    result = _compute_global_mean_by_mu(df, metric="l2_to_benign_mean")

    assert len(result) == 2
    assert set(result["mu"]) == {0.01, 0.05}
    assert "mean" in result.columns
    assert "ci_lower" in result.columns
    assert "ci_upper" in result.columns

    mu_01_row = result[result["mu"] == 0.01].iloc[0]
    expected_mean = np.mean([0.3, 0.4, 0.35, 0.38])
    assert abs(mu_01_row["mean"] - expected_mean) < 0.01


def test_compute_global_mean_by_mu_handles_single_value():
    """Handle single value per mu (no CI possible)."""
    data = {
        "fedprox_mu": [0.01, 0.05],
        "l2_to_benign_mean": [0.3, 0.2],
        "client_id": [0, 0],
        "seed": [42, 42],
    }
    df = pd.DataFrame(data)

    result = _compute_global_mean_by_mu(df, metric="l2_to_benign_mean")

    assert len(result) == 2
    mu_01_row = result[result["mu"] == 0.01].iloc[0]
    assert mu_01_row["mean"] == 0.3
    assert mu_01_row["ci_lower"] == 0.3
    assert mu_01_row["ci_upper"] == 0.3


def test_compute_global_mean_by_mu_orders_by_mu():
    """Return results ordered by mu value ascending."""
    data = {
        "fedprox_mu": [0.1, 0.01, 0.05, 0.1, 0.01, 0.05],
        "l2_to_benign_mean": [0.15, 0.35, 0.25, 0.18, 0.38, 0.28],
        "client_id": [0, 0, 0, 1, 1, 1],
        "seed": [42, 42, 42, 42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _compute_global_mean_by_mu(df, metric="l2_to_benign_mean")

    mu_values = result["mu"].tolist()
    assert mu_values == sorted(mu_values)


def test_render_client_scatter_mu_plot_creates_file():
    """Render scatter plot with global mean overlay and save to file."""
    scatter_data = {
        "fedprox_mu": [0.01] * 4 + [0.05] * 4,
        "l2_to_benign_mean": [0.3, 0.4, 0.35, 0.38, 0.2, 0.25, 0.22, 0.24],
        "jitter": [0.0001, -0.0001, 0.0002, -0.0002, 0.0005, -0.0005, 0.0003, -0.0003],
        "client_id": [0, 1, 0, 1, 0, 1, 0, 1],
        "seed": [42, 42, 43, 43, 42, 42, 43, 43],
    }
    scatter_df = pd.DataFrame(scatter_data)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "fedprox_scatter_l2.png"

        _render_client_scatter_mu_plot(
            scatter_df=scatter_df,
            metric="l2_to_benign_mean",
            output_path=output_path,
            use_log_y=False,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_render_client_scatter_mu_plot_supports_log_scale():
    """Render scatter plot with log y-axis for wide spreads."""
    scatter_data = {
        "fedprox_mu": [0.01, 0.01, 0.05, 0.05],
        "l2_to_benign_mean": [0.01, 10.0, 0.05, 5.0],
        "jitter": [0.0, 0.0, 0.0, 0.0],
        "client_id": [0, 1, 0, 1],
        "seed": [42, 42, 42, 42],
    }
    scatter_df = pd.DataFrame(scatter_data)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "fedprox_scatter_l2_log.png"

        _render_client_scatter_mu_plot(
            scatter_df=scatter_df,
            metric="l2_to_benign_mean",
            output_path=output_path,
            use_log_y=True,
        )

        assert output_path.exists()


def test_render_client_scatter_mu_plot_handles_cosine_similarity():
    """Render scatter plot for cosine similarity metric."""
    scatter_data = {
        "fedprox_mu": [0.01] * 4,
        "cos_to_benign_mean": [0.95, 0.92, 0.94, 0.93],
        "jitter": [0.0001, -0.0001, 0.0002, -0.0002],
        "client_id": [0, 1, 0, 1],
        "seed": [42, 42, 43, 43],
    }
    scatter_df = pd.DataFrame(scatter_data)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "fedprox_scatter_cosine.png"

        _render_client_scatter_mu_plot(
            scatter_df=scatter_df,
            metric="cos_to_benign_mean",
            output_path=output_path,
            use_log_y=False,
        )

        assert output_path.exists()


def test_render_client_scatter_mu_plot_empty_data():
    """Handle empty scatter data gracefully."""
    scatter_df = pd.DataFrame(columns=["fedprox_mu", "l2_to_benign_mean", "jitter", "client_id", "seed"])

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "fedprox_scatter_empty.png"

        with pytest.raises(ValueError, match="empty"):
            _render_client_scatter_mu_plot(
                scatter_df=scatter_df,
                metric="l2_to_benign_mean",
                output_path=output_path,
                use_log_y=False,
            )


def test_prepare_client_scatter_data_multiple_alphas():
    """Handle multiple alpha values in data."""
    data = {
        "round": [5] * 8,
        "client_id": [0, 1, 0, 1, 0, 1, 0, 1],
        "alpha": [0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5],
        "fedprox_mu": [0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05],
        "l2_to_benign_mean": [0.5, 0.6, 0.3, 0.4, 0.4, 0.5, 0.2, 0.3],
        "seed": [42, 42, 42, 42, 42, 42, 42, 42],
    }
    df = pd.DataFrame(data)

    result = _prepare_client_scatter_data(df, metric="l2_to_benign_mean")

    assert "alpha" in result.columns
    assert set(result["alpha"]) == {0.1, 0.5}


def test_compute_global_mean_by_mu_with_many_seeds():
    """Compute robust CI with many seeds."""
    np.random.seed(42)
    n_seeds = 10
    n_clients = 6
    mu_values = [0.01, 0.05, 0.1]

    data_rows = []
    for mu in mu_values:
        for seed in range(n_seeds):
            for client in range(n_clients):
                data_rows.append(
                    {
                        "fedprox_mu": mu,
                        "l2_to_benign_mean": np.random.uniform(0.2, 0.5),
                        "client_id": client,
                        "seed": seed,
                    }
                )
    df = pd.DataFrame(data_rows)

    result = _compute_global_mean_by_mu(df, metric="l2_to_benign_mean")

    assert len(result) == 3
    for _, row in result.iterrows():
        assert row["ci_upper"] > row["mean"]
        assert row["ci_lower"] < row["mean"]
        assert row["ci_upper"] - row["ci_lower"] < 0.5
