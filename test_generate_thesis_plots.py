#!/usr/bin/env python3
"""Unit tests for generate_thesis_plots.py"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from scripts.generate_thesis_plots import (
    _render_cosine_plot,
    _render_l2_plot,
    _render_macro_f1_plot,
    _render_timing_plot,
    compute_confidence_interval,
    compute_server_macro_f1_from_clients,
    perform_statistical_tests,
    plot_privacy_utility,
)


def test_compute_confidence_interval_basic():
    """Test CI computation with basic dataset."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

    assert mean == 3.0
    assert lower < mean
    assert upper > mean
    assert upper - lower > 0  # CI should have non-zero width


def test_compute_confidence_interval_single_value():
    """Test CI computation with single value (edge case)."""
    data = np.array([5.0])
    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

    # With single value, CI is undefined but function should not crash
    assert mean == 5.0
    # CI will be NaN or inf for single value - just check it doesn't crash


def test_compute_server_macro_f1_from_clients():
    """Test server-level macro-F1 aggregation from client CSVs."""
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create mock client CSV files
        for client_id in range(3):
            client_data = pd.DataFrame(
                {
                    "client_id": [client_id] * 5,
                    "round": [0, 1, 2, 3, 4],
                    "macro_f1_after": [0.8, 0.82, 0.85, 0.87, 0.9],
                    "macro_f1_argmax": [0.79, 0.81, 0.84, 0.86, 0.89],
                }
            )
            client_data.to_csv(run_dir / f"client_{client_id}_metrics.csv", index=False)

        # Test round 0
        f1 = compute_server_macro_f1_from_clients(run_dir, 0)
        assert f1 is not None
        assert 0.75 <= f1 <= 0.85  # Should be around 0.8

        # Test round 4
        f1 = compute_server_macro_f1_from_clients(run_dir, 4)
        assert f1 is not None
        assert 0.85 <= f1 <= 0.95  # Should be around 0.9

        # Test non-existent round
        f1 = compute_server_macro_f1_from_clients(run_dir, 999)
        assert f1 is None


def test_compute_server_macro_f1_missing_data():
    """Test macro-F1 computation when no client CSVs exist."""
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        f1 = compute_server_macro_f1_from_clients(run_dir, 0)
        assert f1 is None


def test_l2_spurious_zeros_detection():
    """Test detection of spurious L2 zeros (regression test for median=0 artifact)."""
    # Create dataset with suspiciously many zeros
    l2_data = pd.Series([0.0, 0.0, 0.0, 0.001, 0.002])
    zero_count = (l2_data == 0.0).sum()

    # Should detect that >50% are zeros
    assert zero_count > len(l2_data) * 0.5

    # Create dataset with normal small values
    l2_data_normal = pd.Series([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    zero_count_normal = (l2_data_normal == 0.0).sum()

    # Should NOT detect zeros in normal data
    assert zero_count_normal == 0


def test_perform_statistical_tests_ttest():
    """Test t-test for 2 groups."""
    df = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    result = perform_statistical_tests(df, "group", "value")

    assert result["test"] == "t_test"
    assert "p_value" in result
    assert "statistic" in result
    assert result["p_value"] < 0.05  # Groups are significantly different


def test_perform_statistical_tests_anova():
    """Test ANOVA for >2 groups."""
    df = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C"], "value": [1.0, 2.0, 4.0, 5.0, 7.0, 8.0]})

    result = perform_statistical_tests(df, "group", "value")

    assert result["test"] == "anova"
    assert "p_value" in result
    assert "statistic" in result
    assert "pairwise" in result
    assert len(result["pairwise"]) == 3  # 3 pairwise comparisons for 3 groups


def test_perform_statistical_tests_insufficient_data():
    """Test statistical tests with insufficient data."""
    df = pd.DataFrame({"group": ["A"], "value": [1.0]})

    result = perform_statistical_tests(df, "group", "value")

    assert result["test"] == "insufficient_data"
    assert result["p_value"] is None


def test_render_macro_f1_plot_success():
    """Test macro-F1 renderer with valid data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum"],
            "seed": [1, 2, 1, 2],
            "macro_f1": [0.85, 0.87, 0.90, 0.92],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    assert len(ax.patches) > 0  # Bars were drawn
    assert ax.get_title() == "Detection Performance (Macro-F1, 95% CI)"
    plt.close(fig)


def test_render_macro_f1_plot_missing_column():
    """Test macro-F1 renderer with missing data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame({"aggregation": ["fedavg"], "seed": [1]})
    fig, ax = plt.subplots()

    result = _render_macro_f1_plot(ax, final_rounds, ["fedavg"])

    assert result is False
    assert len(ax.patches) == 0
    plt.close(fig)


def test_render_timing_plot_success():
    """Test timing renderer with valid data."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum"],
            "seed": [1, 2, 1, 2],
            "t_aggregate_ms": [10.0, 12.0, 15.0, 17.0],
        }
    )
    fig, ax = plt.subplots()

    result = _render_timing_plot(ax, df, ["fedavg", "krum"])

    assert result is True
    assert len(ax.patches) > 0
    assert ax.get_title() == "Aggregation Time (95% CI)"
    plt.close(fig)


def test_render_timing_plot_missing_column():
    """Test timing renderer with missing data."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"aggregation": ["fedavg"], "seed": [1]})
    fig, ax = plt.subplots()

    result = _render_timing_plot(ax, df, ["fedavg"])

    assert result is False
    plt.close(fig)


def test_render_l2_plot_success():
    """Test L2 renderer with valid data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "krum", "median"],
            "seed": [1, 1, 1],
            "l2_to_benign_mean": [0.001, 0.002, 0.0015],
        }
    )
    fig, ax = plt.subplots()

    result = _render_l2_plot(ax, final_rounds, ["fedavg", "krum", "median"])

    assert result is True
    assert ax.get_title() == "Model Drift (L2 Distance)"
    plt.close(fig)


def test_render_cosine_plot_success():
    """Test cosine renderer with valid data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "krum"],
            "seed": [1, 1],
            "cos_to_benign_mean": [0.99, 0.98],
        }
    )
    fig, ax = plt.subplots()

    result = _render_cosine_plot(ax, final_rounds, ["fedavg", "krum"])

    assert result is True
    assert ax.get_title() == "Model Alignment (Cosine Similarity)"
    plt.close(fig)


def _write_config(path: Path, seed: int, dp_enabled: bool, dp_noise: float) -> None:
    config = {
        "aggregation": "fedavg",
        "alpha": 0.5,
        "adversary_fraction": 0.0,
        "dp_enabled": dp_enabled,
        "dp_noise_multiplier": dp_noise,
        "personalization_epochs": 0,
        "num_clients": 2,
        "num_rounds": 3,
        "seed": seed,
    }
    path.write_text(json.dumps(config))


@pytest.mark.parametrize("dp_noise", [0.7])
def test_privacy_utility_curve_outputs(tmp_path, dp_noise):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    dp_run_seed42 = runs_dir / "comp_fedavg_alpha0.5_dp1_seed42"
    dp_run_seed43 = runs_dir / "comp_fedavg_alpha0.5_dp1_seed43"
    baseline_run = runs_dir / "comp_fedavg_alpha0.5_dp0_seed42"

    for run_dir in [dp_run_seed42, dp_run_seed43, baseline_run]:
        run_dir.mkdir()

    _write_config(dp_run_seed42 / "config.json", seed=42, dp_enabled=True, dp_noise=dp_noise)
    _write_config(dp_run_seed43 / "config.json", seed=43, dp_enabled=True, dp_noise=dp_noise)
    _write_config(baseline_run / "config.json", seed=42, dp_enabled=False, dp_noise=0.0)

    metrics_df = pd.DataFrame(
        [
            {"round": 1, "dp_enabled": True, "dp_noise_multiplier": dp_noise, "seed": 42},
            {"round": 3, "dp_enabled": True, "dp_noise_multiplier": dp_noise, "seed": 42},
        ]
    )
    metrics_df.to_csv(dp_run_seed42 / "metrics.csv", index=False)
    metrics_df.assign(seed=43).to_csv(dp_run_seed43 / "metrics.csv", index=False)
    metrics_df.assign(dp_enabled=False, dp_noise_multiplier=0.0).to_csv(baseline_run / "metrics.csv", index=False)

    for run_dir, epsilon, f1_values in [
        (dp_run_seed42, 1.5, [0.82, 0.80]),
        (dp_run_seed43, 1.5, [0.78, 0.76]),
        (baseline_run, float("nan"), [0.90, 0.88]),
    ]:
        for idx, f1 in enumerate(f1_values):
            client_df = pd.DataFrame(
                [
                    {"round": 1, "macro_f1_after": f1 - 0.05, "dp_epsilon": epsilon / 3},
                    {"round": 3, "macro_f1_after": f1, "dp_epsilon": epsilon},
                ]
            )
            (run_dir / f"client_{idx}_metrics.csv").write_text(client_df.to_csv(index=False))

    df = pd.DataFrame(
        [
            {
                "dp_enabled": True,
                "dp_noise_multiplier": dp_noise,
                "seed": 42,
                "round": 3,
                "run_dir": str(dp_run_seed42),
            },
            {
                "dp_enabled": True,
                "dp_noise_multiplier": dp_noise,
                "seed": 43,
                "round": 3,
                "run_dir": str(dp_run_seed43),
            },
            {
                "dp_enabled": False,
                "dp_noise_multiplier": 0.0,
                "seed": 42,
                "round": 3,
                "run_dir": str(baseline_run),
            },
        ]
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    plot_privacy_utility(df, output_dir, runs_dir=runs_dir)

    curve_path = output_dir / "privacy_utility_curve.png"
    summary_path = output_dir / "privacy_utility_curve.csv"

    assert curve_path.exists()
    assert summary_path.exists()

    summary_df = pd.read_csv(summary_path)

    assert set(summary_df["is_baseline"].unique()) == {0, 1}

    dp_row = summary_df.loc[summary_df["is_baseline"] == 0].iloc[0]
    baseline_row = summary_df.loc[summary_df["is_baseline"] == 1].iloc[0]

    assert pytest.approx(dp_row["epsilon"], rel=1e-3) == 1.5
    assert dp_row["n"] == 2
    assert pytest.approx(dp_row["macro_f1_mean"], rel=1e-3) == 0.79

    assert pd.isna(baseline_row["epsilon"])
    assert baseline_row["n"] == 1
    assert pytest.approx(baseline_row["macro_f1_mean"], rel=1e-3) == 0.89


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
