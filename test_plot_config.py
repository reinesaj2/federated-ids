#!/usr/bin/env python3
"""
Tests for plot configuration layer.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.plot_config import (
    PALETTES,
    ConfidenceIntervalConfig,
    LayoutConfig,
    MetricDetector,
    PlotStyle,
    SmoothingConfig,
)


def test_plot_style_get_colors_exact():
    """Test getting exact number of colors from palette."""
    style = PlotStyle(palette="colorblind")
    colors = style.get_colors(3)

    assert len(colors) == 3
    assert all(isinstance(c, str) for c in colors)
    assert all(c.startswith("#") for c in colors)


def test_plot_style_get_colors_extended():
    """Test color extension when requesting more than available."""
    style = PlotStyle(palette="default")
    palette_size = len(PALETTES["default"])
    colors = style.get_colors(palette_size + 3)

    assert len(colors) == palette_size + 3


def test_layout_config_compute_grid_auto():
    """Test automatic grid computation for various plot counts."""
    layout = LayoutConfig()

    assert layout.compute_grid(1) == (1, 1)
    assert layout.compute_grid(2) == (1, 2)
    assert layout.compute_grid(3) == (2, 2)
    assert layout.compute_grid(4) == (2, 2)
    assert layout.compute_grid(6) == (2, 3)
    assert layout.compute_grid(7) == (3, 3)
    assert layout.compute_grid(9) == (3, 3)


def test_layout_config_compute_grid_fixed_rows():
    """Test grid computation with fixed rows."""
    layout = LayoutConfig(rows=2)

    assert layout.compute_grid(3) == (2, 2)
    assert layout.compute_grid(5) == (2, 3)


def test_layout_config_compute_grid_fixed_cols():
    """Test grid computation with fixed columns."""
    layout = LayoutConfig(cols=3)

    assert layout.compute_grid(5) == (2, 3)
    assert layout.compute_grid(7) == (3, 3)


def test_layout_config_compute_grid_fixed_both():
    """Test grid computation with both dimensions fixed."""
    layout = LayoutConfig(rows=3, cols=2)

    assert layout.compute_grid(5) == (3, 2)
    assert layout.compute_grid(10) == (3, 2)


def test_metric_detector_server_all_available():
    """Test detection of all server metrics when available."""
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "t_aggregate_ms": [100, 110, 105],
            "l2_to_benign_mean": [0.5, 0.4, 0.3],
            "cos_to_benign_mean": [0.8, 0.85, 0.9],
            "update_norm_mean": [1.0, 1.1, 1.2],
            "pairwise_cosine_mean": [0.7, 0.75, 0.8],
            "l2_dispersion_mean": [0.3, 0.25, 0.2],
        }
    )

    detector = MetricDetector()
    available = detector.detect_available(df, metric_type="server")

    assert available["timing"] == "t_aggregate_ms"
    assert available["robustness"] == "l2_to_benign_mean"
    assert available["norms"] == "update_norm_mean"
    assert available["dispersion"] == "pairwise_cosine_mean"


def test_metric_detector_server_partial_available():
    """Test detection with only some metrics available."""
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "t_aggregate_ms": [100, 110, 105],
            "update_norm_mean": [1.0, 1.1, 1.2],
        }
    )

    detector = MetricDetector()
    available = detector.detect_available(df, metric_type="server")

    assert available["timing"] == "t_aggregate_ms"
    assert available["robustness"] is None
    assert available["norms"] == "update_norm_mean"
    assert available["dispersion"] is None


def test_metric_detector_server_alternate_columns():
    """Test detection using alternate column names."""
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "aggregation_time_ms": [100, 110, 105],  # Alternate name
            "update_norm_std": [0.1, 0.2, 0.15],  # Alternate name
        }
    )

    detector = MetricDetector()
    available = detector.detect_available(df, metric_type="server")

    assert available["timing"] == "aggregation_time_ms"
    assert available["norms"] == "update_norm_std"


def test_metric_detector_server_na_columns():
    """Test that columns with all NaN values are not detected."""
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "t_aggregate_ms": [np.nan, np.nan, np.nan],
            "update_norm_mean": [1.0, 1.1, 1.2],
        }
    )

    detector = MetricDetector()
    available = detector.detect_available(df, metric_type="server")

    assert available["timing"] is None  # All NaN
    assert available["norms"] == "update_norm_mean"


def test_metric_detector_count_available_plots():
    """Test counting available plots."""
    detector = MetricDetector()

    available = {
        "timing": "t_aggregate_ms",
        "robustness": None,
        "norms": "update_norm_mean",
        "dispersion": "pairwise_cosine_mean",
    }

    count = detector.count_available_plots(available)
    assert count == 3  # timing, norms, dispersion (robustness is None)


def test_metric_detector_client_metrics():
    """Test detection of client metrics."""
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "loss_after": [0.5, 0.4, 0.3],
            "acc_after": [0.7, 0.75, 0.8],
            "weight_norm_after": [1.0, 1.1, 1.2],
            "tau_bin": [0.5, 0.5, 0.5],
            "benign_fpr_bin_tau": [0.1, 0.09, 0.08],
        }
    )

    detector = MetricDetector()
    available = detector.detect_available(df, metric_type="client")

    assert available["loss"] == "loss_after"
    assert available["accuracy"] == "acc_after"
    assert available["norms"] == "weight_norm_after"
    assert available["threshold"] == "tau_bin"
    assert available["fpr"] == "benign_fpr_bin_tau"


def test_smoothing_config_rolling_mean():
    """Test rolling mean smoothing."""
    data = [1, 2, 3, 4, 5]
    smoothing = SmoothingConfig(enabled=True, window_size=3, method="rolling_mean")

    smoothed = smoothing.apply(data)

    assert len(smoothed) == len(data)
    # Center value should be average of window
    assert abs(smoothed.iloc[2] - 3.0) < 0.01


def test_smoothing_config_disabled():
    """Test that disabled smoothing returns original data."""
    data = [1, 2, 3, 4, 5]
    smoothing = SmoothingConfig(enabled=False)

    smoothed = smoothing.apply(data)

    assert list(smoothed) == data


def test_smoothing_config_insufficient_data():
    """Test smoothing with insufficient data points."""
    data = [1, 2]
    smoothing = SmoothingConfig(enabled=True, window_size=5)

    smoothed = smoothing.apply(data)

    # Should return original when data length < window
    assert list(smoothed) == data


def test_confidence_interval_config_compute():
    """Test confidence interval computation."""
    data_groups = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.1, 2.1, 3.1, 4.1, 5.1],
        [0.9, 1.9, 2.9, 3.9, 4.9],
    ]

    ci_config = ConfidenceIntervalConfig(enabled=True, confidence=0.95)
    result = ci_config.compute(data_groups)

    assert result is not None
    assert "mean" in result
    assert "lower" in result
    assert "upper" in result

    # Mean should be close to [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(result["mean"][0] - 1.0) < 0.1
    assert abs(result["mean"][2] - 3.0) < 0.1

    # Lower should be less than mean, upper should be greater
    assert all(result["lower"] < result["mean"])
    assert all(result["upper"] > result["mean"])


def test_confidence_interval_config_disabled():
    """Test that disabled CI returns None."""
    data_groups = [[1, 2, 3], [1.1, 2.1, 3.1]]

    ci_config = ConfidenceIntervalConfig(enabled=False)
    result = ci_config.compute(data_groups)

    assert result is None


def test_confidence_interval_config_insufficient_groups():
    """Test CI with insufficient groups."""
    data_groups = [[1, 2, 3]]  # Only 1 group

    ci_config = ConfidenceIntervalConfig(enabled=True)
    result = ci_config.compute(data_groups)

    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
