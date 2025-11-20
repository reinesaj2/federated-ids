#!/usr/bin/env python3
"""Unit tests for caption_tables module."""

import numpy as np
import pandas as pd
import pytest

from scripts.caption_tables import (
    compute_95_ci,
    generate_aggregation_caption_table,
    generate_attack_caption_table,
    generate_heterogeneity_caption_table,
    generate_personalization_caption_table,
    generate_privacy_caption_table,
)


def test_compute_95_ci_single_value():
    """CI for single value should be point estimate."""
    values = np.array([0.5])
    lower, upper = compute_95_ci(values)
    assert lower == 0.5
    assert upper == 0.5


def test_compute_95_ci_multiple_values():
    """CI should bound mean for multiple values."""
    values = np.array([0.8, 0.9, 1.0])
    lower, upper = compute_95_ci(values)
    mean = values.mean()
    assert lower < mean < upper


def test_compute_95_ci_identical_values():
    """CI for identical values should be point estimate."""
    values = np.array([0.9, 0.9, 0.9])
    lower, upper = compute_95_ci(values)
    assert lower == pytest.approx(0.9)
    assert upper == pytest.approx(0.9)


def test_generate_aggregation_caption_table_empty_df():
    """Empty dataframe should return empty result."""
    df = pd.DataFrame()
    result = generate_aggregation_caption_table(df)
    assert len(result) == 0


def test_generate_aggregation_caption_table_missing_columns():
    """Missing required columns should return empty result."""
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    result = generate_aggregation_caption_table(df)
    assert len(result) == 0


def test_generate_aggregation_caption_table_single_method():
    """Single aggregation method should produce one row."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "fedavg"],
            "macro_f1": [0.90, 0.91, 0.92],
        }
    )
    result = generate_aggregation_caption_table(df)
    assert len(result) == 1
    assert result.iloc[0]["Method"] == "Fedavg"
    assert float(result.iloc[0]["Macro-F1"]) == pytest.approx(0.91)
    assert result.iloc[0]["n_seeds"] == 3


def test_generate_aggregation_caption_table_multiple_methods():
    """Multiple methods should each have a row."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum"],
            "macro_f1": [0.90, 0.91, 0.93, 0.94],
        }
    )
    result = generate_aggregation_caption_table(df)
    assert len(result) == 2
    methods = set(result["Method"].values)
    assert "Fedavg" in methods
    assert "Krum" in methods


def test_generate_heterogeneity_caption_table_single_alpha():
    """Single alpha value should produce one row."""
    df = pd.DataFrame({"alpha": [0.5, 0.5, 0.5], "macro_f1": [0.88, 0.89, 0.90]})
    result = generate_heterogeneity_caption_table(df)
    assert len(result) == 1
    assert float(result.iloc[0]["Alpha (heterogeneity)"]) == pytest.approx(0.5)


def test_generate_heterogeneity_caption_table_multiple_alphas():
    """Multiple alpha values should each have a row."""
    df = pd.DataFrame(
        {
            "alpha": [0.1, 0.1, 0.5, 0.5, 1.0, 1.0],
            "macro_f1": [0.85, 0.86, 0.88, 0.89, 0.91, 0.92],
        }
    )
    result = generate_heterogeneity_caption_table(df)
    assert len(result) == 3
    alphas = sorted(result["Alpha (heterogeneity)"].astype(float).values)
    assert alphas == pytest.approx([0.1, 0.5, 1.0])


def test_generate_attack_caption_table_single_config():
    """Single attack config should produce one row."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg"],
            "byzantine_f": [0.0, 0.0],
            "macro_f1": [0.92, 0.93],
        }
    )
    result = generate_attack_caption_table(df)
    assert len(result) == 1
    assert "0%" in result.iloc[0]["Byzantine %"]


def test_generate_attack_caption_table_multiple_configs():
    """Multiple attack levels should have corresponding rows."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg"] * 4 + ["krum"] * 4,
            "byzantine_f": [0.0, 0.1, 0.3, 0.0, 0.0, 0.1, 0.3, 0.0],
            "macro_f1": [0.92, 0.85, 0.70, 0.0, 0.93, 0.87, 0.75, 0.0],
        }
    )
    result = generate_attack_caption_table(df)
    assert len(result) > 0


def test_generate_privacy_caption_table_baseline_only():
    """DP sigma of 0 should be labeled as baseline."""
    df = pd.DataFrame({"dp_sigma": [0.0, 0.0], "macro_f1": [0.92, 0.93]})
    result = generate_privacy_caption_table(df)
    assert len(result) == 1
    assert "none" in result.iloc[0]["DP Sigma"].lower()
    assert "Baseline" in result.iloc[0]["Privacy-Utility"]


def test_generate_privacy_caption_table_multiple_sigmas():
    """Multiple DP noise levels should each have a row."""
    df = pd.DataFrame(
        {
            "dp_sigma": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            "macro_f1": [0.92, 0.93, 0.88, 0.89, 0.85, 0.86],
        }
    )
    result = generate_privacy_caption_table(df)
    assert len(result) == 3


def test_generate_personalization_caption_table_global_only():
    """Personalization with 0 epochs should indicate no personalization."""
    df = pd.DataFrame(
        {
            "personalization_epochs": [0, 0, 0],
            "personalization_gain": [0.0, 0.0, 0.0],
        }
    )
    result = generate_personalization_caption_table(df)
    assert len(result) == 1
    assert result.iloc[0]["Personalization Epochs"] == 0


def test_generate_personalization_caption_table_with_gains():
    """Positive gains should be reported."""
    df = pd.DataFrame(
        {
            "personalization_epochs": [0, 0, 5, 5, 5],
            "personalization_gain": [0.0, 0.0, 0.05, 0.03, -0.02],
        }
    )
    result = generate_personalization_caption_table(df)
    assert len(result) == 2
    personalized_row = result[result["Personalization Epochs"] == 5].iloc[0]
    pct_positive = float(personalized_row["% Positive Gains"].rstrip("%"))
    assert pct_positive > 50.0


def test_generate_aggregation_with_l2_distance():
    """L2 distance should be included when available."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg"],
            "macro_f1": [0.90, 0.91],
            "l2_to_benign_mean": [0.05, 0.04],
        }
    )
    result = generate_aggregation_caption_table(df)
    assert len(result) == 1
    l2_str = result.iloc[0]["L2 Distance"]
    assert l2_str != "N/A"
    assert float(l2_str) == pytest.approx(0.045)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
