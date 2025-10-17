#!/usr/bin/env python3
"""
Tests for comparative analysis framework.
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.comparative_analysis import (
    ATTACK_AGGREGATIONS,
    ComparisonMatrix,
    ExperimentConfig,
    find_available_port,
    is_port_available,
)
from scripts.generate_thesis_plots import perform_statistical_tests
from scripts.plot_metrics_utils import compute_confidence_interval


def test_experiment_config_preset_name():
    """Test that experiment config generates unique preset names."""
    config = ExperimentConfig(
        aggregation="fedavg",
        alpha=1.0,
        adversary_fraction=0.0,
        dp_enabled=False,
        dp_noise_multiplier=0.0,
        personalization_epochs=0,
        num_clients=6,
        num_rounds=20,
        seed=42,
    )

    preset = config.to_preset_name()
    assert "fedavg" in preset
    assert "alpha1.0" in preset
    assert "adv0" in preset
    assert "dp0" in preset
    assert "pers0" in preset
    assert "seed42" in preset


def test_comparison_matrix_aggregation_dimension():
    """Test matrix generation for aggregation dimension only."""
    matrix = ComparisonMatrix(
        aggregation_methods=["fedavg", "krum"],
        seeds=[42, 43],
        num_clients=6,
        num_rounds=10,
    )

    configs = matrix.generate_configs(filter_dimension="aggregation")

    # Exact expected count: 2 aggregation methods × 2 seeds = 4 configs
    assert len(configs) == 4

    # Check all are aggregation variants
    aggregations = {c.aggregation for c in configs}
    assert aggregations == {"fedavg", "krum"}

    # Check other params are fixed to baseline
    for config in configs:
        assert config.alpha == 1.0  # IID baseline
        assert config.adversary_fraction == 0.0
        assert config.dp_enabled is False
        assert config.personalization_epochs == 0


def test_comparison_matrix_heterogeneity_dimension():
    """Test matrix generation for heterogeneity dimension."""
    matrix = ComparisonMatrix(alpha_values=[1.0, 0.5, 0.1], seeds=[42], num_clients=6, num_rounds=10)

    configs = matrix.generate_configs(filter_dimension="heterogeneity")

    # Exact expected count: 3 alpha values × 1 seed = 3 configs
    assert len(configs) == 3

    # Check all use fedavg baseline
    assert all(c.aggregation == "fedavg" for c in configs)

    # Check alpha values
    alphas = {c.alpha for c in configs}
    assert alphas == {1.0, 0.5, 0.1}


def test_comparison_matrix_attack_dimension():
    """Test matrix generation for attack resilience dimension."""
    matrix = ComparisonMatrix(
        aggregation_methods=["fedavg", "krum", "bulyan", "median"],
        adversary_fractions=[0.0, 0.1],
        seeds=[42],
        num_clients=6,
        num_rounds=10,
    )

    configs = matrix.generate_configs(filter_dimension="attack")

    # Exact expected count: 4 agg methods (subset) × 2 adv fractions × 1 seed = 8
    expected_count = len(ATTACK_AGGREGATIONS) * 2 * 1
    assert len(configs) == expected_count

    # Check uses attack aggregations subset
    aggregations = {c.aggregation for c in configs}
    assert aggregations == set(ATTACK_AGGREGATIONS)

    # Check adversary fractions are varied
    adv_fracs = {c.adversary_fraction for c in configs}
    assert adv_fracs == {0.0, 0.1}

    # Check alpha is fixed to moderate non-IID
    assert all(c.alpha == 0.5 for c in configs)

    # Check all attack configs use n=11 clients for Bulyan requirement
    assert all(c.num_clients == 11 for c in configs)


def test_comparison_matrix_privacy_dimension():
    """Test matrix generation for privacy dimension."""
    matrix = ComparisonMatrix(
        dp_configs=[
            {"enabled": False, "noise": 0.0},
            {"enabled": True, "noise": 0.5},
        ],
        seeds=[42],
        num_clients=6,
        num_rounds=10,
    )

    configs = matrix.generate_configs(filter_dimension="privacy")

    # Exact expected count: 2 DP configs × 1 seed = 2 configs
    assert len(configs) == 2

    # Check DP settings
    dp_enabled_configs = [c for c in configs if c.dp_enabled]
    dp_disabled_configs = [c for c in configs if not c.dp_enabled]

    assert len(dp_enabled_configs) == 1
    assert len(dp_disabled_configs) == 1
    assert dp_enabled_configs[0].dp_noise_multiplier == 0.5


def test_comparison_matrix_personalization_dimension():
    """Test matrix generation for personalization dimension."""
    matrix = ComparisonMatrix(personalization_epochs=[0, 5], seeds=[42], num_clients=6, num_rounds=10)

    configs = matrix.generate_configs(filter_dimension="personalization")

    # Exact expected count: 2 personalization settings × 1 seed = 2 configs
    assert len(configs) == 2

    pers_epochs = {c.personalization_epochs for c in configs}
    assert pers_epochs == {0, 5}


def test_comparison_matrix_full_factorial():
    """Test full factorial experiment generation."""
    matrix = ComparisonMatrix(
        aggregation_methods=["fedavg", "krum"],
        alpha_values=[1.0, 0.5],
        adversary_fractions=[0.0],
        dp_configs=[{"enabled": False, "noise": 0.0}],
        personalization_epochs=[0],
        seeds=[42],
        num_clients=6,
        num_rounds=10,
    )

    configs = matrix.generate_configs(filter_dimension=None)  # Full factorial

    # Exact expected count: 2 agg × 2 alpha × 1 adv × 1 dp × 1 pers × 1 seed = 4
    assert len(configs) == 4

    # Verify all combinations present
    agg_alpha_pairs = {(c.aggregation, c.alpha) for c in configs}
    assert len(agg_alpha_pairs) == 4


def test_comparison_matrix_invalid_dimension():
    """Test that invalid dimension raises ValueError."""
    matrix = ComparisonMatrix()

    with pytest.raises(ValueError, match="Invalid dimension"):
        matrix.generate_configs(filter_dimension="invalid_dimension")


def test_comparison_matrix_empty_lists():
    """Test handling of empty configuration lists."""
    matrix = ComparisonMatrix(
        aggregation_methods=[],
        seeds=[42],
    )

    configs = matrix.generate_configs(filter_dimension="aggregation")

    # Should return empty list when no aggregation methods
    assert len(configs) == 0


def test_config_to_preset_name_uniqueness():
    """Test that different configs produce different preset names."""
    config1 = ExperimentConfig(
        aggregation="fedavg",
        alpha=1.0,
        adversary_fraction=0.0,
        dp_enabled=False,
        dp_noise_multiplier=0.0,
        personalization_epochs=0,
        num_clients=6,
        num_rounds=20,
        seed=42,
    )

    config2 = ExperimentConfig(
        aggregation="krum",  # Different
        alpha=1.0,
        adversary_fraction=0.0,
        dp_enabled=False,
        dp_noise_multiplier=0.0,
        personalization_epochs=0,
        num_clients=6,
        num_rounds=20,
        seed=42,
    )

    config3 = ExperimentConfig(
        aggregation="fedavg",
        alpha=0.5,  # Different
        adversary_fraction=0.0,
        dp_enabled=False,
        dp_noise_multiplier=0.0,
        personalization_epochs=0,
        num_clients=6,
        num_rounds=20,
        seed=42,
    )

    preset1 = config1.to_preset_name()
    preset2 = config2.to_preset_name()
    preset3 = config3.to_preset_name()

    # All should be different
    assert preset1 != preset2
    assert preset1 != preset3
    assert preset2 != preset3


def test_experiment_config_serialization():
    """Test that experiment config can be serialized to JSON."""
    config = ExperimentConfig(
        aggregation="krum",
        alpha=0.5,
        adversary_fraction=0.1,
        dp_enabled=True,
        dp_noise_multiplier=0.5,
        personalization_epochs=5,
        num_clients=6,
        num_rounds=20,
        seed=42,
    )

    # Convert to dict and serialize
    config_dict = config.__dict__
    json_str = json.dumps(config_dict)

    # Should not raise
    reloaded = json.loads(json_str)
    assert reloaded["aggregation"] == "krum"
    assert reloaded["alpha"] == 0.5
    assert reloaded["dp_enabled"] is True


def test_is_port_available():
    """Test port availability check."""
    # Port 0 should typically be available (OS assigns ephemeral port)
    # But we can't test specific ports without potential conflicts
    # So just test the function doesn't crash
    result = is_port_available(8080)
    assert isinstance(result, bool)


def test_find_available_port():
    """Test finding available port."""
    # Should find a port in range
    port = find_available_port(start_port=9000, max_attempts=10)
    assert 9000 <= port < 9010

    # Verify the port is actually available
    assert is_port_available(port)


def test_find_available_port_exhaustion():
    """Test port exhaustion raises error."""
    # Try to find port with 0 attempts - should raise
    with pytest.raises(RuntimeError, match="Could not find available port"):
        find_available_port(start_port=8080, max_attempts=0)


def test_compute_confidence_interval():
    """Test confidence interval computation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

    # Check mean is correct
    assert abs(mean - 3.0) < 1e-6

    # Check CI bounds are reasonable
    assert lower < mean < upper
    assert upper - lower > 0

    # Check CI is symmetric around mean (for this symmetric data)
    assert abs((upper - mean) - (mean - lower)) < 1e-6


def test_compute_confidence_interval_single_value():
    """Test confidence interval with single value (edge case)."""
    data = np.array([5.0])

    # Should handle gracefully - returns (mean, mean, mean) for n=1
    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

    assert abs(mean - 5.0) < 1e-6
    # With single value, CI bounds equal the mean (no range)
    assert abs(lower - 5.0) < 1e-6
    assert abs(upper - 5.0) < 1e-6


def test_statistical_tests_ttest():
    """Test t-test for two groups."""
    # Create synthetic data with known difference
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "metric": np.concatenate([np.random.normal(5, 1, 10), np.random.normal(7, 1, 10)]),
        }
    )

    result = perform_statistical_tests(df, "group", "metric")

    assert result["test"] == "t_test"
    assert "statistic" in result
    assert "p_value" in result
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1


def test_statistical_tests_anova():
    """Test ANOVA for multiple groups."""
    # Create synthetic data with 3 groups
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "metric": np.concatenate(
                [
                    np.random.normal(5, 1, 10),
                    np.random.normal(6, 1, 10),
                    np.random.normal(7, 1, 10),
                ]
            ),
        }
    )

    result = perform_statistical_tests(df, "group", "metric")

    assert result["test"] == "anova"
    assert "statistic" in result
    assert "p_value" in result
    assert "pairwise" in result
    assert isinstance(result["pairwise"], dict)
    assert len(result["pairwise"]) == 3  # 3 choose 2 = 3 pairwise comparisons


def test_statistical_tests_insufficient_data():
    """Test handling of insufficient data."""
    df = pd.DataFrame({"group": ["A"] * 5, "metric": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = perform_statistical_tests(df, "group", "metric")

    assert result["test"] == "insufficient_data"
    assert result["p_value"] is None


def test_statistical_tests_empty_groups():
    """Test handling of empty groups after filtering."""
    # Create dataframe with NaN values that will be filtered out
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "metric": [np.nan] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = perform_statistical_tests(df, "group", "metric")

    # Should handle gracefully - only one valid group remains
    assert result["test"] == "insufficient_data"
    assert result["p_value"] is None


def test_comparison_matrix_default_values():
    """Test that default matrix values are sensible."""
    matrix = ComparisonMatrix()

    # Check defaults
    assert len(matrix.aggregation_methods) == 4
    assert "fedavg" in matrix.aggregation_methods
    assert 1.0 in matrix.alpha_values  # IID
    assert 0.0 in matrix.adversary_fractions  # Benign
    assert matrix.num_clients == 6
    assert matrix.num_rounds == 20
    assert len(matrix.seeds) == 3  # Multiple seeds for statistical validity


def test_comparison_matrix_attack_uses_subset():
    """Test that attack dimension uses ATTACK_AGGREGATIONS subset."""
    matrix = ComparisonMatrix(
        aggregation_methods=["fedavg", "krum", "bulyan", "median"],
        adversary_fractions=[0.0, 0.3],
        seeds=[42],
    )

    configs = matrix.generate_configs(filter_dimension="attack")

    # After Issue #70 fix, Bulyan is now included in attack experiments
    aggregations = {c.aggregation for c in configs}
    assert aggregations == set(ATTACK_AGGREGATIONS)
    assert "bulyan" in aggregations  # Included after algorithm fix

    # All attack configs should use n=11 clients for Bulyan requirement (n >= 4f + 3)
    assert all(c.num_clients == 11 for c in configs)
