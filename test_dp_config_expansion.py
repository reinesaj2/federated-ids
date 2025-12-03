#!/usr/bin/env python3
"""
Unit tests for expanded DP configuration in privacy-utility curve analysis.
Tests the enhanced noise level coverage and statistical rigor.
"""

import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent / "scripts"))

from comparative_analysis import ComparisonMatrix  # noqa: E402


class TestDPConfigExpansion:
    """Test expanded DP configuration for privacy-utility curve analysis."""

    def test_dp_configs_include_comprehensive_noise_levels(self):
        """Should include noise levels covering the full privacy-utility spectrum."""
        cm = ComparisonMatrix()

        # Extract noise levels from DP configs
        noise_levels = [config["noise"] for config in cm.dp_configs if config["enabled"]]

        # Should include comprehensive range for privacy-utility analysis
        expected_noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

        for expected_noise in expected_noise_levels:
            assert expected_noise in noise_levels, f"Missing noise level {expected_noise}"

    def test_dp_configs_include_baseline(self):
        """Should include baseline (no DP) configuration for comparison."""
        cm = ComparisonMatrix()

        baseline_configs = [config for config in cm.dp_configs if not config["enabled"]]
        assert len(baseline_configs) == 1
        assert baseline_configs[0]["noise"] == 0.0

    def test_privacy_experiments_generate_correct_count(self):
        """Should generate correct number of privacy experiments."""
        cm = ComparisonMatrix()

        privacy_configs = cm.generate_configs("privacy")

        # Should have 7 DP configs + 1 baseline = 8 total configs
        # With 10 seeds each = 80 total experiments
        expected_total = 8 * 10  # 8 configs * 10 seeds
        assert len(privacy_configs) == expected_total

    def test_privacy_experiments_have_sufficient_seeds(self):
        """Should have at least 5 seeds per configuration for statistical rigor."""
        cm = ComparisonMatrix()

        privacy_configs = cm.generate_configs("privacy")

        # Group by configuration (excluding seed)
        config_groups = {}
        for config in privacy_configs:
            key = (
                config.dp_enabled,
                config.dp_noise_multiplier,
                config.aggregation,
                config.alpha,
                config.adversary_fraction,
                config.personalization_epochs,
                config.fedprox_mu,
                config.num_clients,
                config.num_rounds,
            )
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(config)

        # Each configuration should have at least 5 seeds
        for config_group in config_groups.values():
            assert len(config_group) >= 5, f"Configuration has only {len(config_group)} seeds, need at least 5"

    def test_dp_enabled_configs_have_valid_noise_levels(self):
        """Should have valid noise levels for DP-enabled configurations."""
        cm = ComparisonMatrix()

        dp_configs = [config for config in cm.dp_configs if config["enabled"]]

        for config in dp_configs:
            noise = config["noise"]
            assert isinstance(noise, (int, float))
            assert noise > 0.0, f"DP noise level must be positive, got {noise}"
            assert noise <= 10.0, f"DP noise level seems too high, got {noise}"

    def test_privacy_configs_use_fedavg_aggregation(self):
        """Should use fedavg aggregation for privacy experiments (focused on DP impact)."""
        cm = ComparisonMatrix()

        privacy_configs = cm.generate_configs("privacy")

        # Extract unique aggregation methods
        aggregation_methods = set(config.aggregation for config in privacy_configs)

        # Privacy dimension focuses on DP impact, uses fedavg as baseline
        expected_methods = {"fedavg"}
        assert aggregation_methods == expected_methods

    def test_noise_levels_cover_privacy_utility_spectrum(self):
        """Should cover the full privacy-utility trade-off spectrum."""
        cm = ComparisonMatrix()

        noise_levels = [config["noise"] for config in cm.dp_configs if config["enabled"]]
        noise_levels.sort()

        # Should have low noise (high privacy, low utility)
        assert min(noise_levels) <= 0.2, "Missing low noise levels for high privacy"

        # Should have high noise (low privacy, high utility)
        assert max(noise_levels) >= 1.5, "Missing high noise levels for low privacy"

        # Should have intermediate levels for smooth curve
        assert len(noise_levels) >= 5, "Need at least 5 noise levels for smooth curve"

        # Should have reasonable spacing between levels
        for i in range(1, len(noise_levels)):
            ratio = noise_levels[i] / noise_levels[i - 1]
            assert 1.2 <= ratio <= 3.0, f"Poor spacing between noise levels: {noise_levels[i - 1]} -> {noise_levels[i]}"


class TestDPConfigValidation:
    """Test validation of DP configuration parameters."""

    def test_dp_configs_have_required_fields(self):
        """Should have required fields for each DP configuration."""
        cm = ComparisonMatrix()

        for config in cm.dp_configs:
            assert "enabled" in config, "Missing 'enabled' field"
            assert "noise" in config, "Missing 'noise' field"
            assert isinstance(config["enabled"], bool), "enabled must be boolean"
            assert isinstance(config["noise"], (int, float)), "noise must be numeric"

    def test_dp_configs_no_duplicates(self):
        """Should not have duplicate DP configurations."""
        cm = ComparisonMatrix()

        config_tuples = [(config["enabled"], config["noise"]) for config in cm.dp_configs]
        assert len(config_tuples) == len(set(config_tuples)), "Duplicate DP configurations found"

    def test_dp_configs_ordered_logically(self):
        """Should be ordered logically for privacy-utility analysis."""
        cm = ComparisonMatrix()

        # Baseline should come first
        assert not cm.dp_configs[0]["enabled"], "First config should be baseline"
        assert cm.dp_configs[0]["noise"] == 0.0, "First config should have zero noise"

        # DP configs should be ordered by noise level
        dp_configs = [config for config in cm.dp_configs if config["enabled"]]
        noise_levels = [config["noise"] for config in dp_configs]
        assert noise_levels == sorted(noise_levels), "DP configs should be ordered by noise level"
