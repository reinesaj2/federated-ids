"""Unit tests for experiment matrix validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from validate_experiment_matrix import (  # noqa: E402
    ExperimentConstraints,
    ExperimentMatrixValidator,
)


class TestExperimentConstraints:
    """Test Byzantine resilience constraint validation."""

    def test_bulyan_feasible_with_sufficient_clients(self):
        """Bulyan should pass with n >= 4f + 3."""
        # n=11, f=2 (20% adversaries): 11 >= 4*2+3=11 ✓
        config = ExperimentConstraints(
            aggregation="bulyan", n_clients=11, adversary_fraction=0.2
        )
        is_valid, reason = config.validate()
        assert is_valid is True
        assert reason == "OK"

    def test_bulyan_infeasible_with_insufficient_clients(self):
        """Bulyan should fail with n < 4f + 3."""
        # n=6, f=2 (30% adversaries): 6 < 4*2+3=11 ✗
        config = ExperimentConstraints(
            aggregation="bulyan", n_clients=6, adversary_fraction=0.3
        )
        is_valid, reason = config.validate()
        assert is_valid is False
        assert "Bulyan requires n >= 4f+3" in reason

    def test_krum_feasible_with_sufficient_clients(self):
        """Krum should pass with n >= 2f + 3."""
        # n=6, f=1 (15% adversaries): 6 >= 2*1+3=5 ✓
        config = ExperimentConstraints(
            aggregation="krum", n_clients=6, adversary_fraction=0.15
        )
        is_valid, reason = config.validate()
        assert is_valid is True
        assert reason == "OK"

    def test_krum_infeasible_with_insufficient_clients(self):
        """Krum should fail with n < 2f + 3."""
        # With n=6 and 30% adversaries: f=int(6*0.3)=1
        # Required: n >= 2*1+3 = 5
        # Since 6 >= 5, this is actually FEASIBLE
        # Let's test with more clients having more adversaries
        # n=4, f=2 (50% adversaries): 4 < 2*2+3=7 ✗
        config = ExperimentConstraints(
            aggregation="krum", n_clients=4, adversary_fraction=0.5
        )
        is_valid, reason = config.validate()
        assert is_valid is False
        assert "Krum requires n >= 2f+3" in reason

    def test_median_feasible_with_sufficient_clients(self):
        """Median should pass with n >= 2f + 1."""
        # n=6, f=2 (30% adversaries): 6 >= 2*2+1=5 ✓
        # Wait, f=int(6*0.3)=1, not 2
        # Let me use n=6, f=int(6*0.3)=1, need n >= 2*1+1=3 ✓
        config = ExperimentConstraints(
            aggregation="median", n_clients=6, adversary_fraction=0.3
        )
        is_valid, reason = config.validate()
        assert is_valid is True
        assert reason == "OK"

    def test_median_infeasible_boundary(self):
        """Median should fail at boundary."""
        # n=4, f=2 (50% adversaries): 4 < 2*2+1=5 ✗
        config = ExperimentConstraints(
            aggregation="median", n_clients=4, adversary_fraction=0.5
        )
        is_valid, reason = config.validate()
        assert is_valid is False
        assert "Median requires n >= 2f+1" in reason

    def test_fedavg_always_feasible(self):
        """FedAvg has no Byzantine constraint."""
        for adv_frac in [0.0, 0.1, 0.3, 0.5]:
            config = ExperimentConstraints(
                aggregation="fedavg", n_clients=2, adversary_fraction=adv_frac
            )
            is_valid, reason = config.validate()
            assert is_valid is True
            assert reason == "OK"

    def test_zero_adversaries_all_feasible(self):
        """All aggregations require min clients even with zero adversaries."""
        # FedAvg and Median need min 1 client: n=2 works
        # Krum needs n >= 2*0+3 = 3 clients, so n=2 fails
        # Bulyan needs n >= 4*0+3 = 3 clients, so n=2 fails

        # Test with sufficient clients for all
        for agg in ["fedavg", "krum", "bulyan", "median"]:
            config = ExperimentConstraints(
                aggregation=agg, n_clients=6, adversary_fraction=0.0
            )
            is_valid, reason = config.validate()
            assert (
                is_valid is True
            ), f"{agg} failed with 6 clients and 0 adversaries"


class TestExperimentMatrixValidator:
    """Test batch validation of experiment matrix."""

    def test_validator_counts_viable_and_impossible(self):
        """Validator should count viable and impossible configs correctly."""
        validator = ExperimentMatrixValidator(n_clients=6)
        viable, impossible, impossible_configs = validator.validate_all()

        # With 6 clients:
        # - fedavg: 3 configs all viable
        # - krum: 3 configs all viable (30%: f=1, need n>=5 ✓)
        # - bulyan: 2 viable + 1 impossible (30%: f=1, need n>=7 ✗)
        # - median: 3 configs all viable
        # Total: 11 viable, 1 impossible
        assert viable == 11
        assert impossible == 1
        assert len(impossible_configs) == 1

    def test_impossible_configs_listed(self):
        """Impossible configurations should be listed."""
        validator = ExperimentMatrixValidator(n_clients=6)
        _, _, impossible_configs = validator.validate_all()

        # Should contain Bulyan+30% and Krum+30%
        config_str = " ".join(impossible_configs)
        assert "adv30" in config_str

    def test_print_summary_no_crash(self, capsys):
        """print_summary should not crash."""
        validator = ExperimentMatrixValidator(n_clients=6)
        validator.print_summary()

        captured = capsys.readouterr()
        assert "EXPERIMENT MATRIX VALIDATION" in captured.out
        assert "Viable experiments:" in captured.out
        assert "Impossible:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
