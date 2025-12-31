"""
Unit tests for CIC dataset integration in comparative_analysis.py.

Tests verify:
1. ExperimentConfig.with_dataset() factory method
2. ComparisonMatrix dataset parameter propagation
3. CIC dataset path resolution
"""

import pytest
from scripts.comparative_analysis import ExperimentConfig, ComparisonMatrix


def test_experiment_config_with_dataset_unsw():
    """Verify with_dataset factory creates UNSW config correctly."""
    config = ExperimentConfig.with_dataset(
        dataset="unsw",
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

    assert config.dataset == "unsw"
    assert config.data_path == "data/unsw/UNSW_NB15_training-set.csv"


def test_experiment_config_with_dataset_cic():
    """Verify with_dataset factory creates CIC config correctly."""
    config = ExperimentConfig.with_dataset(
        dataset="cic",
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

    assert config.dataset == "cic"
    assert config.data_path == "data/cic/cic_ids2017_multiclass.csv"


def test_experiment_config_with_dataset_invalid():
    """Verify with_dataset raises error for invalid dataset."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        ExperimentConfig.with_dataset(
            dataset="invalid",
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


def test_comparison_matrix_dataset_propagation():
    """Verify dataset parameter propagates to generated configs."""
    matrix = ComparisonMatrix(
        dataset="cic",
        data_path="data/cic/cic_ids2017_multiclass.csv",
        seeds=[42],
        aggregation_methods=["fedavg"],
    )

    configs = matrix._generate_aggregation_configs()

    assert len(configs) == 1
    assert configs[0].dataset == "cic"
    assert configs[0].data_path == "data/cic/cic_ids2017_multiclass.csv"


def test_comparison_matrix_default_dataset():
    """Verify default dataset is UNSW when not specified."""
    matrix = ComparisonMatrix(seeds=[42], aggregation_methods=["fedavg"])

    configs = matrix._generate_aggregation_configs()

    assert len(configs) == 1
    assert configs[0].dataset == "unsw"


def test_comparison_matrix_custom_data_path():
    """Verify custom data path can be provided."""
    custom_path = "/custom/path/to/dataset.csv"
    matrix = ComparisonMatrix(dataset="cic", data_path=custom_path, seeds=[42], aggregation_methods=["fedavg"])

    configs = matrix._generate_aggregation_configs()

    assert len(configs) == 1
    assert configs[0].data_path == custom_path


def test_experiment_config_to_preset_name_includes_dataset():
    """Verify preset name generation works for both datasets."""
    config_unsw = ExperimentConfig.with_dataset(
        dataset="unsw",
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

    config_cic = ExperimentConfig.with_dataset(
        dataset="cic",
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

    preset_unsw = config_unsw.to_preset_name()
    preset_cic = config_cic.to_preset_name()

    assert preset_unsw.startswith("unsw_")
    assert preset_cic.startswith("cic_")
    assert "comp_fedavg" in preset_unsw
    assert "comp_fedavg" in preset_cic
    assert preset_unsw != preset_cic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
