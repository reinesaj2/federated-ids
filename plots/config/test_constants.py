"""Tests for plotting constants."""

from plots.config.constants import (
    ADVERSARIAL_LEVELS,
    AGGREGATOR_COLORS,
    AGGREGATOR_LABELS,
    AGGREGATOR_ORDER,
    ALPHA_VALUES,
    DATASET_CONFIG,
    DATASETS,
    MU_VALUES,
)


class TestAggregatorConstants:
    def test_aggregator_order_has_four_methods(self):
        assert len(AGGREGATOR_ORDER) == 4

    def test_aggregator_order_contains_expected_methods(self):
        expected = {"fedavg", "krum", "bulyan", "median"}
        assert set(AGGREGATOR_ORDER) == expected

    def test_aggregator_labels_maps_all_methods(self):
        for agg in AGGREGATOR_ORDER:
            assert agg in AGGREGATOR_LABELS
            assert isinstance(AGGREGATOR_LABELS[agg], str)

    def test_aggregator_colors_maps_all_methods(self):
        for agg in AGGREGATOR_ORDER:
            assert agg in AGGREGATOR_COLORS
            assert AGGREGATOR_COLORS[agg].startswith("#")


class TestDatasetConstants:
    def test_datasets_has_three_entries(self):
        assert len(DATASETS) == 3

    def test_datasets_contains_expected_datasets(self):
        expected = {"iiot", "cic", "unsw"}
        assert set(DATASETS) == expected

    def test_dataset_config_has_required_keys(self):
        required_keys = {"label", "color", "marker"}
        for ds in DATASETS:
            assert ds in DATASET_CONFIG
            assert required_keys.issubset(set(DATASET_CONFIG[ds].keys()))

    def test_dataset_config_colors_are_valid_hex(self):
        for ds in DATASETS:
            color = DATASET_CONFIG[ds]["color"]
            assert color.startswith("#")
            assert len(color) == 7


class TestExperimentConstants:
    def test_adversarial_levels_are_sorted(self):
        assert ADVERSARIAL_LEVELS == sorted(ADVERSARIAL_LEVELS)

    def test_adversarial_levels_start_at_zero(self):
        assert ADVERSARIAL_LEVELS[0] == 0

    def test_adversarial_levels_are_percentages(self):
        for level in ADVERSARIAL_LEVELS:
            assert 0 <= level <= 100

    def test_alpha_values_are_sorted(self):
        assert ALPHA_VALUES == sorted(ALPHA_VALUES)

    def test_alpha_values_are_positive(self):
        for alpha in ALPHA_VALUES:
            assert alpha > 0

    def test_mu_values_start_at_zero(self):
        assert MU_VALUES[0] == 0.0

    def test_mu_values_are_sorted(self):
        assert MU_VALUES == sorted(MU_VALUES)
