#!/usr/bin/env python3
"""
Unit tests for load_iiot_data module.

Tests the refactored helper functions for data loading and merging.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from load_iiot_data import (  # noqa: E402
    parse_run_config,
    _load_server_metrics_for_run,
    _load_client_metrics_for_run,
    _create_merged_records,
    load_iiot_data,
)


class TestParseRunConfig:
    """Test parse_run_config function."""

    def test_parse_standard_run_name(self):
        """Test parsing a standard run directory name."""
        run_dir = Path("dsedge-iiotset-nightly_comp_fedavg_alpha0.5_adv10_dp0_pers0_mu0.0_seed42")
        config = parse_run_config(run_dir)

        assert config["dataset"] == "edge-iiotset"
        assert config["aggregation"] == "fedavg"
        assert config["alpha"] == 0.5
        assert config["adv_pct"] == 10
        assert config["pers_epochs"] == 0
        assert config["seed"] == 42

    def test_parse_alpha_inf(self):
        """Test parsing alpha=inf case."""
        run_dir = Path("dsedge-iiotset-nightly_comp_bulyan_alphainf_adv0_dp0_pers3_mu0.0_seed43")
        config = parse_run_config(run_dir)

        assert config["aggregation"] == "bulyan"
        assert config["alpha"] == float("inf")
        assert config["adv_pct"] == 0
        assert config["pers_epochs"] == 3
        assert config["seed"] == 43

    def test_parse_missing_fields_uses_defaults(self):
        """Test that missing fields get default values."""
        run_dir = Path("dsedge-iiotset-nightly_comp_krum")
        config = parse_run_config(run_dir)

        assert config["aggregation"] == "krum"
        assert config.get("adv_pct") == 0
        assert config.get("pers_epochs") == 0
        assert config.get("seed") == 42


class TestLoadServerMetricsForRun:
    """Test _load_server_metrics_for_run function."""

    def test_load_nonexistent_file_returns_none(self, tmp_path):
        """Test that missing metrics.csv returns None."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        result = _load_server_metrics_for_run(run_dir)
        assert result is None

    def test_load_valid_csv_returns_dataframe(self, tmp_path):
        """Test that valid metrics.csv is loaded correctly."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Create test CSV
        test_data = pd.DataFrame(
            {
                "round": [1, 2, 3],
                "l2_to_benign_mean": [0.1, 0.2, 0.3],
                "t_aggregate_ms": [10.0, 11.0, 12.0],
            }
        )
        test_data.to_csv(run_dir / "metrics.csv", index=False)

        result = _load_server_metrics_for_run(run_dir)
        assert result is not None
        assert len(result) == 3
        assert "round" in result.columns
        assert result["round"].tolist() == [1, 2, 3]


class TestLoadClientMetricsForRun:
    """Test _load_client_metrics_for_run function."""

    def test_load_no_client_files_returns_empty_dict(self, tmp_path):
        """Test that no client CSVs returns empty dict."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        result = _load_client_metrics_for_run(run_dir)
        assert result == {}

    def test_load_single_client_groups_by_round(self, tmp_path):
        """Test that client metrics are grouped by round correctly."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Create test client CSV
        test_data = pd.DataFrame(
            {
                "round": [1, 1, 2],
                "macro_f1_before": [0.5, 0.5, 0.6],
                "macro_f1_personalized": [0.55, 0.55, 0.65],
            }
        )
        test_data.to_csv(run_dir / "client_0_metrics.csv", index=False)

        result = _load_client_metrics_for_run(run_dir)

        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2
        assert len(result[2]) == 1
        assert result[1][0]["macro_f1_global"] == 0.5

    def test_fallback_to_macro_f1_global(self, tmp_path):
        """Test fallback from macro_f1_before to macro_f1_global."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # CSV without macro_f1_before
        test_data = pd.DataFrame(
            {
                "round": [1],
                "macro_f1_global": [0.7],
                "macro_f1_personalized": [0.75],
            }
        )
        test_data.to_csv(run_dir / "client_0_metrics.csv", index=False)

        result = _load_client_metrics_for_run(run_dir)

        assert result[1][0]["macro_f1_global"] == 0.7


class TestCreateMergedRecords:
    """Test _create_merged_records function."""

    def test_merge_server_and_client_data(self):
        """Test merging server and client data by round."""
        config = {
            "aggregation": "fedavg",
            "alpha": 0.5,
            "adv_pct": 0,
            "seed": 42,
        }

        df_server = pd.DataFrame(
            {
                "round": [1, 2],
                "l2_to_benign_mean": [0.1, 0.2],
                "t_aggregate_ms": [10.0, 11.0],
            }
        )

        client_data_by_round = {
            1: [{"macro_f1_global": 0.5, "macro_f1_personalized": 0.55}],
            2: [{"macro_f1_global": 0.6, "macro_f1_personalized": 0.65}],
        }

        records = _create_merged_records(config, df_server, client_data_by_round)

        assert len(records) == 2
        assert records[0]["round"] == 1
        assert records[0]["aggregation"] == "fedavg"
        assert records[0]["l2_to_benign_mean"] == 0.1
        assert records[0]["macro_f1_global"] == 0.5
        assert records[1]["macro_f1_global"] == 0.6

    def test_missing_client_data_uses_nan(self):
        """Test that missing client data for a round uses NaN."""
        config = {"aggregation": "krum", "seed": 42}

        df_server = pd.DataFrame(
            {
                "round": [1, 2],
                "l2_to_benign_mean": [0.1, 0.2],
            }
        )

        client_data_by_round = {
            1: [{"macro_f1_global": 0.5, "macro_f1_personalized": 0.55}],
            # Round 2 has no client data
        }

        records = _create_merged_records(config, df_server, client_data_by_round)

        assert len(records) == 2
        assert records[0]["macro_f1_global"] == 0.5
        assert np.isnan(records[1]["macro_f1_global"])
        assert np.isnan(records[1]["macro_f1_personalized"])

    def test_averages_multiple_clients_per_round(self):
        """Test averaging when multiple clients report for same round."""
        config = {"aggregation": "bulyan", "seed": 42}

        df_server = pd.DataFrame(
            {
                "round": [1],
                "l2_to_benign_mean": [0.1],
            }
        )

        client_data_by_round = {
            1: [
                {"macro_f1_global": 0.5, "macro_f1_personalized": 0.55},
                {"macro_f1_global": 0.6, "macro_f1_personalized": 0.65},
                {"macro_f1_global": 0.7, "macro_f1_personalized": 0.75},
            ],
        }

        records = _create_merged_records(config, df_server, client_data_by_round)

        assert len(records) == 1
        # Average of 0.5, 0.6, 0.7 = 0.6
        assert abs(records[0]["macro_f1_global"] - 0.6) < 0.001
        # Average of 0.55, 0.65, 0.75 = 0.65
        assert abs(records[0]["macro_f1_personalized"] - 0.65) < 0.001


class TestLoadIIoTData:
    """Integration tests for load_iiot_data function."""

    def test_load_empty_directory_returns_empty_dataframe(self, tmp_path):
        """Test that empty directory returns empty DataFrame."""
        result = load_iiot_data(tmp_path)
        assert result.empty

    def test_custom_run_pattern(self, tmp_path):
        """Test that custom run_pattern parameter works."""
        # Create a run with custom pattern
        run_dir = tmp_path / "custom_pattern_test_run"
        run_dir.mkdir()

        # Create minimal valid data
        server_data = pd.DataFrame({"round": [1], "l2_to_benign_mean": [0.1]})
        server_data.to_csv(run_dir / "metrics.csv", index=False)

        result = load_iiot_data(tmp_path, run_pattern="custom_pattern*")
        assert not result.empty
        assert len(result) == 1

    def test_personalization_gain_column_computed(self, tmp_path):
        """Test that personalization_gain column is computed."""
        run_dir = tmp_path / "dsedge-iiotset-nightly_test"
        run_dir.mkdir()

        server_data = pd.DataFrame({"round": [1], "l2_to_benign_mean": [0.1]})
        server_data.to_csv(run_dir / "metrics.csv", index=False)

        client_data = pd.DataFrame(
            {
                "round": [1],
                "macro_f1_before": [0.5],
                "macro_f1_personalized": [0.6],
            }
        )
        client_data.to_csv(run_dir / "client_0_metrics.csv", index=False)

        result = load_iiot_data(tmp_path)

        assert "personalization_gain" in result.columns
        assert abs(result.iloc[0]["personalization_gain"] - 0.1) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
