#!/usr/bin/env python3
"""
Integration tests for generate_comprehensive_thesis_plots module.

Tests end-to-end plot generation and validation.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from generate_comprehensive_thesis_plots import (  # noqa: E402
    _validate_dataframe_schema,
    plot_objective1_robustness,
)


class TestValidateDataFrameSchema:
    """Test _validate_dataframe_schema function."""

    def test_valid_schema_passes(self):
        """Test that DataFrame with all required columns passes validation."""
        df = pd.DataFrame(
            {
                "aggregation": ["fedavg"],
                "alpha": [0.5],
                "round": [1],
            }
        )

        # Should not raise
        _validate_dataframe_schema(df, ["aggregation", "alpha", "round"])

    def test_missing_columns_raises_valueerror(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame(
            {
                "aggregation": ["fedavg"],
                "alpha": [0.5],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            _validate_dataframe_schema(df, ["aggregation", "alpha", "round"])

        assert "missing required columns" in str(exc_info.value).lower()
        assert "round" in str(exc_info.value)

    def test_error_message_shows_available_columns(self):
        """Test that error message lists available columns."""
        df = pd.DataFrame(
            {
                "col1": [1],
                "col2": [2],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            _validate_dataframe_schema(df, ["col3"])

        error_msg = str(exc_info.value)
        assert "col1" in error_msg
        assert "col2" in error_msg


class TestPlotObjective1Integration:
    """Integration tests for Objective 1 plot generation."""

    @pytest.fixture
    def minimal_test_data(self):
        """Create minimal test DataFrame for plot generation."""
        np.random.seed(42)

        records = []
        for agg in ["fedavg", "krum", "bulyan", "median"]:
            for adv in [0, 10, 30]:
                for seed in [42, 43]:
                    for round_num in range(1, 21):
                        records.append(
                            {
                                "aggregation": agg,
                                "alpha": 0.5,
                                "adv_pct": adv,
                                "seed": seed,
                                "round": round_num,
                                "l2_to_benign_mean": np.random.rand() * adv / 10,
                                "l2_dispersion_mean": np.random.rand() * 0.5,
                                "t_aggregate_ms": np.random.rand() * 50 + 10,
                                "macro_f1_global": max(0.1, 0.7 - adv / 100 + np.random.rand() * 0.1),
                                "macro_f1_personalized": max(0.1, 0.75 - adv / 100 + np.random.rand() * 0.1),
                                "pers_epochs": 3,
                            }
                        )

        return pd.DataFrame(records)

    def test_plot_objective1_creates_file(self, tmp_path, minimal_test_data):
        """Test that plot_objective1_robustness creates output file."""
        plot_objective1_robustness(minimal_test_data, tmp_path)

        output_file = tmp_path / "obj1_robustness_comprehensive.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 10000  # At least 10KB

    def test_plot_objective1_with_missing_columns_raises_error(self, tmp_path):
        """Test that missing required columns raise validation error."""
        incomplete_df = pd.DataFrame(
            {
                "aggregation": ["fedavg"],
                "alpha": [0.5],
                # Missing other required columns
            }
        )

        with pytest.raises(ValueError) as exc_info:
            plot_objective1_robustness(incomplete_df, tmp_path)

        assert "missing required columns" in str(exc_info.value).lower()

    def test_plot_objective1_handles_empty_filtered_data(self, tmp_path, minimal_test_data):
        """Test that plot handles case where filtering produces empty data."""
        # Create data with no matching aggregators
        df = minimal_test_data.copy()
        df["aggregation"] = "unknown_aggregator"

        # Should complete without error (plots may be empty but file created)
        plot_objective1_robustness(df, tmp_path)

        output_file = tmp_path / "obj1_robustness_comprehensive.png"
        assert output_file.exists()


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_from_csv_to_plots(self, tmp_path):
        """Test complete pipeline: Load CSVs -> Generate plots."""
        # Create mock experiment run directory
        run_dir = tmp_path / "dsedge-iiotset-nightly_comp_fedavg_alpha0.5_adv0_dp0_pers0_mu0.0_seed42"
        run_dir.mkdir()

        # Create server metrics
        server_data = pd.DataFrame(
            {
                "round": list(range(1, 11)),
                "l2_to_benign_mean": np.random.rand(10) * 0.5,
                "l2_dispersion_mean": np.random.rand(10) * 0.3,
                "t_aggregate_ms": np.random.rand(10) * 20 + 5,
            }
        )
        server_data.to_csv(run_dir / "metrics.csv", index=False)

        # Create client metrics
        for client_id in range(3):
            client_data = pd.DataFrame(
                {
                    "round": list(range(1, 11)),
                    "macro_f1_before": 0.6 + np.random.rand(10) * 0.1,
                    "macro_f1_personalized": 0.65 + np.random.rand(10) * 0.1,
                }
            )
            client_data.to_csv(run_dir / f"client_{client_id}_metrics.csv", index=False)

        # Load data
        from load_iiot_data import load_iiot_data

        df = load_iiot_data(tmp_path)

        assert not df.empty
        assert len(df) == 10  # 10 rounds

        # Generate plots (only Objective 1 for integration test)
        output_dir = tmp_path / "test_plots"
        output_dir.mkdir()

        plot_objective1_robustness(df, output_dir)

        # Verify output
        assert (output_dir / "obj1_robustness_comprehensive.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
