"""
Unit tests for privacy-utility curve generation.

Following TDD approach (CLAUDE.md C-1): write tests first, then implement.
Tests cover:
- Curve data preparation from experiment runs
- Epsilon computation with formal DP accounting
- Multi-seed aggregation with confidence intervals
- Edge cases (missing data, no DP, single seeds)
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_thesis_plots import (
    _prepare_privacy_curve_data,
    _render_privacy_curve,
    _extract_macro_f1,
    _compute_epsilon_fallback,
)


class TestExtractMacroF1:
    """Test macro-F1 extraction from result rows."""

    def test_extract_from_macro_f1_after_column(self):
        """Should extract macro_f1_after if present."""
        row = pd.Series(
            {
                "macro_f1_after": 0.85,
                "macro_f1_global": 0.80,
            }
        )
        result = _extract_macro_f1(row)
        assert result == 0.85

    def test_extract_from_macro_f1_global_column(self):
        """Should extract macro_f1_global if _after not present."""
        row = pd.Series(
            {
                "macro_f1_global": 0.80,
                "macro_f1_personalized": 0.82,
            }
        )
        result = _extract_macro_f1(row)
        assert result == 0.80

    def test_extract_returns_none_when_no_macro_columns(self):
        """Should return None when no macro-F1 columns present."""
        row = pd.Series({"some_other_metric": 0.5})
        result = _extract_macro_f1(row)
        assert result is None

    def test_extract_skips_nan_values(self):
        """Should skip NaN values and try next column."""
        row = pd.Series(
            {
                "macro_f1_after": float("nan"),
                "macro_f1_global": 0.80,
            }
        )
        result = _extract_macro_f1(row)
        assert result == 0.80


class TestComputeEpsilonFallback:
    """Test epsilon computation fallback logic."""

    def test_compute_epsilon_from_noise_multiplier(self):
        """Should compute epsilon from noise multiplier in row."""
        row = {
            "dp_noise_multiplier": 1.0,
            "dp_delta": 1e-5,
            "round": 10,
            "dp_sample_rate": 1.0,
        }
        epsilon = _compute_epsilon_fallback(row, None)
        assert epsilon is not None
        assert 0 < epsilon < float("inf")
        assert isinstance(epsilon, float)

    def test_no_epsilon_when_no_noise(self):
        """Should return None when no noise multiplier provided."""
        row = {
            "round": 10,
            "dp_delta": 1e-5,
        }
        epsilon = _compute_epsilon_fallback(row, None)
        assert epsilon is None

    def test_no_epsilon_when_noise_zero(self):
        """Should return None when noise multiplier is 0 or negative."""
        row = {
            "dp_noise_multiplier": 0.0,
            "round": 10,
        }
        epsilon = _compute_epsilon_fallback(row, None)
        assert epsilon is None

    def test_uses_final_row_values_when_available(self):
        """Should prefer dp_sigma from final_row if available."""
        row = {"dp_noise_multiplier": 0.5}
        final_row = pd.Series(
            {
                "dp_sigma": 1.0,
                "round": 20,
                "dp_delta": 1e-5,
                "dp_sample_rate": 1.0,
            }
        )
        epsilon = _compute_epsilon_fallback(row, final_row)
        assert epsilon is not None
        # Noise 1.0 should give different epsilon than 0.5
        epsilon_alt = _compute_epsilon_fallback(row, None)
        assert epsilon != epsilon_alt or epsilon_alt is None


class TestPreparePrivacyCurveData:
    """Test privacy curve data preparation from experiment runs."""

    def test_returns_empty_dataframes_without_run_dir_column(self):
        """Should return empty dataframes when run_dir column missing."""
        final_rounds = pd.DataFrame(
            {
                "seed": [42],
                "dp_enabled": [True],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir)
            dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)
            assert dp_df.empty
            assert baseline_df.empty

    def test_returns_empty_dataframes_when_run_dir_not_found(self):
        """Should return empty dataframes when run directories don't exist."""
        final_rounds = pd.DataFrame(
            {
                "run_dir": ["nonexistent_run"],
                "seed": [42],
                "dp_enabled": [True],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir)
            dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)
            assert dp_df.empty
            assert baseline_df.empty

    def test_builds_baseline_records_when_dp_disabled(self):
        """Should build baseline records when dp_enabled is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir)
            run_dir = runs_root / "test_run"
            run_dir.mkdir(parents=True)

            # Create client metrics files
            client_df = pd.DataFrame(
                {
                    "round": [1, 2, 3],
                    "macro_f1_after": [0.70, 0.75, 0.80],
                }
            )
            (run_dir / "client_0_metrics.csv").write_text(client_df.to_csv(index=False))
            (run_dir / "client_1_metrics.csv").write_text(client_df.to_csv(index=False))

            final_rounds = pd.DataFrame(
                {
                    "run_dir": [str(run_dir)],
                    "seed": [42],
                    "dp_enabled": [False],
                    "dp_noise_multiplier": [0.0],
                }
            )

            dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)

            assert dp_df.empty  # No DP records
            assert not baseline_df.empty
            assert "macro_f1" in baseline_df.columns
            assert baseline_df.iloc[0]["macro_f1"] == 0.80  # Last round F1

    def test_builds_dp_records_with_epsilon(self):
        """Should build DP records with computed epsilon values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir)
            run_dir = runs_root / "dp_run"
            run_dir.mkdir(parents=True)

            # Create client metrics files with DP fields
            client_df = pd.DataFrame(
                {
                    "round": [1, 2, 3],
                    "macro_f1_after": [0.70, 0.75, 0.80],
                    "dp_sigma": [1.0, 1.0, 1.0],
                    "dp_delta": [1e-5, 1e-5, 1e-5],
                }
            )
            (run_dir / "client_0_metrics.csv").write_text(client_df.to_csv(index=False))
            (run_dir / "client_1_metrics.csv").write_text(client_df.to_csv(index=False))

            final_rounds = pd.DataFrame(
                {
                    "run_dir": [str(run_dir)],
                    "seed": [42],
                    "dp_enabled": [True],
                    "dp_noise_multiplier": [1.0],
                    "dp_delta": [1e-5],
                    "round": [3],
                }
            )

            dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)

            assert not dp_df.empty
            assert baseline_df.empty
            assert "epsilon" in dp_df.columns
            assert "macro_f1" in dp_df.columns
            assert 0 < dp_df.iloc[0]["epsilon"] < float("inf")


class TestRenderPrivacyCurve:
    """Test privacy-utility curve rendering and visualization."""

    def test_handles_empty_dp_dataframe(self):
        """Should handle empty DP dataframe gracefully."""
        dp_df = pd.DataFrame()
        baseline_df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Should not raise exception
            _render_privacy_curve(dp_df, baseline_df, output_dir)
            # No output should be created
            assert not list(output_dir.glob("*.csv"))

    def test_generates_summary_csv(self):
        """Should generate privacy_utility_curve.csv summary."""
        dp_df = pd.DataFrame(
            {
                "epsilon": [0.5, 1.0, 2.0, 0.5, 1.0, 2.0],
                "macro_f1": [0.70, 0.75, 0.80, 0.72, 0.76, 0.82],
                "seed": [42, 42, 42, 43, 43, 43],
                "dp_noise_multiplier": [2.0, 1.0, 0.5, 2.0, 1.0, 0.5],
            }
        )
        baseline_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _render_privacy_curve(dp_df, baseline_df, output_dir)

            summary_path = output_dir / "privacy_utility_curve.csv"
            assert summary_path.exists()

            summary_df = pd.read_csv(summary_path)
            assert not summary_df.empty
            assert "epsilon" in summary_df.columns
            assert "macro_f1_mean" in summary_df.columns
            assert "ci_lower" in summary_df.columns
            assert "ci_upper" in summary_df.columns

    def test_confidence_intervals_are_valid(self):
        """Should compute valid confidence intervals (ci_lower <= mean <= ci_upper)."""
        # Multi-seed data for CI computation
        dp_df = pd.DataFrame(
            {
                "epsilon": [1.0] * 5,
                "macro_f1": [0.75, 0.76, 0.74, 0.77, 0.76],
                "seed": [42, 43, 44, 45, 46],
                "dp_noise_multiplier": [1.0] * 5,
            }
        )
        baseline_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _render_privacy_curve(dp_df, baseline_df, output_dir)

            summary_path = output_dir / "privacy_utility_curve.csv"
            summary_df = pd.read_csv(summary_path)

            for _, row in summary_df.iterrows():
                if pd.notna(row["ci_lower"]):
                    assert row["ci_lower"] <= row["macro_f1_mean"]
                    assert row["macro_f1_mean"] <= row["ci_upper"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
