#!/usr/bin/env python3
"""
Unit tests for historical tracking and regression detection.
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.historical_tracking import (
    append_to_baseline,
    detect_regression,
    generate_regression_report,
    load_baseline_window,
    plot_metric_trend_90d,
    trim_baseline_to_window,
)


@pytest.fixture
def temp_baseline_path(tmp_path: Path) -> Path:
    return tmp_path / "historical" / "baselines.csv"


@pytest.fixture
def sample_summary() -> dict:
    return {
        "run_timestamp": "2025-10-23T12:00:00+00:00",
        "raw_analysis_results": {
            "convergence_analysis": {
                "alpha_0.1_mu_0.0": {
                    "alpha": 0.1,
                    "mu": 0.0,
                    "algorithm": "FedAvg",
                    "final_l2_distance": 0.15,
                    "final_cosine_similarity": 0.98,
                    "avg_aggregation_time_ms": 120.5,
                },
                "alpha_0.1_mu_0.01": {
                    "alpha": 0.1,
                    "mu": 0.01,
                    "algorithm": "FedProx",
                    "final_l2_distance": 0.12,
                    "final_cosine_similarity": 0.99,
                    "avg_aggregation_time_ms": 125.0,
                },
            }
        },
    }


@pytest.fixture
def sample_baseline_df() -> pd.DataFrame:
    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    timestamps = [base_time + timedelta(days=i) for i in range(10)]

    return pd.DataFrame(
        {
            "timestamp": [t.isoformat() for t in timestamps],
            "commit_sha": ["abc123"] * 10,
            "alpha": [0.1] * 10,
            "mu": [0.0] * 10,
            "algorithm": ["FedAvg"] * 10,
            "final_l2_distance": [0.15, 0.14, 0.16, 0.15, 0.13, 0.15, 0.14, 0.16, 0.15, 0.14],
            "final_cosine_similarity": [0.98] * 10,
            "avg_aggregation_time_ms": [120.0] * 10,
        }
    )


describe_append_to_baseline = "append_to_baseline"


def test_append_to_baseline_creates_new_file(temp_baseline_path: Path, sample_summary: dict) -> None:
    append_to_baseline(sample_summary, temp_baseline_path, commit_sha="test123")

    assert temp_baseline_path.exists()
    df = pd.read_csv(temp_baseline_path)
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert "commit_sha" in df.columns
    assert "alpha" in df.columns
    assert "mu" in df.columns
    assert df["commit_sha"].iloc[0] == "test123"


def test_append_to_baseline_appends_to_existing(temp_baseline_path: Path, sample_summary: dict) -> None:
    append_to_baseline(sample_summary, temp_baseline_path, commit_sha="first")

    sample_summary2 = sample_summary.copy()
    sample_summary2["run_timestamp"] = "2025-10-24T12:00:00+00:00"
    append_to_baseline(sample_summary2, temp_baseline_path, commit_sha="second")

    df = pd.read_csv(temp_baseline_path)
    assert len(df) == 4
    assert df["commit_sha"].iloc[0] == "first"
    assert df["commit_sha"].iloc[2] == "second"


def test_append_to_baseline_handles_missing_timestamp(temp_baseline_path: Path) -> None:
    summary_no_timestamp = {
        "raw_analysis_results": {
            "convergence_analysis": {
                "alpha_0.1_mu_0.0": {
                    "alpha": 0.1,
                    "mu": 0.0,
                    "algorithm": "FedAvg",
                    "final_l2_distance": 0.15,
                    "final_cosine_similarity": 0.98,
                    "avg_aggregation_time": 120.5,
                }
            }
        }
    }

    append_to_baseline(summary_no_timestamp, temp_baseline_path)

    assert temp_baseline_path.exists()
    df = pd.read_csv(temp_baseline_path)
    assert len(df) == 1
    assert pd.notna(df["timestamp"].iloc[0])


def test_append_to_baseline_handles_missing_metrics(temp_baseline_path: Path) -> None:
    summary_missing = {
        "run_timestamp": "2025-10-23T12:00:00+00:00",
        "raw_analysis_results": {
            "convergence_analysis": {
                "alpha_0.1_mu_0.0": {
                    "alpha": 0.1,
                    "mu": 0.0,
                    "algorithm": "FedAvg",
                    "final_l2_distance": None,
                    "final_cosine_similarity": None,
                    "avg_aggregation_time": None,
                }
            }
        },
    }

    append_to_baseline(summary_missing, temp_baseline_path)

    df = pd.read_csv(temp_baseline_path)
    assert len(df) == 1
    assert math.isnan(df["final_l2_distance"].iloc[0])


def test_append_to_baseline_skips_invalid_entries(temp_baseline_path: Path) -> None:
    summary_invalid = {
        "run_timestamp": "2025-10-23T12:00:00+00:00",
        "raw_analysis_results": {
            "convergence_analysis": {
                "invalid_config": {
                    "alpha": None,
                    "mu": None,
                }
            }
        },
    }

    append_to_baseline(summary_invalid, temp_baseline_path)

    assert not temp_baseline_path.exists() or pd.read_csv(temp_baseline_path).empty


describe_load_baseline_window = "load_baseline_window"


def test_load_baseline_window_returns_empty_for_missing_file(temp_baseline_path: Path) -> None:
    df = load_baseline_window(temp_baseline_path, window_days=90)
    assert df.empty


def test_load_baseline_window_filters_by_time(temp_baseline_path: Path) -> None:
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=100)
    recent_time = now - timedelta(days=30)

    df_baseline = pd.DataFrame(
        {
            "timestamp": [old_time.isoformat(), recent_time.isoformat()],
            "commit_sha": ["old", "recent"],
            "alpha": [0.1, 0.1],
            "mu": [0.0, 0.0],
            "algorithm": ["FedAvg", "FedAvg"],
            "final_l2_distance": [0.15, 0.14],
            "final_cosine_similarity": [0.98, 0.98],
            "avg_aggregation_time_ms": [120.0, 120.0],
        }
    )

    temp_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df_baseline.to_csv(temp_baseline_path, index=False)

    df_loaded = load_baseline_window(temp_baseline_path, window_days=90)

    assert len(df_loaded) == 1
    assert df_loaded["commit_sha"].iloc[0] == "recent"


def test_load_baseline_window_sorts_by_timestamp(temp_baseline_path: Path) -> None:
    now = datetime.now(timezone.utc)
    times = [now - timedelta(days=i) for i in [10, 5, 15]]

    df_baseline = pd.DataFrame(
        {
            "timestamp": [t.isoformat() for t in times],
            "commit_sha": ["third", "second", "first"],
            "alpha": [0.1] * 3,
            "mu": [0.0] * 3,
            "algorithm": ["FedAvg"] * 3,
            "final_l2_distance": [0.15] * 3,
            "final_cosine_similarity": [0.98] * 3,
            "avg_aggregation_time_ms": [120.0] * 3,
        }
    )

    temp_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df_baseline.to_csv(temp_baseline_path, index=False)

    df_loaded = load_baseline_window(temp_baseline_path, window_days=90)

    assert len(df_loaded) == 3
    assert df_loaded["commit_sha"].iloc[0] == "first"
    assert df_loaded["commit_sha"].iloc[1] == "third"
    assert df_loaded["commit_sha"].iloc[2] == "second"


def test_load_baseline_window_handles_missing_timestamp_column(temp_baseline_path: Path) -> None:
    df_no_timestamp = pd.DataFrame({"alpha": [0.1], "mu": [0.0]})

    temp_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df_no_timestamp.to_csv(temp_baseline_path, index=False)

    df_loaded = load_baseline_window(temp_baseline_path, window_days=90)

    assert len(df_loaded) == 1


describe_trim_baseline_to_window = "trim_baseline_to_window"


def test_trim_baseline_to_window_removes_old_data(temp_baseline_path: Path) -> None:
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=100)
    recent_time = now - timedelta(days=30)

    df_baseline = pd.DataFrame(
        {
            "timestamp": [old_time.isoformat(), recent_time.isoformat()],
            "commit_sha": ["old", "recent"],
            "alpha": [0.1, 0.1],
            "mu": [0.0, 0.0],
            "algorithm": ["FedAvg", "FedAvg"],
            "final_l2_distance": [0.15, 0.14],
            "final_cosine_similarity": [0.98, 0.98],
            "avg_aggregation_time_ms": [120.0, 120.0],
        }
    )

    temp_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df_baseline.to_csv(temp_baseline_path, index=False)

    trim_baseline_to_window(temp_baseline_path, window_days=90)

    df_trimmed = pd.read_csv(temp_baseline_path)
    assert len(df_trimmed) == 1
    assert df_trimmed["commit_sha"].iloc[0] == "recent"


def test_trim_baseline_to_window_handles_empty_file(temp_baseline_path: Path) -> None:
    df_empty = pd.DataFrame()
    temp_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df_empty.to_csv(temp_baseline_path, index=False)

    trim_baseline_to_window(temp_baseline_path, window_days=90)

    assert temp_baseline_path.exists()


describe_detect_regression = "detect_regression"


def test_detect_regression_no_regression(sample_baseline_df: pd.DataFrame) -> None:
    current_metrics = {"final_l2_distance": 0.15}

    result = detect_regression(current_metrics, sample_baseline_df, "final_l2_distance", threshold_std=2.0)

    assert result["regression_detected"] is False
    assert result["metric"] == "final_l2_distance"
    assert "z_score" in result
    assert abs(result["z_score"]) < 2.0


def test_detect_regression_with_regression(sample_baseline_df: pd.DataFrame) -> None:
    current_metrics = {"final_l2_distance": 0.50}

    result = detect_regression(current_metrics, sample_baseline_df, "final_l2_distance", threshold_std=2.0)

    assert result["regression_detected"] is True
    assert result["z_score"] > 2.0
    assert result["current"] == 0.50


def test_detect_regression_empty_baseline() -> None:
    empty_df = pd.DataFrame()
    current_metrics = {"final_l2_distance": 0.15}

    result = detect_regression(current_metrics, empty_df, "final_l2_distance")

    assert result["regression_detected"] is False
    assert result["reason"] == "insufficient_baseline_data"


def test_detect_regression_insufficient_samples() -> None:
    single_sample_df = pd.DataFrame({"final_l2_distance": [0.15]})
    current_metrics = {"final_l2_distance": 0.20}

    result = detect_regression(current_metrics, single_sample_df, "final_l2_distance")

    assert result["regression_detected"] is False
    assert result["reason"] == "insufficient_samples"
    assert result["baseline_n"] == 1


def test_detect_regression_zero_variance() -> None:
    zero_var_df = pd.DataFrame({"final_l2_distance": [0.15, 0.15, 0.15]})
    current_metrics = {"final_l2_distance": 0.15}

    result = detect_regression(current_metrics, zero_var_df, "final_l2_distance")

    assert result["regression_detected"] is False
    assert result["reason"] == "zero_variance"


def test_detect_regression_missing_current_value(sample_baseline_df: pd.DataFrame) -> None:
    current_metrics = {"other_metric": 0.15}

    result = detect_regression(current_metrics, sample_baseline_df, "final_l2_distance")

    assert result["regression_detected"] is False
    assert result["reason"] == "missing_current_value"


def test_detect_regression_with_nan_current(sample_baseline_df: pd.DataFrame) -> None:
    current_metrics = {"final_l2_distance": float("nan")}

    result = detect_regression(current_metrics, sample_baseline_df, "final_l2_distance")

    assert result["regression_detected"] is False
    assert result["reason"] == "missing_current_value"


def test_detect_regression_baseline_n_reported(sample_baseline_df: pd.DataFrame) -> None:
    current_metrics = {"final_l2_distance": 0.15}

    result = detect_regression(current_metrics, sample_baseline_df, "final_l2_distance")

    assert result["baseline_n"] == 10


describe_plot_metric_trend_90d = "plot_metric_trend_90d"


def test_plot_metric_trend_90d_creates_file(tmp_path: Path, sample_baseline_df: pd.DataFrame) -> None:
    output_path = tmp_path / "trend.png"
    current_point = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_l2_distance": 0.15,
    }

    plot_metric_trend_90d(sample_baseline_df, current_point, "final_l2_distance", output_path)

    assert output_path.exists()


def test_plot_metric_trend_90d_handles_empty_baseline(tmp_path: Path) -> None:
    output_path = tmp_path / "trend.png"
    empty_df = pd.DataFrame()

    plot_metric_trend_90d(empty_df, None, "final_l2_distance", output_path)

    assert not output_path.exists()


def test_plot_metric_trend_90d_with_custom_label(tmp_path: Path, sample_baseline_df: pd.DataFrame) -> None:
    output_path = tmp_path / "trend.png"

    plot_metric_trend_90d(
        sample_baseline_df,
        None,
        "final_l2_distance",
        output_path,
        metric_label="L2 Distance to Benign Mean",
    )

    assert output_path.exists()


def test_plot_metric_trend_90d_without_current_point(tmp_path: Path, sample_baseline_df: pd.DataFrame) -> None:
    output_path = tmp_path / "trend.png"

    plot_metric_trend_90d(sample_baseline_df, None, "final_l2_distance", output_path)

    assert output_path.exists()


describe_generate_regression_report = "generate_regression_report"


def test_generate_regression_report_no_regression(sample_summary: dict, sample_baseline_df: pd.DataFrame) -> None:
    report = generate_regression_report(sample_summary, sample_baseline_df)

    assert report["any_regression_detected"] is False
    assert report["threshold_std"] == 2.0
    assert report["baseline_window_days"] == 90
    assert len(report["regression_results"]) > 0


def test_generate_regression_report_with_regression(sample_summary: dict) -> None:
    baseline_df = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc).isoformat()] * 10,
            "alpha": [0.1] * 10,
            "mu": [0.0] * 10,
            "final_l2_distance": [0.05, 0.06, 0.05, 0.05, 0.06, 0.05, 0.05, 0.06, 0.05, 0.05],
            "final_cosine_similarity": [0.99] * 10,
            "avg_aggregation_time_ms": [100.0, 101.0, 100.0, 100.0, 101.0, 100.0, 100.0, 101.0, 100.0, 100.0],
        }
    )

    report = generate_regression_report(sample_summary, baseline_df, threshold_std=1.0)

    assert report["any_regression_detected"] is True


def test_generate_regression_report_custom_metrics(sample_summary: dict, sample_baseline_df: pd.DataFrame) -> None:
    report = generate_regression_report(sample_summary, sample_baseline_df, metrics_to_check=["final_l2_distance"])

    assert all(r["metric"] == "final_l2_distance" for r in report["regression_results"])


def test_generate_regression_report_empty_baseline(sample_summary: dict) -> None:
    empty_df = pd.DataFrame()

    report = generate_regression_report(sample_summary, empty_df)

    assert report["any_regression_detected"] is False
    assert len(report["regression_results"]) > 0


def test_generate_regression_report_includes_config_info(
    sample_summary: dict, sample_baseline_df: pd.DataFrame
) -> None:
    report = generate_regression_report(sample_summary, sample_baseline_df)

    for result in report["regression_results"]:
        assert "alpha" in result
        assert "mu" in result
        assert "config_key" in result
