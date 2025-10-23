#!/usr/bin/env python3
"""
Historical tracking and regression detection for FedProx nightly runs.

This module provides functionality to maintain a rolling 90-day baseline of
nightly experiment results and detect statistical regressions compared to
historical performance.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def append_to_baseline(
    current_summary: Dict[str, Any],
    baseline_path: Path,
    commit_sha: str = "unknown",
) -> None:
    """Append current run metrics to historical baseline CSV.

    Args:
        current_summary: Summary dict containing aggregated metrics
        baseline_path: Path to baseline CSV file
        commit_sha: Git commit SHA for tracking
    """
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = current_summary.get("run_timestamp")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    rows: List[Dict[str, Any]] = []
    raw_results = current_summary.get("raw_analysis_results", {})
    convergence = raw_results.get("convergence_analysis", {})

    for config_key, config_data in convergence.items():
        alpha = config_data.get("alpha")
        mu = config_data.get("mu")
        algorithm = config_data.get("algorithm", "unknown")
        l2_distance = config_data.get("final_l2_distance")
        cosine_sim = config_data.get("final_cosine_similarity")
        agg_time = config_data.get("avg_aggregation_time")

        if alpha is None or mu is None:
            continue

        rows.append(
            {
                "timestamp": timestamp,
                "commit_sha": commit_sha,
                "alpha": float(alpha),
                "mu": float(mu),
                "algorithm": algorithm,
                "final_l2_distance": float(l2_distance) if l2_distance is not None else float("nan"),
                "final_cosine_similarity": float(cosine_sim) if cosine_sim is not None else float("nan"),
                "avg_aggregation_time_ms": float(agg_time) if agg_time is not None else float("nan"),
            }
        )

    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if baseline_path.exists():
        df_existing = pd.read_csv(baseline_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(baseline_path, index=False)


def load_baseline_window(baseline_path: Path, window_days: int = 90) -> pd.DataFrame:
    """Load baseline data within specified time window.

    Args:
        baseline_path: Path to baseline CSV file
        window_days: Number of days to include in window

    Returns:
        DataFrame containing baseline data within window
    """
    if not baseline_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(baseline_path)
    if df.empty:
        return df

    if "timestamp" not in df.columns:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    cutoff = cutoff.replace(tzinfo=None)

    df_filtered = df[df["timestamp"] >= cutoff].copy()

    df_filtered = df_filtered.sort_values("timestamp")

    return df_filtered


def trim_baseline_to_window(baseline_path: Path, window_days: int = 90) -> None:
    """Trim baseline CSV to only keep data within window.

    Args:
        baseline_path: Path to baseline CSV file
        window_days: Number of days to retain
    """
    df_window = load_baseline_window(baseline_path, window_days)
    if not df_window.empty:
        df_window.to_csv(baseline_path, index=False)


def detect_regression(
    current_metrics: Dict[str, float],
    baseline_df: pd.DataFrame,
    metric_name: str = "final_l2_distance",
    threshold_std: float = 2.0,
) -> Dict[str, Any]:
    """Detect statistical regression compared to baseline.

    Args:
        current_metrics: Current run metrics dict
        baseline_df: Historical baseline DataFrame
        metric_name: Name of metric to check
        threshold_std: Z-score threshold for regression detection

    Returns:
        Dict containing regression analysis results
    """
    if baseline_df.empty or metric_name not in baseline_df.columns:
        return {
            "metric": metric_name,
            "regression_detected": False,
            "reason": "insufficient_baseline_data",
            "current": current_metrics.get(metric_name, float("nan")),
        }

    metric_values = baseline_df[metric_name].dropna()

    if len(metric_values) < 2:
        return {
            "metric": metric_name,
            "regression_detected": False,
            "reason": "insufficient_samples",
            "current": current_metrics.get(metric_name, float("nan")),
            "baseline_n": len(metric_values),
        }

    baseline_mean = float(metric_values.mean())
    baseline_std = float(metric_values.std(ddof=1))

    if baseline_std == 0.0:
        return {
            "metric": metric_name,
            "regression_detected": False,
            "reason": "zero_variance",
            "current": current_metrics.get(metric_name, float("nan")),
            "baseline_mean": baseline_mean,
        }

    current_value = current_metrics.get(metric_name)
    if current_value is None or math.isnan(current_value):
        return {
            "metric": metric_name,
            "regression_detected": False,
            "reason": "missing_current_value",
        }

    current_value = float(current_value)
    z_score = (current_value - baseline_mean) / baseline_std

    regression_detected = z_score > threshold_std

    return {
        "metric": metric_name,
        "regression_detected": regression_detected,
        "z_score": z_score,
        "current": current_value,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_n": len(metric_values),
        "threshold_std": threshold_std,
    }


def plot_metric_trend_90d(
    baseline_df: pd.DataFrame,
    current_point: Optional[Dict[str, Any]],
    metric: str,
    output_path: Path,
    metric_label: Optional[str] = None,
) -> None:
    """Generate 90-day trend plot for specified metric.

    Args:
        baseline_df: Historical baseline DataFrame
        current_point: Current run data point (optional)
        metric: Column name of metric to plot
        output_path: Path to save plot
        metric_label: Human-readable label for metric (optional)
    """
    if baseline_df.empty or metric not in baseline_df.columns:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = baseline_df.copy()
    if "timestamp" in df_plot.columns:
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if not df_plot.empty and metric in df_plot.columns:
        metric_data = df_plot[[metric]].dropna()
        if not metric_data.empty and "timestamp" in df_plot.columns:
            timestamps = df_plot.loc[metric_data.index, "timestamp"]
            ax.plot(
                timestamps,
                metric_data[metric],
                marker="o",
                alpha=0.6,
                linewidth=1.5,
                markersize=4,
                label="Historical",
            )

            if len(metric_data) >= 7:
                rolling_mean = metric_data[metric].rolling(window=7, center=True).mean()
                rolling_std = metric_data[metric].rolling(window=7, center=True).std()

                valid_indices = rolling_mean.notna() & rolling_std.notna()
                if valid_indices.any():
                    timestamps_valid = df_plot.loc[rolling_mean[valid_indices].index, "timestamp"]
                    ax.fill_between(
                        timestamps_valid,
                        rolling_mean[valid_indices] - rolling_std[valid_indices],
                        rolling_mean[valid_indices] + rolling_std[valid_indices],
                        alpha=0.2,
                        label="Rolling Mean +/- 1 Std",
                    )

    if current_point and metric in current_point:
        current_timestamp = current_point.get("timestamp")
        if current_timestamp:
            if isinstance(current_timestamp, str):
                current_timestamp = pd.to_datetime(current_timestamp)
            ax.scatter(
                [current_timestamp],
                [current_point[metric]],
                s=200,
                color="red",
                marker="*",
                label="Current Run",
                zorder=5,
            )

    display_label = metric_label if metric_label else metric.replace("_", " ").title()
    ax.set_xlabel("Date")
    ax.set_ylabel(display_label)
    ax.set_title(f"{display_label} - 90 Day Trend")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_regression_report(
    current_summary: Dict[str, Any],
    baseline_df: pd.DataFrame,
    metrics_to_check: Optional[List[str]] = None,
    threshold_std: float = 2.0,
) -> Dict[str, Any]:
    """Generate comprehensive regression report.

    Args:
        current_summary: Current run summary dict
        baseline_df: Historical baseline DataFrame
        metrics_to_check: List of metric names to check (optional)
        threshold_std: Z-score threshold for regression

    Returns:
        Dict containing full regression analysis
    """
    if metrics_to_check is None:
        metrics_to_check = ["final_l2_distance", "final_cosine_similarity", "avg_aggregation_time_ms"]

    convergence = current_summary.get("raw_analysis_results", {}).get("convergence_analysis", {})

    regression_results: List[Dict[str, Any]] = []
    any_regression = False

    for config_key, config_data in convergence.items():
        alpha = config_data.get("alpha")
        mu = config_data.get("mu")

        if alpha is None or mu is None:
            continue

        config_baseline = baseline_df[
            (baseline_df["alpha"] == alpha) & (baseline_df["mu"] == mu)
        ].copy()

        for metric_name in metrics_to_check:
            current_value = config_data.get(metric_name)
            if current_value is None:
                continue

            current_metrics = {metric_name: current_value}
            regression_result = detect_regression(
                current_metrics, config_baseline, metric_name, threshold_std
            )

            regression_result["alpha"] = alpha
            regression_result["mu"] = mu
            regression_result["config_key"] = config_key

            if regression_result.get("regression_detected"):
                any_regression = True

            regression_results.append(regression_result)

    return {
        "any_regression_detected": any_regression,
        "threshold_std": threshold_std,
        "baseline_window_days": 90,
        "regression_results": regression_results,
    }
