"""
Data Validation Utilities

Automatic detection of available metrics in DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MetricDetector:
    """Auto-detect available metrics in DataFrame columns."""

    server_metric_map: dict[str, list[str]] = field(
        default_factory=lambda: {
            "timing": ["t_aggregate_ms", "aggregation_time_ms", "t_round_ms"],
            "robustness": ["l2_to_benign_mean", "cos_to_benign_mean"],
            "norms": ["update_norm_mean", "update_norm_std"],
            "dispersion": ["pairwise_cosine_mean", "l2_dispersion_mean"],
        }
    )

    client_metric_map: dict[str, list[str]] = field(
        default_factory=lambda: {
            "loss": ["loss_after", "local_loss"],
            "accuracy": ["acc_after", "local_accuracy"],
            "norms": ["weight_norm_after", "weight_norm"],
            "grad_norms": ["grad_norm_l2"],
            "f1_comparison": ["macro_f1_argmax", "f1_bin_tau"],
            "threshold": ["tau_bin", "threshold_tau"],
            "fpr": ["benign_fpr_bin_tau", "fpr_after"],
        }
    )

    def detect_available(
        self,
        df: pd.DataFrame,
        metric_type: str = "server",
    ) -> dict[str, str | None]:
        """
        Detect which metrics are available in DataFrame.

        Args:
            df: DataFrame to inspect
            metric_type: 'server' or 'client'

        Returns:
            Dict mapping category -> found column name (or None)
        """
        metric_map = (
            self.server_metric_map
            if metric_type == "server"
            else self.client_metric_map
        )
        available = {}

        for category, column_options in metric_map.items():
            found = None
            for col in column_options:
                if col in df.columns and not df[col].isna().all():
                    found = col
                    break
            available[category] = found

        return available

    def count_available_plots(
        self,
        available_metrics: dict[str, str | None],
    ) -> int:
        """Count how many plots can be generated from available metrics."""
        return sum(1 for v in available_metrics.values() if v is not None)


def first_present(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    """
    Return first column that exists in DataFrame.

    Args:
        df: DataFrame to search
        columns: List of column names in priority order

    Returns:
        Series of values from first found column, or None
    """
    for name in columns:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None
