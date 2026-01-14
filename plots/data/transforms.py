"""
Data Transformation Utilities

Functions for aggregating, pivoting, and computing statistics on
experiment data for visualization.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval(
    data: np.ndarray | Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute mean and confidence interval using t-distribution.

    Args:
        data: Array of values (NaN values should be removed before calling)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Raises:
        ValueError: If data is empty
    """
    data_array = np.asarray(data)

    if len(data_array) == 0:
        raise ValueError("Cannot compute confidence interval for empty data")

    if len(data_array) == 1:
        mean = float(data_array[0])
        return mean, mean, mean

    mean = float(np.mean(data_array))
    se = stats.sem(data_array)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)
    return mean, mean - margin, mean + margin


def aggregate_by_seed(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "macro_f1",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate experiment results across seeds with confidence intervals.

    Args:
        df: DataFrame with experiment results including 'seed' column
        group_cols: Columns to group by (excluding seed)
        value_col: Column containing metric values
        confidence: Confidence level for CI computation

    Returns:
        DataFrame with mean, ci_lower, ci_upper columns
    """
    if "seed" not in df.columns:
        return df.groupby(group_cols)[value_col].agg(["mean", "std"]).reset_index()

    def compute_ci(group: pd.Series) -> pd.Series:
        values = group.dropna().values
        if len(values) == 0:
            return pd.Series({"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
        mean, lower, upper = compute_confidence_interval(values, confidence)
        return pd.Series({"mean": mean, "ci_lower": lower, "ci_upper": upper})

    return df.groupby(group_cols)[value_col].apply(compute_ci).unstack().reset_index()


def pivot_for_heatmap(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str = "macro_f1",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """
    Pivot DataFrame for heatmap visualization.

    Args:
        df: Source DataFrame
        index_col: Column for heatmap rows
        columns_col: Column for heatmap columns
        values_col: Column containing values
        aggfunc: Aggregation function ('mean', 'std', etc.)

    Returns:
        Pivoted DataFrame suitable for seaborn heatmap
    """
    return df.pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc=aggfunc,
    )


def prepare_ci_matrix(
    df: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Prepare matrix for confidence interval computation across seeds.

    Args:
        df: DataFrame with 'round' and 'seed' columns
        column: Column to extract values from

    Returns:
        Tuple of (rounds array, values matrix) or None if insufficient data
    """
    if "seed" not in df.columns:
        return None

    pivot = (
        df.pivot_table(index="round", columns="seed", values=column)
        .sort_index()
        .dropna(how="all")
    )
    if pivot.empty:
        return None

    matrix = pivot.to_numpy().T
    valid_rows = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid_rows]

    if matrix.shape[0] < 2:
        return None

    rounds = pivot.index.to_numpy()
    return rounds, matrix
