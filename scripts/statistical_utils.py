"""Reusable statistical utility functions for analysis.

This module provides statistically rigorous functions for computing confidence
intervals, hypothesis tests, and effect sizes with proper handling of edge cases
and NaN values.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def compute_ci(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute confidence interval for a set of values using t-distribution.

    Args:
        values: Sequence of numeric values (NaN values are filtered)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, ci_lower, ci_upper). If insufficient data,
        all three values equal the mean (or NaN if no valid data).
    """
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)

    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(arr.mean())

    if arr.size == 1:
        return mean, mean, mean

    std = float(arr.std(ddof=1))
    if std == 0.0:
        return mean, mean, mean

    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=arr.size - 1))
    margin = t_crit * std / math.sqrt(arr.size)
    return mean, mean - margin, mean + margin


def paired_t_test(
    group1: Sequence[float],
    group2: Sequence[float],
) -> dict[str, object]:
    """Perform paired t-test comparing two groups.

    Filters NaN values and pairs remaining values by position.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary containing:
            - t_stat: t-statistic (NaN if n < 2)
            - p_value: two-tailed p-value (NaN if n < 2)
            - mean_diff: mean difference (group1 - group2)
            - pairs: number of valid paired observations
            - n: same as pairs for consistency
    """
    pairs: list[tuple[float, float]] = []

    for v1, v2 in zip(group1, group2):
        if not math.isnan(v1) and not math.isnan(v2):
            pairs.append((v1, v2))

    n = len(pairs)
    if n == 0:
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "mean_diff": float("nan"),
            "pairs": 0,
            "n": 0,
        }

    arr1 = np.array([v1 for v1, _ in pairs], dtype=float)
    arr2 = np.array([v2 for _, v2 in pairs], dtype=float)

    mean_diff = float(arr1.mean() - arr2.mean())

    if n == 1:
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "mean_diff": mean_diff,
            "pairs": n,
            "n": n,
        }

    diffs = arr1 - arr2
    diff_std = float(diffs.std(ddof=1))
    if diff_std == 0.0:
        t_stat = float("inf") if mean_diff != 0.0 else 0.0
        p_value = 0.0 if mean_diff != 0.0 else 1.0
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": mean_diff,
            "pairs": n,
            "n": n,
        }

    t_stat, p_value = stats.ttest_rel(arr1, arr2)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "pairs": n,
        "n": n,
    }


def cohens_d(
    group1: Sequence[float],
    group2: Sequence[float],
) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Filters NaN values before computation.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size (mean_diff / pooled_std), or NaN if
        insufficient data.
    """
    arr1 = np.array([v for v in group1 if not math.isnan(v)], dtype=float)
    arr2 = np.array([v for v in group2 if not math.isnan(v)], dtype=float)

    if arr1.size == 0 or arr2.size == 0:
        return float("nan")

    mean_diff = float(arr1.mean() - arr2.mean())
    n1, n2 = arr1.size, arr2.size

    if n1 == 1 and n2 == 1:
        return mean_diff

    var1 = float(arr1.var(ddof=1)) if n1 > 1 else 0.0
    var2 = float(arr2.var(ddof=1)) if n2 > 1 else 0.0

    pooled_std = math.sqrt((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2))

    if pooled_std == 0.0:
        return float("nan")

    return mean_diff / pooled_std


def mann_whitney_u(
    group1: Sequence[float],
    group2: Sequence[float],
) -> dict[str, object]:
    """Perform Mann-Whitney U non-parametric test.

    Filters NaN values before computation. Uses two-sided alternative.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary containing:
            - u_stat: U-statistic (NaN if insufficient data)
            - p_value: two-tailed p-value (NaN if insufficient data)
            - n1: number of valid values in group1
            - n2: number of valid values in group2
    """
    arr1 = np.array([v for v in group1 if not math.isnan(v)], dtype=float)
    arr2 = np.array([v for v in group2 if not math.isnan(v)], dtype=float)

    n1, n2 = arr1.size, arr2.size

    if n1 == 0 or n2 == 0 or n1 < 1 or n2 < 1:
        return {
            "u_stat": float("nan"),
            "p_value": float("nan"),
            "n1": n1,
            "n2": n2,
        }

    u_stat, p_value = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
    return {
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "n1": n1,
        "n2": n2,
    }
