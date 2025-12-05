#!/usr/bin/env python3
"""
Statistical Analysis Utilities for IIoT Federated Learning Experiments

Provides:
- Independent t-tests for comparing aggregation methods
- Cohen's d effect size calculation
- Multiple comparison correction (Bonferroni)
- Pairwise comparison matrices
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Result of statistical comparison between two groups."""

    group1: str
    group2: str
    mean1: float
    mean2: float
    std1: float
    std2: float
    n1: int
    n2: int
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool
    alpha: float = 0.05

    def __str__(self) -> str:
        sig_marker = "***" if self.significant else "n.s."
        return (
            f"{self.group1} vs {self.group2}: "
            f"Δμ={self.mean1 - self.mean2:.4f}, "
            f"t={self.t_statistic:.3f}, "
            f"p={self.p_value:.4f} {sig_marker}, "
            f"d={self.cohens_d:.3f}"
        )


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d (positive means group1 > group2)

    Formula:
        d = (mean1 - mean2) / pooled_std
        where pooled_std = sqrt(((n1-1)*std1^2 + (n2-1)*std2^2) / (n1+n2-2))
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (mean1 - mean2) / pooled_std


def independent_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True,
) -> tuple[float, float, float]:
    """
    Perform independent samples t-test.

    Args:
        group1: First group of values
        group2: Second group of values
        alpha: Significance level
        equal_var: Assume equal variances (True for pooled t-test)

    Returns:
        Tuple of (t_statistic, p_value, cohens_d)
    """
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan, np.nan

    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=equal_var)
    cohens_d = compute_cohens_d(group1, group2)

    return float(t_stat), float(p_val), cohens_d


def compare_aggregators(
    df: pd.DataFrame,
    agg1: str,
    agg2: str,
    metric: str = "macro_f1_global",
    condition_filters: dict | None = None,
    alpha: float = 0.05,
) -> StatisticalTestResult | None:
    """
    Compare two aggregation methods using t-test.

    Args:
        df: DataFrame with experiment results
        agg1: First aggregator name
        agg2: Second aggregator name
        metric: Metric column to compare
        condition_filters: Dict of column:value filters (e.g., {"alpha": 0.05, "adv_pct": 0})
        alpha: Significance level

    Returns:
        StatisticalTestResult or None if insufficient data
    """
    df_filtered = df.copy()

    if condition_filters:
        for col, val in condition_filters.items():
            df_filtered = df_filtered[df_filtered[col] == val]

    final_round = df_filtered.groupby(["aggregation", "seed"])["round"].transform("max")
    df_final = df_filtered[df_filtered["round"] == final_round]

    group1_data = df_final[df_final["aggregation"] == agg1][metric].dropna()
    group2_data = df_final[df_final["aggregation"] == agg2][metric].dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        return None

    t_stat, p_val, cohens_d = independent_t_test(group1_data.values, group2_data.values, alpha)

    return StatisticalTestResult(
        group1=agg1,
        group2=agg2,
        mean1=float(group1_data.mean()),
        mean2=float(group2_data.mean()),
        std1=float(group1_data.std()),
        std2=float(group2_data.std()),
        n1=len(group1_data),
        n2=len(group2_data),
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=cohens_d,
        significant=p_val < alpha,
        alpha=alpha,
    )


def pairwise_comparison_matrix(
    df: pd.DataFrame,
    aggregators: list[str],
    metric: str = "macro_f1_global",
    condition_filters: dict | None = None,
    alpha: float = 0.05,
    bonferroni_correction: bool = True,
) -> pd.DataFrame:
    """
    Create pairwise comparison matrix for multiple aggregators.

    Args:
        df: DataFrame with experiment results
        aggregators: List of aggregator names to compare
        metric: Metric to compare
        condition_filters: Dict of column:value filters
        alpha: Base significance level
        bonferroni_correction: Apply Bonferroni correction for multiple comparisons

    Returns:
        DataFrame with rows=aggregators, columns=["mean", "std", "agg1_p", "agg2_p", ...]
        Each cell shows p-value (and effect size in parentheses)
    """
    n_comparisons = len(aggregators) * (len(aggregators) - 1) // 2
    corrected_alpha = alpha / n_comparisons if bonferroni_correction else alpha

    results = []

    for agg in aggregators:
        row = {"aggregator": agg}

        df_filtered = df[df["aggregation"] == agg]
        if condition_filters:
            for col, val in condition_filters.items():
                df_filtered = df_filtered[df_filtered[col] == val]

        final_round = df_filtered.groupby("seed")["round"].transform("max")
        df_final = df_filtered[df_filtered["round"] == final_round]
        values = df_final[metric].dropna()

        row["mean"] = float(values.mean()) if len(values) > 0 else np.nan
        row["std"] = float(values.std()) if len(values) > 0 else np.nan
        row["n"] = len(values)

        for other_agg in aggregators:
            if agg == other_agg:
                row[f"{other_agg}_p"] = "-"
                row[f"{other_agg}_d"] = "-"
            else:
                result = compare_aggregators(df, agg, other_agg, metric, condition_filters, corrected_alpha)
                if result:
                    sig_marker = "***" if result.significant else ""
                    row[f"{other_agg}_p"] = f"{result.p_value:.4f}{sig_marker}"
                    row[f"{other_agg}_d"] = f"{result.cohens_d:.3f}"
                else:
                    row[f"{other_agg}_p"] = "N/A"
                    row[f"{other_agg}_d"] = "N/A"

        results.append(row)

    return pd.DataFrame(results)


def effect_size_interpretation(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        String interpretation (small/medium/large)

    Cohen's guidelines:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def summarize_all_comparisons(
    df: pd.DataFrame,
    aggregators: list[str],
    conditions: list[dict],
    metric: str = "macro_f1_global",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate summary table of all pairwise comparisons across conditions.

    Args:
        df: DataFrame with experiment results
        aggregators: List of aggregators to compare
        conditions: List of condition dicts, each with {column: value} filters
        metric: Metric to compare
        alpha: Significance level

    Returns:
        DataFrame with columns: condition, agg1, agg2, mean_diff, p_value, cohens_d, significant
    """
    all_results = []

    for condition_dict in conditions:
        condition_name = ", ".join(f"{k}={v}" for k, v in condition_dict.items())

        for i, agg1 in enumerate(aggregators):
            for agg2 in aggregators[i + 1 :]:
                result = compare_aggregators(df, agg1, agg2, metric, condition_dict, alpha)

                if result:
                    all_results.append(
                        {
                            "condition": condition_name,
                            "agg1": agg1,
                            "agg2": agg2,
                            "mean1": result.mean1,
                            "mean2": result.mean2,
                            "mean_diff": result.mean1 - result.mean2,
                            "p_value": result.p_value,
                            "cohens_d": result.cohens_d,
                            "effect_size": effect_size_interpretation(result.cohens_d),
                            "significant": result.significant,
                        }
                    )

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("Statistical Analysis Utilities")
    print("Run from main plotting script to generate comparison tables")
