#!/usr/bin/env python3
"""
Generate caption tables with 95% CIs for thesis figures.

Each plot is paired with a compact numeric table of final values suitable for
embedding in or adjacent to figure captions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CaptionStats:
    """Summary statistics for a single metric across seeds."""

    metric_name: str
    mean: float
    ci_lower: float
    ci_upper: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


def compute_95_ci(values: np.ndarray) -> tuple[float, float]:
    """Compute 95% confidence interval for array of values."""
    if len(values) < 2:
        return float(values.mean()), float(values.mean())

    mean = values.mean()
    stderr = stats.sem(values)
    ci = 1.96 * stderr
    return mean - ci, mean + ci


def generate_aggregation_caption_table(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Generate caption table for aggregation comparison plot."""
    if "aggregation" not in df.columns or "macro_f1" not in df.columns:
        return pd.DataFrame()

    method_order = ["fedavg", "krum", "bulyan", "median"]
    available_methods = [m for m in method_order if m in df["aggregation"].unique()]

    # Handle missing seed column
    if "seed" in df.columns:
        final_rounds = df.groupby(["aggregation", "seed"]).tail(1)
    else:
        final_rounds = df

    rows = []

    for method in available_methods:
        method_data = final_rounds[final_rounds["aggregation"] == method]

        macro_f1_vals = method_data["macro_f1"].dropna().values
        if len(macro_f1_vals) == 0:
            continue

        mean_f1, (ci_lower, ci_upper) = macro_f1_vals.mean(), compute_95_ci(macro_f1_vals)

        l2_vals = method_data.get("l2_to_benign_mean", pd.Series(dtype=float)).dropna().values
        mean_l2 = l2_vals.mean() if len(l2_vals) > 0 else None

        rows.append(
            {
                "Method": method.capitalize(),
                "Macro-F1": f"{mean_f1:.3f}",
                "Macro-F1 CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "L2 Distance": f"{mean_l2:.3f}" if mean_l2 is not None else "N/A",
                "n_seeds": len(macro_f1_vals),
            }
        )

    result = pd.DataFrame(rows)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved aggregation caption table: {output_path}")

    return result


def generate_heterogeneity_caption_table(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Generate caption table for heterogeneity comparison plot."""
    if "alpha" not in df.columns or "macro_f1" not in df.columns:
        return pd.DataFrame()

    alpha_order = sorted(df["alpha"].unique())
    rows = []

    for alpha in alpha_order:
        alpha_data = df[df["alpha"] == alpha]
        f1_vals = alpha_data["macro_f1"].dropna().values

        if len(f1_vals) == 0:
            continue

        mean_f1, (ci_lower, ci_upper) = f1_vals.mean(), compute_95_ci(f1_vals)

        rows.append(
            {
                "Alpha (heterogeneity)": f"{alpha:.2f}",
                "Macro-F1": f"{mean_f1:.3f}",
                "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "n_runs": len(f1_vals),
            }
        )

    result = pd.DataFrame(rows)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved heterogeneity caption table: {output_path}")

    return result


def generate_attack_caption_table(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Generate caption table for attack resilience plot."""
    if "byzantine_f" not in df.columns or "macro_f1" not in df.columns:
        return pd.DataFrame()

    byzantine_levels = sorted(df["byzantine_f"].unique())
    methods = sorted(df["aggregation"].unique())

    rows = []
    for method in methods:
        for adv_frac in byzantine_levels:
            subset = df[(df["aggregation"] == method) & (df["byzantine_f"] == adv_frac)]
            f1_vals = subset["macro_f1"].dropna().values

            if len(f1_vals) == 0:
                continue

            mean_f1, (ci_lower, ci_upper) = f1_vals.mean(), compute_95_ci(f1_vals)
            degradation = 100.0 * (1.0 - mean_f1)

            rows.append(
                {
                    "Method": method.capitalize(),
                    "Byzantine %": f"{adv_frac * 100:.0f}%",
                    "Macro-F1": f"{mean_f1:.3f}",
                    "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                    "Degradation %": f"{degradation:.1f}%",
                    "n_seeds": len(f1_vals),
                }
            )

    result = pd.DataFrame(rows)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved attack caption table: {output_path}")

    return result


def generate_privacy_caption_table(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Generate caption table for privacy-utility tradeoff plot."""
    if "dp_sigma" not in df.columns or "macro_f1" not in df.columns:
        return pd.DataFrame()

    sigma_values = sorted(df["dp_sigma"].dropna().unique())
    rows = []

    for sigma in sigma_values:
        sigma_label = "None (no DP)" if sigma == 0.0 else f"{sigma:.2f}"
        sigma_data = df[df["dp_sigma"] == sigma]
        f1_vals = sigma_data["macro_f1"].dropna().values

        if len(f1_vals) == 0:
            continue

        mean_f1, (ci_lower, ci_upper) = f1_vals.mean(), compute_95_ci(f1_vals)

        rows.append(
            {
                "DP Sigma": sigma_label,
                "Macro-F1": f"{mean_f1:.3f}",
                "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "Privacy-Utility": "Baseline" if sigma == 0.0 else "Degraded",
                "n_runs": len(f1_vals),
            }
        )

    result = pd.DataFrame(rows)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved privacy caption table: {output_path}")

    return result


def generate_personalization_caption_table(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Generate caption table for personalization benefit plot."""
    if "personalization_epochs" not in df.columns:
        return pd.DataFrame()

    epoch_values = sorted(df["personalization_epochs"].unique())
    rows = []

    for epochs in epoch_values:
        epoch_data = df[df["personalization_epochs"] == epochs]
        gain_vals = epoch_data.get("personalization_gain", pd.Series(dtype=float)).dropna().values

        if len(gain_vals) == 0:
            continue

        mean_gain, (ci_lower, ci_upper) = gain_vals.mean(), compute_95_ci(gain_vals)
        pct_positive = 100.0 * (gain_vals > 0).sum() / len(gain_vals)

        rows.append(
            {
                "Personalization Epochs": int(epochs),
                "Mean Gain": f"{mean_gain:.3f}",
                "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "% Positive Gains": f"{pct_positive:.0f}%",
                "n_clients": len(gain_vals),
            }
        )

    result = pd.DataFrame(rows)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved personalization caption table: {output_path}")

    return result
