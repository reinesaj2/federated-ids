"""
Objective 1: Robust Aggregation Figures

Generates plots comparing aggregation methods (FedAvg, Bulyan, Krum, Median)
under adversarial conditions (0%, 10%, 30% Byzantine clients).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from plots.config.constants import (
    ADVERSARIAL_LEVELS,
    AGGREGATOR_COLORS,
    AGGREGATOR_LABELS,
    AGGREGATOR_ORDER,
    DATASET_CONFIG,
    DATASETS,
)
from plots.config.style import ThesisStyle
from plots.figures.primitives import save_figure

ADVERSARIAL_LABELS: dict[int, str] = {
    0: "No Attack (0%)",
    10: "10% Byzantine",
    30: "30% Byzantine",
}


def plot_f1_over_rounds(
    ax: Axes,
    df: pd.DataFrame,
    dataset: str,
    adv_pct: int,
) -> None:
    """
    Plot Macro F1 over training rounds for all aggregators.

    Args:
        ax: Matplotlib axes
        df: DataFrame with columns [round, aggregator, macro_f1, seed]
        dataset: Dataset identifier ('iiot', 'cic', 'unsw')
        adv_pct: Adversarial percentage to filter
    """
    style = ThesisStyle()
    subset = df[(df["dataset"] == dataset) & (df["adv_pct"] == adv_pct)]

    for agg in AGGREGATOR_ORDER:
        agg_data = subset[subset["aggregator"] == agg]
        if agg_data.empty:
            continue

        grouped = agg_data.groupby("round")["macro_f1"].agg(["mean", "std", "count"])
        rounds = grouped.index.values
        means = grouped["mean"].values

        ci_lower = means - 1.96 * grouped["std"].values / np.sqrt(grouped["count"].values)
        ci_upper = means + 1.96 * grouped["std"].values / np.sqrt(grouped["count"].values)

        ax.plot(
            rounds,
            means,
            label=AGGREGATOR_LABELS[agg],
            color=AGGREGATOR_COLORS[agg],
            linewidth=style.linewidth,
            marker="o",
            markersize=style.markersize / 2,
            markevery=max(1, len(rounds) // 10),
        )
        ax.fill_between(rounds, ci_lower, ci_upper, color=AGGREGATOR_COLORS[agg], alpha=0.15)

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"{DATASET_CONFIG[dataset]['label']} - {ADVERSARIAL_LABELS[adv_pct]}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def plot_attack_resilience_comparison(
    ax: Axes,
    df: pd.DataFrame,
    dataset: str,
) -> None:
    """
    Plot aggregator performance degradation under attack.

    Args:
        ax: Matplotlib axes
        df: DataFrame with experiment results
        dataset: Dataset to plot
    """
    style = ThesisStyle()
    subset = df[df["dataset"] == dataset]

    for agg in AGGREGATOR_ORDER:
        agg_data = subset[subset["aggregator"] == agg]
        if agg_data.empty:
            continue

        means = []
        for adv in ADVERSARIAL_LEVELS:
            adv_data = agg_data[agg_data["adv_pct"] == adv]["macro_f1"]
            means.append(adv_data.mean() if len(adv_data) > 0 else 0)

        ax.plot(
            ADVERSARIAL_LEVELS,
            means,
            label=AGGREGATOR_LABELS[agg],
            color=AGGREGATOR_COLORS[agg],
            linewidth=style.linewidth,
            marker="o",
            markersize=style.markersize,
        )

    ax.set_xlabel("Byzantine Client Percentage (%)")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"{DATASET_CONFIG[dataset]['label']} - Attack Resilience", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ADVERSARIAL_LEVELS)
    ax.grid(True, alpha=0.3)


def plot_f1_retention_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    attack_level: int = 30,
) -> None:
    """
    Plot heatmap of F1 retention at specific attack level.

    Args:
        ax: Matplotlib axes
        df: DataFrame with experiment results
        attack_level: Attack percentage for retention calculation
    """
    import seaborn as sns

    retention_data = []
    for dataset in DATASETS:
        row = []
        for agg in AGGREGATOR_ORDER:
            baseline = df[(df["dataset"] == dataset) & (df["aggregator"] == agg) & (df["adv_pct"] == 0)]["macro_f1"].mean()
            attacked = df[(df["dataset"] == dataset) & (df["aggregator"] == agg) & (df["adv_pct"] == attack_level)]["macro_f1"].mean()
            retention = (attacked / baseline * 100) if baseline > 0 else 0
            row.append(retention)
        retention_data.append(row)

    retention_df = pd.DataFrame(
        retention_data,
        index=[DATASET_CONFIG[d]["label"] for d in DATASETS],
        columns=[AGGREGATOR_LABELS[a] for a in AGGREGATOR_ORDER],
    )

    sns.heatmap(
        retention_df,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=50,
        vmax=100,
        linewidths=0.5,
    )
    ax.set_title(f"F1 Retention at {attack_level}% Attack (%)", fontweight="bold")


def generate_objective1_figures(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Generate all Objective 1 (Robustness) figures.

    Args:
        df: DataFrame with experiment results
        output_dir: Output directory for figures
        formats: Output formats (default: ['png', 'pdf'])

    Returns:
        Dict mapping figure name -> list of saved paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    style = ThesisStyle()
    style.apply()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Path]] = {}

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Objective 1: Aggregator Robustness - {DATASET_CONFIG[dataset]['label']}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for i, adv_pct in enumerate(ADVERSARIAL_LEVELS):
            plot_f1_over_rounds(axes[i], df, dataset, adv_pct)

        plt.tight_layout()
        paths = save_figure(fig, output_dir / f"obj1_{dataset}_f1_rounds", formats)
        results[f"obj1_{dataset}_f1_rounds"] = paths

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))
    fig.suptitle("Objective 1: Attack Resilience Comparison", fontsize=14, fontweight="bold", y=1.02)

    for i, dataset in enumerate(DATASETS):
        plot_attack_resilience_comparison(axes[i], df, dataset)

    plt.tight_layout()
    paths = save_figure(fig, output_dir / "obj1_attack_resilience", formats)
    results["obj1_attack_resilience"] = paths

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_f1_retention_heatmap(ax, df, attack_level=30)
    plt.tight_layout()
    paths = save_figure(fig, output_dir / "obj1_retention_heatmap", formats)
    results["obj1_retention_heatmap"] = paths

    return results
