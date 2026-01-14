"""
Objective 2: Handling Non-IID Data Figures

Generates plots analyzing FedProx effectiveness under data heterogeneity,
including convergence analysis, mu parameter sensitivity, and client drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from plots.config.constants import (
    AGGREGATOR_COLORS,
    ALPHA_VALUES,
    DATASET_CONFIG,
    DATASETS,
    MU_VALUES,
)
from plots.config.style import PALETTES, ThesisStyle
from plots.figures.primitives import save_figure


def plot_heterogeneity_impact(
    ax: Axes,
    df: pd.DataFrame,
    dataset: str,
    aggregator: str = "fedavg",
) -> None:
    """
    Plot F1 performance across different heterogeneity levels (alpha values).

    Args:
        ax: Matplotlib axes
        df: DataFrame with columns [alpha, macro_f1, seed]
        dataset: Dataset identifier
        aggregator: Aggregator to analyze
    """
    style = ThesisStyle()
    subset = df[(df["dataset"] == dataset) & (df["aggregator"] == aggregator) & (df["adv_pct"] == 0)]

    grouped = subset.groupby("alpha")["macro_f1"].agg(["mean", "std", "count"])
    valid_alphas = [a for a in ALPHA_VALUES if a in grouped.index]

    if not valid_alphas:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    means = [grouped.loc[a, "mean"] for a in valid_alphas]
    stds = [grouped.loc[a, "std"] for a in valid_alphas]

    ax.errorbar(
        valid_alphas,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        color=DATASET_CONFIG[dataset]["color"],
        linewidth=style.linewidth,
        markersize=style.markersize,
    )

    ax.set_xlabel("Dirichlet Alpha (lower = more heterogeneous)")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"{DATASET_CONFIG[dataset]['label']} - Heterogeneity Impact", fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def plot_fedprox_mu_sensitivity(
    ax: Axes,
    df: pd.DataFrame,
    dataset: str,
    alpha: float = 0.1,
) -> None:
    """
    Plot FedProx performance across different mu values.

    Args:
        ax: Matplotlib axes
        df: DataFrame with columns [mu, macro_f1, seed]
        dataset: Dataset identifier
        alpha: Heterogeneity level to analyze
    """
    style = ThesisStyle()
    subset = df[(df["dataset"] == dataset) & (df["alpha"] == alpha) & (df["adv_pct"] == 0)]

    colors = PALETTES["colorblind"]
    {mu: colors[i % len(colors)] for i, mu in enumerate(MU_VALUES)}

    grouped = subset.groupby("mu")["macro_f1"].agg(["mean", "std", "count"])
    valid_mus = [m for m in MU_VALUES if m in grouped.index]

    if not valid_mus:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    means = [grouped.loc[m, "mean"] for m in valid_mus]
    stds = [grouped.loc[m, "std"] for m in valid_mus]

    ax.errorbar(
        valid_mus,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        color=DATASET_CONFIG[dataset]["color"],
        linewidth=style.linewidth,
        markersize=style.markersize,
    )

    ax.axhline(y=means[0] if valid_mus[0] == 0.0 else 0, color="gray", linestyle="--", alpha=0.5, label="FedAvg baseline")
    ax.set_xlabel("FedProx mu Parameter")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"{DATASET_CONFIG[dataset]['label']} - FedProx mu Sensitivity (alpha={alpha})", fontweight="bold")
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_fedavg_vs_fedprox(
    ax: Axes,
    df: pd.DataFrame,
    dataset: str,
    optimal_mu: float = 0.01,
) -> None:
    """
    Compare FedAvg vs FedProx across heterogeneity levels.

    Args:
        ax: Matplotlib axes
        df: DataFrame with experiment results
        dataset: Dataset identifier
        optimal_mu: Optimal mu value for FedProx comparison
    """
    style = ThesisStyle()
    subset = df[(df["dataset"] == dataset) & (df["adv_pct"] == 0)]

    fedavg = subset[subset["mu"] == 0.0].groupby("alpha")["macro_f1"].mean()
    fedprox = subset[subset["mu"] == optimal_mu].groupby("alpha")["macro_f1"].mean()

    valid_alphas = sorted(set(fedavg.index) & set(fedprox.index))
    if not valid_alphas:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(
        valid_alphas,
        [fedavg.loc[a] for a in valid_alphas],
        marker="o",
        label="FedAvg",
        color=AGGREGATOR_COLORS["fedavg"],
        linewidth=style.linewidth,
    )
    ax.plot(
        valid_alphas,
        [fedprox.loc[a] for a in valid_alphas],
        marker="s",
        label=f"FedProx (mu={optimal_mu})",
        color=PALETTES["colorblind"][1],
        linewidth=style.linewidth,
    )

    ax.set_xlabel("Dirichlet Alpha")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"{DATASET_CONFIG[dataset]['label']} - FedAvg vs FedProx", fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def plot_fedprox_improvement_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    optimal_mu: float = 0.01,
) -> None:
    """
    Plot heatmap of FedProx improvement over FedAvg.

    Args:
        ax: Matplotlib axes
        df: DataFrame with experiment results
        optimal_mu: Mu value for FedProx
    """
    improvement_data = []
    valid_alphas = [a for a in ALPHA_VALUES if a <= 0.5]

    for alpha in valid_alphas:
        row = []
        for dataset in DATASETS:
            subset = df[(df["dataset"] == dataset) & (df["alpha"] == alpha) & (df["adv_pct"] == 0)]
            fedavg_f1 = subset[subset["mu"] == 0.0]["macro_f1"].mean()
            fedprox_f1 = subset[subset["mu"] == optimal_mu]["macro_f1"].mean()
            improvement = ((fedprox_f1 - fedavg_f1) / fedavg_f1 * 100) if fedavg_f1 > 0 else 0
            row.append(improvement)
        improvement_data.append(row)

    improvement_df = pd.DataFrame(
        improvement_data,
        index=[f"alpha={a}" for a in valid_alphas],
        columns=[DATASET_CONFIG[d]["label"] for d in DATASETS],
    )

    sns.heatmap(
        improvement_df,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
    )
    ax.set_title(f"FedProx Improvement over FedAvg (%) at mu={optimal_mu}", fontweight="bold")


def plot_convergence_comparison(
    ax: Axes,
    df: pd.DataFrame,
    dataset: str,
    alpha: float = 0.1,
) -> None:
    """
    Plot convergence curves for FedAvg vs FedProx.

    Args:
        ax: Matplotlib axes
        df: DataFrame with round-level metrics
        dataset: Dataset identifier
        alpha: Heterogeneity level
    """
    style = ThesisStyle()
    subset = df[(df["dataset"] == dataset) & (df["alpha"] == alpha) & (df["adv_pct"] == 0)]

    fedavg = subset[subset["mu"] == 0.0]
    fedprox = subset[subset["mu"] == 0.01]

    if "round" in fedavg.columns and not fedavg.empty:
        fedavg_by_round = fedavg.groupby("round")["macro_f1"].mean()
        ax.plot(fedavg_by_round.index, fedavg_by_round.values, label="FedAvg", color=AGGREGATOR_COLORS["fedavg"], linewidth=style.linewidth)

    if "round" in fedprox.columns and not fedprox.empty:
        fedprox_by_round = fedprox.groupby("round")["macro_f1"].mean()
        ax.plot(
            fedprox_by_round.index, fedprox_by_round.values, label="FedProx", color=PALETTES["colorblind"][1], linewidth=style.linewidth
        )

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"{DATASET_CONFIG[dataset]['label']} - Convergence (alpha={alpha})", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def generate_objective2_figures(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Generate all Objective 2 (Heterogeneity) figures.

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

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))
    fig.suptitle("Objective 2: Heterogeneity Impact on Performance", fontsize=14, fontweight="bold", y=1.02)
    for i, dataset in enumerate(DATASETS):
        plot_heterogeneity_impact(axes[i], df, dataset)
    plt.tight_layout()
    paths = save_figure(fig, output_dir / "obj2_heterogeneity_impact", formats)
    results["obj2_heterogeneity_impact"] = paths

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))
    fig.suptitle("Objective 2: FedProx mu Parameter Sensitivity", fontsize=14, fontweight="bold", y=1.02)
    for i, dataset in enumerate(DATASETS):
        plot_fedprox_mu_sensitivity(axes[i], df, dataset)
    plt.tight_layout()
    paths = save_figure(fig, output_dir / "obj2_mu_sensitivity", formats)
    results["obj2_mu_sensitivity"] = paths

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))
    fig.suptitle("Objective 2: FedAvg vs FedProx Comparison", fontsize=14, fontweight="bold", y=1.02)
    for i, dataset in enumerate(DATASETS):
        plot_fedavg_vs_fedprox(axes[i], df, dataset)
    plt.tight_layout()
    paths = save_figure(fig, output_dir / "obj2_fedavg_vs_fedprox", formats)
    results["obj2_fedavg_vs_fedprox"] = paths

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fedprox_improvement_heatmap(ax, df)
    plt.tight_layout()
    paths = save_figure(fig, output_dir / "obj2_improvement_heatmap", formats)
    results["obj2_improvement_heatmap"] = paths

    return results
