"""
Chapter 4 Plot Suite

Orchestrates generation of all Chapter 4 thesis figures:
- Figure 4.1: Overall Performance
- Figure 4.2: Aggregation Comparison
- Figure 4.3: Adversarial Resilience
- Figure 4.4: Heterogeneity Analysis
- Figure 4.5: FedProx Analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plots.config.constants import (
    ADVERSARIAL_LEVELS,
    AGGREGATOR_COLORS,
    AGGREGATOR_LABELS,
    AGGREGATOR_ORDER,
    ALPHA_VALUES,
    DATASET_CONFIG,
    DATASETS,
    MU_VALUES,
)
from plots.config.style import ThesisStyle
from plots.figures.primitives import save_figure


def _plot_fig41_overall_performance(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Generate Figure 4.1: Overall Federated Performance by Dataset."""
    ThesisStyle()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), layout="tight")
    fig.suptitle("Figure 4.1: Overall Federated Performance by Dataset", fontsize=14, fontweight="bold", y=1.02)

    benign_df = df[(df["adv_pct"] == 0) & (df["mu"] == 0.0)]

    ax = axes[0]
    means, stds, counts = [], [], []
    for ds in DATASETS:
        ds_data = benign_df[benign_df["dataset"] == ds]["macro_f1"].dropna()
        means.append(ds_data.mean() if len(ds_data) > 0 else 0)
        stds.append(ds_data.std() if len(ds_data) > 0 else 0)
        counts.append(len(ds_data))

    x = np.arange(len(DATASETS))
    colors = [DATASET_CONFIG[d]["color"] for d in DATASETS]
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", capsize=5)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(a) Mean Macro F1 (Benign)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_CONFIG[d]["label"] for d in DATASETS], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        if stds[i] > 0:
            ax.annotate(f"n={cnt}", xy=(bar.get_x() + bar.get_width() / 2, means[i] + stds[i] + 0.02), ha="center", fontsize=8)

    ax = axes[1]
    box_data = []
    for ds in DATASETS:
        ds_data = benign_df[benign_df["dataset"] == ds]["macro_f1"].dropna()
        box_data.append(ds_data.values if len(ds_data) > 0 else [0])
    bp = ax.boxplot(box_data, patch_artist=True, labels=[DATASET_CONFIG[d]["label"] for d in DATASETS])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(b) F1 Distribution (Benign)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[2]
    for ds in DATASETS:
        ds_data = df[(df["dataset"] == ds) & (df["adv_pct"] == 0)]
        alpha_means = ds_data.groupby("alpha")["macro_f1"].mean()
        valid_alphas = [a for a in alpha_means.index if not np.isinf(a) and a in ALPHA_VALUES]
        if valid_alphas:
            ax.plot(
                valid_alphas,
                [alpha_means[a] for a in valid_alphas],
                marker=DATASET_CONFIG[ds]["marker"],
                label=DATASET_CONFIG[ds]["label"],
                color=DATASET_CONFIG[ds]["color"],
                linewidth=2,
            )
    ax.set_xlabel("Dirichlet Alpha")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(c) Heterogeneity Impact", fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)

    ax = axes[3]
    width = 0.25
    x = np.arange(len(ADVERSARIAL_LEVELS))
    for i, ds in enumerate(DATASETS):
        adv_means = []
        for adv in ADVERSARIAL_LEVELS:
            adv_data = df[(df["dataset"] == ds) & (df["adv_pct"] == adv)]["macro_f1"]
            adv_means.append(adv_data.mean() if len(adv_data) > 0 else 0)
        ax.bar(x + i * width, adv_means, width, label=DATASET_CONFIG[ds]["label"], color=DATASET_CONFIG[ds]["color"])
    ax.set_xlabel("Byzantine Client %")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(d) Attack Impact", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{a}%" for a in ADVERSARIAL_LEVELS])
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)

    ax = axes[4]
    summary_text = "Summary Statistics:\n\n"
    for ds in DATASETS:
        ds_data = benign_df[benign_df["dataset"] == ds]["macro_f1"]
        if len(ds_data) > 0:
            summary_text += f"{DATASET_CONFIG[ds]['label']}:\n"
            summary_text += f"  Mean: {ds_data.mean():.3f}\n"
            summary_text += f"  Std:  {ds_data.std():.3f}\n"
            summary_text += f"  n:    {len(ds_data)}\n\n"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace")
    ax.set_title("(e) Summary Statistics", fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4.1_overall_performance", formats)


def _plot_fig42_aggregation_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Generate Figure 4.2: Aggregation Method Comparison."""
    ThesisStyle()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Figure 4.2: Aggregation Method Comparison", fontsize=14, fontweight="bold", y=1.02)

    benign_df = df[(df["adv_pct"] == 0) & (df["mu"] == 0.0)]

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        ds_data = benign_df[benign_df["dataset"] == ds]
        agg_means = [ds_data[ds_data["aggregator"] == agg]["macro_f1"].mean() for agg in AGGREGATOR_ORDER]
        agg_stds = [ds_data[ds_data["aggregator"] == agg]["macro_f1"].std() for agg in AGGREGATOR_ORDER]

        x = np.arange(len(AGGREGATOR_ORDER))
        ax.bar(x, agg_means, yerr=agg_stds, color=[AGGREGATOR_COLORS[a] for a in AGGREGATOR_ORDER], edgecolor="black", capsize=5)
        ax.set_ylabel("Macro F1 Score")
        ax.set_title(f"({chr(97 + i)}) {DATASET_CONFIG[ds]['label']}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([AGGREGATOR_LABELS[a] for a in AGGREGATOR_ORDER], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)

    ax = axes[3]
    x = np.arange(len(DATASETS))
    width = 0.2
    for j, agg in enumerate(AGGREGATOR_ORDER):
        agg_means = [benign_df[(benign_df["dataset"] == ds) & (benign_df["aggregator"] == agg)]["macro_f1"].mean() for ds in DATASETS]
        ax.bar(x + j * width, agg_means, width, label=AGGREGATOR_LABELS[agg], color=AGGREGATOR_COLORS[agg])
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(d) Cross-Dataset Comparison", fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DATASET_CONFIG[d]["label"] for d in DATASETS], rotation=15, ha="right")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)

    ax = axes[4]
    cv_data = []
    for ds in DATASETS:
        row = []
        for agg in AGGREGATOR_ORDER:
            agg_data = benign_df[(benign_df["dataset"] == ds) & (benign_df["aggregator"] == agg)]["macro_f1"]
            cv = (agg_data.std() / agg_data.mean() * 100) if agg_data.mean() > 0 else 0
            row.append(cv)
        cv_data.append(row)
    cv_df = pd.DataFrame(
        cv_data, index=[DATASET_CONFIG[d]["label"] for d in DATASETS], columns=[AGGREGATOR_LABELS[a] for a in AGGREGATOR_ORDER]
    )
    sns.heatmap(cv_df, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5, vmin=0, vmax=20)
    ax.set_title("(e) Stability (CV %)", fontweight="bold")

    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4.2_aggregation_comparison", formats)


def _plot_fig43_adversarial_resilience(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Generate Figure 4.3: Adversarial Resilience."""
    ThesisStyle()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Figure 4.3: Adversarial Resilience", fontsize=14, fontweight="bold", y=1.02)

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        for agg in AGGREGATOR_ORDER:
            agg_means = [
                df[(df["dataset"] == ds) & (df["aggregator"] == agg) & (df["adv_pct"] == adv)]["macro_f1"].mean()
                for adv in ADVERSARIAL_LEVELS
            ]
            ax.plot(ADVERSARIAL_LEVELS, agg_means, marker="o", label=AGGREGATOR_LABELS[agg], color=AGGREGATOR_COLORS[agg], linewidth=2)
        ax.set_xlabel("Byzantine %")
        ax.set_ylabel("Macro F1 Score")
        ax.set_title(f"({chr(97 + i)}) {DATASET_CONFIG[ds]['label']}", fontweight="bold")
        ax.set_xticks(ADVERSARIAL_LEVELS)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(0, 1.05)

    ax = axes[3]
    retention_data = []
    for ds in DATASETS:
        row = []
        for agg in AGGREGATOR_ORDER:
            baseline = df[(df["dataset"] == ds) & (df["aggregator"] == agg) & (df["adv_pct"] == 0)]["macro_f1"].mean()
            attacked = df[(df["dataset"] == ds) & (df["aggregator"] == agg) & (df["adv_pct"] == 30)]["macro_f1"].mean()
            retention = (attacked / baseline * 100) if baseline > 0 else 0
            row.append(retention)
        retention_data.append(row)
    retention_df = pd.DataFrame(
        retention_data, index=[DATASET_CONFIG[d]["label"] for d in DATASETS], columns=[AGGREGATOR_LABELS[a] for a in AGGREGATOR_ORDER]
    )
    sns.heatmap(retention_df, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn", vmin=50, vmax=100, linewidths=0.5)
    ax.set_title("(d) F1 Retention at 30% Attack", fontweight="bold")

    ax = axes[4]
    degradation_data = []
    for agg in AGGREGATOR_ORDER:
        agg_degs = []
        for ds in DATASETS:
            baseline = df[(df["dataset"] == ds) & (df["aggregator"] == agg) & (df["adv_pct"] == 0)]["macro_f1"].mean()
            attacked = df[(df["dataset"] == ds) & (df["aggregator"] == agg) & (df["adv_pct"] == 30)]["macro_f1"].mean()
            deg = ((baseline - attacked) / baseline * 100) if baseline > 0 else 0
            agg_degs.append(deg)
        degradation_data.append(np.mean(agg_degs))
    x = np.arange(len(AGGREGATOR_ORDER))
    ax.bar(x, degradation_data, color=[AGGREGATOR_COLORS[a] for a in AGGREGATOR_ORDER], edgecolor="black")
    ax.set_ylabel("Avg Degradation (%)")
    ax.set_title("(e) Performance Degradation", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([AGGREGATOR_LABELS[a] for a in AGGREGATOR_ORDER], rotation=15, ha="right")

    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4.3_adversarial_resilience", formats)


def _plot_fig44_heterogeneity_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Generate Figure 4.4: Heterogeneity Analysis."""
    ThesisStyle()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Figure 4.4: Heterogeneity Analysis", fontsize=14, fontweight="bold", y=1.02)

    benign_df = df[(df["adv_pct"] == 0) & (df["mu"] == 0.0)]

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        ds_data = benign_df[benign_df["dataset"] == ds]
        alpha_grouped = ds_data.groupby("alpha")["macro_f1"].agg(["mean", "std"])
        valid_alphas = [a for a in ALPHA_VALUES if a in alpha_grouped.index]
        if valid_alphas:
            means = [alpha_grouped.loc[a, "mean"] for a in valid_alphas]
            stds = [alpha_grouped.loc[a, "std"] for a in valid_alphas]
            ax.errorbar(valid_alphas, means, yerr=stds, marker="o", capsize=5, color=DATASET_CONFIG[ds]["color"], linewidth=2)
        ax.set_xlabel("Dirichlet Alpha")
        ax.set_ylabel("Macro F1 Score")
        ax.set_title(f"({chr(97 + i)}) {DATASET_CONFIG[ds]['label']}", fontweight="bold")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)

    ax = axes[3]
    for ds in DATASETS:
        ds_data = benign_df[benign_df["dataset"] == ds]
        alpha_means = ds_data.groupby("alpha")["macro_f1"].mean()
        valid_alphas = [a for a in ALPHA_VALUES if a in alpha_means.index]
        if valid_alphas:
            ax.plot(
                valid_alphas,
                [alpha_means[a] for a in valid_alphas],
                marker=DATASET_CONFIG[ds]["marker"],
                label=DATASET_CONFIG[ds]["label"],
                color=DATASET_CONFIG[ds]["color"],
                linewidth=2,
            )
    ax.set_xlabel("Dirichlet Alpha")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(d) Cross-Dataset Comparison", fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)

    ax = axes[4]
    sensitivity_text = "Heterogeneity Sensitivity:\n\n"
    for ds in DATASETS:
        ds_data = benign_df[benign_df["dataset"] == ds]
        alpha_means = ds_data.groupby("alpha")["macro_f1"].mean()
        if len(alpha_means) >= 2:
            high_alpha = alpha_means.get(1.0, alpha_means.iloc[-1])
            low_alpha = alpha_means.get(0.02, alpha_means.iloc[0])
            drop = ((high_alpha - low_alpha) / high_alpha * 100) if high_alpha > 0 else 0
            sensitivity_text += f"{DATASET_CONFIG[ds]['label']}:\n  Drop: {drop:.1f}%\n\n"
    ax.text(0.1, 0.9, sensitivity_text, transform=ax.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace")
    ax.set_title("(e) Sensitivity Summary", fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4.4_heterogeneity_analysis", formats)


def _plot_fig45_fedprox_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Generate Figure 4.5: FedProx Analysis."""
    ThesisStyle()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Figure 4.5: FedProx Analysis", fontsize=14, fontweight="bold", y=1.02)

    benign_df = df[(df["adv_pct"] == 0)]

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        ds_data = benign_df[benign_df["dataset"] == ds]
        mu_grouped = ds_data.groupby("mu")["macro_f1"].agg(["mean", "std"])
        valid_mus = [m for m in MU_VALUES if m in mu_grouped.index]
        if valid_mus:
            means = [mu_grouped.loc[m, "mean"] for m in valid_mus]
            stds = [mu_grouped.loc[m, "std"] for m in valid_mus]
            x = np.arange(len(valid_mus))
            ax.bar(x, means, yerr=stds, color=DATASET_CONFIG[ds]["color"], edgecolor="black", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([str(m) for m in valid_mus], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("FedProx mu")
        ax.set_ylabel("Macro F1 Score")
        ax.set_title(f"({chr(97 + i)}) {DATASET_CONFIG[ds]['label']}", fontweight="bold")
        ax.set_ylim(0, 1.05)

    ax = axes[3]
    improvement_data = []
    test_mus = [0.01, 0.02, 0.05]
    for mu in test_mus:
        row = []
        for ds in DATASETS:
            fedavg = benign_df[(benign_df["dataset"] == ds) & (benign_df["mu"] == 0.0)]["macro_f1"].mean()
            fedprox = benign_df[(benign_df["dataset"] == ds) & (benign_df["mu"] == mu)]["macro_f1"].mean()
            improvement = ((fedprox - fedavg) / fedavg * 100) if fedavg > 0 else 0
            row.append(improvement)
        improvement_data.append(row)
    improvement_df = pd.DataFrame(
        improvement_data, index=[f"mu={m}" for m in test_mus], columns=[DATASET_CONFIG[d]["label"] for d in DATASETS]
    )
    sns.heatmap(improvement_df, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn", center=0, linewidths=0.5)
    ax.set_title("(d) FedProx Improvement (%)", fontweight="bold")

    ax = axes[4]
    optimal_text = "Optimal mu Configuration:\n\n"
    for ds in DATASETS:
        ds_data = benign_df[benign_df["dataset"] == ds]
        mu_means = ds_data.groupby("mu")["macro_f1"].mean()
        if len(mu_means) > 0:
            best_mu = mu_means.idxmax()
            best_f1 = mu_means.max()
            optimal_text += f"{DATASET_CONFIG[ds]['label']}:\n  Best mu: {best_mu}\n  F1: {best_f1:.3f}\n\n"
    ax.text(0.1, 0.9, optimal_text, transform=ax.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace")
    ax.set_title("(e) Optimal Configuration", fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4.5_fedprox_analysis", formats)


def generate_chapter4_figures(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str] | None = None,
    figures: list[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Generate all Chapter 4 thesis figures.

    Args:
        df: DataFrame with experiment results
        output_dir: Output directory for figures
        formats: Output formats (default: ['png', 'pdf'])
        figures: Specific figures to generate (default: all)
                 Options: ['4.1', '4.2', '4.3', '4.4', '4.5']

    Returns:
        Dict mapping figure name -> list of saved paths
    """
    if formats is None:
        formats = ["png", "pdf"]
    if figures is None:
        figures = ["4.1", "4.2", "4.3", "4.4", "4.5"]

    style = ThesisStyle()
    style.apply()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Path]] = {}

    figure_funcs = {
        "4.1": _plot_fig41_overall_performance,
        "4.2": _plot_fig42_aggregation_comparison,
        "4.3": _plot_fig43_adversarial_resilience,
        "4.4": _plot_fig44_heterogeneity_analysis,
        "4.5": _plot_fig45_fedprox_analysis,
    }

    for fig_id in figures:
        if fig_id in figure_funcs:
            paths = figure_funcs[fig_id](df, output_dir, formats)
            results[f"fig{fig_id}"] = paths
            print(f"Generated Figure {fig_id}: {[p.name for p in paths]}")

    return results
