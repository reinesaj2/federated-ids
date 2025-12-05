#!/usr/bin/env python3
"""
Comprehensive IIoT Plotting with Statistical Analysis

Generates publication-quality plots for ALL IIoT experiments including:
1. Per-class F1 visualizations (heatmap, bar, box, line)
2. Statistical significance testing (t-tests, Cohen's d)
3. Bulyan baseline comparison panels
4. All pairwise aggregator comparisons
5. Alpha sweep × attack sweep analysis
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from load_iiot_data_enhanced import expand_per_class_f1, load_iiot_data_enhanced  # noqa: E402
from plot_metrics_utils import compute_confidence_interval  # noqa: E402
from statistical_analysis import (  # noqa: E402
    pairwise_comparison_matrix,
    summarize_all_comparisons,
)

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "figure.dpi": 300,
    }
)

COLORS = {
    "fedavg": "#1f77b4",
    "krum": "#ff7f0e",
    "bulyan": "#2ca02c",
    "median": "#d62728",
    "fedprox": "#9467bd",
}

MINORITY_CLASSES = {
    "DDOS_ICMP",
    "VULNERABILITY_SCANNER",
    "DDOS_UDP",
    "DDOS_HTTP",
    "SQL_INJECTION",
    "UPLOADING",
    "BACKDOOR",
    "RANSOMWARE",
    "MITM",
    "FINGERPRINTING",
}


def plot_perclass_heatmap(df_expanded: pd.DataFrame, output_dir: Path, config_filter: dict):
    """
    Heatmap: Aggregator × Class showing final F1 scores.

    Args:
        df_expanded: DataFrame with per-class F1 expanded
        output_dir: Output directory
        config_filter: Dict with alpha, adv_pct to filter
    """
    df_filtered = df_expanded.copy()
    for col, val in config_filter.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    final_round = df_filtered.groupby(["aggregation", "seed"])["round"].transform("max")
    df_final = df_filtered[df_filtered["round"] == final_round]

    aggregators = ["fedavg", "krum", "bulyan", "median"]
    pivot_data = []

    all_classes = sorted(df_final["class_name"].unique())

    for agg in aggregators:
        agg_data = df_final[df_final["aggregation"] == agg]
        row = []
        for class_name in all_classes:
            class_data = agg_data[agg_data["class_name"] == class_name]["class_f1"]
            row.append(class_data.mean() if len(class_data) > 0 else np.nan)
        pivot_data.append(row)

    pivot_df = pd.DataFrame(
        pivot_data,
        index=[a.capitalize() for a in aggregators],
        columns=all_classes,
    )

    fig, ax = plt.subplots(figsize=(20, 6))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "Mean F1 Score"},
        linewidths=0.5,
        linecolor="gray",
    )

    for i, class_name in enumerate(all_classes):
        if class_name in MINORITY_CLASSES:
            ax.add_patch(
                plt.Rectangle(
                    (i, 0),
                    1,
                    len(aggregators),
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                    linestyle="--",
                )
            )

    config_str = ", ".join(f"{k}={v}" for k, v in config_filter.items())
    ax.set_title(
        f"Per-Class F1 Performance Heatmap ({config_str})\n" "Red boxes indicate minority attack classes",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Attack Class", fontsize=12)
    ax.set_ylabel("Aggregator", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    filename = f"perclass_heatmap_alpha{config_filter.get('alpha', 'all')}_adv{config_filter.get('adv_pct', 'all')}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_perclass_bars_minority(df_expanded: pd.DataFrame, output_dir: Path, config_filter: dict):
    """
    Bar chart showing per-class F1 with minority classes highlighted.

    Args:
        df_expanded: DataFrame with per-class F1 expanded
        output_dir: Output directory
        config_filter: Dict with alpha, adv_pct to filter
    """
    df_filtered = df_expanded.copy()
    for col, val in config_filter.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    final_round = df_filtered.groupby(["aggregation", "seed"])["round"].transform("max")
    df_final = df_filtered[df_filtered["round"] == final_round]

    aggregators = ["fedavg", "krum", "bulyan", "median"]
    all_classes = sorted(df_final["class_name"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for idx, agg in enumerate(aggregators):
        ax = axes[idx]
        agg_data = df_final[df_final["aggregation"] == agg]

        means = []
        errors = []
        colors = []

        for class_name in all_classes:
            class_data = agg_data[agg_data["class_name"] == class_name]["class_f1"].dropna()
            if len(class_data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(class_data)
                means.append(mean)
                errors.append([[mean - ci_low], [ci_up - mean]])
                colors.append("red" if class_name in MINORITY_CLASSES else "steelblue")
            else:
                means.append(0)
                errors.append([[0], [0]])
                colors.append("gray")

        x = np.arange(len(all_classes))
        ax.bar(
            x,
            means,
            yerr=np.array(errors).squeeze().T,
            color=colors,
            alpha=0.7,
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_title(f"{agg.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1 Score (95% CI)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)

        if idx == 0:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="red", alpha=0.7, label="Minority Class"),
                Patch(facecolor="steelblue", alpha=0.7, label="Majority Class"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    config_str = ", ".join(f"{k}={v}" for k, v in config_filter.items())
    fig.suptitle(
        f"Per-Class F1 Scores with Minority Class Highlighting ({config_str})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = f"perclass_bars_alpha{config_filter.get('alpha', 'all')}_adv{config_filter.get('adv_pct', 'all')}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_perclass_boxplots(df_expanded: pd.DataFrame, output_dir: Path, config_filter: dict):
    """
    Box plots showing distribution of per-class F1 across seeds.

    Args:
        df_expanded: DataFrame with per-class F1 expanded
        output_dir: Output directory
        config_filter: Dict with alpha, adv_pct to filter
    """
    df_filtered = df_expanded.copy()
    for col, val in config_filter.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    final_round = df_filtered.groupby(["aggregation", "seed"])["round"].transform("max")
    df_final = df_filtered[df_filtered["round"] == final_round]

    minority_classes_present = [c for c in MINORITY_CLASSES if c in df_final["class_name"].unique()][:9]

    if len(minority_classes_present) == 0:
        print("Warning: No minority classes found for box plots")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, class_name in enumerate(minority_classes_present):
        ax = axes[idx]
        class_data = df_final[df_final["class_name"] == class_name]

        data_by_agg = []
        labels = []

        for agg in ["fedavg", "krum", "bulyan", "median"]:
            agg_values = class_data[class_data["aggregation"] == agg]["class_f1"].dropna()
            if len(agg_values) > 0:
                data_by_agg.append(agg_values)
                labels.append(agg.capitalize())

        if data_by_agg:
            bp = ax.boxplot(
                data_by_agg,
                labels=labels,
                patch_artist=True,
                notch=True,
                showmeans=True,
            )

            for patch, agg in zip(bp["boxes"], ["fedavg", "krum", "bulyan", "median"]):
                patch.set_facecolor(COLORS.get(agg, "gray"))
                patch.set_alpha(0.6)

        ax.set_title(f"{class_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)

    for idx in range(len(minority_classes_present), 9):
        axes[idx].axis("off")

    config_str = ", ".join(f"{k}={v}" for k, v in config_filter.items())
    fig.suptitle(
        f"Minority Class F1 Distribution Across Seeds ({config_str})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    alpha_val = config_filter.get("alpha", "all")
    adv_val = config_filter.get("adv_pct", "all")
    filename = f"perclass_boxplots_alpha{alpha_val}_adv{adv_val}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_perclass_lines_convergence(df_expanded: pd.DataFrame, output_dir: Path, config_filter: dict):
    """
    Line plots showing how per-class F1 evolves over rounds.

    Args:
        df_expanded: DataFrame with per-class F1 expanded
        output_dir: Output directory
        config_filter: Dict with alpha, adv_pct to filter
    """
    df_filtered = df_expanded.copy()
    for col, val in config_filter.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    minority_classes_present = [c for c in MINORITY_CLASSES if c in df_filtered["class_name"].unique()][:6]

    if len(minority_classes_present) == 0:
        print("Warning: No minority classes found for line plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, class_name in enumerate(minority_classes_present):
        ax = axes[idx]
        class_data = df_filtered[df_filtered["class_name"] == class_name]

        for agg in ["fedavg", "krum", "bulyan", "median"]:
            agg_data = class_data[class_data["aggregation"] == agg]

            round_stats = agg_data.groupby("round")["class_f1"].apply(
                lambda x: compute_confidence_interval(x.dropna())
            ).apply(pd.Series)
            if round_stats.empty:
                continue

            round_stats.columns = ["mean", "ci_low", "ci_up"]
            rounds = round_stats.index

            ax.plot(
                rounds,
                round_stats["mean"],
                label=agg.capitalize(),
                color=COLORS.get(agg, "gray"),
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.fill_between(
                rounds,
                round_stats["ci_low"],
                round_stats["ci_up"],
                color=COLORS.get(agg, "gray"),
                alpha=0.15,
            )

        ax.set_title(f"{class_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Communication Round", fontsize=9)
        ax.set_ylabel("F1 Score", fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    config_str = ", ".join(f"{k}={v}" for k, v in config_filter.items())
    fig.suptitle(
        f"Minority Class F1 Convergence Over Rounds ({config_str})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    alpha_val = config_filter.get("alpha", "all")
    adv_val = config_filter.get("adv_pct", "all")
    filename = f"perclass_lines_alpha{alpha_val}_adv{adv_val}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_bulyan_baseline_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Dedicated 5-panel figure highlighting Bulyan performance.

    Panels:
    1. Bulyan vs FedAvg across alpha (benign)
    2. Bulyan vs FedAvg at different alpha values (bars)
    3. Statistical significance matrix
    4. Per-class F1 comparison (Bulyan vs FedAvg)
    5. Convergence speed comparison
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "Bulyan Baseline: Comprehensive Performance Analysis (Benign Scenarios)",
        fontsize=18,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(gs[0, :2])
    _plot_bulyan_vs_fedavg_alpha(df, ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    _plot_bulyan_vs_fedavg_bars_multi_alpha(df, ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    _plot_statistical_matrix(df, ax3, output_dir)

    ax4 = fig.add_subplot(gs[1, 1:])
    _plot_convergence_comparison(df, ax4)

    plt.savefig(output_dir / "bulyan_baseline_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'bulyan_baseline_comprehensive.png'}")


def _plot_bulyan_vs_fedavg_alpha(df: pd.DataFrame, ax: plt.Axes):
    """Panel 1: Bulyan vs FedAvg across alpha sweep (benign)."""
    benign_df = df[df["adv_pct"] == 0]
    max_round = benign_df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")
    final_df = benign_df[benign_df["round"] == max_round]

    for agg in ["fedavg", "bulyan"]:
        agg_data = final_df[final_df["aggregation"] == agg]
        alphas = []
        means = []
        ci_lows = []
        ci_ups = []

        for alpha in sorted(agg_data["alpha"].unique()):
            if alpha == float("inf"):
                continue
            data = agg_data[agg_data["alpha"] == alpha]["macro_f1_global"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                alphas.append(alpha)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)

        if alphas:
            ax.plot(
                alphas,
                means,
                marker="o",
                label=agg.capitalize(),
                color=COLORS[agg],
                linewidth=3,
                markersize=10,
            )
            ax.fill_between(alphas, ci_lows, ci_ups, color=COLORS[agg], alpha=0.2)

    ax.set_xlabel("Dirichlet α (Heterogeneity)", fontsize=11)
    ax.set_ylabel("Final Macro F1", fontsize=11)
    ax.set_title("Bulyan vs FedAvg: Heterogeneity Robustness (Benign)", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_bulyan_vs_fedavg_bars_multi_alpha(df: pd.DataFrame, ax: plt.Axes):
    """Panel 2: Bulyan vs FedAvg at multiple alpha values."""
    benign_df = df[df["adv_pct"] == 0]
    max_round = benign_df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")
    final_df = benign_df[benign_df["round"] == max_round]

    alphas_to_show = [0.05, 0.1, 0.2, 1.0]
    x = np.arange(len(alphas_to_show))
    width = 0.35

    for i, agg in enumerate(["fedavg", "bulyan"]):
        means = []
        errs = []
        for alpha in alphas_to_show:
            data = final_df[(final_df["aggregation"] == agg) & (final_df["alpha"] == alpha)]["macro_f1_global"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                means.append(mean)
                errs.append([[mean - ci_low], [ci_up - mean]])
            else:
                means.append(0)
                errs.append([[0], [0]])

        ax.bar(
            x + i * width,
            means,
            width,
            yerr=np.array(errs).squeeze().T if means else None,
            label=agg.capitalize(),
            color=COLORS[agg],
            alpha=0.7,
            capsize=4,
            edgecolor="black",
            linewidth=2 if agg == "bulyan" else 1,
        )

    ax.set_ylabel("Final Macro F1", fontsize=10)
    ax.set_title("Bulyan vs FedAvg Across α Values", fontsize=11, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"α={a}" for a in alphas_to_show])
    ax.set_ylim([0, 1.0])
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def _plot_statistical_matrix(df: pd.DataFrame, ax: plt.Axes, output_dir: Path):
    """Panel 3: Statistical significance matrix."""
    from statistical_analysis import pairwise_comparison_matrix

    matrix_df = pairwise_comparison_matrix(
        df,
        ["fedavg", "krum", "bulyan", "median"],
        metric="macro_f1_global",
        condition_filters={"alpha": 0.05, "adv_pct": 0},
        alpha=0.05,
        bonferroni_correction=True,
    )

    p_value_cols = [col for col in matrix_df.columns if col.endswith("_p")]
    p_values_only = matrix_df[["aggregator"] + p_value_cols].copy()

    p_values_numeric = []
    for _, row in p_values_only.iterrows():
        row_values = []
        for col in p_value_cols:
            val = row[col]
            if val == "-":
                row_values.append(np.nan)
            elif val == "N/A":
                row_values.append(1.0)
            else:
                try:
                    p_val = float(val.rstrip("*"))
                    row_values.append(p_val)
                except ValueError:
                    row_values.append(1.0)
        p_values_numeric.append(row_values)

    p_matrix = pd.DataFrame(
        p_values_numeric,
        index=p_values_only["aggregator"],
        columns=[col.replace("_p", "").capitalize() for col in p_value_cols],
    )

    sns.heatmap(
        p_matrix,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        ax=ax,
        cbar_kws={"label": "p-value"},
        linewidths=1,
        linecolor="black",
    )

    ax.set_title("Statistical Significance\n(Bonferroni-corrected)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Compared to", fontsize=9)
    ax.set_ylabel("Aggregator", fontsize=9)


def _plot_perclass_bulyan_fedavg(df: pd.DataFrame, ax: plt.Axes):
    """Panel 4: Per-class F1 comparison between Bulyan and FedAvg."""
    from load_iiot_data_enhanced import expand_per_class_f1

    df_expanded = expand_per_class_f1(df)
    filtered = df_expanded[(df_expanded["alpha"] == 0.05) & (df_expanded["adv_pct"] == 0)]
    final_df = filtered[filtered["round"] == filtered.groupby(["aggregation", "seed"])["round"].transform("max")]

    all_classes = sorted(final_df["class_name"].unique())
    x = np.arange(len(all_classes))
    width = 0.35

    for i, agg in enumerate(["fedavg", "bulyan"]):
        means = []
        for class_name in all_classes:
            class_data = final_df[(final_df["aggregation"] == agg) & (final_df["class_name"] == class_name)]["class_f1"]
            means.append(class_data.mean() if len(class_data) > 0 else 0)

        ax.bar(
            x + i * width,
            means,
            width,
            label=agg.capitalize(),
            color=COLORS[agg],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_ylabel("Mean F1 Score", fontsize=10)
    ax.set_title("Per-Class Performance: Bulyan vs FedAvg (α=0.05)", fontsize=11, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis="y", alpha=0.3)


def _plot_convergence_comparison(df: pd.DataFrame, ax: plt.Axes):
    """Panel 5: Convergence speed comparison."""
    filtered = df[(df["alpha"] == 0.05) & (df["adv_pct"] == 0)]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = filtered[filtered["aggregation"] == agg]

        round_stats = agg_data.groupby("round")["macro_f1_global"].apply(
            lambda x: compute_confidence_interval(x.dropna())
        ).apply(pd.Series)
        if round_stats.empty:
            continue

        round_stats.columns = ["mean", "ci_low", "ci_up"]
        rounds = round_stats.index

        linestyle = "-" if agg == "bulyan" else "--"
        linewidth = 3 if agg == "bulyan" else 2

        ax.plot(
            rounds,
            round_stats["mean"],
            label=agg.capitalize(),
            color=COLORS.get(agg, "gray"),
            linewidth=linewidth,
            linestyle=linestyle,
            marker="o" if agg == "bulyan" else None,
            markersize=5 if agg == "bulyan" else 0,
        )
        ax.fill_between(
            rounds,
            round_stats["ci_low"],
            round_stats["ci_up"],
            color=COLORS.get(agg, "gray"),
            alpha=0.15,
        )

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("Convergence Comparison at α=0.05 (Bulyan emphasized)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_robustness_attack(df: pd.DataFrame, ax: plt.Axes):
    """Panel 6: Robustness under attack."""
    filtered = df[df["alpha"] == 0.05]
    max_round = filtered.groupby(["aggregation", "adv_pct", "seed"])["round"].transform("max")
    final_df = filtered[filtered["round"] == max_round]

    for agg in ["fedavg", "bulyan"]:
        agg_data = final_df[final_df["aggregation"] == agg]

        adv_levels = []
        means = []
        ci_lows = []
        ci_ups = []

        for adv in sorted(agg_data["adv_pct"].unique()):
            data = agg_data[agg_data["adv_pct"] == adv]["macro_f1_global"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                adv_levels.append(adv)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)

        if adv_levels:
            linestyle = "-" if agg == "bulyan" else "--"
            linewidth = 3 if agg == "bulyan" else 2

            ax.plot(
                adv_levels,
                means,
                marker="o",
                label=agg.capitalize(),
                color=COLORS[agg],
                linewidth=linewidth,
                linestyle=linestyle,
                markersize=10 if agg == "bulyan" else 8,
            )
            ax.fill_between(adv_levels, ci_lows, ci_ups, color=COLORS[agg], alpha=0.2)

    ax.set_xlabel("Adversary %", fontsize=10)
    ax.set_ylabel("Final F1", fontsize=10)
    ax.set_title("Attack Resilience (α=0.05)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])


def save_statistical_tables(df: pd.DataFrame, output_dir: Path):
    """Save comprehensive statistical comparison tables."""
    print("\n" + "=" * 80)
    print("GENERATING STATISTICAL COMPARISON TABLES")
    print("=" * 80)

    # All comparisons: only FedAvg and Bulyan available across all alphas
    all_alpha_conditions = [
        {"alpha": 0.05, "adv_pct": 0},
        {"alpha": 0.1, "adv_pct": 0},
        {"alpha": 0.2, "adv_pct": 0},
        {"alpha": 0.5, "adv_pct": 0},
        {"alpha": 1.0, "adv_pct": 0},
    ]

    # Summary for FedAvg vs Bulyan across all alphas
    summary_fedavg_bulyan = summarize_all_comparisons(
        df,
        ["fedavg", "bulyan"],
        all_alpha_conditions,
        metric="macro_f1_global",
        alpha=0.05,
    )

    summary_fedavg_bulyan.to_csv(output_dir / "statistical_comparisons_fedavg_bulyan.csv", index=False)
    print(f"Saved: {output_dir / 'statistical_comparisons_fedavg_bulyan.csv'}")

    # At alpha=1.0, all 4 aggregators available
    summary_all_aggs = summarize_all_comparisons(
        df,
        ["fedavg", "krum", "bulyan", "median"],
        [{"alpha": 1.0, "adv_pct": 0}],
        metric="macro_f1_global",
        alpha=0.05,
    )

    summary_all_aggs.to_csv(
        output_dir / "statistical_comparisons_all_aggregators_alpha1.0.csv",
        index=False,
    )
    print(f"Saved: {output_dir / 'statistical_comparisons_all_aggregators_alpha1.0.csv'}")

    # Pairwise matrices for each alpha (FedAvg vs Bulyan)
    for alpha_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
        matrix_df = pairwise_comparison_matrix(
            df,
            ["fedavg", "bulyan"],
            metric="macro_f1_global",
            condition_filters={"alpha": alpha_val, "adv_pct": 0},
            alpha=0.05,
            bonferroni_correction=True,
        )

        filename = f"pairwise_matrix_alpha{alpha_val}_fedavg_bulyan.csv"
        matrix_df.to_csv(output_dir / filename, index=False)
        print(f"Saved: {output_dir / filename}")

    # Pairwise matrix for all 4 aggregators at alpha=1.0
    matrix_all = pairwise_comparison_matrix(
        df,
        ["fedavg", "krum", "bulyan", "median"],
        metric="macro_f1_global",
        condition_filters={"alpha": 1.0, "adv_pct": 0},
        alpha=0.05,
        bonferroni_correction=True,
    )

    matrix_all.to_csv(output_dir / "pairwise_matrix_alpha1.0_all_aggregators.csv", index=False)
    print(f"Saved: {output_dir / 'pairwise_matrix_alpha1.0_all_aggregators.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive IIoT plots with statistical analysis")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Directory containing experiment runs")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("iiot_comprehensive_plots"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IIOT COMPREHENSIVE PLOTTING WITH STATISTICAL ANALYSIS")
    print("=" * 80)

    print(f"\nLoading experiments from {args.runs_dir}...")
    df = load_iiot_data_enhanced(args.runs_dir)

    if df.empty:
        print("ERROR: No data loaded!")
        return

    print(f"SUCCESS: Loaded {len(df)} records from {df['seed'].nunique()} seeds")
    print(f"  - Aggregators: {sorted(df['aggregation'].unique())}")
    print(f"  - Alpha values: {sorted(df['alpha'].unique())}")
    print(f"  - Attack levels: {sorted(df['adv_pct'].unique())}")

    print("\nExpanding per-class F1 scores...")
    df_expanded = expand_per_class_f1(df)
    print(f"Expanded to {len(df_expanded)} records across {df_expanded['class_name'].nunique()} classes")

    print("\n" + "=" * 80)
    print("GENERATING BULYAN BASELINE COMPARISON")
    print("=" * 80)
    plot_bulyan_baseline_comparison(df, args.output_dir)

    print("\n" + "=" * 80)
    print("GENERATING PER-CLASS VISUALIZATIONS")
    print("=" * 80)

    configs = [
        {"alpha": 0.05, "adv_pct": 0},
        {"alpha": 0.1, "adv_pct": 0},
        {"alpha": 0.2, "adv_pct": 0},
        {"alpha": 0.5, "adv_pct": 0},
        {"alpha": 1.0, "adv_pct": 0},
    ]

    for config in configs:
        print(f"\nProcessing config: {config}")
        plot_perclass_heatmap(df_expanded, args.output_dir, config)
        plot_perclass_bars_minority(df_expanded, args.output_dir, config)
        plot_perclass_boxplots(df_expanded, args.output_dir, config)
        plot_perclass_lines_convergence(df_expanded, args.output_dir, config)

    save_statistical_tables(df, args.output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"All plots and tables saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
