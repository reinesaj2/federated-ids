#!/usr/bin/env python3
"""
Comprehensive Thesis Plotting Framework for IIoT Federated Learning

Generates publication-quality multi-panel figures for ALL thesis objectives:
1. Robustness: Aggregation strategies under attack
2. Heterogeneity: Handling non-IID data
3. Personalization: Local fine-tuning gains
4. System Overhead: Computational cost vs security
5. Multi-class Efficacy: Performance on 43 IIoT attack classes

Each figure contains 6 panels with rich visualizations showing trends across
multiple experimental dimensions.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Import our robust data loader and utilities
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from load_iiot_data import load_iiot_data  # noqa: E402
from plot_metrics_utils import compute_confidence_interval  # noqa: E402

# Publication-quality styling
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
    "fedavg": "#1f77b4",  # blue
    "krum": "#ff7f0e",  # orange
    "bulyan": "#2ca02c",  # green
    "median": "#d62728",  # red
    "fedprox": "#9467bd",  # purple
}


def get_max_attack_level_for_aggregator(agg: str) -> int:
    """
    Get maximum attack level for aggregator based on Byzantine resilience constraints.

    Bulyan requires n >= 4f + 3 (El Mhamdi et al. 2018).
    With n=11 clients, max f=2, so max adversary fraction = 2/11 = 18.2% ≈ 20%.
    Other methods lack formal Byzantine guarantees and can test up to 30%.

    Args:
        agg: Aggregation method name

    Returns:
        Maximum attack percentage (0-100)
    """
    if agg.lower() == "bulyan":
        return 20  # Theoretical maximum for n=11 (satisfies n >= 4(2) + 3 = 11)
    return 30  # Other methods tested to higher attack levels


def get_attack_levels_for_aggregator(agg: str) -> list[int]:
    """
    Get attack levels to evaluate for aggregator.

    Args:
        agg: Aggregation method name

    Returns:
        List of attack percentages
    """
    max_level = get_max_attack_level_for_aggregator(agg)
    if max_level == 20:
        return [0, 10, 20]
    return [0, 10, 30]


def _validate_dataframe_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}. " f"Available columns: {df.columns.tolist()}")


def plot_objective1_robustness(df: pd.DataFrame, output_dir: Path):
    """Objective 1: Robust Aggregation Under Attack (6-panel figure)."""
    _validate_dataframe_schema(
        df,
        [
            "aggregation",
            "alpha",
            "adv_pct",
            "round",
            "l2_to_benign_mean",
            "macro_f1_global",
        ],
    )

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "Objective 1: Robust Aggregation Strategies for Byzantine-Resilient IIoT IDS",
        fontsize=18,
        fontweight="bold",
    )

    # Filter to relevant aggregators and attack levels
    agg_df = df[(df["aggregation"].isin(["fedavg", "krum", "bulyan", "median"])) & (df["alpha"] == 0.5)].copy()  # Standard heterogeneity

    # Panel 1: L2 Distance vs Attack Level (top-left, wider)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_robustness_l2_vs_attack(agg_df, ax1)

    # Panel 2: F1 Score vs Attack Level (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_utility_vs_attack(agg_df, ax2)

    # Panel 3: Convergence trajectories (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_convergence_trajectories(agg_df, ax3)

    # Panel 4: Attack resilience heatmap (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_attack_resilience_heatmap(agg_df, ax4)

    # Panel 5: Robustness vs utility tradeoff (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_robustness_utility_tradeoff(agg_df, ax5)

    # Panel 6: Per-aggregator performance breakdown (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    plot_aggregator_performance_bars(agg_df, ax6)

    plt.savefig(output_dir / "obj1_robustness_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'obj1_robustness_comprehensive.png'}")


def plot_robustness_l2_vs_attack(df: pd.DataFrame, ax: plt.Axes):
    """L2 distance to benign mean across attack levels."""
    if "l2_to_benign_mean" not in df.columns:
        ax.text(0.5, 0.5, "L2 data not available", ha="center", va="center")
        return

    # Group by aggregation and adversary percentage
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = df[df["aggregation"] == agg]

        adv_levels = []
        means = []
        ci_lowers = []
        ci_uppers = []

        for adv in sorted(agg_data["adv_pct"].unique()):
            data = agg_data[agg_data["adv_pct"] == adv]["l2_to_benign_mean"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                adv_levels.append(adv)
                means.append(mean)
                ci_lowers.append(ci_low)
                ci_uppers.append(ci_up)

        if len(adv_levels) > 0:
            ax.plot(adv_levels, means, marker="o", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2.5, markersize=8)
            ax.fill_between(adv_levels, ci_lowers, ci_uppers, color=COLORS.get(agg, "gray"), alpha=0.2)

    ax.set_xlabel("Adversary Percentage (%)", fontsize=11)
    ax.set_ylabel("L2 Distance to Benign Mean", fontsize=11)
    ax.set_title("Robustness: Distance from Poisoned Consensus", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)


def plot_utility_vs_attack(df: pd.DataFrame, ax: plt.Axes):
    """F1 score degradation under attack."""
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = df[df["aggregation"] == agg]

        adv_levels = []
        means = []
        ci_lowers = []
        ci_uppers = []

        for adv in sorted(agg_data["adv_pct"].unique()):
            # Get final round F1 scores
            data = agg_data[
                (agg_data["adv_pct"] == adv) & (agg_data["round"] == agg_data.groupby(["seed", "adv_pct"])["round"].transform("max"))
            ]["macro_f1_global"].dropna()

            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                adv_levels.append(adv)
                means.append(mean)
                ci_lowers.append(ci_low)
                ci_uppers.append(ci_up)

        if len(adv_levels) > 0:
            ax.plot(adv_levels, means, marker="s", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2.5, markersize=8)
            ax.fill_between(adv_levels, ci_lowers, ci_uppers, color=COLORS.get(agg, "gray"), alpha=0.2)

    ax.set_xlabel("Adversary Percentage (%)", fontsize=11)
    ax.set_ylabel("Final Macro F1 Score", fontsize=11)
    ax.set_title("Utility Under Attack", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])


def plot_convergence_trajectories(df: pd.DataFrame, ax: plt.Axes):
    """F1 convergence over rounds for each aggregator at max attack level."""
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        max_adv = get_max_attack_level_for_aggregator(agg)
        agg_data = df[(df["aggregation"] == agg) & (df["adv_pct"] == max_adv)]

        if len(agg_data) == 0:
            continue

        # Group by round and compute mean F1
        round_stats = agg_data.groupby("round")["macro_f1_global"].apply(lambda x: compute_confidence_interval(x.dropna())).apply(pd.Series)
        round_stats.columns = ["mean", "ci_low", "ci_up"]

        rounds = round_stats.index
        label = f"{agg.capitalize()} ({max_adv}%)"
        ax.plot(rounds, round_stats["mean"], label=label, color=COLORS.get(agg, "gray"), linewidth=2)
        ax.fill_between(rounds, round_stats["ci_low"], round_stats["ci_up"], color=COLORS.get(agg, "gray"), alpha=0.15)

    ax.set_xlabel("Communication Round", fontsize=10)
    ax.set_ylabel("Macro F1 Score", fontsize=10)
    ax.set_title("Convergence Under Max Attack", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98,
        0.02,
        "Bulyan: 20% max (n≥4f+3 constraint)",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        style="italic",
        color="gray",
    )


def plot_attack_resilience_heatmap(df: pd.DataFrame, ax: plt.Axes):
    """Heatmap: Aggregator × Attack Level → Final F1."""
    # Prepare data for heatmap with hybrid attack levels
    pivot_data = []

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        row_data = []
        attack_levels = get_attack_levels_for_aggregator(agg)

        for adv in [0, 10, 20, 30]:
            if adv not in attack_levels:
                row_data.append(np.nan)
            else:
                data = df[
                    (df["aggregation"] == agg)
                    & (df["adv_pct"] == adv)
                    & (df["round"] == df.groupby(["aggregation", "adv_pct", "seed"])["round"].transform("max"))
                ]["macro_f1_global"].dropna()

                if len(data) > 0:
                    row_data.append(data.mean())
                else:
                    row_data.append(np.nan)
        pivot_data.append(row_data)

    pivot_df = pd.DataFrame(
        pivot_data,
        index=["FedAvg", "Krum", "Bulyan", "Median"],
        columns=["0%", "10%", "20%", "30%"],
    )

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "Final F1 Score"},
        mask=pivot_df.isna(),
    )
    ax.set_xlabel("Adversary Level", fontsize=10)
    ax.set_ylabel("Aggregator", fontsize=10)
    ax.set_title("Attack Resilience Matrix", fontsize=11, fontweight="bold")
    ax.text(
        0.98,
        -0.15,
        "Bulyan limited to 20% by Byzantine constraint (n≥4f+3)",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        style="italic",
        color="gray",
    )


def plot_robustness_utility_tradeoff(df: pd.DataFrame, ax: plt.Axes):
    """Scatter: L2 distance (robustness) vs F1 (utility) for each aggregator."""
    # Get final round data
    final_df = df[df["round"] == df.groupby(["aggregation", "adv_pct", "seed"])["round"].transform("max")]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = final_df[final_df["aggregation"] == agg]

        # Scatter with color by adversary percentage
        scatter = ax.scatter(
            agg_data["l2_to_benign_mean"],
            agg_data["macro_f1_global"],
            c=agg_data["adv_pct"],
            cmap="RdYlGn_r",
            marker="o" if agg == "fedavg" else ("s" if agg == "krum" else ("D" if agg == "bulyan" else "^")),
            s=100,
            alpha=0.7,
            label=agg.capitalize(),
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("L2 Distance (Robustness →)", fontsize=10)
    ax.set_ylabel("Final F1 Score (Utility →)", fontsize=10)
    ax.set_title("Robustness-Utility Tradeoff", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Adv %", fontsize=9)


def plot_aggregator_performance_bars(df: pd.DataFrame, ax: plt.Axes):
    """Grouped bar chart: Aggregator performance across attack levels."""
    # Compute mean final F1 for each (aggregator, adv_pct)
    final_df = df[df["round"] == df.groupby(["aggregation", "adv_pct", "seed"])["round"].transform("max")]

    stats_data = []
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        attack_levels = get_attack_levels_for_aggregator(agg)
        for adv in attack_levels:
            agg_mask = (final_df["aggregation"] == agg) & (final_df["adv_pct"] == adv)
            data = final_df[agg_mask]["macro_f1_global"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                stats_data.append(
                    {
                        "aggregator": agg.capitalize(),
                        "adv_pct": f"{adv}%",
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_up": ci_up,
                        "n": len(data),
                    }
                )

    stats_df = pd.DataFrame(stats_data)

    # Plot grouped bars
    if len(stats_df) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(4)  # 4 aggregators
    width = 0.22

    # Use all attack levels that appear in data
    all_attack_levels = sorted(stats_df["adv_pct"].unique(), key=lambda x: int(x.rstrip("%")))

    for i, adv in enumerate(all_attack_levels):
        subset = stats_df[stats_df["adv_pct"] == adv].copy()

        means = []
        err_lower = []
        err_upper = []

        for agg in ["FedAvg", "Krum", "Bulyan", "Median"]:
            agg_data = subset[subset["aggregator"] == agg]
            if len(agg_data) > 0:
                mean_val = agg_data["mean"].values[0]
                ci_low_val = agg_data["ci_low"].values[0]
                ci_up_val = agg_data["ci_up"].values[0]
                means.append(mean_val)
                err_lower.append(mean_val - ci_low_val)
                err_upper.append(ci_up_val - mean_val)
            else:
                means.append(np.nan)
                err_lower.append(0)
                err_upper.append(0)

        ax.bar(
            x + i * width,
            means,
            width,
            label=f"Adv={adv}",
            yerr=[err_lower, err_upper],
            capsize=4,
            alpha=0.8,
        )

    ax.set_xlabel("Aggregation Strategy", fontsize=11)
    ax.set_ylabel("Final Macro F1 Score", fontsize=11)
    ax.set_title("Performance Comparison Across Attack Intensities (95% CI)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (len(all_attack_levels) - 1) / 2)
    ax.set_xticklabels(["FedAvg", "Krum", "Bulyan", "Median"])
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.0])
    ax.text(
        0.98,
        0.02,
        "Bulyan max: 20% (n≥4f+3)",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        style="italic",
        color="gray",
    )


def plot_objective2_heterogeneity(df: pd.DataFrame, output_dir: Path):
    """Objective 2: Heterogeneity Handling (Alpha sweep)."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "Objective 2: Handling Data Heterogeneity in IIoT Federated IDS",
        fontsize=18,
        fontweight="bold",
    )

    # Filter to benign runs (Adv=0) across alpha values
    het_df = df[(df["adv_pct"] == 0)].copy()

    # Panel 1: F1 vs Alpha (top, wider)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_f1_vs_alpha(het_df, ax1)

    # Panel 2: Convergence speed vs Alpha (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_convergence_speed(het_df, ax2)

    # Panel 3: Client drift by alpha (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_drift_by_alpha(het_df, ax3)

    # Panel 4: Alpha heatmap (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_alpha_heatmap(het_df, ax4)

    # Panel 5: IID vs Non-IID comparison (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_iid_vs_noniid(het_df, ax5)

    # Panel 6: Per-alpha convergence trajectories (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    plot_alpha_convergence_trajectories(het_df, ax6)

    plt.savefig(output_dir / "obj2_heterogeneity_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'obj2_heterogeneity_comprehensive.png'}")


def plot_f1_vs_alpha(df: pd.DataFrame, ax: plt.Axes):
    """Final F1 score vs alpha for each aggregator."""
    # Get final round F1
    final_df = df[df["round"] == df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = final_df[final_df["aggregation"] == agg]

        alphas = []
        means = []
        ci_lows = []
        ci_ups = []

        for alpha in sorted(agg_data["alpha"].unique()):
            if alpha == float("inf"):
                continue  # Skip infinity for plotting
            data = agg_data[agg_data["alpha"] == alpha]["macro_f1_global"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                alphas.append(alpha)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)

        if len(alphas) > 0:
            ax.plot(alphas, means, marker="o", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2.5, markersize=8)
            ax.fill_between(alphas, ci_lows, ci_ups, color=COLORS.get(agg, "gray"), alpha=0.2)

    ax.set_xlabel("Dirichlet α (Heterogeneity: 0=high, 1=low)", fontsize=11)
    ax.set_ylabel("Final Macro F1 Score", fontsize=11)
    ax.set_title("Performance vs Data Heterogeneity (Benign)", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color="gray", linestyle=":", label="IID threshold")


def plot_convergence_speed(df: pd.DataFrame, ax: plt.Axes):
    """Rounds to reach 90% of final F1."""
    # Compute for each run
    results = []

    for (agg, alpha, seed), group in df.groupby(["aggregation", "alpha", "seed"]):
        if alpha == float("inf"):
            continue
        group = group.sort_values("round")
        final_f1 = group["macro_f1_global"].iloc[-1]
        target = 0.9 * final_f1

        rounds_to_target = None
        for _, row in group.iterrows():
            if row["macro_f1_global"] >= target:
                rounds_to_target = row["round"]
                break

        if rounds_to_target:
            results.append({"aggregation": agg, "alpha": alpha, "rounds": rounds_to_target})

    if results:
        results_df = pd.DataFrame(results)
        for agg in ["fedavg", "krum", "bulyan", "median"]:
            agg_data = results_df[results_df["aggregation"] == agg]
            if len(agg_data) == 0:
                continue

            alphas = sorted(agg_data["alpha"].unique())
            means = [agg_data[agg_data["alpha"] == a]["rounds"].mean() for a in alphas]

            ax.plot(alphas, means, marker="s", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2)

    ax.set_xlabel("Dirichlet α", fontsize=10)
    ax.set_ylabel("Rounds to 90% Final F1", fontsize=10)
    ax.set_title("Convergence Speed", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_drift_by_alpha(df: pd.DataFrame, ax: plt.Axes):
    """Final L2 dispersion by alpha."""
    final_df = df[df["round"] == df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = final_df[final_df["aggregation"] == agg]

        alphas = []
        means = []

        for alpha in sorted(agg_data["alpha"].unique()):
            if alpha == float("inf"):
                continue
            data = agg_data[agg_data["alpha"] == alpha]["l2_dispersion_mean"].dropna()
            if len(data) > 0:
                alphas.append(alpha)
                means.append(data.mean())

        if len(alphas) > 0:
            ax.plot(alphas, means, marker="D", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2)

    ax.set_xlabel("Dirichlet α", fontsize=10)
    ax.set_ylabel("Final L2 Dispersion", fontsize=10)
    ax.set_title("Client Model Drift", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_alpha_heatmap(df: pd.DataFrame, ax: plt.Axes):
    """Heatmap: Aggregator × Alpha → Final F1."""
    final_df = df[df["round"] == df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    alphas_to_show = [0.02, 0.05, 0.1, 0.5, 1.0]
    pivot_data = []

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        row = []
        for alpha in alphas_to_show:
            data = final_df[(final_df["aggregation"] == agg) & (final_df["alpha"] == alpha)]["macro_f1_global"].dropna()
            row.append(data.mean() if len(data) > 0 else np.nan)
        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data, index=["FedAvg", "Krum", "Bulyan", "Median"], columns=[f"α={a}" for a in alphas_to_show])

    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.5, vmax=0.7, ax=ax, cbar_kws={"label": "Final F1"})
    ax.set_title("Heterogeneity Impact Matrix", fontsize=11, fontweight="bold")


def plot_iid_vs_noniid(df: pd.DataFrame, ax: plt.Axes):
    """Bar comparison: IID (α=1.0) vs Non-IID (α=0.1)."""
    final_df = df[df["round"] == df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    x = np.arange(4)
    width = 0.35

    for i, alpha in enumerate([1.0, 0.1]):
        means = []
        errs = []
        for agg in ["fedavg", "krum", "bulyan", "median"]:
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
            yerr=np.array(errs).squeeze().T,
            width=width,
            label=f"{'IID' if alpha == 1.0 else 'Non-IID'} (α={alpha})",
            capsize=4,
            alpha=0.8,
        )

    ax.set_xlabel("Aggregator", fontsize=10)
    ax.set_ylabel("Final F1 Score", fontsize=10)
    ax.set_title("IID vs Non-IID Performance", fontsize=11, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["FedAvg", "Krum", "Bulyan", "Median"])
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_alpha_convergence_trajectories(df: pd.DataFrame, ax: plt.Axes):
    """Convergence for FedAvg across different alphas."""
    alphas_to_show = [0.02, 0.1, 0.5, 1.0]
    fedavg_df = df[df["aggregation"] == "fedavg"]

    for alpha in alphas_to_show:
        alpha_data = fedavg_df[fedavg_df["alpha"] == alpha]

        round_stats = (
            alpha_data.groupby("round")["macro_f1_global"].apply(lambda x: compute_confidence_interval(x.dropna())).apply(pd.Series)
        )
        round_stats.columns = ["mean", "ci_low", "ci_up"]

        rounds = round_stats.index
        ax.plot(rounds, round_stats["mean"], label=f"α={alpha}", linewidth=2, marker="o", markersize=4)
        ax.fill_between(rounds, round_stats["ci_low"], round_stats["ci_up"], alpha=0.15)

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("FedAvg Convergence Under Different Heterogeneity Levels", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def plot_objective3_personalization(df: pd.DataFrame, output_dir: Path):
    """Objective 3: Personalization Gains & Risks."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "Objective 3: Personalization for Client-Specific IIoT Attack Detection",
        fontsize=18,
        fontweight="bold",
    )

    # Filter to personalization experiments
    pers_df = df[df["pers_epochs"] > 0].copy()

    # Panel 1: Gain vs adversary level (top, wider)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_gain_vs_adversary(pers_df, ax1)

    # Panel 2: Gain vs heterogeneity (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_gain_vs_heterogeneity(pers_df, ax2)

    # Panel 3: Gain by epochs (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_gain_by_epochs(pers_df, ax3)

    # Panel 4: Risk under attack (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_personalization_risk(pers_df, ax4)

    # Panel 5: Per-aggregator gains (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_gain_by_aggregator(pers_df, ax5)

    # Panel 6: Before/after comparison (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    plot_before_after_comparison(pers_df, ax6)

    plt.savefig(output_dir / "obj3_personalization_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'obj3_personalization_comprehensive.png'}")


def plot_gain_vs_adversary(df: pd.DataFrame, ax: plt.Axes):
    """Personalization gain vs attack level."""
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = df[df["aggregation"] == agg]

        adv_levels = []
        means = []
        ci_lows = []
        ci_ups = []

        for adv in sorted(agg_data["adv_pct"].unique()):
            data = agg_data[agg_data["adv_pct"] == adv]["personalization_gain"].dropna()
            if len(data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(data)
                adv_levels.append(adv)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)

        if len(adv_levels) > 0:
            ax.plot(adv_levels, means, marker="o", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2.5, markersize=8)
            ax.fill_between(adv_levels, ci_lows, ci_ups, color=COLORS.get(agg, "gray"), alpha=0.2)

    ax.set_xlabel("Adversary Percentage (%)", fontsize=11)
    ax.set_ylabel("Mean Personalization Gain (ΔF1)", fontsize=11)
    ax.set_title("Personalization Benefit vs Attack Intensity", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle=":", linewidth=1.5, label="Break-even")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def plot_gain_vs_heterogeneity(df: pd.DataFrame, ax: plt.Axes):
    """Personalization gain vs alpha (benign only)."""
    benign_df = df[df["adv_pct"] == 0]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = benign_df[benign_df["aggregation"] == agg]

        alphas = []
        means = []

        for alpha in sorted(agg_data["alpha"].unique()):
            if alpha == float("inf"):
                continue
            data = agg_data[agg_data["alpha"] == alpha]["personalization_gain"].dropna()
            if len(data) > 0:
                alphas.append(alpha)
                means.append(data.mean())

        if len(alphas) > 0:
            ax.plot(alphas, means, marker="s", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2)

    ax.set_xlabel("Dirichlet α", fontsize=10)
    ax.set_ylabel("Mean Gain (ΔF1)", fontsize=10)
    ax.set_title("Gain vs Heterogeneity", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.axhline(y=0, color="black", linestyle=":", linewidth=1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_gain_by_epochs(df: pd.DataFrame, ax: plt.Axes):
    """Personalization gain by number of epochs (benign)."""
    benign_df = df[df["adv_pct"] == 0]

    epochs_list = sorted(benign_df["pers_epochs"].unique())

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = benign_df[benign_df["aggregation"] == agg]

        means = []
        for epochs in epochs_list:
            data = agg_data[agg_data["pers_epochs"] == epochs]["personalization_gain"].dropna()
            if len(data) > 0:
                means.append(data.mean())
            else:
                means.append(np.nan)

        if len([m for m in means if not np.isnan(m)]) > 0:
            ax.plot(epochs_list, means, marker="D", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2, markersize=8)

    ax.set_xlabel("Personalization Epochs", fontsize=10)
    ax.set_ylabel("Mean Gain (ΔF1)", fontsize=10)
    ax.set_title("Gain by Training Duration", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_personalization_risk(df: pd.DataFrame, ax: plt.Axes):
    """Show percentage of clients with negative gain under attack."""
    results = []

    for adv in sorted(df["adv_pct"].unique()):
        adv_data = df[df["adv_pct"] == adv]["personalization_gain"].dropna()
        if len(adv_data) > 0:
            pct_negative = (adv_data < -0.01).sum() / len(adv_data) * 100
            pct_positive = (adv_data > 0.01).sum() / len(adv_data) * 100
            results.append({"adv": adv, "negative": pct_negative, "positive": pct_positive, "neutral": 100 - pct_negative - pct_positive})

    if results:
        results_df = pd.DataFrame(results)

        x = results_df["adv"]
        ax.bar(x, results_df["positive"], label="Positive Gain", color="green", alpha=0.7)
        ax.bar(x, results_df["neutral"], bottom=results_df["positive"], label="Neutral", color="gray", alpha=0.5)
        ax.bar(
            x,
            results_df["negative"],
            bottom=results_df["positive"] + results_df["neutral"],
            label="Negative Gain (Risk)",
            color="red",
            alpha=0.7,
        )

    ax.set_xlabel("Adversary %", fontsize=10)
    ax.set_ylabel("Client Distribution (%)", fontsize=10)
    ax.set_title("Personalization Risk Profile", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim([0, 100])
    ax.grid(axis="y", alpha=0.3)


def plot_gain_by_aggregator(df: pd.DataFrame, ax: plt.Axes):
    """Box plot: gain distribution by aggregator."""
    benign_df = df[df["adv_pct"] == 0]

    data_to_plot = []
    labels = []

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = benign_df[benign_df["aggregation"] == agg]["personalization_gain"].dropna()
        if len(agg_data) > 0:
            data_to_plot.append(agg_data)
            labels.append(agg.capitalize())

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

        for patch, agg in zip(bp['boxes'], ["fedavg", "krum", "bulyan", "median"]):
            patch.set_facecolor(COLORS.get(agg, "gray"))
            patch.set_alpha(0.6)

    ax.set_ylabel("Personalization Gain (ΔF1)", fontsize=10)
    ax.set_title("Gain Distribution (Benign)", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle=":", linewidth=1)
    ax.grid(axis="y", alpha=0.3)


def plot_before_after_comparison(df: pd.DataFrame, ax: plt.Axes):
    """Scatter: global F1 vs personalized F1 (benign)."""
    benign_df = df[df["adv_pct"] == 0]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = benign_df[benign_df["aggregation"] == agg]

        ax.scatter(
            agg_data["macro_f1_global"],
            agg_data["macro_f1_personalized"],
            label=agg.capitalize(),
            color=COLORS.get(agg, "gray"),
            alpha=0.6,
            s=50,
        )

    # Diagonal line (no improvement)
    lims = [0.5, 0.8]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label="No Improvement")

    ax.set_xlabel("Global Model F1 (Before)", fontsize=11)
    ax.set_ylabel("Personalized Model F1 (After)", fontsize=11)
    ax.set_title("Personalization Impact: Before vs After (Benign Conditions)", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_objective4_system_overhead(df: pd.DataFrame, output_dir: Path):
    """Objective 4: System Overhead & Cost-Benefit Analysis."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "Objective 4: Computational Overhead vs Security in IIoT Federated IDS",
        fontsize=18,
        fontweight="bold",
    )

    # Panel 1: Aggregation time by method (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_aggregation_time(df, ax1)

    # Panel 2: Time vs attack level (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_time_vs_attack(df, ax2)

    # Panel 3: Time vs alpha (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_time_vs_alpha(df, ax3)

    # Panel 4: Cost-benefit scatter (middle-left, wider)
    ax4 = fig.add_subplot(gs[1, :2])
    plot_cost_benefit(df, ax4)

    # Panel 5: Overhead comparison (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_overhead_comparison(df, ax5)

    # Panel 6: Cumulative time over rounds (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    plot_cumulative_time(df, ax6)

    plt.savefig(output_dir / "obj4_system_overhead_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'obj4_system_overhead_comprehensive.png'}")


def plot_aggregation_time(df: pd.DataFrame, ax: plt.Axes):
    """Box plot: aggregation time by method."""
    data_to_plot = []
    labels = []

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = df[df["aggregation"] == agg]["t_aggregate_ms"].dropna()
        if len(agg_data) > 0:
            data_to_plot.append(agg_data)
            labels.append(agg.capitalize())

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

        for patch, agg in zip(bp['boxes'], ["fedavg", "krum", "bulyan", "median"]):
            patch.set_facecolor(COLORS.get(agg, "gray"))
            patch.set_alpha(0.6)

    ax.set_ylabel("Aggregation Time (ms)", fontsize=10)
    ax.set_title("Aggregation Overhead", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)


def plot_time_vs_attack(df: pd.DataFrame, ax: plt.Axes):
    """Aggregation time vs attack level."""
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = df[df["aggregation"] == agg]

        adv_levels = []
        means = []

        for adv in sorted(agg_data["adv_pct"].unique()):
            data = agg_data[agg_data["adv_pct"] == adv]["t_aggregate_ms"].dropna()
            if len(data) > 0:
                adv_levels.append(adv)
                means.append(data.mean())

        if len(adv_levels) > 0:
            ax.plot(adv_levels, means, marker="o", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2)

    ax.set_xlabel("Adversary %", fontsize=10)
    ax.set_ylabel("Mean Time (ms)", fontsize=10)
    ax.set_title("Overhead vs Attack Level", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_time_vs_alpha(df: pd.DataFrame, ax: plt.Axes):
    """Aggregation time vs heterogeneity."""
    benign_df = df[df["adv_pct"] == 0]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = benign_df[benign_df["aggregation"] == agg]

        alphas = []
        means = []

        for alpha in sorted(agg_data["alpha"].unique()):
            if alpha == float("inf"):
                continue
            data = agg_data[agg_data["alpha"] == alpha]["t_aggregate_ms"].dropna()
            if len(data) > 0:
                alphas.append(alpha)
                means.append(data.mean())

        if len(alphas) > 0:
            ax.plot(alphas, means, marker="s", label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2)

    ax.set_xlabel("Dirichlet α", fontsize=10)
    ax.set_ylabel("Mean Time (ms)", fontsize=10)
    ax.set_title("Overhead vs Heterogeneity", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cost_benefit(df: pd.DataFrame, ax: plt.Axes):
    """Scatter: aggregation time vs F1 score under attack."""
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        max_adv = get_max_attack_level_for_aggregator(agg)
        agg_data = df[(df["aggregation"] == agg) & (df["adv_pct"] == max_adv)]

        label = f"{agg.capitalize()} ({max_adv}%)"
        ax.scatter(
            agg_data["t_aggregate_ms"],
            agg_data["macro_f1_global"],
            label=label,
            color=COLORS.get(agg, "gray"),
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidths=1,
        )

    ax.set_xlabel("Aggregation Time (ms, log scale)", fontsize=11)
    ax.set_ylabel("F1 Score Under Max Attack", fontsize=11)
    ax.set_title("Cost-Benefit Tradeoff: Computational Cost vs Security", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        "Bulyan: 20% max (n≥4f+3 constraint)",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="top",
        style="italic",
        color="gray",
    )


def plot_overhead_comparison(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: relative overhead compared to FedAvg."""
    benign_df = df[df["adv_pct"] == 0]

    fedavg_time = benign_df[benign_df["aggregation"] == "fedavg"]["t_aggregate_ms"].mean()

    aggs = []
    overheads = []

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_time = benign_df[benign_df["aggregation"] == agg]["t_aggregate_ms"].mean()
        if not np.isnan(agg_time):
            aggs.append(agg.capitalize())
            overheads.append(agg_time / fedavg_time if fedavg_time > 0 else 0)

    if aggs:
        bars = ax.bar(aggs, overheads, color=[COLORS.get(a.lower(), "gray") for a in aggs], alpha=0.7)

        # Annotate bars
        for bar, overhead in zip(bars, overheads):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{overhead:.1f}×', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Relative Overhead (× FedAvg)", fontsize=10)
    ax.set_title("Overhead Multiplier", fontsize=11, fontweight="bold")
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, label="FedAvg Baseline")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_cumulative_time(df: pd.DataFrame, ax: plt.Axes):
    """Cumulative aggregation time over rounds."""
    alpha_05 = df[(df["alpha"] == 0.5) & (df["adv_pct"] == 0)]

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = alpha_05[alpha_05["aggregation"] == agg]

        # Group by seed and compute cumulative time
        cumulative_times = []
        all_rounds = []

        for seed in agg_data["seed"].unique():
            seed_data = agg_data[agg_data["seed"] == seed].sort_values("round")
            cumsum = seed_data["t_aggregate_ms"].cumsum()
            cumulative_times.append(cumsum.values)
            all_rounds.append(seed_data["round"].values)

        if cumulative_times:
            # Find minimum length across all seeds to handle variable-length runs
            min_len = min(len(ct) for ct in cumulative_times)
            cumulative_times_trimmed = [ct[:min_len] for ct in cumulative_times]
            rounds_trimmed = all_rounds[0][:min_len]  # Use first seed's rounds as reference
            mean_cumulative = np.mean(cumulative_times_trimmed, axis=0)
            ax.plot(rounds_trimmed, mean_cumulative, label=agg.capitalize(), color=COLORS.get(agg, "gray"), linewidth=2.5)

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Cumulative Aggregation Time (ms)", fontsize=11)
    ax.set_title("Total Computational Cost Over Training (α=0.5, Benign)", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive thesis plots for IIoT")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Directory containing experiment runs")
    parser.add_argument("--output_dir", type=Path, default=Path("thesis_plots_iiot"), help="Output directory for plots")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE THESIS PLOTTING FRAMEWORK")
    print("=" * 80)

    print(f"\nLoading all experiments from {args.runs_dir}...")
    df = load_iiot_data(args.runs_dir)

    if df.empty:
        print("ERROR: No data loaded!")
        return

    print(f"SUCCESS: Loaded {len(df)} records from {df['seed'].nunique()} seeds")
    print(f"  - Aggregators: {sorted(df['aggregation'].unique())}")
    print(f"  - Alpha values: {sorted(df['alpha'].unique())}")
    print(f"  - Attack levels: {sorted(df['adv_pct'].unique())}")

    # Validate schema
    _validate_dataframe_schema(
        df,
        [
            "aggregation",
            "alpha",
            "adv_pct",
            "pers_epochs",
            "seed",
            "round",
            "l2_to_benign_mean",
            "l2_dispersion_mean",
            "t_aggregate_ms",
            "macro_f1_global",
            "macro_f1_personalized",
        ],
    )

    print("\n" + "=" * 80)
    print("GENERATING OBJECTIVE 1: ROBUSTNESS PLOTS")
    print("=" * 80)
    plot_objective1_robustness(df, args.output_dir)

    print("\n" + "=" * 80)
    print("GENERATING OBJECTIVE 2: HETEROGENEITY PLOTS")
    print("=" * 80)
    plot_objective2_heterogeneity(df, args.output_dir)

    print("\n" + "=" * 80)
    print("GENERATING OBJECTIVE 3: PERSONALIZATION PLOTS")
    print("=" * 80)
    plot_objective3_personalization(df, args.output_dir)

    print("\n" + "=" * 80)
    print("GENERATING OBJECTIVE 4: SYSTEM OVERHEAD PLOTS")
    print("=" * 80)
    plot_objective4_system_overhead(df, args.output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"All 4 comprehensive plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
