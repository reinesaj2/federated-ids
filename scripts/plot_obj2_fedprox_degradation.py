#!/usr/bin/env python3
"""
Objective 2: FedProx Performance Degradation Analysis

Generates comprehensive visualizations showing how FedProx proximal term strength (mu)
affects performance across different heterogeneity levels (alpha).

Key findings to visualize:
1. FedProx with strong mu (0.05, 0.1) degrades performance vs FedAvg
2. Degradation is worse at low alpha (high heterogeneity)
3. Weak mu (0.01) performs similarly to FedAvg
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("results/comparative_analysis")
THESIS_OUTPUT_DIR = Path("results/thesis_plots_package/objective2_heterogeneity")


def parse_run_config(run_name: str) -> dict:
    """Extract configuration from run directory name."""
    config = {
        "aggregation": "unknown",
        "alpha": None,
        "adversary": 0,
        "seed": 0,
        "mu": 0.0,
    }

    if "comp_fedavg" in run_name:
        config["aggregation"] = "FedAvg"
    elif "comp_fedprox" in run_name or "fedprox" in run_name.lower():
        config["aggregation"] = "FedProx"
    elif "comp_krum" in run_name:
        config["aggregation"] = "Krum"
    elif "comp_bulyan" in run_name:
        config["aggregation"] = "Bulyan"
    elif "comp_median" in run_name:
        config["aggregation"] = "Median"

    alpha_match = re.search(r"_alpha([\d.]+|inf)_", run_name)
    if alpha_match:
        val = alpha_match.group(1)
        config["alpha"] = float("inf") if val == "inf" else float(val)

    adv_match = re.search(r"_adv(\d+)_", run_name)
    if adv_match:
        config["adversary"] = int(adv_match.group(1))

    seed_match = re.search(r"_seed(\d+)", run_name)
    if seed_match:
        config["seed"] = int(seed_match.group(1))

    mu_match = re.search(r"_mu([\d.]+)_", run_name)
    if mu_match:
        config["mu"] = float(mu_match.group(1))

    return config


def load_client_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load and combine all client metrics."""
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None

    dfs = []
    for cf in client_files:
        try:
            df = pd.read_csv(cf)
            client_match = re.search(r"client_(\d+)", cf.name)
            if client_match:
                df["client_id"] = int(client_match.group(1))
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def collect_fedprox_data():
    """Collect FedAvg and FedProx data across alpha and mu values."""
    all_data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = parse_run_config(run_dir.name)

        if config["alpha"] is None or config["alpha"] > 100:
            continue

        if config["adversary"] != 0:
            continue

        if config["aggregation"] not in ["FedAvg", "FedProx"]:
            continue

        client_df = load_client_metrics(run_dir)

        if client_df is not None and "macro_f1_after" in client_df.columns:
            final_round = client_df["round"].max()
            final_data = client_df[client_df["round"] == final_round]

            for _, row in final_data.iterrows():
                all_data.append({
                    "aggregation": config["aggregation"],
                    "alpha": config["alpha"],
                    "mu": config["mu"],
                    "seed": config["seed"],
                    "client_id": row["client_id"],
                    "round": row["round"],
                    "f1": row["macro_f1_after"],
                })

    return pd.DataFrame(all_data)


def plot_fedprox_degradation(df: pd.DataFrame):
    """Create comprehensive FedProx degradation analysis plots."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "FedProx Performance Degradation: How Proximal Term Strength (μ) Affects Heterogeneous IDS Training",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    alpha_values = sorted([a for a in df["alpha"].unique() if 0 < a < 100])
    mu_values = sorted(df["mu"].unique())

    print(f"Alpha values: {alpha_values}")
    print(f"Mu values: {mu_values}")

    colors_mu = {
        0.0: "#1f77b4",     # FedAvg - blue
        0.01: "#2ca02c",    # weak mu - green
        0.05: "#ff7f0e",    # medium mu - orange
        0.1: "#d62728",     # strong mu - red
    }

    # Panel 1: F1 vs Alpha for different mu values (main finding)
    ax1 = fig.add_subplot(gs[0, :2])

    summary = df.groupby(["alpha", "mu"]).agg(
        f1_mean=("f1", "mean"),
        f1_sem=("f1", "sem"),
        f1_std=("f1", "std"),
        n=("f1", "count")
    ).reset_index()

    for mu in mu_values:
        subset = summary[summary["mu"] == mu].sort_values("alpha")
        if subset.empty:
            continue

        label = "FedAvg (μ=0)" if mu == 0.0 else f"FedProx (μ={mu})"
        marker = "o" if mu == 0.0 else ("s" if mu == 0.01 else ("D" if mu == 0.05 else "^"))

        ax1.plot(
            subset["alpha"],
            subset["f1_mean"],
            marker=marker,
            color=colors_mu.get(mu, "#333"),
            label=label,
            linewidth=2.5,
            markersize=10,
        )
        ax1.fill_between(
            subset["alpha"],
            subset["f1_mean"] - 1.96 * subset["f1_sem"],
            subset["f1_mean"] + 1.96 * subset["f1_sem"],
            color=colors_mu.get(mu, "#333"),
            alpha=0.15,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Dirichlet α (Lower = More Heterogeneous)", fontsize=14)
    ax1.set_ylabel("Final Macro-F1 Score", fontsize=14)
    ax1.set_title("(A) Performance vs Heterogeneity: FedAvg vs FedProx with Different μ", fontweight="bold", fontsize=14)
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.30, 0.75)

    # Panel 2: Relative degradation from FedAvg baseline
    ax2 = fig.add_subplot(gs[0, 2])

    fedavg_baseline = summary[summary["mu"] == 0.0][["alpha", "f1_mean"]].rename(columns={"f1_mean": "fedavg_f1"})
    merged = summary.merge(fedavg_baseline, on="alpha", how="left")
    merged["degradation_pct"] = ((merged["f1_mean"] - merged["fedavg_f1"]) / merged["fedavg_f1"]) * 100

    for mu in [0.01, 0.05, 0.1]:
        subset = merged[merged["mu"] == mu].sort_values("alpha")
        if subset.empty:
            continue

        ax2.plot(
            subset["alpha"],
            subset["degradation_pct"],
            marker="o",
            color=colors_mu.get(mu, "#333"),
            label=f"μ={mu}",
            linewidth=2,
            markersize=8,
        )

    ax2.set_xscale("log")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="FedAvg Baseline")
    ax2.set_xlabel("Dirichlet α", fontsize=12)
    ax2.set_ylabel("Performance Change (%)", fontsize=12)
    ax2.set_title("(B) Relative Degradation\nvs FedAvg", fontweight="bold", fontsize=12)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Alpha vs Mu Heatmap
    ax3 = fig.add_subplot(gs[1, 0])

    heatmap_data = summary.pivot(index="mu", columns="alpha", values="f1_mean")

    sns.heatmap(
        heatmap_data,
        ax=ax3,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.30,
        vmax=0.70,
        cbar_kws={"label": "Final Macro-F1"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax3.set_xlabel("Dirichlet α (Heterogeneity Level)", fontsize=13)
    ax3.set_ylabel("FedProx μ (Proximal Term Strength)", fontsize=13)
    ax3.set_title("(C) Performance Heatmap: α vs μ", fontweight="bold", fontsize=14)

    # Panel 4: Per-alpha comparison (bar chart)
    ax4 = fig.add_subplot(gs[1, 1])

    plot_alphas = [0.05, 0.2, 1.0]
    x_pos = np.arange(len(mu_values))
    width = 0.25

    for i, alpha in enumerate(plot_alphas):
        alpha_data = summary[np.isclose(summary["alpha"], alpha, rtol=0.01)]
        alpha_data = alpha_data.sort_values("mu")

        if not alpha_data.empty:
            ax4.bar(
                x_pos + i * width,
                alpha_data["f1_mean"],
                width,
                yerr=1.96 * alpha_data["f1_sem"],
                label=f"α={alpha}",
                capsize=3,
                alpha=0.8,
            )

    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels([f"{mu}" for mu in mu_values])
    ax4.set_xlabel("FedProx μ", fontsize=12)
    ax4.set_ylabel("Final Macro-F1", fontsize=12)
    ax4.set_title("(D) Per-α Comparison", fontweight="bold", fontsize=12)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim(0.30, 0.75)

    # Panel 5: Degradation heatmap (percentage change from FedAvg)
    ax5 = fig.add_subplot(gs[1, 2])

    degradation_matrix = merged[merged["mu"] > 0].pivot(index="mu", columns="alpha", values="degradation_pct")

    sns.heatmap(
        degradation_matrix,
        ax=ax5,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0,
        vmin=-35,
        vmax=5,
        cbar_kws={"label": "Performance Change (%)"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax5.set_xlabel("Dirichlet α (Heterogeneity Level)", fontsize=13)
    ax5.set_ylabel("FedProx μ (Proximal Term Strength)", fontsize=13)
    ax5.set_title("(E) Performance Degradation Heatmap: % Change from FedAvg Baseline", fontweight="bold", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_DIR / "obj2_fedprox_degradation_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "obj2_fedprox_degradation_analysis.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {OUTPUT_DIR / 'obj2_fedprox_degradation_analysis.pdf'}")


def plot_seed_spread(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-seed macro-F1 across alpha/mu to show variance explicitly."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"

    per_seed = (
        df.groupby(["alpha", "mu", "aggregation", "seed"])
        .agg(f1_mean=("f1", "mean"))
        .reset_index()
    )

    alpha_values = sorted([a for a in per_seed["alpha"].unique() if np.isfinite(a)])
    mu_values = sorted(per_seed["mu"].unique())

    colors_mu = {
        0.0: "#1f77b4",
        0.01: "#2ca02c",
        0.05: "#ff7f0e",
        0.1: "#d62728",
    }

    fig, axes = plt.subplots(
        1, len(alpha_values), figsize=(3.4 * len(alpha_values), 5), sharey=True
    )
    if len(alpha_values) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alpha_values):
        alpha_mask = np.isclose(per_seed["alpha"], alpha, rtol=1e-4, atol=1e-6)
        alpha_data = per_seed[alpha_mask]
        if alpha_data.empty:
            continue

        sns.stripplot(
            data=alpha_data,
            hue="mu",
            x="mu",
            y="f1_mean",
            order=mu_values,
            palette=colors_mu,
            jitter=0.15,
            size=6,
            alpha=0.65,
            ax=ax,
            dodge=False,
            legend=False,
        )

        summary = (
            alpha_data.groupby("mu")["f1_mean"]
            .agg(["mean", "sem"])
            .reindex(mu_values)
            .reset_index()
        )
        ax.errorbar(
            x=np.arange(len(mu_values)),
            y=summary["mean"],
            yerr=1.96 * summary["sem"],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            markersize=6,
        )

        ax.set_title(f"α = {alpha}", fontsize=12, fontweight="bold")
        ax.set_xlabel("FedProx μ")
        ax.set_xticks(np.arange(len(mu_values)))
        ax.set_xticklabels([str(mu) for mu in mu_values])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Final Macro-F1")
    fig.suptitle("Seed-level Macro-F1 by α and μ", fontsize=16, fontweight="bold", y=1.02)

    handles = [
        Line2D([], [], marker="o", color=colors_mu.get(mu, "#333"), linestyle="", markersize=8, label=f"μ={mu}")
        for mu in mu_values
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(mu_values), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "obj2_fedprox_seed_spread.png"
    fig.savefig(seed_path, dpi=300, bbox_inches="tight")
    fig.savefig(seed_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {seed_path}")
    print(f"Saved: {seed_path.with_suffix('.pdf')}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting FedAvg and FedProx data...")
    df = collect_fedprox_data()

    if df.empty:
        print("No data found!")
        return

    print(f"\nLoaded {len(df)} records")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Alpha values: {sorted(df['alpha'].unique())}")
    print(f"Mu values: {sorted(df['mu'].unique())}")
    print(f"Aggregations: {df['aggregation'].unique()}")

    print("\nSummary statistics:")
    summary = df.groupby(["aggregation", "alpha", "mu"])["f1"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print("\nGenerating FedProx degradation analysis plots...")
    plot_fedprox_degradation(df)
    plot_seed_spread(df, THESIS_OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
