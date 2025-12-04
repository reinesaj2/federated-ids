#!/usr/bin/env python3
"""
Objective 2 - Plot 3: Client Drift Analysis Under Heterogeneity

Shows how data heterogeneity affects model divergence between clients
using L2 dispersion metrics.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("thesis_plots_iiot")


def load_drift_data():
    """Load L2 dispersion data from all valid experiments."""
    data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = {}

        if "comp_fedavg" in run_dir.name:
            config["aggregation"] = "FedAvg"
        elif "fedprox" in run_dir.name.lower():
            config["aggregation"] = "FedProx"
        elif "krum" in run_dir.name:
            config["aggregation"] = "Krum"
        elif "bulyan" in run_dir.name:
            config["aggregation"] = "Bulyan"
        elif "median" in run_dir.name:
            config["aggregation"] = "Median"
        else:
            continue

        m = re.search(r"alpha([\d.]+|inf)", run_dir.name)
        if m:
            config["alpha"] = float("inf") if m.group(1) == "inf" else float(m.group(1))
        else:
            continue

        m = re.search(r"adv(\d+)", run_dir.name)
        if int(m.group(1)) if m else 0 != 0:
            continue  # Skip attack runs

        m = re.search(r"seed(\d+)", run_dir.name)
        config["seed"] = int(m.group(1)) if m else 0

        # Load server metrics for L2 dispersion
        metrics_file = run_dir / "metrics.csv"
        client_file = run_dir / "client_0_metrics.csv"

        if metrics_file.exists() and client_file.exists():
            try:
                # Verify 15-class
                cdf = pd.read_csv(client_file)
                if cdf["n_classes"].iloc[0] != 15:
                    continue

                config["final_f1"] = cdf["macro_f1_after"].iloc[-1]

                sdf = pd.read_csv(metrics_file)
                if "l2_dispersion_mean" in sdf.columns:
                    config["l2_final"] = sdf["l2_dispersion_mean"].iloc[-1]
                    config["l2_round1"] = sdf["l2_dispersion_mean"].iloc[0] if len(sdf) > 0 else None

                    # Round-by-round L2 data
                    for _, row in sdf.iterrows():
                        data.append({
                            **config,
                            "round": row["round"],
                            "l2": row["l2_dispersion_mean"],
                        })
            except Exception:
                pass

    return pd.DataFrame(data)


def plot_client_drift(df: pd.DataFrame, output_path: Path):
    """Generate the 4-panel client drift analysis figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Objective 2: Client Model Drift Increases with Heterogeneity",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "FedAvg": "#1f77b4",
        "FedProx": "#ff7f0e",
        "Krum": "#2ca02c",
        "Bulyan": "#d62728",
        "Median": "#9467bd",
    }

    fedavg = df[df["aggregation"] == "FedAvg"]

    # Panel A: L2 Dispersion vs Alpha (FedAvg)
    ax1 = axes[0, 0]

    # Get final round L2 for each run
    final_l2 = fedavg.groupby(["alpha", "seed"]).apply(
        lambda x: x[x["round"] == x["round"].max()]["l2"].mean()
    ).reset_index(name="l2_final")

    l2_summary = final_l2.groupby("alpha").agg(
        l2_mean=("l2_final", "mean"),
        l2_sem=("l2_final", "sem"),
        n=("l2_final", "count"),
    ).reset_index()
    l2_summary = l2_summary[l2_summary["alpha"] < 100].sort_values("alpha")

    ax1.errorbar(
        l2_summary["alpha"],
        l2_summary["l2_mean"],
        yerr=1.96 * l2_summary["l2_sem"],
        marker="o",
        color=colors["FedAvg"],
        linewidth=2,
        markersize=8,
        capsize=4,
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Dirichlet $\\alpha$ (lower = more heterogeneous)")
    ax1.set_ylabel("L2 Dispersion (Model Drift)")
    ax1.set_title("A) Client Model Drift vs Heterogeneity (FedAvg)", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add correlation annotation
    if len(l2_summary) > 2:
        corr, p = stats.spearmanr(l2_summary["alpha"], l2_summary["l2_mean"])
        ax1.annotate(
            f"Spearman r = {corr:.3f}\np = {p:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Panel B: L2 Evolution Over Rounds at Different Alpha
    ax2 = axes[0, 1]

    alpha_colors_plot = {0.02: "#1f77b4", 0.1: "#ff7f0e", 0.5: "#2ca02c", 1.0: "#d62728"}

    for alpha in [0.02, 0.1, 0.5, 1.0]:
        subset = fedavg[np.isclose(fedavg["alpha"], alpha, rtol=0.1)]
        if len(subset) == 0:
            continue

        round_summary = subset.groupby("round").agg(
            l2_mean=("l2", "mean"),
            l2_sem=("l2", "sem"),
        ).reset_index()

        ax2.plot(
            round_summary["round"],
            round_summary["l2_mean"],
            marker="o",
            color=alpha_colors_plot[alpha],
            label=f"$\\alpha$={alpha}",
            linewidth=2,
            markersize=5,
        )

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("L2 Dispersion")
    ax2.set_title("B) Model Drift Evolution Over Training", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel C: FedAvg vs FedProx L2 Comparison
    ax3 = axes[1, 0]

    fedprox = df[df["aggregation"] == "FedProx"]

    comparison_data = []
    for alpha in sorted(set(fedavg["alpha"].unique()) & set(fedprox["alpha"].unique())):
        if alpha > 10:
            continue

        fa_l2 = fedavg[np.isclose(fedavg["alpha"], alpha, rtol=0.1)]
        fp_l2 = fedprox[np.isclose(fedprox["alpha"], alpha, rtol=0.1)]

        if len(fa_l2) > 0 and len(fp_l2) > 0:
            # Get final round L2
            fa_final = fa_l2.groupby("seed").apply(
                lambda x: x[x["round"] == x["round"].max()]["l2"].mean()
            )
            fp_final = fp_l2.groupby("seed").apply(
                lambda x: x[x["round"] == x["round"].max()]["l2"].mean()
            )

            comparison_data.append({
                "alpha": alpha,
                "FedAvg_l2": fa_final.mean(),
                "FedAvg_sem": fa_final.sem(),
                "FedProx_l2": fp_final.mean(),
                "FedProx_sem": fp_final.sem(),
            })

    comp_df = pd.DataFrame(comparison_data)

    if len(comp_df) > 0:
        x = np.arange(len(comp_df))
        width = 0.35

        ax3.bar(x - width/2, comp_df["FedAvg_l2"], width,
                yerr=1.96 * comp_df["FedAvg_sem"],
                label="FedAvg", color=colors["FedAvg"], capsize=3)
        ax3.bar(x + width/2, comp_df["FedProx_l2"], width,
                yerr=1.96 * comp_df["FedProx_sem"],
                label="FedProx", color=colors["FedProx"], capsize=3)

        ax3.set_xticks(x)
        ax3.set_xticklabels([f"$\\alpha$={a}" for a in comp_df["alpha"]])
        ax3.set_ylabel("L2 Dispersion (Final Round)")
        ax3.set_title("C) Model Drift: FedAvg vs FedProx", fontweight="bold")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3, axis="y")

    # Panel D: L2 vs F1 Scatter
    ax4 = axes[1, 1]

    # Get final values for scatter
    final_data = df.groupby(["aggregation", "alpha", "seed"]).apply(
        lambda x: pd.Series({
            "l2_final": x[x["round"] == x["round"].max()]["l2"].mean(),
            "f1_final": x["final_f1"].iloc[0],
        })
    ).reset_index()

    for agg in ["FedAvg", "FedProx"]:
        subset = final_data[final_data["aggregation"] == agg]
        if len(subset) > 0:
            ax4.scatter(
                subset["l2_final"],
                subset["f1_final"],
                color=colors[agg],
                label=agg,
                alpha=0.6,
                s=50,
            )

    ax4.set_xlabel("L2 Dispersion (Model Drift)")
    ax4.set_ylabel("Final Macro F1 Score")
    ax4.set_title("D) Model Drift vs Performance", fontweight="bold")
    ax4.legend(loc="lower left")
    ax4.grid(True, alpha=0.3)

    # Add correlation
    all_l2 = final_data["l2_final"].dropna()
    all_f1 = final_data.loc[all_l2.index, "f1_final"]
    if len(all_l2) > 5:
        corr, p = stats.pearsonr(all_l2, all_f1)
        ax4.annotate(
            f"Pearson r = {corr:.3f}\np = {p:.4f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading L2 dispersion data...")
    df = load_drift_data()

    if df.empty:
        print("No valid data found!")
        return

    print(f"Loaded {len(df)} records with L2 data")

    output_path = OUTPUT_DIR / "obj2_client_drift.png"
    plot_client_drift(df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
