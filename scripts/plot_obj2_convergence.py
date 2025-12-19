#!/usr/bin/env python3
"""
Objective 2 - Plot 2: Convergence Analysis Under Heterogeneity

Shows how data heterogeneity affects convergence speed and learning curves.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("thesis_plots_iiot")


def load_round_data():
    """Load round-by-round data from all valid experiments."""
    all_data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = {}

        if "comp_fedavg" in run_dir.name:
            config["aggregation"] = "FedAvg"
        elif "fedprox" in run_dir.name.lower():
            config["aggregation"] = "FedProx"
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

        # Load all client metrics for this run
        client_files = list(run_dir.glob("client_*_metrics.csv"))
        if not client_files:
            continue

        try:
            # Use client 0 as representative
            df = pd.read_csv(client_files[0])
            if df["n_classes"].iloc[0] != 15:
                continue

            for _, row in df.iterrows():
                all_data.append({
                    **config,
                    "round": row["round"],
                    "f1": row["macro_f1_after"],
                })
        except Exception:
            pass

    return pd.DataFrame(all_data)


def plot_convergence_analysis(df: pd.DataFrame, output_path: Path):
    """Generate the 4-panel convergence analysis figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Objective 2: Convergence Speed Is Unaffected by Data Heterogeneity",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    alpha_colors = {
        0.02: "#1f77b4",
        0.1: "#ff7f0e",
        0.5: "#2ca02c",
        1.0: "#d62728",
    }

    fedavg = df[df["aggregation"] == "FedAvg"]

    # Panel A: FedAvg Learning Curves at Different Alpha
    ax1 = axes[0, 0]

    for alpha in [0.02, 0.1, 0.5, 1.0]:
        subset = fedavg[np.isclose(fedavg["alpha"], alpha, rtol=0.1)]
        if len(subset) == 0:
            continue

        round_summary = subset.groupby("round").agg(
            f1_mean=("f1", "mean"),
            f1_sem=("f1", "sem"),
        ).reset_index()

        ax1.plot(
            round_summary["round"],
            round_summary["f1_mean"],
            marker="o",
            color=alpha_colors[alpha],
            label=f"$\\alpha$={alpha}",
            linewidth=2,
            markersize=5,
        )
        ax1.fill_between(
            round_summary["round"],
            round_summary["f1_mean"] - 1.96 * round_summary["f1_sem"],
            round_summary["f1_mean"] + 1.96 * round_summary["f1_sem"],
            color=alpha_colors[alpha],
            alpha=0.2,
        )

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("A) FedAvg Learning Curves by Heterogeneity Level", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 0.80)
    ax1.grid(True, alpha=0.3)

    # Panel B: Rounds to 90% of Final F1
    ax2 = axes[0, 1]

    def rounds_to_threshold(group, threshold=0.9):
        final_f1 = group["f1"].max()
        target = final_f1 * threshold
        for rnd in sorted(group["round"].unique()):
            if group[group["round"] == rnd]["f1"].mean() >= target:
                return rnd
        return group["round"].max()

    convergence_data = []
    for (alpha, seed), group in fedavg.groupby(["alpha", "seed"]):
        if alpha > 10:
            continue
        r90 = rounds_to_threshold(group, 0.9)
        convergence_data.append({"alpha": alpha, "seed": seed, "rounds_to_90": r90})

    conv_df = pd.DataFrame(convergence_data)
    conv_summary = conv_df.groupby("alpha").agg(
        r90_mean=("rounds_to_90", "mean"),
        r90_sem=("rounds_to_90", "sem"),
        n=("rounds_to_90", "count"),
    ).reset_index()

    ax2.errorbar(
        conv_summary["alpha"],
        conv_summary["r90_mean"],
        yerr=1.96 * conv_summary["r90_sem"],
        marker="o",
        color="#1f77b4",
        linewidth=2,
        markersize=8,
        capsize=4,
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("Dirichlet $\\alpha$")
    ax2.set_ylabel("Rounds to 90% of Final F1")
    ax2.set_title("B) Convergence Speed vs Heterogeneity", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add trend annotation
    if len(conv_summary) > 2:
        r90_range = conv_summary["r90_mean"].max() - conv_summary["r90_mean"].min()
        ax2.annotate(
            f"Range: {r90_range:.1f} rounds\n(minimal variation)",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Panel C: Early vs Late Performance Gap
    ax3 = axes[1, 0]

    early_late_data = []
    for alpha in sorted(fedavg["alpha"].unique()):
        if alpha > 10:
            continue
        subset = fedavg[np.isclose(fedavg["alpha"], alpha, rtol=0.1)]
        if len(subset) == 0:
            continue

        # Early = round 3, Late = final round
        early = subset[subset["round"] == 3]["f1"].mean() if 3 in subset["round"].values else np.nan
        late = subset[subset["round"] == subset["round"].max()]["f1"].mean()

        if not np.isnan(early):
            early_late_data.append({
                "alpha": alpha,
                "early_f1": early,
                "late_f1": late,
                "gap": late - early,
            })

    el_df = pd.DataFrame(early_late_data)

    if len(el_df) > 0:
        x = np.arange(len(el_df))
        width = 0.35

        ax3.bar(x - width/2, el_df["early_f1"], width, label="Round 3", color="#1f77b4")
        ax3.bar(x + width/2, el_df["late_f1"], width, label="Final Round", color="#2ca02c")

        ax3.set_xticks(x)
        ax3.set_xticklabels([f"$\\alpha$={a}" for a in el_df["alpha"]])
        ax3.set_ylabel("Macro F1 Score")
        ax3.set_title("C) Early vs Final Performance by Heterogeneity", fontweight="bold")
        ax3.legend(loc="lower right")
        ax3.set_ylim(0, 0.80)
        ax3.grid(True, alpha=0.3, axis="y")

    # Panel D: FedAvg vs FedProx Convergence at Extreme Non-IID
    ax4 = axes[1, 1]

    extreme_alpha = 0.02
    fedavg_extreme = fedavg[np.isclose(fedavg["alpha"], extreme_alpha, rtol=0.1)]
    fedprox_extreme = df[(df["aggregation"] == "FedProx") & 
                         np.isclose(df["alpha"], extreme_alpha, rtol=0.1)]

    for agg_df, label, color in [(fedavg_extreme, "FedAvg", "#1f77b4"), 
                                  (fedprox_extreme, "FedProx", "#ff7f0e")]:
        if len(agg_df) == 0:
            continue

        round_summary = agg_df.groupby("round").agg(
            f1_mean=("f1", "mean"),
            f1_sem=("f1", "sem"),
        ).reset_index()

        ax4.plot(
            round_summary["round"],
            round_summary["f1_mean"],
            marker="o",
            color=color,
            label=label,
            linewidth=2,
            markersize=5,
        )
        ax4.fill_between(
            round_summary["round"],
            round_summary["f1_mean"] - 1.96 * round_summary["f1_sem"],
            round_summary["f1_mean"] + 1.96 * round_summary["f1_sem"],
            color=color,
            alpha=0.2,
        )

    ax4.set_xlabel("Communication Round")
    ax4.set_ylabel("Macro F1 Score")
    ax4.set_title(f"D) FedAvg vs FedProx at Extreme Non-IID ($\\alpha$={extreme_alpha})", fontweight="bold")
    ax4.legend(loc="lower right")
    ax4.set_ylim(0, 0.80)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading round-by-round data...")
    df = load_round_data()

    if df.empty:
        print("No valid data found!")
        return

    print(f"Loaded {len(df)} round records from {df['seed'].nunique()} seeds")

    output_path = OUTPUT_DIR / "obj2_convergence_analysis.png"
    plot_convergence_analysis(df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
