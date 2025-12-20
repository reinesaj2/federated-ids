#!/usr/bin/env python3
"""
Objective 2 - Plot 4: FedProx Mu Sensitivity Analysis

Shows whether the proximal term strength (mu) affects performance
under different heterogeneity levels.
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


def load_fedprox_data():
    """Load FedProx experiments with mu values."""
    data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        if "fedprox" not in run_dir.name.lower():
            continue

        config = {"aggregation": "FedProx"}

        m = re.search(r"alpha([\d.]+|inf)", run_dir.name)
        if m:
            config["alpha"] = float("inf") if m.group(1) == "inf" else float(m.group(1))
        else:
            continue

        m = re.search(r"adv(\d+)", run_dir.name)
        if int(m.group(1)) if m else 0 != 0:
            continue  # Skip attack runs

        m = re.search(r"mu([\d.]+)", run_dir.name)
        config["mu"] = float(m.group(1)) if m else 0.0

        m = re.search(r"seed(\d+)", run_dir.name)
        config["seed"] = int(m.group(1)) if m else 0

        client_file = run_dir / "client_0_metrics.csv"
        if client_file.exists():
            try:
                df = pd.read_csv(client_file)
                if df["n_classes"].iloc[0] != 15:
                    continue

                config["final_f1"] = df["macro_f1_after"].iloc[-1]
                config["rounds"] = df["round"].max()
                data.append(config)
            except Exception:
                pass

    return pd.DataFrame(data)


def load_fedavg_baseline():
    """Load FedAvg baseline for comparison."""
    data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        if "comp_fedavg" not in run_dir.name:
            continue

        config = {"aggregation": "FedAvg", "mu": 0.0}

        m = re.search(r"alpha([\d.]+|inf)", run_dir.name)
        if m:
            config["alpha"] = float("inf") if m.group(1) == "inf" else float(m.group(1))
        else:
            continue

        m = re.search(r"adv(\d+)", run_dir.name)
        if int(m.group(1)) if m else 0 != 0:
            continue

        m = re.search(r"seed(\d+)", run_dir.name)
        config["seed"] = int(m.group(1)) if m else 0

        client_file = run_dir / "client_0_metrics.csv"
        if client_file.exists():
            try:
                df = pd.read_csv(client_file)
                if df["n_classes"].iloc[0] != 15:
                    continue

                config["final_f1"] = df["macro_f1_after"].iloc[-1]
                data.append(config)
            except Exception:
                pass

    return pd.DataFrame(data)


def plot_fedprox_mu(fedprox_df: pd.DataFrame, fedavg_df: pd.DataFrame, output_path: Path):
    """Generate the 4-panel FedProx mu analysis figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Objective 2: FedProx Proximal Term (mu) Has No Benefit on IIoT Data",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: F1 vs Mu (all data)
    ax1 = axes[0, 0]

    mu_summary = (
        fedprox_df.groupby("mu")
        .agg(
            f1_mean=("final_f1", "mean"),
            f1_sem=("final_f1", "sem"),
            n=("final_f1", "count"),
        )
        .reset_index()
    )

    # Add FedAvg baseline
    fedavg_mean = fedavg_df["final_f1"].mean()
    fedavg_sem = fedavg_df["final_f1"].sem()

    ax1.axhline(fedavg_mean, color="gray", linestyle="--", alpha=0.7, label="FedAvg baseline")
    ax1.axhspan(fedavg_mean - 1.96 * fedavg_sem, fedavg_mean + 1.96 * fedavg_sem, color="gray", alpha=0.1)

    ax1.errorbar(
        mu_summary["mu"],
        mu_summary["f1_mean"],
        yerr=1.96 * mu_summary["f1_sem"],
        marker="o",
        color="#ff7f0e",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="FedProx",
    )

    ax1.set_xlabel("Proximal Term Strength ($\\mu$)")
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("A) F1 vs Proximal Term Strength", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.60, 0.80)
    ax1.grid(True, alpha=0.3)

    # Add sample sizes
    for _, row in mu_summary.iterrows():
        ax1.annotate(f"n={int(row['n'])}", xy=(row["mu"], row["f1_mean"] + 0.02), ha="center", fontsize=8, rotation=45)

    # Panel B: Mu effect at different alpha levels
    ax2 = axes[0, 1]

    # Heatmap of F1 by mu and alpha
    pivot_data = fedprox_df.groupby(["mu", "alpha"])["final_f1"].mean().unstack()

    # Filter to reasonable alpha values
    alpha_cols = [c for c in pivot_data.columns if c < 10]
    if len(alpha_cols) > 0:
        pivot_filtered = pivot_data[alpha_cols]

        sns.heatmap(
            pivot_filtered,
            ax=ax2,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.60,
            vmax=0.75,
            cbar_kws={"label": "F1 Score"},
        )
        ax2.set_xlabel("Dirichlet $\\alpha$")
        ax2.set_ylabel("Proximal Term ($\\mu$)")
        ax2.set_title("B) F1 Heatmap: $\\mu$ vs $\\alpha$", fontweight="bold")

    # Panel C: Best mu per alpha level
    ax3 = axes[1, 0]

    best_mu_data = []
    for alpha in sorted(fedprox_df["alpha"].unique()):
        if alpha > 10:
            continue
        subset = fedprox_df[np.isclose(fedprox_df["alpha"], alpha, rtol=0.1)]
        if len(subset) > 0:
            mu_perf = subset.groupby("mu")["final_f1"].mean()
            best_mu = mu_perf.idxmax()
            best_f1 = mu_perf.max()

            # Compare to mu=0 (essentially FedAvg behavior)
            mu0_f1 = mu_perf.get(0.0, np.nan)

            best_mu_data.append(
                {
                    "alpha": alpha,
                    "best_mu": best_mu,
                    "best_f1": best_f1,
                    "mu0_f1": mu0_f1,
                    "gain": (best_f1 - mu0_f1) * 100 if not np.isnan(mu0_f1) else 0,
                }
            )

    best_df = pd.DataFrame(best_mu_data)

    if len(best_df) > 0:
        ax3.bar(range(len(best_df)), best_df["best_mu"], color="#ff7f0e")
        ax3.set_xticks(range(len(best_df)))
        ax3.set_xticklabels([f"$\\alpha$={a}" for a in best_df["alpha"]], rotation=45, ha="right")
        ax3.set_ylabel("Optimal $\\mu$ Value")
        ax3.set_title("C) Optimal $\\mu$ by Heterogeneity Level", fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        # Add gain annotation
        avg_gain = best_df["gain"].mean()
        ax3.annotate(
            f"Avg gain over $\\mu$=0:\n{avg_gain:+.2f}%",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Panel D: Statistical Summary
    ax4 = axes[1, 1]

    summary_text = "FEDPROX MU ANALYSIS SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"

    summary_text += f"Total FedProx runs: {len(fedprox_df)}\n"
    summary_text += f"Mu values tested: {sorted(fedprox_df['mu'].unique())}\n\n"

    summary_text += "Performance by Mu:\n"
    for _, row in mu_summary.iterrows():
        summary_text += f"  $\\mu$={row['mu']:.3f}: F1={row['f1_mean']:.4f} (n={int(row['n'])})\n"

    summary_text += f"\nFedAvg baseline: F1={fedavg_mean:.4f}\n\n"

    # ANOVA test
    groups = [
        fedprox_df[fedprox_df["mu"] == mu]["final_f1"].values
        for mu in fedprox_df["mu"].unique()
        if len(fedprox_df[fedprox_df["mu"] == mu]) > 2
    ]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        summary_text += "ANOVA across mu values:\n"
        summary_text += f"  F = {f_stat:.2f}, p = {p_val:.4f}\n"
        if p_val > 0.05:
            summary_text += "  Result: NO significant effect of mu\n"
        else:
            summary_text += "  Result: Significant effect detected\n"

    summary_text += "\n" + "=" * 40 + "\n"
    summary_text += "CONCLUSION:\n"
    summary_text += "Proximal term strength has\n"
    summary_text += "minimal impact on IIoT IDS\n"
    summary_text += "performance across all tested\n"
    summary_text += "heterogeneity levels."

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax4.axis("off")
    ax4.set_title("D) Statistical Summary", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FedProx data...")
    fedprox_df = load_fedprox_data()
    fedavg_df = load_fedavg_baseline()

    if fedprox_df.empty:
        print("No FedProx data found!")
        return

    print(f"Loaded {len(fedprox_df)} FedProx runs with {fedprox_df['mu'].nunique()} mu values")
    print(f"Loaded {len(fedavg_df)} FedAvg baseline runs")

    output_path = OUTPUT_DIR / "obj2_fedprox_mu_analysis.png"
    plot_fedprox_mu(fedprox_df, fedavg_df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
