#!/usr/bin/env python3
"""
Generate Personalization Gains Visualization for IIoT Data

Adapted for Edge-IIoTset experiment naming convention:
dsedge-iiotset-nightly_comp_fedavg_alpha0.5_adv0_dp0_pers3_mu0.0_seed42
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plot_metrics_utils import compute_confidence_interval  # noqa: E402

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 14


def parse_iiot_experiment_name(exp_dir: Path) -> dict:
    """
    Parse IIoT experiment directory name.

    Example: 'dsedge-iiotset-nightly_comp_fedavg_alpha0.5_adv0_dp0_pers3_mu0.0_seed42'
    """
    name = exp_dir.name
    config = {}

    # Extract dataset
    if "edge-iiotset" in name:
        config["dataset"] = "edge-iiotset"
    else:
        config["dataset"] = "unknown"

    # Extract aggregation
    if "comp_fedavg" in name:
        config["aggregation"] = "fedavg"
    elif "comp_krum" in name:
        config["aggregation"] = "krum"
    elif "comp_bulyan" in name:
        config["aggregation"] = "bulyan"
    elif "comp_median" in name:
        config["aggregation"] = "median"
    else:
        config["aggregation"] = "unknown"

    # Extract alpha
    alpha_match = re.search(r"alpha([0-9.]+)", name)
    if alpha_match:
        config["alpha"] = float(alpha_match.group(1))

    # Extract adversary fraction
    adv_match = re.search(r"adv(\d+)", name)
    if adv_match:
        config["adv_pct"] = int(adv_match.group(1))

    # Extract personalization epochs
    pers_match = re.search(r"pers(\d+)", name)
    if pers_match:
        config["pers_epochs"] = int(pers_match.group(1))
    else:
        config["pers_epochs"] = 0

    # Extract seed
    seed_match = re.search(r"seed(\d+)", name)
    if seed_match:
        config["seed"] = int(seed_match.group(1))

    return config


def load_personalization_data(runs_dir: Path) -> pd.DataFrame:
    """Load all personalization metrics from IIoT experiment runs."""
    all_data = []

    # Look for personalization experiments (pers > 0)
    for exp_dir in runs_dir.glob("*pers[1-9]*"):
        if not exp_dir.is_dir():
            continue

        config = parse_iiot_experiment_name(exp_dir)
        if not config or config.get("pers_epochs", 0) == 0:
            continue

        # Load all client metrics
        for client_csv in exp_dir.glob("client_*_metrics.csv"):
            try:
                client_id = int(re.search(r"client_(\d+)", client_csv.name).group(1))
            except:
                continue

            try:
                df = pd.read_csv(client_csv)
                if df.empty:
                    continue

                # Take last round
                last_row = df.iloc[-1]

                # Extract personalization metrics
                global_f1 = last_row.get("macro_f1_global", None)
                pers_f1 = last_row.get("macro_f1_personalized", None)

                # Try alternative column names
                if pd.isna(global_f1):
                    global_f1 = last_row.get("macro_f1_before", None)
                if pd.isna(pers_f1):
                    pers_f1 = last_row.get("macro_f1_after", None)

                if pd.isna(global_f1) or pd.isna(pers_f1):
                    continue

                global_f1 = float(global_f1)
                pers_f1 = float(pers_f1)
                gain = pers_f1 - global_f1

                all_data.append(
                    {
                        "dataset": config.get("dataset", "unknown"),
                        "aggregation": config.get("aggregation", "unknown"),
                        "alpha": config.get("alpha", 1.0),
                        "adv_pct": config.get("adv_pct", 0),
                        "pers_epochs": config.get("pers_epochs", 0),
                        "seed": config.get("seed", 42),
                        "client_id": client_id,
                        "global_f1": global_f1,
                        "personalized_f1": pers_f1,
                        "personalization_gain": gain,
                        "exp_dir": exp_dir.name,
                    }
                )
            except Exception as e:
                print(f"Warning: Error processing {client_csv}: {e}")
                continue

    if not all_data:
        return pd.DataFrame()

    return pd.DataFrame(all_data)


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for personalization gains."""
    summary = {}

    # Overall statistics
    summary["overall"] = {
        "mean_gain": float(df["personalization_gain"].mean()),
        "median_gain": float(df["personalization_gain"].median()),
        "std_gain": float(df["personalization_gain"].std()),
        "max_gain": float(df["personalization_gain"].max()),
        "min_gain": float(df["personalization_gain"].min()),
        "pct_positive_gains": float((df["personalization_gain"] > 0.01).sum() / len(df) * 100),
        "pct_negative_gains": float((df["personalization_gain"] < -0.01).sum() / len(df) * 100),
    }

    # By adversary percentage
    summary["by_adv_pct"] = {}
    for adv in sorted(df["adv_pct"].unique()):
        adv_df = df[df["adv_pct"] == adv]
        summary["by_adv_pct"][int(adv)] = {
            "mean_gain": float(adv_df["personalization_gain"].mean()),
            "n_clients": int(len(adv_df)),
        }

    # By alpha (heterogeneity)
    summary["by_alpha"] = {}
    for alpha in sorted(df["alpha"].unique()):
        alpha_df = df[df["alpha"] == alpha]
        summary["by_alpha"][float(alpha)] = {
            "mean_gain": float(alpha_df["personalization_gain"].mean()),
            "n_clients": int(len(alpha_df)),
        }

    # By personalization epochs
    summary["by_epochs"] = {}
    for epochs in sorted(df["pers_epochs"].unique()):
        epochs_df = df[df["pers_epochs"] == epochs]
        summary["by_epochs"][int(epochs)] = {
            "mean_gain": float(epochs_df["personalization_gain"].mean()),
            "n_clients": int(len(epochs_df)),
        }

    return summary


def plot_personalization_gains(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive personalization gains visualization."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "IIoT Personalization Gains Analysis - Benign Setting (Adv=0%)\n" "Thesis Objective 3",
        fontsize=15,
        fontweight="bold",
    )

    # Plot 1: Gains by configuration (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_gains_by_config(df, ax1)

    # Plot 2: Gains vs adversary percentage (middle-left)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_gains_vs_adversary(df, ax2)

    # Plot 3: Gains by epochs (middle-center)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_gains_by_epochs(df, ax3)

    # Plot 4: Gains vs heterogeneity (middle-right)
    ax4 = fig.add_subplot(gs[1, 2])
    plot_gains_vs_alpha(df, ax4)

    # Plot 5: Per-client scatter (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    plot_per_client_scatter(df, ax5)

    # Plot 6: Global vs Personalized comparison (bottom-right, wider)
    ax6 = fig.add_subplot(gs[2, 1:])
    plot_global_vs_personalized(df, ax6)

    plt.savefig(output_dir / "personalization_gains_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_dir / 'personalization_gains_analysis.png'}")


def plot_gains_by_config(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Bar chart showing mean gain by configuration with 95% CIs."""
    # Group by configuration
    config_groups = df.groupby(["alpha", "adv_pct", "pers_epochs"])["personalization_gain"].apply(list).reset_index()
    config_groups.columns = ["alpha", "adv_pct", "pers_epochs", "values"]

    # Compute CI for each configuration
    ci_data = []
    for _, row in config_groups.iterrows():
        values = np.array([v for v in row["values"] if not np.isnan(v)])
        if len(values) == 0:
            continue
        mean, ci_lower, ci_upper = compute_confidence_interval(values, confidence=0.95)
        ci_data.append(
            {
                "alpha": row["alpha"],
                "adv_pct": row["adv_pct"],
                "pers_epochs": row["pers_epochs"],
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(values),
            }
        )

    config_df = pd.DataFrame(ci_data)

    # Create labels
    config_df["config"] = config_df.apply(lambda r: f"α={r['alpha']:.1f}\nAdv={r['adv_pct']}%\nEp={r['pers_epochs']}", axis=1)

    # Sort by epochs (to group similar configs), then by mean gain
    config_df = config_df.sort_values(["pers_epochs", "mean"], ascending=[True, False])

    # Compute error bars
    yerr_lower = config_df["mean"] - config_df["ci_lower"]
    yerr_upper = config_df["ci_upper"] - config_df["mean"]

    # Plot
    bars = ax.bar(
        range(len(config_df)),
        config_df["mean"],
        yerr=[yerr_lower, yerr_upper],
        capsize=5,
        alpha=0.7,
    )

    # Color bars based on gain sign
    for bar, gain in zip(bars, config_df["mean"]):
        if gain > 0.01:
            bar.set_color("green")
            bar.set_alpha(0.7)
        elif gain < -0.01:
            bar.set_color("red")
            bar.set_alpha(0.7)
        else:
            bar.set_color("gray")
            bar.set_alpha(0.3)

    ax.set_xticks(range(len(config_df)))
    ax.set_xticklabels(config_df["config"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Personalization Gain (F1)")
    ax.set_title("Personalization Gains by Configuration (95% CI)")

    # Add n annotations
    for i, (_, row) in enumerate(config_df.iterrows()):
        y_pos = row["mean"] + yerr_upper.iloc[i] + 0.005
        ax.text(i, y_pos, f"n={row['n']}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    # Add warning about attack conditions
    warning_text = (
        "NOTE: Positive gains shown here are for BENIGN conditions (Adv=0%).\n"
        "Under attack (Adv=30%), personalization can show NEGATIVE gains.\n"
        "See attack resilience analysis for adversarial performance."
    )
    ax.text(
        0.98,
        0.02,
        warning_text,
        transform=ax.transAxes,
        fontsize=8,
        va='bottom',
        ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='orange', linewidth=2),
    )


def plot_gains_vs_adversary(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Box plot: personalization gain vs adversary percentage."""
    sns.boxplot(data=df, x="adv_pct", y="personalization_gain", ax=ax)
    ax.set_xlabel("Adversary Percentage (%)")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain vs Attack Intensity")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(axis="y", alpha=0.3)


def plot_gains_by_epochs(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Box plot: gain distribution by personalization epochs."""
    sns.boxplot(data=df, x="pers_epochs", y="personalization_gain", ax=ax)
    ax.set_xlabel("Personalization Epochs")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain by Training Epochs")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(axis="y", alpha=0.3)


def plot_gains_vs_alpha(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter plot: personalization gain vs alpha (heterogeneity)."""
    # Scatter with color by adversary percentage
    scatter = ax.scatter(
        df["alpha"],
        df["personalization_gain"],
        c=df["adv_pct"],
        cmap="RdYlGn_r",
        alpha=0.6,
        s=60,
    )
    plt.colorbar(scatter, ax=ax, label="Adv %")

    ax.set_xlabel("Dirichlet α (Heterogeneity)")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain vs Data Heterogeneity")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)


def plot_per_client_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter plot: global F1 vs personalization gain."""
    # Color by adversary percentage
    scatter = ax.scatter(
        df["global_f1"],
        df["personalization_gain"],
        c=df["adv_pct"],
        cmap="RdYlGn_r",
        alpha=0.6,
        s=60,
    )
    plt.colorbar(scatter, ax=ax, label="Adv %")

    ax.set_xlabel("Global Model F1")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain vs Global Performance")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)


def plot_global_vs_personalized(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Paired comparison: global vs personalized F1 scores."""
    # Sample subset if too many data points for readability
    if len(df) > 40:
        df_plot = df.sample(n=40, random_state=42).sort_values("personalization_gain")
    else:
        df_plot = df.sort_values("personalization_gain")

    df_plot = df_plot.copy()
    df_plot["client_label"] = df_plot["exp_dir"].str[:20] + "_c" + df_plot["client_id"].astype(str)

    x_pos = np.arange(len(df_plot))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        df_plot["global_f1"],
        width,
        label="Global Model",
        alpha=0.7,
        color="steelblue",
    )
    ax.bar(
        x_pos + width / 2,
        df_plot["personalized_f1"],
        width,
        label="Personalized Model",
        alpha=0.7,
        color="forestgreen",
    )

    ax.set_xticks(x_pos[:: max(1, len(df_plot) // 20)])
    ax.set_xticklabels(
        df_plot["client_label"].iloc[:: max(1, len(df_plot) // 20)],
        rotation=90,
        ha="right",
        fontsize=6,
    )
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"Global vs Personalized Performance (n={len(df_plot)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IIoT personalization gains visualization")
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing IIoT experiment runs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("thesis_plots_iiot"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading IIoT personalization data from {args.runs_dir}...")
    df = load_personalization_data(args.runs_dir)

    if df.empty:
        print("ERROR: No personalization data found!")
        print(f"Looked in: {args.runs_dir}")
        print("Expected directories matching: *pers[1-9]*")
        return

    n_experiments = df["exp_dir"].nunique()
    n_seeds = df["seed"].nunique()
    print(f"Loaded {len(df)} client metrics from {n_experiments} experiments across {n_seeds} seeds")

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary_stats(df)

    # Save summary
    summary_file = args.output_dir / "personalization_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to {summary_file}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"Overall mean gain: {summary['overall']['mean_gain']:.4f}")
    print(f"Overall median gain: {summary['overall']['median_gain']:.4f}")
    print(f"Clients with positive gains (>0.01): {summary['overall']['pct_positive_gains']:.1f}%")
    print(f"Clients with negative gains (<-0.01): {summary['overall']['pct_negative_gains']:.1f}%")
    print(f"Max gain observed: {summary['overall']['max_gain']:.4f}")
    print(f"Min gain observed: {summary['overall']['min_gain']:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_personalization_gains(df, args.output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
