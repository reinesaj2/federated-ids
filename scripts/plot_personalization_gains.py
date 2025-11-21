#!/usr/bin/env python3
"""
Generate Personalization Gains Visualization

Creates publication-ready plots showing personalization benefits across:
- Different datasets (UNSW-NB15, CIC-IDS2017)
- Heterogeneity levels (Dirichlet alpha)
- Personalization epochs
- Learning rates

Outputs:
- Multi-panel figure with bar charts, scatter plots, box plots
- LaTeX table for thesis inclusion
- Summary statistics JSON
"""

import argparse
import csv
import json
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


def parse_experiment_name(exp_dir: Path) -> dict[str, any]:
    """
    Parse experiment directory name into components.

    Example: 'unsw_alpha0p1_pers5_lr0p01' ->
        {'dataset': 'unsw', 'alpha': 0.1, 'pers_epochs': 5, 'lr': 0.01}
    """
    parts = exp_dir.name.split("_")
    config = {}

    for i, part in enumerate(parts):
        if part.startswith("alpha"):
            config["alpha"] = float(part.replace("alpha", "").replace("p", "."))
        elif part.startswith("pers"):
            config["pers_epochs"] = int(part.replace("pers", ""))
        elif part.startswith("lr"):
            config["lr"] = float(part.replace("lr", "").replace("p", "."))
        elif i == 0:  # First part is dataset name
            config["dataset"] = part

    return config


def load_personalization_data(logs_dir: Path) -> pd.DataFrame:
    """Load all personalization metrics from experiment directories."""
    all_data = []

    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        config = parse_experiment_name(exp_dir)
        if not config:
            continue

        # Load all client metrics
        for client_csv in exp_dir.glob("client_*_metrics.csv"):
            client_id = int(client_csv.stem.split("_")[1])

            with open(client_csv, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue

                # Take last round
                last_row = rows[-1]

                # Extract personalization metrics
                try:
                    global_f1 = float(last_row.get("macro_f1_global", 0) or 0)
                    pers_f1 = float(last_row.get("macro_f1_personalized", 0) or 0)
                    gain = float(last_row.get("personalization_gain", 0) or 0)

                    # Fallback for baseline (personalization_epochs=0)
                    if global_f1 == 0 and "macro_f1_after" in last_row:
                        global_f1 = float(last_row["macro_f1_after"])
                        pers_f1 = global_f1
                        gain = 0.0

                    all_data.append(
                        {
                            "dataset": config.get("dataset", "unknown"),
                            "alpha": config.get("alpha", 1.0),
                            "pers_epochs": config.get("pers_epochs", 0),
                            "lr": config.get("lr", 0.01),
                            "client_id": client_id,
                            "global_f1": global_f1,
                            "personalized_f1": pers_f1,
                            "personalization_gain": gain,
                            "exp_dir": exp_dir.name,
                        }
                    )
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping {client_csv} due to error: {e}")
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
    }

    # By dataset
    summary["by_dataset"] = {}
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        summary["by_dataset"][dataset] = {
            "mean_gain": float(dataset_df["personalization_gain"].mean()),
            "n_clients": int(len(dataset_df)),
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
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "Personalization Gains Analysis (Thesis Objective 3)",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Gains by experiment configuration (top-left, wider)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_gains_by_config(df, ax1)

    # Plot 2: Gains vs heterogeneity (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_gains_vs_alpha(df, ax2)

    # Plot 3: Gains by personalization epochs (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_gains_by_epochs(df, ax3)

    # Plot 4: Gains by dataset (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_gains_by_dataset(df, ax4)

    # Plot 5: Per-client scatter (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_per_client_scatter(df, ax5)

    # Plot 6: Global vs Personalized F1 comparison (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    plot_global_vs_personalized(df, ax6)

    plt.savefig(output_dir / "personalization_gains_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_dir / 'personalization_gains_analysis.png'}")


def plot_gains_by_config(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Bar chart showing mean gain by experiment configuration with 95% CIs."""
    # Aggregate by configuration - collect raw values for CI computation
    config_groups = df.groupby(["dataset", "alpha", "pers_epochs"])["personalization_gain"].apply(list).reset_index()
    config_groups.columns = ["dataset", "alpha", "pers_epochs", "values"]

    # Compute CI for each configuration
    ci_data = []
    for _, row in config_groups.iterrows():
        values = np.array(row["values"])
        # Remove NaN values before CI computation
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue  # Skip configurations with no valid data
        mean, ci_lower, ci_upper = compute_confidence_interval(values, confidence=0.95)
        ci_data.append(
            {
                "dataset": row["dataset"],
                "alpha": row["alpha"],
                "pers_epochs": row["pers_epochs"],
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(values),
            }
        )

    config_df = pd.DataFrame(ci_data)

    # Create configuration labels
    config_df["config"] = config_df.apply(lambda r: f"{r['dataset']}\nα={r['alpha']}\nep={r['pers_epochs']}", axis=1)

    # Sort by mean gain
    config_df = config_df.sort_values("mean", ascending=False)

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
        color=sns.color_palette("viridis", len(config_df)),
    )

    # Color bars: green for positive, red for near-zero
    for _, (bar, gain) in enumerate(zip(bars, config_df["mean"], strict=False)):
        if gain > 0.01:
            bar.set_color("green")
            bar.set_alpha(0.7)
        elif gain < 0.01:
            bar.set_color("red")
            bar.set_alpha(0.3)

    ax.set_xticks(range(len(config_df)))
    ax.set_xticklabels(config_df["config"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Personalization Gain (F1)")
    ax.set_title("Personalization Gains by Configuration (95% CI)")

    # Add n annotations
    for i, row in config_df.iterrows():
        y_pos = row["mean"] + yerr_upper.iloc[i] + 0.005
        ax.text(i, y_pos, f"n={row['n']}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=0.01, color="orange", linestyle="--", linewidth=1, label="Threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot_gains_vs_alpha(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter plot: personalization gain vs alpha (heterogeneity)."""
    # Filter to experiments with personalization enabled
    pers_df = df[df["pers_epochs"] > 0]

    if len(pers_df) == 0:
        ax.text(0.5, 0.5, "No personalization data", ha="center", va="center", fontsize=12)
        return

    # Scatter with jitter
    for dataset in pers_df["dataset"].unique():
        dataset_df = pers_df[pers_df["dataset"] == dataset]
        # Add jitter to alpha for visibility
        alpha_jitter = dataset_df["alpha"] + np.random.normal(0, 0.005, len(dataset_df))
        ax.scatter(
            alpha_jitter,
            dataset_df["personalization_gain"],
            alpha=0.6,
            s=60,
            label=dataset.upper(),
        )

    # Add trend line
    if len(pers_df) > 3:
        z = np.polyfit(pers_df["alpha"], pers_df["personalization_gain"], 1)
        p = np.poly1d(z)
        alpha_range = np.linspace(pers_df["alpha"].min(), pers_df["alpha"].max(), 100)
        ax.plot(alpha_range, p(alpha_range), "r--", linewidth=2, alpha=0.5, label="Trend")

    ax.set_xlabel("Dirichlet α (Heterogeneity)")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain vs Data Heterogeneity")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_gains_by_epochs(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Box plot: gain distribution by personalization epochs."""
    # Filter to experiments with personalization enabled
    pers_df = df[df["pers_epochs"] > 0]

    if len(pers_df) == 0:
        ax.text(0.5, 0.5, "No personalization data", ha="center", va="center", fontsize=12)
        return

    # Check for sparse data in pers_epochs groups
    min_recommended_samples = 3
    sparse_groups = []
    for epochs in pers_df["pers_epochs"].unique():
        count = len(pers_df[pers_df["pers_epochs"] == epochs])
        if count < min_recommended_samples:
            sparse_groups.append(f"epochs={int(epochs)}(n={count})")

    if sparse_groups:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Sparse data in personalization gains by epochs: {', '.join(sparse_groups)}. "
            f"Epochs with n<{min_recommended_samples} may show unreliable statistics."
        )

    sns.boxplot(data=pers_df, x="pers_epochs", y="personalization_gain", ax=ax)
    ax.set_xlabel("Personalization Epochs")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain by Training Epochs")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    # Add sample size annotations
    for i, epochs in enumerate(sorted(pers_df["pers_epochs"].unique())):
        epoch_data = pers_df[pers_df["pers_epochs"] == epochs]["personalization_gain"]
        n = len(epoch_data)
        y_max = epoch_data.max() if n > 0 else 0
        ax.text(i, y_max * 1.05, f'n={n}', ha='center', va='bottom', fontsize=8)


def plot_gains_by_dataset(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Violin plot: gain distribution by dataset."""
    # Filter to experiments with personalization enabled
    pers_df = df[df["pers_epochs"] > 0]

    if len(pers_df) == 0:
        ax.text(0.5, 0.5, "No personalization data", ha="center", va="center", fontsize=12)
        return

    # Check for sparse data in dataset groups
    min_recommended_samples = 3
    sparse_groups = []
    for dataset in pers_df["dataset"].unique():
        count = len(pers_df[pers_df["dataset"] == dataset])
        if count < min_recommended_samples:
            sparse_groups.append(f"{dataset}(n={count})")

    if sparse_groups:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Sparse data in personalization gains by dataset: {', '.join(sparse_groups)}. "
            f"Datasets with n<{min_recommended_samples} may show unreliable statistics."
        )

    sns.violinplot(data=pers_df, x="dataset", y="personalization_gain", ax=ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain by Dataset")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    # Add sample size annotations
    for i, dataset in enumerate(pers_df["dataset"].unique()):
        dataset_data = pers_df[pers_df["dataset"] == dataset]["personalization_gain"]
        n = len(dataset_data)
        y_max = dataset_data.max() if n > 0 else 0
        ax.text(i, y_max * 1.05, f'n={n}', ha='center', va='bottom', fontsize=8)


def plot_per_client_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter plot: global F1 vs personalization gain."""
    # Filter to experiments with personalization enabled
    pers_df = df[df["pers_epochs"] > 0]

    if len(pers_df) == 0:
        ax.text(0.5, 0.5, "No personalization data", ha="center", va="center", fontsize=12)
        return

    # Color by alpha
    scatter = ax.scatter(
        pers_df["global_f1"],
        pers_df["personalization_gain"],
        c=pers_df["alpha"],
        cmap="viridis",
        alpha=0.6,
        s=60,
    )
    plt.colorbar(scatter, ax=ax, label="α (heterogeneity)")

    ax.set_xlabel("Global Model F1")
    ax.set_ylabel("Personalization Gain")
    ax.set_title("Gain vs Global Performance")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)


def plot_global_vs_personalized(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Paired comparison: global vs personalized F1 scores."""
    # Filter to experiments with personalization enabled
    pers_df = df[df["pers_epochs"] > 0]

    if len(pers_df) == 0:
        ax.text(0.5, 0.5, "No personalization data", ha="center", va="center", fontsize=12)
        return

    # Create client labels
    pers_df = pers_df.copy()
    pers_df["client_label"] = pers_df["exp_dir"] + "_c" + pers_df["client_id"].astype(str)

    # Sort by gain
    pers_df = pers_df.sort_values("personalization_gain")

    x_pos = np.arange(len(pers_df))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        pers_df["global_f1"],
        width,
        label="Global Model",
        alpha=0.7,
        color="steelblue",
    )
    ax.bar(
        x_pos + width / 2,
        pers_df["personalized_f1"],
        width,
        label="Personalized Model",
        alpha=0.7,
        color="forestgreen",
    )

    # Draw arrows for significant improvements
    for i, row in enumerate(pers_df.itertuples()):
        if row.personalization_gain > 0.01:
            ax.annotate(
                "",
                xy=(i + width / 2, row.personalized_f1),
                xytext=(i - width / 2, row.global_f1),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.5),
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pers_df["client_label"], rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Global vs Personalized Model Performance (sorted by gain)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def generate_latex_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate LaTeX table for thesis inclusion."""
    # Aggregate by configuration
    config_df = (
        df.groupby(["dataset", "alpha", "pers_epochs"])
        .agg(
            {
                "global_f1": "mean",
                "personalized_f1": "mean",
                "personalization_gain": ["mean", "std"],
            }
        )
        .reset_index()
    )

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Personalization Gains Across Experimental Configurations}")
    latex_lines.append("\\label{tab:personalization-gains}")
    latex_lines.append("\\begin{tabular}{llrrrrr}")
    latex_lines.append("\\hline")
    latex_lines.append("Dataset & $\\alpha$ & Pers. Epochs & Global F1 & " "Personalized F1 & Gain (Mean) & Gain (Std) \\\\")
    latex_lines.append("\\hline")

    for row in config_df.itertuples():
        dataset = row[1].upper()
        alpha = row[2]
        epochs = row[3]
        global_f1 = row[4]
        pers_f1 = row[5]
        gain_mean = row[6]
        gain_std = row[7]

        latex_lines.append(
            f"{dataset} & {alpha:.2f} & {epochs} & {global_f1:.4f} & {pers_f1:.4f} & " f"{gain_mean:.4f} & {gain_std:.4f} \\\\"
        )

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    output_file = output_dir / "personalization_gains_table.tex"
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"Saved LaTeX table to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate personalization gains visualization")
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=Path("logs_debug"),
        help="Directory containing experiment logs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/personalization"),
        help="Output directory for plots and tables",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading personalization data from {args.logs_dir}...")
    df = load_personalization_data(args.logs_dir)

    if df.empty:
        print("ERROR: No personalization data found!")
        return

    n_experiments = df["exp_dir"].nunique()
    print(f"Loaded {len(df)} client metrics from {n_experiments} experiments")

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
    pct_positive = summary["overall"]["pct_positive_gains"]
    print(f"Clients with meaningful gains (>0.01): {pct_positive:.1f}%")
    print(f"Max gain observed: {summary['overall']['max_gain']:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_personalization_gains(df, args.output_dir)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(df, args.output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
