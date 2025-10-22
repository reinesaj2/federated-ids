#!/usr/bin/env python3
"""
Generate comparison plots for robust aggregation experiments.

Creates visualizations comparing algorithm performance under different adversary rates.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


RUN_NAME_PATTERN = re.compile(
    r"robust_agg_(?P<aggregation>\w+)_adv(?P<adv_fraction>[0-9.]+)_seed(?P<seed>\d+)"
)


def load_summary_data(artifacts_dir: Path) -> pd.DataFrame:
    """Load summary data from artifacts directory."""
    all_data = []

    # Look for summary JSON files in artifacts
    for json_file in artifacts_dir.rglob("robust_agg_summary.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    if not all_data:
        print("Warning: No summary data found, trying to load from detailed CSV files")
        return load_from_detailed_csvs(artifacts_dir)

    return pd.DataFrame(all_data)


def load_from_detailed_csvs(artifacts_dir: Path) -> pd.DataFrame:
    """Load data from detailed CSV files as fallback."""
    all_data = []

    for csv_file in artifacts_dir.rglob("robust_agg_detailed.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Compute summary statistics
    grouped = combined.groupby(["aggregation", "adv_fraction"]).agg({
        "macro_f1": ["mean", "std", "count"],
        "accuracy": ["mean", "std"],
        "l2_distance": ["mean", "std"],
        "cosine_similarity": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    grouped.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in grouped.columns]

    return grouped


def plot_f1_vs_adversary(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot F1 score vs adversary fraction for each algorithm."""
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = sorted(df["aggregation"].unique())
    colors = sns.color_palette("husl", len(algorithms))

    for i, algo in enumerate(algorithms):
        algo_data = df[df["aggregation"] == algo].sort_values("adv_fraction")

        x = algo_data["adv_fraction"]
        y = algo_data["macro_f1_mean"]
        yerr = algo_data.get("macro_f1_std", 0)

        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", markersize=8,
            linewidth=2, capsize=5,
            label=algo.upper(),
            color=colors[i]
        )

    ax.set_xlabel("Adversary Fraction", fontsize=13, fontweight="bold")
    ax.set_ylabel("Macro F1 Score", fontsize=13, fontweight="bold")
    ax.set_title("Algorithm Robustness: F1 Score vs Adversary Rate", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Add horizontal line at 0.7 (acceptable threshold)
    ax.axhline(y=0.7, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="Acceptable Threshold (0.7)")

    plt.tight_layout()
    output_path = output_dir / "f1_vs_adversary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_l2_vs_adversary(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot L2 distance vs adversary fraction for each algorithm."""
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = sorted(df["aggregation"].unique())
    colors = sns.color_palette("husl", len(algorithms))

    for i, algo in enumerate(algorithms):
        algo_data = df[df["aggregation"] == algo].sort_values("adv_fraction")

        x = algo_data["adv_fraction"]
        y = algo_data["l2_distance_mean"]
        yerr = algo_data.get("l2_distance_std", 0)

        ax.errorbar(
            x, y, yerr=yerr,
            marker="s", markersize=8,
            linewidth=2, capsize=5,
            label=algo.upper(),
            color=colors[i]
        )

    ax.set_xlabel("Adversary Fraction", fontsize=13, fontweight="bold")
    ax.set_ylabel("L2 Distance to Benign Mean", fontsize=13, fontweight="bold")
    ax.set_title("Algorithm Robustness: L2 Distance vs Adversary Rate", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "l2_vs_adversary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_algorithm_comparison_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmap comparing algorithms across adversary rates."""
    # Pivot data for heatmap
    pivot = df.pivot(index="aggregation", columns="adv_fraction", values="macro_f1_mean")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        cbar_kws={"label": "Macro F1 Score"},
        ax=ax
    )

    ax.set_xlabel("Adversary Fraction", fontsize=13, fontweight="bold")
    ax.set_ylabel("Aggregation Algorithm", fontsize=13, fontweight="bold")
    ax.set_title("Algorithm Performance Heatmap", fontsize=15, fontweight="bold")

    # Convert y-axis labels to uppercase
    yticklabels = [label.get_text().upper() for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, rotation=0)

    plt.tight_layout()
    output_path = output_dir / "algorithm_comparison_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_robustness_degradation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot relative performance degradation compared to no-adversary baseline."""
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = sorted(df["aggregation"].unique())
    colors = sns.color_palette("husl", len(algorithms))

    for i, algo in enumerate(algorithms):
        algo_data = df[df["aggregation"] == algo].sort_values("adv_fraction")

        # Get baseline (0% adversary)
        baseline_data = algo_data[algo_data["adv_fraction"] == 0.0]
        if len(baseline_data) == 0:
            print(f"Warning: No baseline data for {algo}")
            continue

        baseline_f1 = baseline_data["macro_f1_mean"].iloc[0]

        # Compute relative degradation
        x = algo_data["adv_fraction"]
        y = (algo_data["macro_f1_mean"] - baseline_f1) / baseline_f1 * 100

        ax.plot(
            x, y,
            marker="o", markersize=8,
            linewidth=2,
            label=algo.upper(),
            color=colors[i]
        )

    ax.set_xlabel("Adversary Fraction", fontsize=13, fontweight="bold")
    ax.set_ylabel("Relative Performance Degradation (%)", fontsize=13, fontweight="bold")
    ax.set_title("Algorithm Robustness Degradation", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    plt.tight_layout()
    output_path = output_dir / "robustness_degradation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_algorithm_bar_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart comparing algorithms at each adversary rate."""
    adv_fractions = sorted(df["adv_fraction"].unique())

    fig, axes = plt.subplots(1, len(adv_fractions), figsize=(5 * len(adv_fractions), 6))

    if len(adv_fractions) == 1:
        axes = [axes]

    for i, adv_frac in enumerate(adv_fractions):
        ax = axes[i]
        data = df[df["adv_fraction"] == adv_frac].sort_values("macro_f1_mean", ascending=False)

        algorithms = data["aggregation"].str.upper()
        f1_scores = data["macro_f1_mean"]
        errors = data.get("macro_f1_std", [0] * len(data))

        colors = sns.color_palette("husl", len(algorithms))
        bars = ax.bar(range(len(algorithms)), f1_scores, yerr=errors, capsize=5, color=colors)

        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha="right")
        ax.set_ylabel("Macro F1 Score" if i == 0 else "", fontsize=12, fontweight="bold")
        ax.set_title(f"Adversary Rate: {adv_frac:.1%}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{f1:.3f}",
                ha="center", va="bottom", fontsize=10
            )

    plt.tight_layout()
    output_path = output_dir / "algorithm_bar_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate robust aggregation comparison plots"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
        help="Directory containing experiment artifacts",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/robust_agg_weekly/consolidated"),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Robust Aggregation Comparison Plots")
    print("=" * 60)

    # Load summary data
    df = load_summary_data(args.artifacts_dir)

    if df.empty:
        print("Error: No summary data found")
        return

    print(f"\nLoaded data:")
    print(f"  Algorithms: {sorted(df['aggregation'].unique())}")
    print(f"  Adversary rates: {sorted(df['adv_fraction'].unique())}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_f1_vs_adversary(df, args.output_dir)
    plot_l2_vs_adversary(df, args.output_dir)
    plot_algorithm_comparison_heatmap(df, args.output_dir)
    plot_robustness_degradation(df, args.output_dir)
    plot_algorithm_bar_comparison(df, args.output_dir)

    print("\n" + "=" * 60)
    print("Plot generation completed")
    print(f"Plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
