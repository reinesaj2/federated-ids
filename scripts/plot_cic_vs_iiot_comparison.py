#!/usr/bin/env python3
"""
Cross-Dataset Comparison: CIC-IDS2017 vs Edge-IIoTset

Generates comprehensive comparison plots contrasting federated learning
performance across two benchmark intrusion detection datasets.

Key Comparisons:
- Objective 1: Robust aggregation under Byzantine attacks
- Objective 2: Heterogeneity resilience (alpha values)
- Objective 3: Personalization gains
- Objective 4: Privacy-utility tradeoffs
- Overall: Baseline performance characteristics

Data Sources:
- CIC: 436 experimental runs with '_datasetcic' suffix
- IIOT: 344 experimental runs with 'dsedge-iiotset' prefix
"""

import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("plots/cross_dataset_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_run_config(run_name: str) -> Dict:
    """Extract configuration from run directory name."""
    config = {}

    # Dataset - CIC has explicit suffix, IIOT has prefix
    if "_datasetcic" in run_name.lower() or run_name.endswith("datasetcic"):
        config["dataset"] = "CIC-IDS2017"
    elif run_name.startswith("dsedge-iiotset") or "_datasetedge-iiotset" in run_name:
        config["dataset"] = "Edge-IIoTset"
    else:
        return None

    # Aggregation - check for all variants
    if "_fedavg_" in run_name or "comp_fedavg" in run_name:
        config["aggregation"] = "FedAvg"
    elif "_krum_" in run_name or "comp_krum" in run_name:
        config["aggregation"] = "Krum"
    elif "_bulyan_" in run_name or "comp_bulyan" in run_name:
        config["aggregation"] = "Bulyan"
    elif "_median_" in run_name or "comp_median" in run_name:
        config["aggregation"] = "Median"
    elif "fedprox" in run_name.lower():
        config["aggregation"] = "FedProx"
    else:
        return None

    # Alpha (heterogeneity)
    m = re.search(r"alpha([\d.]+|inf)", run_name)
    if m:
        config["alpha"] = float("inf") if m.group(1) == "inf" else float(m.group(1))

    # Adversary percentage
    m = re.search(r"adv(\d+)", run_name)
    config["adv_pct"] = int(m.group(1)) if m else 0

    # Differential privacy
    m = re.search(r"dp(\d+)", run_name)
    config["dp"] = int(m.group(1)) if m else 0

    # Personalization epochs
    m = re.search(r"pers(\d+)", run_name)
    config["pers_epochs"] = int(m.group(1)) if m else 0

    # FedProx mu
    m = re.search(r"mu([\d.]+)", run_name)
    config["mu"] = float(m.group(1)) if m else 0.0

    # Seed
    m = re.search(r"seed(\d+)", run_name)
    config["seed"] = int(m.group(1)) if m else 0

    return config


def load_cross_dataset_data() -> pd.DataFrame:
    """Load and parse all CIC and IIOT experimental runs."""
    data = []
    parse_failed = {"no_dataset": 0, "no_agg": 0}
    load_failed = {"no_csv": 0, "empty_csv": 0, "parse_error": 0}
    iiot_total = 0

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = parse_run_config(run_dir.name)
        if config is None:
            if run_dir.name.startswith("dsedge-iiotset") or "_datasetcic" in run_dir.name:
                parse_failed["no_agg"] += 1
            continue
        if "dataset" not in config:
            parse_failed["no_dataset"] += 1
            continue

        # Track IIOT runs after successful parsing
        if config.get("dataset") == "Edge-IIoTset":
            iiot_total += 1

        # Try client_0_metrics.csv first
        client_file = run_dir / "client_0_metrics.csv"
        if not client_file.exists():
            load_failed["no_csv"] += 1
            continue

        try:
            df = pd.read_csv(client_file)
            if df.empty:
                load_failed["empty_csv"] += 1
                continue

            # Extract final metrics
            config["final_f1"] = df["macro_f1_after"].iloc[-1]

            # Try to get baseline F1 if no personalization
            if config["pers_epochs"] == 0 and "macro_f1_before" in df.columns:
                config["baseline_f1"] = df["macro_f1_before"].iloc[-1]
            else:
                config["baseline_f1"] = config["final_f1"]

            # Number of classes
            if "n_classes" in df.columns:
                config["n_classes"] = df["n_classes"].iloc[0]

            data.append(config)

        except Exception as e:
            load_failed["parse_error"] += 1
            # Only print first few errors to avoid spam
            if load_failed["parse_error"] <= 5:
                print(f"Error loading {run_dir.name}: {e}")
            continue

    df = pd.DataFrame(data)
    iiot_loaded = len(df[df['dataset'] == 'Edge-IIoTset'])
    cic_loaded = len(df[df['dataset'] == 'CIC-IDS2017'])

    print(f"\nSuccessfully loaded {len(df)} runs total:")
    print(f"  - CIC-IDS2017: {cic_loaded} runs")
    print(f"  - Edge-IIoTset: {iiot_loaded} runs ({iiot_loaded/3.44:.1f}% of 344 available)")
    print("\nLoad statistics:")
    print(f"  - Missing CSV files: {load_failed['no_csv']}")
    print(f"  - Empty CSV files: {load_failed['empty_csv']}")
    print(f"  - CSV parse errors: {load_failed['parse_error']}")
    print(f"  - Total usable: {len(df)}")

    return df


def plot_obj1_attack_resilience_comparison(df: pd.DataFrame):
    """Compare robust aggregation performance under Byzantine attacks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Objective 1: Attack Resilience - CIC-IDS2017 vs Edge-IIoTset", fontsize=16, fontweight="bold")

    colors = {
        "FedAvg": "#1f77b4",
        "Krum": "#2ca02c",
        "Bulyan": "#d62728",
        "Median": "#9467bd",
    }

    datasets = ["CIC-IDS2017", "Edge-IIoTset"]
    agg_order = ["FedAvg", "Krum", "Bulyan", "Median"]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[(df["dataset"] == dataset) & (df["aggregation"].isin(agg_order))]

        if dataset_df.empty:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha="center", va="center", transform=ax.transAxes)
            continue

        # Plot each aggregation method
        for agg in agg_order:
            agg_data = dataset_df[dataset_df["aggregation"] == agg]
            summary = (
                agg_data.groupby("adv_pct")
                .agg(mean_f1=("final_f1", "mean"), std_f1=("final_f1", "std"), count=("final_f1", "count"))
                .reset_index()
            )

            if summary.empty:
                continue

            # Calculate 95% confidence intervals
            summary["ci"] = 1.96 * summary["std_f1"] / np.sqrt(summary["count"])

            ax.plot(summary["adv_pct"], summary["mean_f1"], marker="o", label=agg, color=colors.get(agg), linewidth=2, markersize=8)
            ax.fill_between(
                summary["adv_pct"], summary["mean_f1"] - summary["ci"], summary["mean_f1"] + summary["ci"], alpha=0.2, color=colors.get(agg)
            )

        ax.set_xlabel("Adversary Percentage (%)", fontsize=12)
        ax.set_ylabel("Macro-F1 Score", fontsize=12)
        ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj1_attack_resilience_cic_vs_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_obj2_heterogeneity_comparison(df: pd.DataFrame):
    """Compare performance across heterogeneity levels (alpha values)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Objective 2: Heterogeneity Resilience - CIC-IDS2017 vs Edge-IIoTset", fontsize=16, fontweight="bold")

    datasets = ["CIC-IDS2017", "Edge-IIoTset"]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[
            (df["dataset"] == dataset)
            & (df["aggregation"] == "FedAvg")
            & (df["adv_pct"] == 0)
            & (df["mu"] == 0.0)
            & (df["alpha"] < float("inf"))
        ]

        if dataset_df.empty:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha="center", va="center", transform=ax.transAxes)
            continue

        summary = (
            dataset_df.groupby("alpha")
            .agg(mean_f1=("final_f1", "mean"), std_f1=("final_f1", "std"), count=("final_f1", "count"))
            .reset_index()
        )

        summary["ci"] = 1.96 * summary["std_f1"] / np.sqrt(summary["count"])

        ax.errorbar(
            summary["alpha"], summary["mean_f1"], yerr=summary["ci"], marker="o", linewidth=2, markersize=8, capsize=5, label="FedAvg"
        )

        ax.set_xlabel("Alpha (heterogeneity parameter)", fontsize=12)
        ax.set_ylabel("Macro-F1 Score", fontsize=12)
        ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        ax.legend()

    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj2_heterogeneity_cic_vs_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_baseline_performance_comparison(df: pd.DataFrame):
    """Compare baseline performance characteristics across datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Baseline Performance Comparison: CIC-IDS2017 vs Edge-IIoTset", fontsize=16, fontweight="bold")

    # Filter for benign baseline conditions
    baseline_df = df[
        (df["adv_pct"] == 0) & (df["dp"] == 0) & (df["pers_epochs"] == 0) & (df["mu"] == 0.0) & (df["aggregation"] == "FedAvg")
    ]

    # Panel A: F1 Distribution by Dataset
    ax = axes[0, 0]
    for dataset in ["CIC-IDS2017", "Edge-IIoTset"]:
        data = baseline_df[baseline_df["dataset"] == dataset]["final_f1"]
        if not data.empty:
            ax.hist(data, alpha=0.6, label=dataset, bins=20)
    ax.set_xlabel("Macro-F1 Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("F1 Score Distribution (Baseline)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Box Plot Comparison
    ax = axes[0, 1]
    plot_data = []
    labels = []
    for dataset in ["CIC-IDS2017", "Edge-IIoTset"]:
        data = baseline_df[baseline_df["dataset"] == dataset]["final_f1"]
        if not data.empty:
            plot_data.append(data)
            labels.append(dataset)

    if plot_data:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
            patch.set_facecolor(color)
    ax.set_ylabel("Macro-F1 Score", fontsize=12)
    ax.set_title("F1 Score Comparison (Baseline)", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Statistical Summary Table
    ax = axes[1, 0]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for dataset in ["CIC-IDS2017", "Edge-IIoTset"]:
        data = baseline_df[baseline_df["dataset"] == dataset]["final_f1"]
        if not data.empty:
            table_data.append(
                [dataset, f"{data.mean():.4f}", f"{data.std():.4f}", f"{data.min():.4f}", f"{data.max():.4f}", f"{len(data)}"]
            )

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Dataset", "Mean F1", "Std F1", "Min F1", "Max F1", "N"],
            cellLoc="center",
            loc="center",
            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    ax.set_title("Statistical Summary", fontsize=14)

    # Panel D: Sample Size by Alpha
    ax = axes[1, 1]
    for dataset in ["CIC-IDS2017", "Edge-IIoTset"]:
        dataset_df = baseline_df[baseline_df["dataset"] == dataset]
        if not dataset_df.empty:
            counts = dataset_df.groupby("alpha").size()
            ax.plot(counts.index, counts.values, marker="o", label=dataset, linewidth=2, markersize=8)
    ax.set_xlabel("Alpha (heterogeneity)", fontsize=12)
    ax.set_ylabel("Number of Experiments", fontsize=12)
    ax.set_title("Sample Size Distribution", fontsize=14)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "baseline_performance_cic_vs_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_aggregation_heatmap_comparison(df: pd.DataFrame):
    """Create heatmap comparing aggregation performance across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Aggregation Method Performance Heatmap: CIC vs IIOT", fontsize=16, fontweight="bold")

    agg_order = ["FedAvg", "Krum", "Bulyan", "Median"]
    adv_levels = [0, 10, 20, 30]

    for idx, dataset in enumerate(["CIC-IDS2017", "Edge-IIoTset"]):
        ax = axes[idx]

        # Create pivot table
        dataset_df = df[(df["dataset"] == dataset) & (df["aggregation"].isin(agg_order)) & (df["adv_pct"].isin(adv_levels))]

        if dataset_df.empty:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha="center", va="center", transform=ax.transAxes)
            continue

        pivot = dataset_df.pivot_table(values="final_f1", index="aggregation", columns="adv_pct", aggfunc="mean")

        # Reindex to ensure consistent ordering
        pivot = pivot.reindex(agg_order)
        # Only keep adversary levels that exist in the data
        available_levels = [lvl for lvl in adv_levels if lvl in pivot.columns]
        if available_levels:
            pivot = pivot[available_levels]

        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1.0, ax=ax, cbar_kws={"label": "Macro-F1"})
        ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Adversary Percentage (%)", fontsize=12)
        ax.set_ylabel("Aggregation Method", fontsize=12)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aggregation_heatmap_cic_vs_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def generate_statistical_comparison_report(df: pd.DataFrame):
    """Generate comprehensive statistical comparison report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CROSS-DATASET STATISTICAL COMPARISON REPORT")
    report_lines.append("CIC-IDS2017 vs Edge-IIoTset")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall dataset statistics
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 80)
    for dataset in ["CIC-IDS2017", "Edge-IIoTset"]:
        dataset_df = df[df["dataset"] == dataset]
        report_lines.append(f"\n{dataset}:")
        report_lines.append(f"  Total Experiments: {len(dataset_df)}")
        report_lines.append(f"  Mean F1: {dataset_df['final_f1'].mean():.4f} (SD: {dataset_df['final_f1'].std():.4f})")
        report_lines.append(f"  Min F1: {dataset_df['final_f1'].min():.4f}")
        report_lines.append(f"  Max F1: {dataset_df['final_f1'].max():.4f}")

    # Objective 1: Attack Resilience
    report_lines.append("\n" + "=" * 80)
    report_lines.append("OBJECTIVE 1: ROBUST AGGREGATION (30% Adversaries)")
    report_lines.append("-" * 80)

    attack_df = df[df["adv_pct"] == 30]
    for agg in ["FedAvg", "Krum", "Bulyan", "Median"]:
        report_lines.append(f"\n{agg}:")
        for dataset in ["CIC-IDS2017", "Edge-IIoTset"]:
            data = attack_df[(attack_df["dataset"] == dataset) & (attack_df["aggregation"] == agg)]["final_f1"]
            if not data.empty:
                report_lines.append(f"  {dataset}: {data.mean():.4f} +/- {data.std():.4f} (n={len(data)})")

    # Statistical tests
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
    report_lines.append("-" * 80)

    # Compare benign baselines between datasets
    baseline_cic = df[(df["dataset"] == "CIC-IDS2017") & (df["adv_pct"] == 0) & (df["aggregation"] == "FedAvg")]["final_f1"]
    baseline_iiot = df[(df["dataset"] == "Edge-IIoTset") & (df["adv_pct"] == 0) & (df["aggregation"] == "FedAvg")]["final_f1"]

    if len(baseline_cic) > 0 and len(baseline_iiot) > 0:
        t_stat, p_value = stats.ttest_ind(baseline_cic, baseline_iiot)
        cohen_d = (baseline_cic.mean() - baseline_iiot.mean()) / np.sqrt(
            (baseline_cic.std()**2 + baseline_iiot.std()**2) / 2
        )
        report_lines.append("\nBaseline FedAvg (benign):")
        report_lines.append(f"  CIC mean: {baseline_cic.mean():.4f}")
        report_lines.append(f"  IIOT mean: {baseline_iiot.mean():.4f}")
        report_lines.append(f"  t-statistic: {t_stat:.4f}")
        report_lines.append(f"  p-value: {p_value:.6f}")
        report_lines.append(f"  Cohen's d: {cohen_d:.4f}")

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        report_lines.append(f"  Significance: {sig}")

    report_lines.append("\n" + "=" * 80)

    # Save report
    report_path = OUTPUT_DIR / "statistical_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Saved: {report_path}")

    # Print to console
    print("\n".join(report_lines))


def main():
    """Generate all cross-dataset comparison plots."""
    print("Loading cross-dataset experimental data...")
    df = load_cross_dataset_data()

    if df.empty:
        print("ERROR: No data loaded. Check runs directory.")
        return

    print("\nDataset breakdown:")
    print(df.groupby("dataset").size())

    print("\nGenerating comparison plots...")

    # Objective 1: Attack Resilience
    print("\n1. Generating Objective 1 comparison (attack resilience)...")
    plot_obj1_attack_resilience_comparison(df)

    # Objective 2: Heterogeneity
    print("2. Generating Objective 2 comparison (heterogeneity)...")
    plot_obj2_heterogeneity_comparison(df)

    # Baseline Performance
    print("3. Generating baseline performance comparison...")
    plot_baseline_performance_comparison(df)

    # Aggregation Heatmap
    print("4. Generating aggregation heatmap comparison...")
    plot_aggregation_heatmap_comparison(df)

    # Statistical Report
    print("5. Generating statistical comparison report...")
    generate_statistical_comparison_report(df)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("DONE!")


if __name__ == "__main__":
    main()
