#!/usr/bin/env python3
"""
Plot Comprehensive Analysis Results from FULL_IIOT_COMPREHENSIVE_ANALYSIS.md

Generates publication-quality plots highlighting:
1. Metric computation discrepancy (Reported vs TRUE global F1)
2. Aggregator robustness under Byzantine attacks
3. Data heterogeneity impact (Alpha sweep)
4. FedProx mu parameter analysis
5. Majority vs Minority class performance comparison
6. Top 5 majority classes detailed performance
7. Training dynamics over rounds
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_metrics_utils import compute_confidence_interval

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
        "figure.titlesize": 14,
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

TOP_5_MAJORITY_CLASSES = [
    "BENIGN",
    "DDOS_UDP",
    "DDOS_ICMP",
    "SQL_INJECTION",
    "VULNERABILITY_SCANNER",
]

CLASS_SAMPLE_COUNTS = {
    "BENIGN": 1238765,
    "DDOS_UDP": 93254,
    "DDOS_ICMP": 89329,
    "SQL_INJECTION": 39273,
    "VULNERABILITY_SCANNER": 38503,
    "DDOS_TCP": 38461,
    "PASSWORD": 38448,
    "DDOS_HTTP": 38316,
    "UPLOADING": 28785,
    "BACKDOOR": 18984,
    "PORT_SCANNING": 17314,
    "XSS": 12199,
    "RANSOMWARE": 8368,
    "MITM": 928,
    "FINGERPRINTING": 764,
}


def compute_true_global_f1_from_confusion_matrices(run_dir: Path, num_clients: int = 10):
    """
    Compute TRUE global F1 from aggregated confusion matrices.

    Args:
        run_dir: Path to experiment run directory
        num_clients: Number of clients

    Returns:
        Dictionary with per-round TRUE global F1 scores, or None if data unavailable
    """
    try:
        global_cm = None

        for client_id in range(num_clients):
            client_file = run_dir / f"client_{client_id}_metrics.csv"
            if not client_file.exists():
                return None

            df = pd.read_csv(client_file)

            if "confusion_matrix_counts" not in df.columns:
                return None

            last_round_cm_str = df.iloc[-1]["confusion_matrix_counts"]

            if pd.isna(last_round_cm_str):
                return None

            cm = json.loads(last_round_cm_str)
            cm_array = np.array(cm)

            if global_cm is None:
                global_cm = cm_array
            else:
                global_cm += cm_array

        if global_cm is None:
            return None

        num_classes = global_cm.shape[0]
        per_class_f1 = []

        for class_idx in range(num_classes):
            tp = global_cm[class_idx, class_idx]
            fp = global_cm[:, class_idx].sum() - tp
            fn = global_cm[class_idx, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_f1.append(f1)

        return np.mean(per_class_f1)

    except Exception as e:
        print(f"Error computing TRUE global F1 for {run_dir}: {e}")
        return None


def plot_metric_discrepancy(runs_dir: Path, output_dir: Path):
    """
    Plot Reported vs TRUE global F1 for FedProx mu=1.0, alpha=0.1 across seeds.

    This visualizes Section 1.3 of the comprehensive analysis.
    """
    seeds = list(range(42, 50))
    reported_f1s = []
    true_f1s = []

    for seed in seeds:
        run_dir = runs_dir / f"dsedge-iiotset-full_comp_fedprox_alpha0.1_adv0_dp0_pers0_mu1.0_seed{seed}_datasetedge-iiotset-full"

        if not run_dir.exists():
            print(f"Warning: Run directory not found for seed {seed}")
            continue

        metrics_file = run_dir / "metrics.csv"
        if not metrics_file.exists():
            print(f"Warning: metrics.csv not found for seed {seed}")
            continue

        df = pd.read_csv(metrics_file)
        reported_f1 = df.iloc[-1]["global_macro_f1_test"]
        reported_f1s.append(reported_f1)

        true_f1 = compute_true_global_f1_from_confusion_matrices(run_dir)
        if true_f1 is not None:
            true_f1s.append(true_f1)
        else:
            print(f"Warning: Could not compute TRUE F1 for seed {seed}")
            true_f1s.append(reported_f1)

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(seeds))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        reported_f1s,
        width,
        label="Reported F1 (Weighted Client Avg)",
        color="#d62728",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    bars2 = ax.bar(
        x + width / 2,
        true_f1s,
        width,
        label="TRUE Global F1 (Aggregated CM)",
        color="#2ca02c",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    for i, (reported, true) in enumerate(zip(reported_f1s, true_f1s)):
        improvement = ((true - reported) / reported) * 100
        ax.text(
            x[i],
            max(reported, true) + 0.02,
            f"+{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Random Seed", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Critical Finding: Reported Metric Underestimates TRUE System Performance\n" "FedProx (mu=1.0, alpha=0.1, 0% adversaries)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.9, color="green", linestyle="--", linewidth=2, alpha=0.5, label="90% threshold")

    mean_reported = np.mean(reported_f1s)
    mean_true = np.mean(true_f1s)
    textstr = f"Mean Reported: {mean_reported:.2%}\nMean TRUE: {mean_true:.2%}\nImprovement: +{((mean_true - mean_reported) / mean_reported) * 100:.1f}%"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "metric_discrepancy_reported_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'metric_discrepancy_reported_vs_true.png'}")


def plot_byzantine_robustness(df: pd.DataFrame, output_dir: Path):
    """
    Plot aggregator performance under Byzantine attacks (0%, 10%, 20%, 30%).

    This visualizes Section 3.1 of the comprehensive analysis.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    attack_levels = [0, 10, 20, 30]

    for agg in ["fedavg", "krum", "median", "bulyan"]:
        means = []
        ci_lows = []
        ci_ups = []

        for adv_pct in attack_levels:
            agg_data = df[(df["aggregator"] == agg) & (df["adv_percent"] == adv_pct)]

            if len(agg_data) == 0:
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_ups.append(np.nan)
                continue

            final_round = agg_data.groupby("seed")["round"].transform("max")
            final_data = agg_data[agg_data["round"] == final_round]["macro_f1"].dropna()

            if len(final_data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(final_data)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)
            else:
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_ups.append(np.nan)

        valid_mask = ~np.isnan(means)
        if np.any(valid_mask):
            valid_attacks = [a for a, v in zip(attack_levels, valid_mask) if v]
            valid_means = [m for m, v in zip(means, valid_mask) if v]
            valid_ci_lows = [c for c, v in zip(ci_lows, valid_mask) if v]
            valid_ci_ups = [c for c, v in zip(ci_ups, valid_mask) if v]

            linestyle = "-" if agg == "median" else "--" if agg == "fedavg" else "-"
            linewidth = 3 if agg in ["median", "fedavg"] else 2
            marker = "o" if agg == "median" else "s" if agg == "fedavg" else "^"

            ax.plot(
                valid_attacks,
                valid_means,
                label=agg.capitalize(),
                color=COLORS.get(agg, "gray"),
                linewidth=linewidth,
                linestyle=linestyle,
                marker=marker,
                markersize=10,
            )
            ax.fill_between(
                valid_attacks,
                valid_ci_lows,
                valid_ci_ups,
                color=COLORS.get(agg, "gray"),
                alpha=0.15,
            )

    ax.set_xlabel("Adversarial Clients (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Robust Aggregation: Performance Under Byzantine Attacks\n" "Success: Median maintains 3x better F1 than FedAvg at 30% attack",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(attack_levels)
    ax.set_ylim([0, 0.8])
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="green", linestyle=":", linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "byzantine_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'byzantine_robustness.png'}")


def plot_heterogeneity_impact(df: pd.DataFrame, output_dir: Path):
    """
    Plot performance by Alpha (data heterogeneity) for benign scenarios.

    This visualizes Section 4.1 of the comprehensive analysis.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    benign_df = df[df["adv_percent"] == 0]

    alphas = sorted([a for a in benign_df["alpha"].unique() if a != float("inf")])

    for agg in ["fedavg", "krum", "median", "bulyan", "fedprox"]:
        means = []
        ci_lows = []
        ci_ups = []
        valid_alphas = []

        for alpha in alphas:
            agg_data = benign_df[(benign_df["aggregator"] == agg) & (benign_df["alpha"] == alpha)]

            if len(agg_data) == 0:
                continue

            final_round = agg_data.groupby("seed")["round"].transform("max")
            final_data = agg_data[agg_data["round"] == final_round]["macro_f1"].dropna()

            if len(final_data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(final_data)
                valid_alphas.append(alpha)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)

        if len(valid_alphas) > 0:
            linestyle = "-" if agg == "fedprox" else "--"
            linewidth = 3 if agg == "fedprox" else 2

            ax.plot(
                valid_alphas,
                means,
                label=agg.capitalize(),
                color=COLORS.get(agg, "gray"),
                linewidth=linewidth,
                linestyle=linestyle,
                marker="o",
                markersize=8,
            )
            ax.fill_between(
                valid_alphas,
                ci_lows,
                ci_ups,
                color=COLORS.get(agg, "gray"),
                alpha=0.15,
            )

    ax.set_xlabel("Dirichlet Alpha (Data Heterogeneity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Data Heterogeneity Impact: Performance Across Non-IID Levels\n" "Lower alpha = More heterogeneous | Higher alpha = More IID",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_ylim([0, 0.8])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        "Extreme\nNon-IID",
        xy=(0.02, 0.1),
        xytext=(0.01, 0.25),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    ax.annotate(
        "Mild\nNon-IID",
        xy=(1.0, 0.45),
        xytext=(2.0, 0.6),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "heterogeneity_impact.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'heterogeneity_impact.png'}")


def plot_fedprox_mu_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Plot FedProx mu parameter impact on performance.

    This visualizes Section 4.1.3 of the comprehensive analysis.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    fedprox_df = df[(df["aggregator"] == "fedprox") & (df["adv_percent"] == 0)]

    mu_values = sorted([m for m in fedprox_df["mu"].unique() if not pd.isna(m)])

    for alpha in [0.05, 0.1, 0.2]:
        means = []
        ci_lows = []
        ci_ups = []
        valid_mus = []

        for mu in mu_values:
            mu_data = fedprox_df[(fedprox_df["mu"] == mu) & (fedprox_df["alpha"] == alpha)]

            if len(mu_data) == 0:
                continue

            final_round = mu_data.groupby("seed")["round"].transform("max")
            final_data = mu_data[mu_data["round"] == final_round]["macro_f1"].dropna()

            if len(final_data) > 0:
                mean, ci_low, ci_up = compute_confidence_interval(final_data)
                valid_mus.append(mu)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_ups.append(ci_up)

        if len(valid_mus) > 0:
            ax.plot(
                valid_mus,
                means,
                label=f"Alpha={alpha}",
                linewidth=3,
                marker="o",
                markersize=8,
            )
            ax.fill_between(
                valid_mus,
                ci_lows,
                ci_ups,
                alpha=0.15,
            )

    ax.set_xlabel("FedProx Mu (Proximal Term Strength)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "FedProx Mu Parameter Analysis: Constraining Local Model Drift\n"
        "Success: High mu (0.5-1.0) dramatically improves non-IID performance",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_ylim([0, 0.9])
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.axvline(x=1.0, color="green", linestyle="--", linewidth=2, alpha=0.5)
    ax.annotate(
        "Optimal\nmu=1.0",
        xy=(1.0, 0.7),
        xytext=(2.0, 0.8),
        fontsize=10,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fedprox_mu_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'fedprox_mu_analysis.png'}")


def plot_majority_vs_minority_classes(runs_dir: Path, output_dir: Path):
    """
    Plot Majority vs Minority class macro F1 comparison.

    Uses best experiment from Section 6.1 of the comprehensive analysis.
    """
    best_run_dir = runs_dir / "dsedge-iiotset-full_comp_fedprox_alpha0.1_adv0_dp0_pers0_mu0.08_seed42_datasetedge-iiotset-full"

    if not best_run_dir.exists():
        print(f"Warning: Best run directory not found: {best_run_dir}")
        return

    all_f1s = {}

    for client_id in range(10):
        client_file = best_run_dir / f"client_{client_id}_metrics.csv"
        if not client_file.exists():
            continue

        df = pd.read_csv(client_file)

        if "f1_per_class_after" not in df.columns:
            continue

        last_round = df.iloc[-1]
        per_class_str = last_round["f1_per_class_after"]

        if pd.isna(per_class_str):
            continue

        per_class = json.loads(per_class_str)

        for class_name, f1_score in per_class.items():
            class_upper = class_name.upper()
            if class_upper not in all_f1s:
                all_f1s[class_upper] = []
            all_f1s[class_upper].append(f1_score)

    majority_classes = []
    majority_f1s = []
    minority_classes = []
    minority_f1s = []

    for class_name, f1_list in all_f1s.items():
        mean_f1 = np.mean(f1_list)

        if class_name in TOP_5_MAJORITY_CLASSES:
            majority_classes.append(class_name)
            majority_f1s.append(mean_f1)
        else:
            minority_classes.append(class_name)
            minority_f1s.append(mean_f1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x_maj = np.arange(len(majority_classes))
    bars1 = ax1.bar(
        x_maj,
        majority_f1s,
        color="#2ca02c",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    for i, (bar, f1) in enumerate(zip(bars1, majority_f1s)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{f1:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("Attack Class", fontsize=11, fontweight="bold")
    ax1.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax1.set_title("Top 5 Majority Classes (SUCCESS)\nAll achieve >95% F1", fontsize=12, fontweight="bold", color="green")
    ax1.set_xticks(x_maj)
    ax1.set_xticklabels(majority_classes, rotation=45, ha="right")
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.95, color="green", linestyle="--", linewidth=2, alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    x_min = np.arange(len(minority_classes))
    colors_min = ["#ff7f0e" if f1 < 0.9 else "#2ca02c" for f1 in minority_f1s]
    bars2 = ax2.bar(
        x_min,
        minority_f1s,
        color=colors_min,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    for i, (bar, f1) in enumerate(zip(bars2, minority_f1s)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{f1:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_xlabel("Attack Class", fontsize=11, fontweight="bold")
    ax2.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax2.set_title("Minority Classes (MIXED RESULTS)\nGreen: Success | Orange: Needs Improvement", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_min)
    ax2.set_xticklabels(minority_classes, rotation=45, ha="right")
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.9, color="green", linestyle="--", linewidth=2, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Per-Class Performance Analysis: Majority vs Minority Classes\n" "Best Configuration: FedProx (mu=0.08, alpha=0.1, seed=42)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "majority_vs_minority_classes.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'majority_vs_minority_classes.png'}")


def plot_top5_detailed_performance(runs_dir: Path, output_dir: Path):
    """
    Plot detailed performance metrics for top 5 majority classes.

    Shows Precision, Recall, and F1 for each class.
    """
    best_run_dir = runs_dir / "dsedge-iiotset-full_comp_fedprox_alpha0.1_adv0_dp0_pers0_mu0.08_seed42_datasetedge-iiotset-full"

    if not best_run_dir.exists():
        print(f"Warning: Best run directory not found: {best_run_dir}")
        return

    class_metrics = {class_name: {"precision": [], "recall": [], "f1": []} for class_name in TOP_5_MAJORITY_CLASSES}

    for client_id in range(10):
        client_file = best_run_dir / f"client_{client_id}_metrics.csv"
        if not client_file.exists():
            continue

        df = pd.read_csv(client_file)
        last_round = df.iloc[-1]

        metric_mapping = {
            "precision_per_class": "precision",
            "recall_per_class": "recall",
            "f1_per_class_after": "f1",
        }

        for col_name, metric_key in metric_mapping.items():
            if col_name not in df.columns:
                continue

            metric_str = last_round[col_name]
            if pd.isna(metric_str):
                continue

            metric_dict = json.loads(metric_str)

            for class_name, value in metric_dict.items():
                class_upper = class_name.upper()
                if class_upper in class_metrics:
                    class_metrics[class_upper][metric_key].append(value)

    classes = []
    precisions = []
    recalls = []
    f1s = []

    for class_name in TOP_5_MAJORITY_CLASSES:
        if class_name in class_metrics:
            classes.append(class_name)
            precisions.append(np.mean(class_metrics[class_name]["precision"]) if class_metrics[class_name]["precision"] else 0)
            recalls.append(np.mean(class_metrics[class_name]["recall"]) if class_metrics[class_name]["recall"] else 0)
            f1s.append(np.mean(class_metrics[class_name]["f1"]) if class_metrics[class_name]["f1"] else 0)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(classes))
    width = 0.25

    bars1 = ax.bar(x - width, precisions, width, label="Precision", color="#1f77b4", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x, recalls, width, label="Recall", color="#ff7f0e", alpha=0.8, edgecolor="black")
    bars3 = ax.bar(x + width, f1s, width, label="F1 Score", color="#2ca02c", alpha=0.8, edgecolor="black")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2%}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Attack Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Top 5 Majority Classes: Detailed Performance Metrics\n" "SUCCESS: All metrics >95% demonstrating excellent detection capability",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.95, color="green", linestyle="--", linewidth=2, alpha=0.5)

    for class_name, sample_count in CLASS_SAMPLE_COUNTS.items():
        if class_name in classes:
            idx = classes.index(class_name)
            ax.text(
                idx,
                0.05,
                f"n={sample_count:,}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="darkgray",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "top5_detailed_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'top5_detailed_performance.png'}")


def plot_training_dynamics(runs_dir: Path, output_dir: Path):
    """
    Plot training dynamics over rounds for best configuration.

    This visualizes Section 5.3 of the comprehensive analysis.
    """
    best_run_dir = runs_dir / "dsedge-iiotset-full_comp_fedprox_alpha0.1_adv0_dp0_pers0_mu1.0_seed42_datasetedge-iiotset-full"

    if not best_run_dir.exists():
        print(f"Warning: Best run directory not found: {best_run_dir}")
        return

    metrics_file = best_run_dir / "metrics.csv"
    if not metrics_file.exists():
        print(f"Warning: metrics.csv not found")
        return

    df = pd.read_csv(metrics_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    rounds = df["round"]
    macro_f1 = df["global_macro_f1_test"]

    ax1.plot(rounds, macro_f1, color="#2ca02c", linewidth=3, marker="o", markersize=8, label="Macro F1")
    ax1.set_xlabel("Communication Round", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
    ax1.set_title("F1 Score Convergence Over Training", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])
    ax1.axhline(y=0.9, color="green", linestyle="--", linewidth=2, alpha=0.5, label="90% threshold")
    ax1.legend(loc="lower right", fontsize=10)

    improvement_rounds = [1, 5, 10, 15, 18, 20]
    for i, r in enumerate(improvement_rounds):
        if r <= len(df):
            f1_val = df[df["round"] == r]["global_macro_f1_test"].values[0]
            ax1.annotate(
                f"R{r}: {f1_val:.1%}",
                xy=(r, f1_val),
                xytext=(r + 1, f1_val + 0.05),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
            )

    if "global_macro_f1_val" in df.columns:
        macro_f1_val = df["global_macro_f1_val"]
        ax2.plot(rounds, macro_f1_val, color="#1f77b4", linewidth=3, marker="^", markersize=8, label="Validation F1")
        ax2.plot(rounds, macro_f1, color="#2ca02c", linewidth=3, marker="o", markersize=8, label="Test F1", alpha=0.7)
        ax2.set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
        ax2.set_title("Validation vs Test Performance", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])
        ax2.legend(loc="lower right", fontsize=10)
    else:
        ax2.plot(rounds, macro_f1, color="#2ca02c", linewidth=3, marker="o", markersize=8, label="Test F1")
        ax2.set_xlabel("Communication Round", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
        ax2.set_title("F1 Score Progression", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])
        ax2.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "Training Dynamics: FedProx (mu=1.0, alpha=0.1, seed=42)\n" "Rapid initial improvement with continued gains through round 18",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "training_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'training_dynamics.png'}")


def main():
    runs_dir = Path("runs")
    output_dir = Path("comprehensive_analysis_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING COMPREHENSIVE ANALYSIS PLOTS")
    print("=" * 80)

    print("\n1. Loading experiment data...")
    df = pd.read_csv("full_iiot_all_results.csv")
    print(f"   Loaded {len(df)} records")

    print("\n2. Plotting Metric Discrepancy (Reported vs TRUE)...")
    plot_metric_discrepancy(runs_dir, output_dir)

    print("\n3. Plotting Byzantine Robustness...")
    plot_byzantine_robustness(df, output_dir)

    print("\n4. Plotting Heterogeneity Impact...")
    plot_heterogeneity_impact(df, output_dir)

    print("\n5. Plotting FedProx Mu Analysis...")
    plot_fedprox_mu_analysis(df, output_dir)

    print("\n6. Plotting Majority vs Minority Classes...")
    plot_majority_vs_minority_classes(runs_dir, output_dir)

    print("\n7. Plotting Top 5 Detailed Performance...")
    plot_top5_detailed_performance(runs_dir, output_dir)

    print("\n8. Plotting Training Dynamics...")
    plot_training_dynamics(runs_dir, output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"All plots saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  1. metric_discrepancy_reported_vs_true.png - Critical finding on metric computation")
    print("  2. byzantine_robustness.png - Aggregator performance under attacks")
    print("  3. heterogeneity_impact.png - Data heterogeneity effects")
    print("  4. fedprox_mu_analysis.png - FedProx parameter tuning")
    print("  5. majority_vs_minority_classes.png - Class-wise comparison")
    print("  6. top5_detailed_performance.png - Top 5 majority classes metrics")
    print("  7. training_dynamics.png - Convergence analysis")


if __name__ == "__main__":
    main()
