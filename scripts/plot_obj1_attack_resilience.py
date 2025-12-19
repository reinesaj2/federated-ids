#!/usr/bin/env python3
"""
Objective 1: Robust Aggregation - Attack Resilience in IIoT Federated IDS

Shows how Krum, Bulyan, and Median defend against Byzantine adversaries
compared to standard FedAvg.

Based on 85 attack experiments + benign baselines.
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


def load_valid_data():
    """Load all valid 15-class experiments."""
    data = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = {}

        if "comp_fedavg" in run_dir.name:
            config["aggregation"] = "FedAvg"
        elif "comp_krum" in run_dir.name:
            config["aggregation"] = "Krum"
        elif "comp_bulyan" in run_dir.name:
            config["aggregation"] = "Bulyan"
        elif "comp_median" in run_dir.name:
            config["aggregation"] = "Median"
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
        config["adv_pct"] = int(m.group(1)) if m else 0

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


def plot_attack_resilience(df: pd.DataFrame, output_path: Path):
    """Generate the 4-panel attack resilience figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Objective 1: Robust Aggregation Defends Against Byzantine Adversaries",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "FedAvg": "#1f77b4",
        "Krum": "#2ca02c",
        "Bulyan": "#d62728",
        "Median": "#9467bd",
    }

    agg_order = ["FedAvg", "Krum", "Bulyan", "Median"]
    adv_levels = [0, 10, 20, 30]

    # Panel A: F1 vs Adversary Percentage (line plot)
    ax1 = axes[0, 0]

    for agg in agg_order:
        agg_data = df[df["aggregation"] == agg]
        summary = (
            agg_data.groupby("adv_pct")
            .agg(
                f1_mean=("final_f1", "mean"),
                f1_sem=("final_f1", "sem"),
                n=("final_f1", "count"),
            )
            .reset_index()
        )

        if len(summary) >= 2:
            ax1.errorbar(
                summary["adv_pct"],
                summary["f1_mean"],
                yerr=1.96 * summary["f1_sem"],
                marker="o",
                color=colors[agg],
                linewidth=2,
                markersize=8,
                capsize=4,
                label=agg,
            )

    ax1.set_xlabel("Adversary Percentage (%)")
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("A) Performance Degradation Under Attack", fontweight="bold")
    ax1.legend(loc="lower left")
    ax1.set_xlim(-2, 35)
    ax1.set_ylim(0.40, 0.80)
    ax1.grid(True, alpha=0.3)

    # Panel B: Resilience Score (% retained from baseline)
    ax2 = axes[0, 1]

    resilience_data = []
    for agg in agg_order:
        agg_df = df[df["aggregation"] == agg]
        baseline = agg_df[agg_df["adv_pct"] == 0]["final_f1"].mean()

        for adv in [10, 20, 30]:
            attack_f1 = agg_df[agg_df["adv_pct"] == adv]["final_f1"]
            if len(attack_f1) > 0 and baseline > 0:
                retained = (attack_f1.mean() / baseline) * 100
                resilience_data.append(
                    {
                        "aggregation": agg,
                        "adv_pct": adv,
                        "retained": retained,
                        "n": len(attack_f1),
                    }
                )

    res_df = pd.DataFrame(resilience_data)

    x = np.arange(3)  # 10%, 20%, 30%
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, agg in enumerate(agg_order):
        agg_res = res_df[res_df["aggregation"] == agg].sort_values("adv_pct")
        if len(agg_res) > 0:
            # Map adversary percentages to x positions
            x_positions = []
            y_values = []
            for _, row in agg_res.iterrows():
                if row["adv_pct"] == 10:
                    x_positions.append(0 + offsets[i] * width)
                elif row["adv_pct"] == 20:
                    x_positions.append(1 + offsets[i] * width)
                elif row["adv_pct"] == 30:
                    x_positions.append(2 + offsets[i] * width)
                y_values.append(row["retained"])

            ax2.bar(
                x_positions,
                y_values,
                width,
                color=colors[agg],
                label=agg,
            )

    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["10%", "20%", "30%"])
    ax2.set_xlabel("Adversary Percentage")
    ax2.set_ylabel("Performance Retained (%)")
    ax2.set_title("B) Resilience: % of Baseline Performance Retained", fontweight="bold")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set_ylim(50, 110)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel C: Performance at 10% Adversaries (bar chart)
    ax3 = axes[1, 0]
    adv10 = df[df["adv_pct"] == 10]

    adv10_data = []
    for agg in agg_order:
        subset = adv10[adv10["aggregation"] == agg]
        if len(subset) > 0:
            adv10_data.append(
                {
                    "aggregation": agg,
                    "f1_mean": subset["final_f1"].mean(),
                    "f1_sem": subset["final_f1"].sem(),
                    "n": len(subset),
                }
            )

    adv10_df = pd.DataFrame(adv10_data)

    if len(adv10_df) > 0:
        bars = ax3.bar(
            range(len(adv10_df)),
            adv10_df["f1_mean"],
            yerr=1.96 * adv10_df["f1_sem"],
            color=[colors[a] for a in adv10_df["aggregation"]],
            capsize=4,
        )

        ax3.set_xticks(range(len(adv10_df)))
        ax3.set_xticklabels(adv10_df["aggregation"])

        # Add sample sizes
        for i, row in adv10_df.iterrows():
            ax3.annotate(
                f"n={row['n']}",
                xy=(i, row["f1_mean"] + 1.96 * row["f1_sem"] + 0.01),
                ha="center",
                fontsize=9,
            )

    ax3.set_ylabel("Macro F1 Score")
    ax3.set_title("C) Performance at 10% Adversaries", fontweight="bold")
    ax3.set_ylim(0.50, 0.80)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel D: Performance at 30% Adversaries
    ax4 = axes[1, 1]
    adv30 = df[df["adv_pct"] == 30]

    adv30_data = []
    for agg in agg_order:
        subset = adv30[adv30["aggregation"] == agg]
        if len(subset) > 0:
            adv30_data.append(
                {
                    "aggregation": agg,
                    "f1_mean": subset["final_f1"].mean(),
                    "f1_sem": subset["final_f1"].sem(),
                    "n": len(subset),
                }
            )

    adv30_df = pd.DataFrame(adv30_data)

    if len(adv30_df) > 0:
        bars = ax4.bar(
            range(len(adv30_df)),
            adv30_df["f1_mean"],
            yerr=1.96 * adv30_df["f1_sem"],
            color=[colors[a] for a in adv30_df["aggregation"]],
            capsize=4,
        )

        ax4.set_xticks(range(len(adv30_df)))
        ax4.set_xticklabels(adv30_df["aggregation"])

        for i, row in adv30_df.iterrows():
            ax4.annotate(
                f"n={row['n']}",
                xy=(i, row["f1_mean"] + 1.96 * row["f1_sem"] + 0.01),
                ha="center",
                fontsize=9,
            )

        # Add improvement annotation
        if "FedAvg" in adv30_df["aggregation"].values:
            fedavg_f1 = adv30_df[adv30_df["aggregation"] == "FedAvg"]["f1_mean"].values[0]
            best_robust = adv30_df[adv30_df["aggregation"] != "FedAvg"]["f1_mean"].max()
            improvement = ((best_robust - fedavg_f1) / fedavg_f1) * 100
            ax4.annotate(
                f"Best robust agg:\n+{improvement:.1f}% vs FedAvg",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    ax4.set_ylabel("Macro F1 Score")
    ax4.set_title("D) Performance at 30% Adversaries (Severe Attack)", fontweight="bold")
    ax4.set_ylim(0.40, 0.80)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading valid 15-class experiments...")
    df = load_valid_data()

    if df.empty:
        print("No valid data found!")
        return

    attack_runs = len(df[df["adv_pct"] > 0])
    print(f"Loaded {len(df)} valid experiments ({attack_runs} with adversaries)")

    output_path = OUTPUT_DIR / "obj1_attack_resilience.png"
    plot_attack_resilience(df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
