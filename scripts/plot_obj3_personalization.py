#!/usr/bin/env python3
"""
Objective 3: Personalization Benefit in IIoT Federated IDS

Shows how local fine-tuning after federated training improves 
individual client performance.

Based on 30 personalization experiments + baselines.
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
    """Load all valid 15-class experiments with personalization info."""
    data = []
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
        config["adv_pct"] = int(m.group(1)) if m else 0

        m = re.search(r"pers(\d+)", run_dir.name)
        config["pers_epochs"] = int(m.group(1)) if m else 0

        m = re.search(r"dp(\d+)", run_dir.name)
        config["dp"] = int(m.group(1)) if m else 0

        m = re.search(r"seed(\d+)", run_dir.name)
        config["seed"] = int(m.group(1)) if m else 0

        # Load both global and personalized metrics if available
        client_file = run_dir / "client_0_metrics.csv"
        if client_file.exists():
            try:
                df = pd.read_csv(client_file)
                if df["n_classes"].iloc[0] != 15:
                    continue

                config["final_f1"] = df["macro_f1_after"].iloc[-1]

                # Check for personalized F1
                if "macro_f1_personalized" in df.columns:
                    pers_f1 = df["macro_f1_personalized"].iloc[-1]
                    if pd.notna(pers_f1) and pers_f1 > 0:
                        config["pers_f1"] = pers_f1

                data.append(config)
            except Exception:
                pass

    return pd.DataFrame(data)


def plot_personalization(df: pd.DataFrame, output_path: Path):
    """Generate the 4-panel personalization figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Objective 3: Personalization Improves Local Client Performance",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = {"Global": "#1f77b4", "Personalized": "#2ca02c", "FedAvg": "#1f77b4", "FedProx": "#ff7f0e"}

    # Filter to benign, no DP
    benign = df[(df["adv_pct"] == 0) & (df["dp"] == 0)]

    # Panel A: Global vs Personalized F1 by Personalization Epochs
    ax1 = axes[0, 0]

    pers_levels = [0, 3, 5]
    global_means = []
    global_sems = []
    pers_means = []
    pers_sems = []
    counts = []

    for epochs in pers_levels:
        subset = benign[benign["pers_epochs"] == epochs]
        if len(subset) > 0:
            global_means.append(subset["final_f1"].mean())
            global_sems.append(subset["final_f1"].sem())
            if "pers_f1" in subset.columns and epochs > 0:
                pers_vals = subset["pers_f1"].dropna()
                if len(pers_vals) > 0:
                    pers_means.append(pers_vals.mean())
                    pers_sems.append(pers_vals.sem())
                else:
                    pers_means.append(np.nan)
                    pers_sems.append(np.nan)
            else:
                pers_means.append(np.nan)
                pers_sems.append(np.nan)
            counts.append(len(subset))
        else:
            global_means.append(np.nan)
            global_sems.append(np.nan)
            pers_means.append(np.nan)
            pers_sems.append(np.nan)
            counts.append(0)

    x = np.arange(len(pers_levels))
    width = 0.35

    ax1.bar(x - width/2, global_means, width, yerr=[1.96*s for s in global_sems],
            label="Global Model", color=colors["Global"], capsize=4)
    
    # Only plot personalized where we have data
    pers_valid = [p if not np.isnan(p) else 0 for p in pers_means]
    pers_sem_valid = [s if not np.isnan(s) else 0 for s in pers_sems]
    ax1.bar(x + width/2, pers_valid, width, yerr=[1.96*s for s in pers_sem_valid],
            label="Personalized Model", color=colors["Personalized"], capsize=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{e} epochs" for e in pers_levels])
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("A) Effect of Personalization Epochs", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.60, 0.80)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add sample sizes
    for i, n in enumerate(counts):
        ax1.annotate(f"n={n}", xy=(i, 0.61), ha="center", fontsize=9)

    # Panel B: Personalization Gain Distribution
    ax2 = axes[0, 1]

    pers_df = benign[(benign["pers_epochs"] > 0) & (benign["pers_f1"].notna())]
    if len(pers_df) > 0:
        pers_df = pers_df.copy()
        pers_df["gain"] = (pers_df["pers_f1"] - pers_df["final_f1"]) * 100

        sns.histplot(pers_df["gain"], kde=True, ax=ax2, color=colors["Personalized"])
        ax2.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax2.axvline(pers_df["gain"].mean(), color="red", linestyle="-", linewidth=2,
                    label=f"Mean: {pers_df['gain'].mean():.2f}%")

        ax2.set_xlabel("Personalization Gain (%)")
        ax2.set_ylabel("Count")
        ax2.set_title("B) Distribution of Personalization Gains", fontweight="bold")
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "No personalization data\navailable", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title("B) Distribution of Personalization Gains", fontweight="bold")

    # Panel C: Personalization by Heterogeneity Level
    ax3 = axes[1, 0]

    # Compare personalization benefit at different alpha levels
    pers_by_alpha = []
    for alpha in sorted(benign["alpha"].unique()):
        if alpha > 10:
            continue
        subset = benign[(benign["alpha"] == alpha) & (benign["pers_epochs"] > 0)]
        if len(subset) > 0 and "pers_f1" in subset.columns:
            pers_vals = subset["pers_f1"].dropna()
            global_vals = subset["final_f1"]
            if len(pers_vals) > 0:
                gain = (pers_vals.mean() - global_vals.mean()) * 100
                pers_by_alpha.append({
                    "alpha": alpha,
                    "gain": gain,
                    "n": len(pers_vals),
                })

    if len(pers_by_alpha) > 0:
        alpha_df = pd.DataFrame(pers_by_alpha)
        ax3.bar(range(len(alpha_df)), alpha_df["gain"], color=colors["Personalized"])
        ax3.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax3.set_xticks(range(len(alpha_df)))
        ax3.set_xticklabels([f"$\\alpha$={a}" for a in alpha_df["alpha"]])
        ax3.set_ylabel("Personalization Gain (%)")
        ax3.set_title("C) Personalization Benefit by Heterogeneity", fontweight="bold")
    else:
        # Show baseline comparison instead
        baseline = benign[benign["pers_epochs"] == 0]
        pers = benign[benign["pers_epochs"] > 0]
        
        compare_data = [
            {"condition": "No Pers (baseline)", "f1": baseline["final_f1"].mean(), 
             "sem": baseline["final_f1"].sem(), "n": len(baseline)},
            {"condition": "With Pers (3-5 ep)", "f1": pers["final_f1"].mean(),
             "sem": pers["final_f1"].sem(), "n": len(pers)},
        ]
        comp_df = pd.DataFrame(compare_data)
        
        ax3.bar(range(len(comp_df)), comp_df["f1"], yerr=1.96*comp_df["sem"],
                color=[colors["Global"], colors["Personalized"]], capsize=4)
        ax3.set_xticks(range(len(comp_df)))
        ax3.set_xticklabels(comp_df["condition"])
        ax3.set_ylabel("Macro F1 Score")
        ax3.set_title("C) Baseline vs Personalized Runs", fontweight="bold")
        ax3.set_ylim(0.60, 0.80)

    ax3.grid(True, alpha=0.3, axis="y")

    # Panel D: Summary Statistics
    ax4 = axes[1, 1]

    # Create summary table as text
    baseline = benign[benign["pers_epochs"] == 0]
    pers3 = benign[benign["pers_epochs"] == 3]
    pers5 = benign[benign["pers_epochs"] == 5]

    summary_text = "PERSONALIZATION SUMMARY\n"
    summary_text += "=" * 35 + "\n\n"

    summary_text += f"Baseline (0 epochs):\n"
    summary_text += f"  N = {len(baseline)}\n"
    summary_text += f"  F1 = {baseline['final_f1'].mean():.4f} (+/- {baseline['final_f1'].std():.4f})\n\n"

    if len(pers3) > 0:
        summary_text += f"Personalized (3 epochs):\n"
        summary_text += f"  N = {len(pers3)}\n"
        summary_text += f"  F1 = {pers3['final_f1'].mean():.4f} (+/- {pers3['final_f1'].std():.4f})\n\n"

    if len(pers5) > 0:
        summary_text += f"Personalized (5 epochs):\n"
        summary_text += f"  N = {len(pers5)}\n"
        summary_text += f"  F1 = {pers5['final_f1'].mean():.4f} (+/- {pers5['final_f1'].std():.4f})\n\n"

    # Statistical test
    if len(baseline) > 0 and len(pers3) > 0:
        t, p = stats.ttest_ind(pers3["final_f1"], baseline["final_f1"])
        diff = pers3["final_f1"].mean() - baseline["final_f1"].mean()
        summary_text += f"Statistical Test (3 ep vs baseline):\n"
        summary_text += f"  Difference: {diff*100:+.2f}%\n"
        summary_text += f"  p-value: {p:.4f}\n"
        if p < 0.05:
            summary_text += "  Result: SIGNIFICANT"
        else:
            summary_text += "  Result: Not significant"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax4.axis("off")
    ax4.set_title("D) Statistical Summary", fontweight="bold")

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

    pers_runs = len(df[df["pers_epochs"] > 0])
    print(f"Loaded {len(df)} valid experiments ({pers_runs} with personalization)")

    output_path = OUTPUT_DIR / "obj3_personalization.png"
    plot_personalization(df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
