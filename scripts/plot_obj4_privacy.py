#!/usr/bin/env python3
"""
Objective 4: Privacy-Utility Tradeoff in IIoT Federated IDS

Shows the cost of differential privacy on model performance.

Based on 16 DP experiments + baselines.
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
    """Load all valid 15-class experiments with DP info."""
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
        config["dp_enabled"] = int(m.group(1)) if m else 0

        m = re.search(r"seed(\d+)", run_dir.name)
        config["seed"] = int(m.group(1)) if m else 0

        client_file = run_dir / "client_0_metrics.csv"
        if client_file.exists():
            try:
                df = pd.read_csv(client_file)
                if df["n_classes"].iloc[0] != 15:
                    continue

                config["final_f1"] = df["macro_f1_after"].iloc[-1]
                config["final_acc"] = df["acc_after"].iloc[-1] if "acc_after" in df.columns else None

                # Get DP epsilon if available
                if "dp_epsilon" in df.columns:
                    eps = df["dp_epsilon"].iloc[-1]
                    if pd.notna(eps):
                        config["dp_epsilon"] = eps

                data.append(config)
            except Exception:
                pass

    return pd.DataFrame(data)


def plot_privacy_utility(df: pd.DataFrame, output_path: Path):
    """Generate the 4-panel privacy-utility figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Objective 4: Privacy-Utility Tradeoff with Differential Privacy",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = {"No DP": "#1f77b4", "With DP": "#d62728"}

    # Filter to benign, no personalization
    benign = df[(df["adv_pct"] == 0) & (df["pers_epochs"] == 0)]
    fedavg = benign[benign["aggregation"] == "FedAvg"]

    no_dp = fedavg[fedavg["dp_enabled"] == 0]
    with_dp = fedavg[fedavg["dp_enabled"] == 1]

    # Panel A: F1 Comparison (No DP vs With DP)
    ax1 = axes[0, 0]

    compare_data = [
        {"condition": "No DP", "f1_mean": no_dp["final_f1"].mean(), 
         "f1_sem": no_dp["final_f1"].sem(), "n": len(no_dp), "color": colors["No DP"]},
        {"condition": "With DP", "f1_mean": with_dp["final_f1"].mean(),
         "f1_sem": with_dp["final_f1"].sem(), "n": len(with_dp), "color": colors["With DP"]},
    ]
    comp_df = pd.DataFrame(compare_data)

    bars = ax1.bar(
        range(len(comp_df)),
        comp_df["f1_mean"],
        yerr=1.96 * comp_df["f1_sem"],
        color=comp_df["color"].tolist(),
        capsize=4,
    )

    ax1.set_xticks(range(len(comp_df)))
    ax1.set_xticklabels(comp_df["condition"])
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("A) Performance: No DP vs With DP", fontweight="bold")
    ax1.set_ylim(0.50, 0.80)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add sample sizes and difference
    for i, row in comp_df.iterrows():
        ax1.annotate(f"n={row['n']}", xy=(i, row["f1_mean"] + 1.96*row["f1_sem"] + 0.01),
                     ha="center", fontsize=10)

    if len(no_dp) > 0 and len(with_dp) > 0:
        diff = (with_dp["final_f1"].mean() - no_dp["final_f1"].mean()) * 100
        ax1.annotate(
            f"Privacy cost: {diff:.1f}%",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Panel B: Distribution Comparison
    ax2 = axes[0, 1]

    if len(no_dp) > 0:
        sns.kdeplot(no_dp["final_f1"], ax=ax2, color=colors["No DP"], 
                    label=f"No DP (n={len(no_dp)})", fill=True, alpha=0.3)
    if len(with_dp) > 0:
        sns.kdeplot(with_dp["final_f1"], ax=ax2, color=colors["With DP"],
                    label=f"With DP (n={len(with_dp)})", fill=True, alpha=0.3)

    ax2.set_xlabel("Macro F1 Score")
    ax2.set_ylabel("Density")
    ax2.set_title("B) F1 Score Distributions", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel C: Privacy Cost by Heterogeneity
    ax3 = axes[1, 0]

    # Compare DP impact at different alpha levels
    privacy_cost = []
    for alpha in sorted(fedavg["alpha"].unique()):
        if alpha > 10:
            continue
        no_dp_alpha = fedavg[(fedavg["alpha"] == alpha) & (fedavg["dp_enabled"] == 0)]
        dp_alpha = fedavg[(fedavg["alpha"] == alpha) & (fedavg["dp_enabled"] == 1)]

        if len(no_dp_alpha) > 0 and len(dp_alpha) > 0:
            cost = (no_dp_alpha["final_f1"].mean() - dp_alpha["final_f1"].mean()) * 100
            privacy_cost.append({
                "alpha": alpha,
                "cost": cost,
                "no_dp_n": len(no_dp_alpha),
                "dp_n": len(dp_alpha),
            })

    if len(privacy_cost) > 0:
        cost_df = pd.DataFrame(privacy_cost)
        ax3.bar(range(len(cost_df)), cost_df["cost"], color=colors["With DP"])
        ax3.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax3.set_xticks(range(len(cost_df)))
        ax3.set_xticklabels([f"$\\alpha$={a}" for a in cost_df["alpha"]])
        ax3.set_ylabel("Privacy Cost (% F1 reduction)")
        ax3.set_title("C) Privacy Cost by Heterogeneity Level", fontweight="bold")
    else:
        # Show overall comparison with error bars
        ax3.text(0.5, 0.5, "Limited DP data\nacross alpha levels",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("C) Privacy Cost by Heterogeneity Level", fontweight="bold")

    ax3.grid(True, alpha=0.3, axis="y")

    # Panel D: Summary Statistics
    ax4 = axes[1, 1]

    summary_text = "PRIVACY-UTILITY SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"

    summary_text += f"Without Differential Privacy:\n"
    summary_text += f"  N = {len(no_dp)}\n"
    summary_text += f"  F1 = {no_dp['final_f1'].mean():.4f} (+/- {no_dp['final_f1'].std():.4f})\n\n"

    summary_text += f"With Differential Privacy:\n"
    summary_text += f"  N = {len(with_dp)}\n"
    if len(with_dp) > 0:
        summary_text += f"  F1 = {with_dp['final_f1'].mean():.4f} (+/- {with_dp['final_f1'].std():.4f})\n"
        if "dp_epsilon" in with_dp.columns:
            eps_vals = with_dp["dp_epsilon"].dropna()
            if len(eps_vals) > 0:
                summary_text += f"  Epsilon = {eps_vals.mean():.2f}\n"
    summary_text += "\n"

    # Statistical test
    if len(no_dp) > 0 and len(with_dp) > 0:
        t, p = stats.ttest_ind(with_dp["final_f1"], no_dp["final_f1"])
        diff = with_dp["final_f1"].mean() - no_dp["final_f1"].mean()
        summary_text += f"Statistical Test:\n"
        summary_text += f"  Difference: {diff*100:+.2f}%\n"
        summary_text += f"  p-value: {p:.4f}\n"
        if p < 0.05:
            summary_text += "  Result: SIGNIFICANT privacy cost\n"
        else:
            summary_text += "  Result: Not significant\n"

        # Practical interpretation
        summary_text += "\n" + "=" * 40 + "\n"
        summary_text += "INTERPRETATION:\n"
        if abs(diff) < 0.05:
            summary_text += "Privacy cost is ACCEPTABLE (<5%)\n"
            summary_text += "DP can be used with minimal impact"
        else:
            summary_text += f"Privacy cost is NOTABLE ({abs(diff)*100:.1f}%)\n"
            summary_text += "Consider privacy vs utility tradeoff"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
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

    dp_runs = len(df[df["dp_enabled"] > 0])
    print(f"Loaded {len(df)} valid experiments ({dp_runs} with DP)")

    output_path = OUTPUT_DIR / "obj4_privacy_utility.png"
    plot_privacy_utility(df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
