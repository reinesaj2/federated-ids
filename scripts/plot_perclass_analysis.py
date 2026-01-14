#!/usr/bin/env python3
"""
Per-Class F1 Analysis Plots for Chapter 4.

Generates publication-quality plots analyzing per-class macro F1 scores:
1. Per-class F1 convergence lines over rounds (like perclass_lines_alpha*.png)
2. Per-class F1 heatmaps by aggregator (like perclass_heatmap_alpha*.png)
3. Per-class F1 boxplots showing seed variability
4. Per-class F1 bars with minority class highlighting (like perclass_bars_alpha*.png)
5. Comprehensive multi-panel figures (like obj1_robustness_comprehensive.png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

COLORBLIND_PALETTE = [
    "#0173B2",  # Blue (FedAvg)
    "#DE8F05",  # Orange (Krum)
    "#029E73",  # Green (Bulyan)
    "#CC78BC",  # Purple (Median)
    "#ECE133",  # Yellow
]

AGGREGATOR_LABELS = {
    "fedavg": "FedAvg",
    "bulyan": "Bulyan",
    "krum": "Krum",
    "median": "Median",
    "fedprox": "FedProx",
}

AGGREGATOR_ORDER = ["fedavg", "krum", "bulyan", "median"]
AGGREGATOR_COLORS = {agg: COLORBLIND_PALETTE[i] for i, agg in enumerate(AGGREGATOR_ORDER)}

IIOT_MINORITY_CLASSES = [
    "RANSOMWARE", "FINGERPRINTING", "XSS", "UPLOADING", "PORT_SCANNING", "PASSWORD"
]

CIC_MINORITY_CLASSES = [
    "Heartbleed", "Web Attack Sql Injection", "Infiltration"
]

UNSW_MINORITY_CLASSES = [
    "WORMS", "SHELLCODE", "BACKDOOR", "ANALYSIS"
]

DATASET_CONFIG = {
    "iiot": {
        "label": "Edge-IIoTset",
        "color": COLORBLIND_PALETTE[0],
        "minority": IIOT_MINORITY_CLASSES,
    },
    "cic": {
        "label": "CIC-IDS2017",
        "color": COLORBLIND_PALETTE[1],
        "minority": CIC_MINORITY_CLASSES,
    },
    "unsw": {
        "label": "UNSW-NB15",
        "color": COLORBLIND_PALETTE[2],
        "minority": UNSW_MINORITY_CLASSES,
    },
}

ALPHA_VALUES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def setup_thesis_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_perclass_lines(conv_df: pd.DataFrame, dataset: str, alpha: float, adv: int, output_dir: Path):
    """Plot F1 convergence over rounds for each class."""
    setup_thesis_style()
    
    subset = conv_df[(conv_df["dataset"] == dataset) & 
                     (conv_df["alpha"] == alpha) & 
                     (conv_df["adv_pct"] == adv) &
                     (conv_df["mu"] == 0.0)]
    
    if subset.empty:
        print(f"    No data for {dataset} alpha={alpha} adv={adv}")
        return
    
    classes = sorted(subset["class_name"].unique())
    n_classes = len(classes)
    
    if n_classes == 0:
        return
    
    ncols = 3
    nrows = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    fig.suptitle(
        f"Per-Class F1 Convergence Over Rounds (alpha={alpha}, adv_pct={adv})\n{DATASET_CONFIG[dataset]['label']}",
        fontsize=14, fontweight="bold", y=1.02
    )
    
    for idx, class_name in enumerate(classes):
        ax = axes[idx]
        class_data = subset[subset["class_name"] == class_name]
        
        for agg in AGGREGATOR_ORDER:
            agg_data = class_data[class_data["aggregator"] == agg]
            if agg_data.empty:
                continue
            
            rounds = agg_data["round"].values
            means = agg_data["f1_mean"].values
            stds = agg_data["f1_std"].values
            
            ax.plot(rounds, means, marker="o", markersize=4, linewidth=2,
                    label=AGGREGATOR_LABELS[agg], color=AGGREGATOR_COLORS[agg])
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.2, color=AGGREGATOR_COLORS[agg])
        
        ax.set_title(class_name.upper(), fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_classes, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    fname = f"perclass_lines_{dataset}_alpha{alpha}_adv{adv}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def plot_perclass_heatmap(final_df: pd.DataFrame, dataset: str, alpha: float, adv: int, output_dir: Path):
    """Plot per-class F1 heatmap by aggregator."""
    setup_thesis_style()
    
    subset = final_df[(final_df["dataset"] == dataset) & 
                      (final_df["alpha"] == alpha) & 
                      (final_df["adv_pct"] == adv) &
                      (final_df["mu"] == 0.0)]
    
    if subset.empty:
        print(f"    No data for heatmap {dataset} alpha={alpha} adv={adv}")
        return
    
    pivot = subset.pivot_table(
        index="aggregator",
        columns="class_name",
        values="f1_mean",
        aggfunc="mean"
    )
    
    pivot = pivot.reindex([a for a in AGGREGATOR_ORDER if a in pivot.index])
    classes = sorted(pivot.columns.tolist())
    pivot = pivot[classes]
    
    if pivot.empty:
        return
    
    minority_classes = DATASET_CONFIG[dataset]["minority"]
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([AGGREGATOR_LABELS.get(a, a) for a in pivot.index], fontsize=10)
    
    for i in range(len(pivot.index)):
        for j in range(len(classes)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color)
    
    for j, class_name in enumerate(classes):
        if class_name in minority_classes:
            rect = mpatches.Rectangle(
                (j - 0.5, -0.5), 1, len(pivot.index),
                linewidth=2, edgecolor="red", facecolor="none",
                linestyle="--"
            )
            ax.add_patch(rect)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean F1 Score", fontsize=10)
    
    ax.set_xlabel("Attack Class", fontsize=11)
    ax.set_ylabel("Aggregator", fontsize=11)
    ax.set_title(
        f"Per-Class F1 Performance Heatmap (alpha={alpha}, adv_pct={adv})\n"
        f"{DATASET_CONFIG[dataset]['label']} - Red boxes indicate minority attack classes",
        fontsize=12, fontweight="bold"
    )
    
    plt.tight_layout()
    fname = f"perclass_heatmap_{dataset}_alpha{alpha}_adv{adv}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def plot_perclass_bars(final_df: pd.DataFrame, dataset: str, alpha: float, adv: int, output_dir: Path):
    """Plot per-class F1 bars with minority highlighting."""
    setup_thesis_style()
    
    subset = final_df[(final_df["dataset"] == dataset) & 
                      (final_df["alpha"] == alpha) & 
                      (final_df["adv_pct"] == adv) &
                      (final_df["mu"] == 0.0)]
    
    if subset.empty:
        print(f"    No data for bars {dataset} alpha={alpha} adv={adv}")
        return
    
    aggregators = [a for a in AGGREGATOR_ORDER if a in subset["aggregator"].unique()]
    if len(aggregators) == 0:
        return
    
    ncols = 2
    nrows = (len(aggregators) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten() if len(aggregators) > 1 else [axes]
    
    fig.suptitle(
        f"Per-Class F1 Scores with Minority Class Highlighting (alpha={alpha}, adv_pct={adv})\n"
        f"{DATASET_CONFIG[dataset]['label']}",
        fontsize=14, fontweight="bold", y=1.02
    )
    
    minority_classes = DATASET_CONFIG[dataset]["minority"]
    
    for idx, agg in enumerate(aggregators):
        ax = axes[idx]
        agg_data = subset[subset["aggregator"] == agg].sort_values("class_name")
        
        classes = agg_data["class_name"].tolist()
        means = agg_data["f1_mean"].values
        stds = agg_data["f1_std"].values
        
        colors = ["red" if c in minority_classes else "steelblue" for c in classes]
        
        x = np.arange(len(classes))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors,
               edgecolor="black", linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 Score (95% CI)")
        ax.set_ylim(0, 1.1)
        ax.set_title(AGGREGATOR_LABELS[agg], fontweight="bold", fontsize=12)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3, axis="y")
        
        minority_patch = mpatches.Patch(color="red", label="Minority Class")
        majority_patch = mpatches.Patch(color="steelblue", label="Majority Class")
        ax.legend(handles=[minority_patch, majority_patch], loc="upper right", fontsize=8)
    
    for idx in range(len(aggregators), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    fname = f"perclass_bars_{dataset}_alpha{alpha}_adv{adv}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def plot_comprehensive_perclass_figure(final_df: pd.DataFrame, conv_df: pd.DataFrame, 
                                       dataset: str, output_dir: Path):
    """Create comprehensive per-class analysis figure (6-panel layout)."""
    setup_thesis_style()
    
    fig = plt.figure(figsize=(20, 16))
    
    fig.suptitle(
        f"Per-Class F1 Comprehensive Analysis: {DATASET_CONFIG[dataset]['label']}",
        fontsize=16, fontweight="bold", y=0.98
    )
    
    minority_classes = DATASET_CONFIG[dataset]["minority"]
    
    benign_final = final_df[(final_df["dataset"] == dataset) & 
                            (final_df["adv_pct"] == 0) & 
                            (final_df["mu"] == 0.0)]
    
    benign_conv = conv_df[(conv_df["dataset"] == dataset) & 
                          (conv_df["adv_pct"] == 0) & 
                          (conv_df["mu"] == 0.0)]
    
    if benign_final.empty:
        print(f"    No data for comprehensive figure {dataset}")
        plt.close(fig)
        return
    
    ax1 = fig.add_subplot(2, 3, 1)
    pivot = benign_final.pivot_table(index="aggregator", columns="class_name", 
                                      values="f1_mean", aggfunc="mean")
    pivot = pivot.reindex([a for a in AGGREGATOR_ORDER if a in pivot.index])
    classes = sorted(pivot.columns.tolist())
    pivot = pivot[classes]
    
    if not pivot.empty:
        im = ax1.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
        ax1.set_xticks(range(len(classes)))
        ax1.set_yticks(range(len(pivot.index)))
        ax1.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax1.set_yticklabels([AGGREGATOR_LABELS.get(a, a) for a in pivot.index])
        
        for i in range(len(pivot.index)):
            for j in range(len(classes)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > 0.5 else "black"
                    ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)
        
        fig.colorbar(im, ax=ax1, shrink=0.6)
    ax1.set_title("(A) Per-Class Heatmap (Benign)", fontweight="bold")
    
    ax2 = fig.add_subplot(2, 3, 2)
    class_means = benign_final.groupby("class_name")["f1_mean"].mean().sort_values()
    colors = ["red" if c in minority_classes else "steelblue" for c in class_means.index]
    ax2.barh(range(len(class_means)), class_means.values, color=colors)
    ax2.set_yticks(range(len(class_means)))
    ax2.set_yticklabels(class_means.index, fontsize=8)
    ax2.set_xlabel("Mean F1 Score")
    ax2.set_xlim(0, 1.05)
    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("(B) Ranked Class Performance", fontweight="bold")
    
    ax3 = fig.add_subplot(2, 3, 3)
    minority_data = benign_final[benign_final["class_name"].isin(minority_classes)]
    majority_data = benign_final[~benign_final["class_name"].isin(minority_classes)]
    
    agg_minority = minority_data.groupby("aggregator")["f1_mean"].mean()
    agg_majority = majority_data.groupby("aggregator")["f1_mean"].mean()
    
    x = np.arange(len(AGGREGATOR_ORDER))
    width = 0.35
    
    for i, agg in enumerate(AGGREGATOR_ORDER):
        if agg in agg_minority.index:
            ax3.bar(i - width/2, agg_minority.get(agg, 0), width, color="red", 
                    label="Minority" if i == 0 else "")
        if agg in agg_majority.index:
            ax3.bar(i + width/2, agg_majority.get(agg, 0), width, color="steelblue", 
                    label="Majority" if i == 0 else "")
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([AGGREGATOR_LABELS[a] for a in AGGREGATOR_ORDER], rotation=15)
    ax3.set_ylabel("Mean F1 Score")
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper right")
    ax3.set_title("(C) Minority vs Majority Classes", fontweight="bold")
    
    ax4 = fig.add_subplot(2, 3, 4)
    worst_classes = class_means.head(5).index.tolist()
    
    for class_name in worst_classes:
        class_subset = benign_conv[benign_conv["class_name"] == class_name]
        round_means = class_subset.groupby("round")["f1_mean"].mean()
        if not round_means.empty:
            ax4.plot(round_means.index, round_means.values, marker="o", markersize=3,
                     linewidth=2, label=class_name)
    
    ax4.set_xlabel("Communication Round")
    ax4.set_ylabel("F1 Score")
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc="best", fontsize=7)
    ax4.set_title("(D) Worst 5 Classes Convergence", fontweight="bold")
    
    ax5 = fig.add_subplot(2, 3, 5)
    alpha_class_perf = []
    for alpha in ALPHA_VALUES:
        alpha_data = final_df[(final_df["dataset"] == dataset) & 
                              (final_df["alpha"] == alpha) & 
                              (final_df["adv_pct"] == 0)]
        
        if not alpha_data.empty:
            minority_f1 = alpha_data[alpha_data["class_name"].isin(minority_classes)]["f1_mean"].mean()
            majority_f1 = alpha_data[~alpha_data["class_name"].isin(minority_classes)]["f1_mean"].mean()
            alpha_class_perf.append({"alpha": alpha, "minority": minority_f1, "majority": majority_f1})
    
    if alpha_class_perf:
        perf_df = pd.DataFrame(alpha_class_perf)
        ax5.plot(perf_df["alpha"], perf_df["minority"], "ro-", linewidth=2, label="Minority Classes")
        ax5.plot(perf_df["alpha"], perf_df["majority"], "bs-", linewidth=2, label="Majority Classes")
        ax5.set_xscale("log")
        ax5.set_xlabel("Dirichlet Alpha (log)")
        ax5.set_ylabel("Mean F1 Score")
        ax5.set_ylim(0, 1.05)
        ax5.legend(loc="best")
    ax5.set_title("(E) Heterogeneity Impact by Class Type", fontweight="bold")
    
    ax6 = fig.add_subplot(2, 3, 6)
    class_cv = benign_final.groupby("class_name").apply(
        lambda x: x["f1_std"].mean() / x["f1_mean"].mean() if x["f1_mean"].mean() > 0 else 0
    ).sort_values(ascending=False).head(10)
    colors = ["red" if c in minority_classes else "steelblue" for c in class_cv.index]
    ax6.barh(range(len(class_cv)), class_cv.values, color=colors)
    ax6.set_yticks(range(len(class_cv)))
    ax6.set_yticklabels(class_cv.index, fontsize=8)
    ax6.set_xlabel("Coefficient of Variation")
    ax6.set_title("(F) Most Variable Classes", fontweight="bold")
    
    plt.tight_layout()
    fname = f"perclass_comprehensive_{dataset}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def plot_cross_dataset_perclass(final_df: pd.DataFrame, output_dir: Path):
    """Compare per-class performance across all 3 datasets."""
    setup_thesis_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Cross-Dataset Per-Class Analysis: Minority Class Challenge",
        fontsize=16, fontweight="bold", y=1.02
    )
    
    benign_data = final_df[(final_df["adv_pct"] == 0) & (final_df["mu"] == 0.0)]
    
    ax = axes[0, 0]
    minority_means = []
    majority_means = []
    dataset_labels = []
    
    for dataset in ["iiot", "cic", "unsw"]:
        ds_final = benign_data[benign_data["dataset"] == dataset]
        if ds_final.empty:
            continue
        
        minority_classes = DATASET_CONFIG[dataset]["minority"]
        
        minority_f1 = ds_final[ds_final["class_name"].isin(minority_classes)]["f1_mean"].mean()
        majority_f1 = ds_final[~ds_final["class_name"].isin(minority_classes)]["f1_mean"].mean()
        
        minority_means.append(minority_f1 if not np.isnan(minority_f1) else 0)
        majority_means.append(majority_f1 if not np.isnan(majority_f1) else 0)
        dataset_labels.append(DATASET_CONFIG[dataset]["label"])
    
    if dataset_labels:
        x = np.arange(len(dataset_labels))
        width = 0.35
        ax.bar(x - width/2, minority_means, width, color="red", label="Minority")
        ax.bar(x + width/2, majority_means, width, color="steelblue", label="Majority")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, rotation=15)
        ax.set_ylabel("Mean F1 Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
    ax.set_title("(A) Minority vs Majority Gap", fontweight="bold")
    
    ax = axes[0, 1]
    if minority_means and majority_means:
        gaps = [maj - mino for mino, maj in zip(minority_means, majority_means)]
        colors = [DATASET_CONFIG[ds]["color"] for ds in ["iiot", "cic", "unsw"] 
                  if DATASET_CONFIG[ds]["label"] in dataset_labels]
        ax.bar(range(len(gaps)), gaps, color=colors[:len(gaps)])
        ax.set_xticks(range(len(dataset_labels)))
        ax.set_xticklabels(dataset_labels, rotation=15)
        ax.set_ylabel("F1 Gap (Majority - Minority)")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1)
    ax.set_title("(B) Class Imbalance Impact", fontweight="bold")
    
    ax = axes[0, 2]
    summary_text = []
    for dataset in ["iiot", "cic", "unsw"]:
        ds_final = benign_data[benign_data["dataset"] == dataset]
        if ds_final.empty:
            continue
        minority_classes = DATASET_CONFIG[dataset]["minority"]
        
        worst = ds_final.groupby("class_name")["f1_mean"].mean().nsmallest(3)
        summary_text.append(f"{DATASET_CONFIG[dataset]['label']}:")
        for cls, f1 in worst.items():
            marker = "*" if cls in minority_classes else ""
            summary_text.append(f"  {cls}{marker}: {f1:.3f}")
        summary_text.append("")
    
    ax.text(0.1, 0.9, "\n".join(summary_text), transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace")
    ax.set_title("(C) Worst Performing Classes", fontweight="bold")
    ax.axis("off")
    
    for idx, dataset in enumerate(["iiot", "cic", "unsw"]):
        ax = axes[1, idx]
        ds_final = benign_data[benign_data["dataset"] == dataset]
        if ds_final.empty:
            ax.axis("off")
            continue
        
        minority_classes = DATASET_CONFIG[dataset]["minority"]
        
        class_means = ds_final.groupby("class_name")["f1_mean"].mean().sort_values()
        colors = ["red" if c in minority_classes else "steelblue" for c in class_means.index]
        
        ax.barh(range(len(class_means)), class_means.values, color=colors)
        ax.set_yticks(range(len(class_means)))
        ax.set_yticklabels(class_means.index, fontsize=7)
        ax.set_xlabel("Mean F1 Score")
        ax.set_xlim(0, 1.05)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"(D{idx+1}) {DATASET_CONFIG[dataset]['label']}", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_dir / "perclass_cross_dataset_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "perclass_cross_dataset_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    Saved: perclass_cross_dataset_analysis.png/pdf")


def plot_attack_impact_perclass(final_df: pd.DataFrame, dataset: str, output_dir: Path):
    """Show how attacks affect minority vs majority classes."""
    setup_thesis_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Attack Impact on Per-Class Performance: {DATASET_CONFIG[dataset]['label']}",
        fontsize=16, fontweight="bold", y=1.02
    )
    
    minority_classes = DATASET_CONFIG[dataset]["minority"]
    ds_data = final_df[(final_df["dataset"] == dataset) & (final_df["mu"] == 0.0)]
    
    ax = axes[0, 0]
    adv_levels = [0, 10, 20, 30]
    for agg in AGGREGATOR_ORDER:
        agg_data = ds_data[ds_data["aggregator"] == agg]
        minority_means = []
        for adv in adv_levels:
            adv_data = agg_data[agg_data["adv_pct"] == adv]
            minority_f1 = adv_data[adv_data["class_name"].isin(minority_classes)]["f1_mean"].mean()
            minority_means.append(minority_f1 if not np.isnan(minority_f1) else 0)
        if any(m > 0 for m in minority_means):
            ax.plot(adv_levels, minority_means, marker="o", linewidth=2, 
                    label=AGGREGATOR_LABELS[agg], color=AGGREGATOR_COLORS[agg])
    ax.set_xlabel("Byzantine Clients (%)")
    ax.set_ylabel("Mean F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.set_title("(A) Minority Classes Under Attack", fontweight="bold")
    
    ax = axes[0, 1]
    for agg in AGGREGATOR_ORDER:
        agg_data = ds_data[ds_data["aggregator"] == agg]
        majority_means = []
        for adv in adv_levels:
            adv_data = agg_data[agg_data["adv_pct"] == adv]
            majority_f1 = adv_data[~adv_data["class_name"].isin(minority_classes)]["f1_mean"].mean()
            majority_means.append(majority_f1 if not np.isnan(majority_f1) else 0)
        if any(m > 0 for m in majority_means):
            ax.plot(adv_levels, majority_means, marker="s", linewidth=2, 
                    label=AGGREGATOR_LABELS[agg], color=AGGREGATOR_COLORS[agg])
    ax.set_xlabel("Byzantine Clients (%)")
    ax.set_ylabel("Mean F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.set_title("(B) Majority Classes Under Attack", fontweight="bold")
    
    ax = axes[1, 0]
    benign = ds_data[ds_data["adv_pct"] == 0]
    attack30 = ds_data[ds_data["adv_pct"] == 30]
    
    degradation_data = []
    for class_name in ds_data["class_name"].unique():
        benign_f1 = benign[benign["class_name"] == class_name]["f1_mean"].mean()
        attack_f1 = attack30[attack30["class_name"] == class_name]["f1_mean"].mean()
        if benign_f1 > 0 and not np.isnan(attack_f1):
            deg = ((benign_f1 - attack_f1) / benign_f1) * 100
            degradation_data.append({"class": class_name, "degradation": deg,
                                     "is_minority": class_name in minority_classes})
    
    if degradation_data:
        deg_df = pd.DataFrame(degradation_data).sort_values("degradation", ascending=False)
        colors = ["red" if row["is_minority"] else "steelblue" 
                  for _, row in deg_df.iterrows()]
        ax.barh(range(len(deg_df)), deg_df["degradation"].values, color=colors)
        ax.set_yticks(range(len(deg_df)))
        ax.set_yticklabels(deg_df["class"].values, fontsize=7)
        ax.set_xlabel("F1 Degradation (%)")
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=1)
    ax.set_title("(C) Per-Class Degradation (30% Attack)", fontweight="bold")
    
    ax = axes[1, 1]
    summary_lines = []
    for adv in [0, 10, 30]:
        adv_data = ds_data[ds_data["adv_pct"] == adv]
        minority_f1 = adv_data[adv_data["class_name"].isin(minority_classes)]["f1_mean"].mean()
        majority_f1 = adv_data[~adv_data["class_name"].isin(minority_classes)]["f1_mean"].mean()
        gap = majority_f1 - minority_f1 if not np.isnan(minority_f1) and not np.isnan(majority_f1) else 0
        summary_lines.append(f"Adv={adv}%:")
        summary_lines.append(f"  Minority: {minority_f1:.3f}")
        summary_lines.append(f"  Majority: {majority_f1:.3f}")
        summary_lines.append(f"  Gap: {gap:.3f}")
        summary_lines.append("")
    
    ax.text(0.1, 0.9, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, va="top", fontfamily="monospace")
    ax.set_title("(D) Summary Statistics", fontweight="bold")
    ax.axis("off")
    
    plt.tight_layout()
    fname = f"perclass_attack_impact_{dataset}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def main():
    base_path = Path("/Users/abrahamreines/Documents/Thesis")
    final_csv = base_path / "cluster-experiments" / "perclass_final_summary.csv"
    conv_csv = base_path / "cluster-experiments" / "perclass_convergence_summary.csv"
    output_dir = base_path / "cluster-experiments" / "thesis_plots" / "chapter4" / "perclass"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PER-CLASS F1 ANALYSIS PLOTS")
    print("=" * 70)
    print(f"Final summary: {final_csv}")
    print(f"Convergence: {conv_csv}")
    print(f"Output: {output_dir}")

    if not final_csv.exists():
        print(f"\nERROR: Data file not found: {final_csv}")
        print("Run aggregate_perclass_metrics.py first.")
        return

    print("\nLoading data...")
    final_df = pd.read_csv(final_csv)
    conv_df = pd.read_csv(conv_csv)
    
    print(f"  Final summary records: {len(final_df)}")
    print(f"  Convergence records: {len(conv_df)}")
    
    for dataset in ["iiot", "cic", "unsw"]:
        ds_count = len(final_df[final_df["dataset"] == dataset])
        print(f"    {DATASET_CONFIG[dataset]['label']}: {ds_count} configurations")

    print("\n" + "=" * 70)
    print("GENERATING PER-CLASS PLOTS")
    print("=" * 70)

    key_configs = [
        (1.0, 0),
        (0.5, 0),
        (0.1, 0),
        (0.05, 0),
        (1.0, 10),
        (1.0, 30),
    ]

    for dataset in ["iiot", "cic", "unsw"]:
        print(f"\n--- {DATASET_CONFIG[dataset]['label']} ---")
        
        print("\n  Generating comprehensive figure...")
        plot_comprehensive_perclass_figure(final_df, conv_df, dataset, output_dir)
        
        print("\n  Generating attack impact figure...")
        plot_attack_impact_perclass(final_df, dataset, output_dir)
        
        for alpha, adv in key_configs:
            print(f"\n  Config: alpha={alpha}, adv={adv}")
            plot_perclass_heatmap(final_df, dataset, alpha, adv, output_dir)
            plot_perclass_bars(final_df, dataset, alpha, adv, output_dir)
            
            if adv == 0 and alpha in [1.0, 0.1, 0.05]:
                plot_perclass_lines(conv_df, dataset, alpha, adv, output_dir)

    print("\n--- Cross-Dataset Analysis ---")
    plot_cross_dataset_perclass(final_df, output_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
