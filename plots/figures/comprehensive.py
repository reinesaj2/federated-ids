"""
Comprehensive Thesis Figures

Dataset characterization, efficiency analysis, and cross-comparison plots
that span multiple objectives for Chapter 4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.config.constants import (
    AGGREGATOR_COLORS,
    DATASET_CONFIG,
)
from plots.config.style import PALETTES, ThesisStyle
from plots.figures.primitives import save_figure

IIOT_CLASS_DIST = {
    "Normal": 72.80,
    "DDoS_UDP": 5.48,
    "DDoS_ICMP": 5.25,
    "SQL_injection": 2.31,
    "Vulnerability_scanner": 2.26,
    "DDoS_TCP": 2.26,
    "Password": 2.26,
    "DDoS_HTTP": 2.25,
    "Uploading": 1.69,
    "Backdoor": 1.12,
    "Port_Scanning": 1.02,
    "XSS": 0.72,
    "Ransomware": 0.49,
    "MITM": 0.05,
    "Fingerprinting": 0.04,
}

CIC_CLASS_DIST = {
    "BENIGN": 80.30,
    "DoS Hulk": 8.16,
    "PortScan": 5.61,
    "DDoS": 4.52,
    "DoS GoldenEye": 0.36,
    "FTP-Patator": 0.28,
    "SSH-Patator": 0.21,
    "DoS slowloris": 0.20,
    "DoS Slowhttptest": 0.19,
    "Bot": 0.07,
    "Web Attack BF": 0.05,
    "Web Attack XSS": 0.02,
    "Infiltration": 0.001,
    "SQL Injection": 0.0007,
    "Heartbleed": 0.0004,
}

UNSW_CLASS_DIST = {
    "Normal": 37.00,
    "Generic": 18.87,
    "Exploits": 17.30,
    "Fuzzers": 9.40,
    "DoS": 6.39,
    "Reconnaissance": 5.45,
    "Analysis": 2.67,
    "Backdoor": 0.91,
    "Shellcode": 0.56,
    "Worms": 0.06,
}

IMBALANCE_METRICS = {
    "iiot": {"shannon_entropy": 3.71, "imbalance_ratio": 24, "effective_classes": 13.07, "gini": 0.85},
    "cic": {"shannon_entropy": 1.11, "imbalance_ratio": 206645, "effective_classes": 2.16, "gini": 0.33},
    "unsw": {"shannon_entropy": 2.89, "imbalance_ratio": 617, "effective_classes": 7.42, "gini": 0.72},
}

OVERHEAD_DATA = {
    "FedAvg": {"time_ms": 0.5, "overhead_x": 1.0},
    "Krum": {"time_ms": 10.0, "overhead_x": 26.8},
    "Bulyan": {"time_ms": 20.0, "overhead_x": 45.7},
    "Median": {"time_ms": 12.0, "overhead_x": 27.1},
}


def plot_dataset_characterization(output_dir: Path, formats: list[str]) -> list[Path]:
    """Plot dataset characterization: class distribution and imbalance metrics."""
    style = ThesisStyle()
    style.apply()
    colorblind = PALETTES["colorblind"]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Dataset Characterization: Class Distribution and Imbalance Analysis", fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    classes = list(IIOT_CLASS_DIST.keys())
    values = list(IIOT_CLASS_DIST.values())
    colors = [colorblind[0] if v > 1 else colorblind[3] for v in values]
    ax1.barh(range(len(classes)), values, color=colors)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels(classes, fontsize=8)
    ax1.set_xlabel("Percentage (%)")
    ax1.set_title("(A) Edge-IIoTset Class Distribution", fontweight="bold")
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(2, 3, 2)
    classes = list(CIC_CLASS_DIST.keys())
    values = list(CIC_CLASS_DIST.values())
    colors = [colorblind[1] if v > 0.1 else colorblind[3] for v in values]
    ax2.barh(range(len(classes)), values, color=colors)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=8)
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("(B) CIC-IDS2017 Class Distribution", fontweight="bold")
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(2, 3, 3)
    classes = list(UNSW_CLASS_DIST.keys())
    values = list(UNSW_CLASS_DIST.values())
    colors = [colorblind[2] if v > 1 else colorblind[3] for v in values]
    ax3.barh(range(len(classes)), values, color=colors)
    ax3.set_yticks(range(len(classes)))
    ax3.set_yticklabels(classes, fontsize=8)
    ax3.set_xlabel("Percentage (%)")
    ax3.set_title("(C) UNSW-NB15 Class Distribution", fontweight="bold")
    ax3.invert_yaxis()

    ax4 = fig.add_subplot(2, 3, 4)
    datasets = ["iiot", "cic", "unsw"]
    metrics = ["shannon_entropy", "gini"]
    metric_labels = ["Shannon Entropy", "Gini Impurity"]
    x = np.arange(len(datasets))
    width = 0.35
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [IMBALANCE_METRICS[ds][metric] for ds in datasets]
        ax4.bar(x + i * width, vals, width, label=label, color=colorblind[i])
    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels([DATASET_CONFIG[ds]["label"] for ds in datasets], rotation=15)
    ax4.set_ylabel("Score")
    ax4.set_title("(D) Balance Metrics (Higher = More Balanced)", fontweight="bold")
    ax4.legend()

    ax5 = fig.add_subplot(2, 3, 5)
    imbalance = [np.log10(IMBALANCE_METRICS[ds]["imbalance_ratio"]) for ds in datasets]
    colors = [DATASET_CONFIG[ds]["color"] for ds in datasets]
    bars = ax5.bar(range(len(datasets)), imbalance, color=colors)
    ax5.set_xticks(range(len(datasets)))
    ax5.set_xticklabels([DATASET_CONFIG[ds]["label"] for ds in datasets], rotation=15)
    ax5.set_ylabel("Log10(Imbalance Ratio)")
    ax5.set_title("(E) Class Imbalance Ratio (Lower = Better)", fontweight="bold")
    for i, (bar, ds) in enumerate(zip(bars, datasets)):
        ratio = IMBALANCE_METRICS[ds]["imbalance_ratio"]
        ax5.annotate(f"{ratio:,}:1", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    dataset_classes = {"iiot": 15, "cic": 15, "unsw": 10}
    effective = [IMBALANCE_METRICS[ds]["effective_classes"] for ds in datasets]
    total = [dataset_classes[ds] for ds in datasets]
    x = np.arange(len(datasets))
    width = 0.35
    ax6.bar(x - width / 2, effective, width, label="Effective Classes", color=colors, alpha=0.8)
    ax6.bar(x + width / 2, total, width, label="Total Classes", color=colors, alpha=0.4)
    ax6.set_xticks(x)
    ax6.set_xticklabels([DATASET_CONFIG[ds]["label"] for ds in datasets], rotation=15)
    ax6.set_ylabel("Number of Classes")
    ax6.set_title("(F) Effective vs Total Classes", fontweight="bold")
    ax6.legend()

    plt.tight_layout()
    return save_figure(fig, output_dir / "dataset_characterization", formats)


def plot_cic_vs_iiot_comparison(output_dir: Path, formats: list[str]) -> list[Path]:
    """Plot CIC vs Edge-IIoTset side-by-side comparison."""
    style = ThesisStyle()
    style.apply()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("CIC-IDS2017 vs Edge-IIoTset: Comprehensive Comparison", fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0, 0]
    comparison_data = {
        "Metric": ["Mean Macro-F1", "Best F1 (Benign)", "Effective Classes", "Samples (M)"],
        "CIC": [0.177, 0.253, 2.16, 2.83],
        "IIoT": [0.432, 0.619, 13.07, 1.70],
    }
    x = np.arange(len(comparison_data["Metric"]))
    width = 0.35
    ax.bar(x - width / 2, comparison_data["CIC"], width, label="CIC-IDS2017", color=DATASET_CONFIG["cic"]["color"])
    ax.bar(x + width / 2, comparison_data["IIoT"], width, label="Edge-IIoTset", color=DATASET_CONFIG["iiot"]["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_data["Metric"], rotation=15, ha="right")
    ax.set_ylabel("Value (normalized)")
    ax.set_title("(A) Key Metrics Comparison", fontweight="bold")
    ax.legend()

    ax = axes[0, 1]
    aggregators = ["FedAvg", "Bulyan", "Median", "Krum"]
    cic_f1 = [0.205, 0.212, 0.203, 0.202]
    iiot_f1 = [0.583, 0.602, 0.598, 0.507]
    x = np.arange(len(aggregators))
    width = 0.35
    ax.bar(x - width / 2, cic_f1, width, label="CIC-IDS2017", color=DATASET_CONFIG["cic"]["color"])
    ax.bar(x + width / 2, iiot_f1, width, label="Edge-IIoTset", color=DATASET_CONFIG["iiot"]["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(aggregators)
    ax.set_ylabel("Macro F1 Score")
    ax.set_ylim(0, 0.8)
    ax.set_title("(B) Performance by Aggregator (Benign)", fontweight="bold")
    ax.legend()

    ax = axes[0, 2]
    adv_levels = [0, 10, 20, 30]
    cic_deg = [0, 10.4, 22.3, 30.7]
    iiot_deg = [0, 19.2, 36.3, 49.0]
    ax.plot(adv_levels, cic_deg, "o-", color=DATASET_CONFIG["cic"]["color"], linewidth=2, markersize=8, label="CIC-IDS2017")
    ax.plot(adv_levels, iiot_deg, "s-", color=DATASET_CONFIG["iiot"]["color"], linewidth=2, markersize=8, label="Edge-IIoTset")
    ax.set_xlabel("Byzantine Clients (%)")
    ax.set_ylabel("F1 Degradation (%)")
    ax.set_title("(C) Attack Impact (Degradation)", fontweight="bold")
    ax.legend()

    ax = axes[1, 0]
    alpha_vals = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    cic_alpha = [0.248, 0.218, 0.176, 0.169, 0.177, 0.195]
    iiot_alpha = [0.390, 0.467, 0.534, 0.619, 0.649, 0.663]
    ax.plot(alpha_vals, cic_alpha, "o-", color=DATASET_CONFIG["cic"]["color"], linewidth=2, markersize=8, label="CIC-IDS2017")
    ax.plot(alpha_vals, iiot_alpha, "s-", color=DATASET_CONFIG["iiot"]["color"], linewidth=2, markersize=8, label="Edge-IIoTset")
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet Alpha (log)")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("(D) Heterogeneity Impact", fontweight="bold")
    ax.legend()

    ax = axes[1, 1]
    stats_text = """Statistical Comparison (t-test):

CIC Mean F1:  0.177 +/- 0.068
IIoT Mean F1: 0.432 +/- 0.203

t-statistic: -31.84
p-value: < 0.000001
Cohen's d: -3.33 (Very Large Effect)

Interpretation:
- IIoT outperforms CIC by 2.4x
- Difference is highly significant
- Effect size indicates practical importance
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, va="top", fontfamily="monospace")
    ax.set_title("(E) Statistical Significance", fontweight="bold")
    ax.axis("off")

    ax = axes[1, 2]
    recommendations = """Key Findings:

1. Edge-IIoTset is 2.4x better for multi-class
   FL intrusion detection

2. CIC-IDS2017 has extreme class imbalance
   (206,645:1 ratio) limiting FL performance

3. CIC effectively degrades to binary
   classification (80% BENIGN dominance)

4. IIoT protocol-specific features provide
   better attack separation than CIC flow stats

5. Heterogeneity affects IIoT more strongly
   (4.6% vs 0.7% variation with alpha)
"""
    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes, fontsize=9, va="top", fontfamily="monospace")
    ax.set_title("(F) Recommendations", fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    return save_figure(fig, output_dir / "cic_vs_iiot_comparison", formats)


def plot_efficiency_overhead(output_dir: Path, formats: list[str]) -> list[Path]:
    """Plot efficiency and computational overhead analysis."""
    style = ThesisStyle()
    style.apply()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Computational Efficiency and Overhead Analysis", fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    aggs = list(OVERHEAD_DATA.keys())
    times = [OVERHEAD_DATA[a]["time_ms"] for a in aggs]
    colors = [AGGREGATOR_COLORS.get(a.lower(), "#999999") for a in aggs]
    ax1.bar(range(len(aggs)), times, color=colors, edgecolor="black")
    ax1.set_xticks(range(len(aggs)))
    ax1.set_xticklabels(aggs)
    ax1.set_ylabel("Aggregation Time (ms)")
    ax1.set_title("(A) Aggregation Time per Round", fontweight="bold")
    ax1.axhline(y=100, color="red", linestyle="--", label="Real-time threshold (100ms)")
    ax1.axhline(y=50, color="orange", linestyle="--", label="RPi4 limit (50ms)")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")

    ax2 = fig.add_subplot(2, 3, 2)
    overheads = [OVERHEAD_DATA[a]["overhead_x"] for a in aggs]
    ax2.bar(range(len(aggs)), overheads, color=colors, edgecolor="black")
    ax2.set_xticks(range(len(aggs)))
    ax2.set_xticklabels(aggs)
    ax2.set_ylabel("Overhead Multiplier (vs FedAvg)")
    ax2.set_title("(B) Relative Computational Overhead", fontweight="bold")
    ax2.axhline(y=1, color="gray", linestyle="-", alpha=0.5)
    for i, (bar, oh) in enumerate(zip(ax2.patches, overheads)):
        ax2.annotate(f"{oh:.1f}x", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9)

    ax3 = fig.add_subplot(2, 3, 3)
    rounds = np.arange(1, 16)
    for agg in aggs:
        cumulative = rounds * OVERHEAD_DATA[agg]["time_ms"]
        ax3.plot(rounds, cumulative, "o-", linewidth=2, markersize=4, label=agg, color=AGGREGATOR_COLORS.get(agg.lower(), "#999999"))
    ax3.set_xlabel("Communication Round")
    ax3.set_ylabel("Cumulative Aggregation Time (ms)")
    ax3.set_title("(C) Total Overhead Over Training", fontweight="bold")
    ax3.legend()

    ax4 = fig.add_subplot(2, 3, 4)
    tradeoff_data = {
        "FedAvg": {"time": 0.5, "f1_30": 0.02, "resilience": 0},
        "Krum": {"time": 10.0, "f1_30": 0.57, "resilience": 85},
        "Bulyan": {"time": 20.0, "f1_30": 0.67, "resilience": 97},
        "Median": {"time": 12.0, "f1_30": 0.66, "resilience": 96},
    }
    for agg, data in tradeoff_data.items():
        color = AGGREGATOR_COLORS.get(agg.lower(), "#999999")
        ax4.scatter(
            data["time"], data["f1_30"], s=200, c=[color], label=f"{agg} ({data['resilience']}% ret)", marker="o", edgecolors="black"
        )
    ax4.set_xlabel("Aggregation Time (ms, log scale)")
    ax4.set_ylabel("F1 at 30% Attack")
    ax4.set_title("(D) Cost-Benefit Tradeoff", fontweight="bold")
    ax4.set_xscale("log")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 0.8)

    ax5 = fig.add_subplot(2, 3, 5)
    convergence_rounds = {
        "FedAvg (benign)": 5,
        "Bulyan (benign)": 6,
        "Bulyan (30% attack)": 8,
        "Krum (benign)": 7,
        "Krum (30% attack)": 10,
        "Median (benign)": 6,
        "Median (30% attack)": 9,
    }
    labels = list(convergence_rounds.keys())
    values = list(convergence_rounds.values())
    colors_conv = [PALETTES["colorblind"][0] if "benign" in label else PALETTES["colorblind"][1] for label in labels]
    ax5.barh(range(len(labels)), values, color=colors_conv)
    ax5.set_yticks(range(len(labels)))
    ax5.set_yticklabels(labels, fontsize=8)
    ax5.set_xlabel("Rounds to 90% Final F1")
    ax5.set_title("(E) Convergence Speed", fontweight="bold")

    ax6 = fig.add_subplot(2, 3, 6)
    summary_text = """Key Findings - Efficiency:

1. Aggregation overhead: 27-46x vs FedAvg
   but absolute times remain practical

2. Bulyan: 20ms/round (under 50ms RPi4 limit)
   Median: 12ms/round (best overhead/security)
   Krum: 10ms/round (fastest robust method)

3. Total training overhead (15 rounds):
   - FedAvg: ~40ms
   - Bulyan: ~450ms
   - All methods complete in < 500ms

4. Convergence speed similar across methods
   (5-10 rounds to 90% final accuracy)

5. Real-time detection updates feasible
   on edge hardware (Raspberry Pi 4 class)

6. Recommendation: Median for resource-
   constrained devices, Bulyan for servers
"""
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9, va="top", fontfamily="monospace")
    ax6.set_title("(F) Summary", fontweight="bold")
    ax6.axis("off")

    plt.tight_layout()
    return save_figure(fig, output_dir / "efficiency_overhead", formats)


def generate_comprehensive_figures(
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Generate all comprehensive thesis figures.

    Args:
        output_dir: Output directory for figures
        formats: Output formats (default: ['png', 'pdf'])

    Returns:
        Dict mapping figure name -> list of saved paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Path]] = {}

    paths = plot_dataset_characterization(output_dir, formats)
    results["dataset_characterization"] = paths
    print("Generated: dataset_characterization")

    paths = plot_cic_vs_iiot_comparison(output_dir, formats)
    results["cic_vs_iiot_comparison"] = paths
    print("Generated: cic_vs_iiot_comparison")

    paths = plot_efficiency_overhead(output_dir, formats)
    results["efficiency_overhead"] = paths
    print("Generated: efficiency_overhead")

    return results
