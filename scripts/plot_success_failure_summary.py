#!/usr/bin/env python3
"""
Generate success/failure summary visualization.

Creates a simple dashboard showing what works vs what fails in the system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

output_dir = Path("comprehensive_analysis_plots")
output_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(16, 10))

fig.suptitle(
    "Edge-IIoTset Full: Success vs Failure Analysis\n" "Publication-Ready Summary of 2,871 Experiments",
    fontsize=18,
    fontweight="bold",
)

success_items = [
    ("TRUE Global F1", "92.04% mean across seeds", "vs 63.09% reported (+45.9%)"),
    ("Best Configuration", "95.71% macro F1 (seed 47)", "FedProx mu=1.0, alpha=0.1"),
    ("Top 5 Majority Classes", "All achieve >95% F1", "BENIGN, DDoS_UDP, DDoS_ICMP, SQL_INJ, VULN_SCAN"),
    ("Robust Aggregation", "3.2x better than FedAvg", "Median maintains 40% F1 at 30% attack"),
    ("Class Coverage", "7 out of 15 classes >95% F1", "Including smallest class MITM (928 samples)"),
]

mixed_items = [
    ("Minority Classes", "Range: 60.85% - 95.77% F1", "6 classes below 90% F1"),
    ("Moderate Heterogeneity", "Alpha=0.1 viable with tuning", "Requires mu>=0.5 for FedProx"),
    ("Training Stability", "Seed variance: 86-96% F1", "9.36% range across seeds"),
]

failure_items = [
    ("FedAvg Under Attack", "12.31% F1 at 30% adversaries", "Catastrophic failure"),
    ("Extreme Heterogeneity", "28.57% F1 at alpha=0.02", "Insufficient samples per class"),
    ("PORT_SCANNING Recall", "45.48% recall", "High precision (91.91%) but conservative"),
    ("RANSOMWARE Recall", "54.72% recall", "High precision (98.05%) but many false negatives"),
    ("Late Training Degradation", "2.38% drop R18→R20", "Suggests need for LR decay"),
]

ax_success = fig.add_subplot(3, 1, 1)
ax_mixed = fig.add_subplot(3, 1, 2)
ax_failure = fig.add_subplot(3, 1, 3)

y_pos = 0.9
for title, metric, detail in success_items:
    ax_success.text(0.02, y_pos, f"{title}:", fontsize=12, fontweight="bold", color="darkgreen")
    ax_success.text(0.3, y_pos, metric, fontsize=11, color="green")
    ax_success.text(0.65, y_pos, detail, fontsize=10, color="gray", style="italic")
    y_pos -= 0.18

ax_success.set_xlim(0, 1)
ax_success.set_ylim(0, 1)
ax_success.axis("off")
ax_success.set_title("SUCCESS: What Works Well", fontsize=14, fontweight="bold", color="green", loc="left")
ax_success.add_patch(mpatches.Rectangle((0, 0), 1, 1, fill=True, facecolor="lightgreen", alpha=0.2, transform=ax_success.transAxes))

y_pos = 0.9
for title, metric, detail in mixed_items:
    ax_mixed.text(0.02, y_pos, f"{title}:", fontsize=12, fontweight="bold", color="darkorange")
    ax_mixed.text(0.3, y_pos, metric, fontsize=11, color="orange")
    ax_mixed.text(0.65, y_pos, detail, fontsize=10, color="gray", style="italic")
    y_pos -= 0.25

ax_mixed.set_xlim(0, 1)
ax_mixed.set_ylim(0, 1)
ax_mixed.axis("off")
ax_mixed.set_title("MIXED: Needs Improvement", fontsize=14, fontweight="bold", color="orange", loc="left")
ax_mixed.add_patch(mpatches.Rectangle((0, 0), 1, 1, fill=True, facecolor="yellow", alpha=0.2, transform=ax_mixed.transAxes))

y_pos = 0.9
for title, metric, detail in failure_items:
    ax_failure.text(0.02, y_pos, f"{title}:", fontsize=12, fontweight="bold", color="darkred")
    ax_failure.text(0.3, y_pos, metric, fontsize=11, color="red")
    ax_failure.text(0.65, y_pos, detail, fontsize=10, color="gray", style="italic")
    y_pos -= 0.18

ax_failure.set_xlim(0, 1)
ax_failure.set_ylim(0, 1)
ax_failure.axis("off")
ax_failure.set_title("FAILURE: Critical Issues", fontsize=14, fontweight="bold", color="red", loc="left")
ax_failure.add_patch(mpatches.Rectangle((0, 0), 1, 1, fill=True, facecolor="lightcoral", alpha=0.2, transform=ax_failure.transAxes))

plt.tight_layout()
plt.savefig(output_dir / "success_failure_summary.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'success_failure_summary.png'}")
