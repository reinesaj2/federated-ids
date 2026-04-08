#!/usr/bin/env python3
"""
Generate neurips-format thesis plots from summary_all_clean.csv.

Outputs:
  results/neurips_plots_fixed/plot01_aggregation_baseline.png  -- adv=0 baseline F1
  results/neurips_plots_fixed/plot02_attack_resilience.png     -- F1 vs adversary fraction
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SUMMARY_CSV = Path("results/summary_all_clean.csv")
OUTPUT_DIR = Path("results/neurips_plots_fixed")

AGGREGATORS = ["fedavg", "krum", "bulyan", "median"]
AGG_LABELS = {
    "fedavg": "FedAvg",
    "krum": "Krum",
    "bulyan": "Bulyan",
    "median": "Median",
}
COLORS = {
    "fedavg": "#1f77b4",
    "krum": "#ff7f0e",
    "bulyan": "#2ca02c",
    "median": "#d62728",
}
MARKERS = {
    "fedavg": "o",
    "krum": "s",
    "bulyan": "^",
    "median": "D",
}
DATASETS = [
    ("iiot", "Edge-IIoTset"),
    ("cic", "CIC-IDS2017"),
    ("unsw", "UNSW-NB15"),
]
ALPHA_ATTACK = 0.5


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
        }
    )


def _ci95_margin(values: np.ndarray) -> float:
    """Return 95% CI half-width using t-distribution."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    return float(stats.sem(values) * stats.t.ppf(0.975, len(values) - 1))


def _plot_baseline_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dataset_label: str,
    alpha_val: float,
) -> None:
    """Bar chart of baseline (adv=0) F1 for each aggregator."""
    subset = df[(df["adv_pct"] == 0) & (df["alpha"] == alpha_val)]
    x = np.arange(len(AGGREGATORS))
    width = 0.55

    means, yerrs = [], []
    for agg in AGGREGATORS:
        vals = subset[subset["aggregator"] == agg]["macro_f1"].dropna().values
        means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
        yerrs.append(_ci95_margin(vals) if len(vals) > 0 else 0.0)

    colors = [COLORS[a] for a in AGGREGATORS]
    ax.bar(
        x,
        means,
        width,
        yerr=yerrs,
        color=colors,
        capsize=4,
        error_kw={"linewidth": 1.2},
        alpha=0.85,
    )

    for i, (mean, err) in enumerate(zip(means, yerrs)):
        if not np.isnan(mean):
            ax.text(
                x[i],
                mean + err + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([AGG_LABELS[a] for a in AGGREGATORS])
    ax.set_title(dataset_label, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def generate_plot01(df: pd.DataFrame, output_dir: Path) -> None:
    """Baseline aggregation performance (adv=0, alpha=0.5), one panel per dataset."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Aggregation Baseline Performance (adv=0%, alpha=0.5)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        _plot_baseline_panel(ax, df[df["dataset"] == ds_key], ds_label, ALPHA_ATTACK)

    for ax in axes[1:]:
        ax.set_ylabel("")

    plt.tight_layout()
    out = output_dir / "plot01_aggregation_baseline.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _plot_resilience_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dataset_label: str,
    alpha_val: float,
) -> list:
    """Line+CI plot of F1 vs adv_pct for each aggregator. Returns handles for legend."""
    subset = df[df["alpha"] == alpha_val]
    adv_levels = sorted(subset["adv_pct"].unique())
    handles = []

    for agg in AGGREGATORS:
        agg_data = subset[subset["aggregator"] == agg]
        xs, ys, errs = [], [], []
        for adv in adv_levels:
            vals = agg_data[agg_data["adv_pct"] == adv]["macro_f1"].dropna().values
            if len(vals) == 0:
                xs.append(adv)
                ys.append(np.nan)
                errs.append(np.nan)
                continue
            xs.append(adv)
            ys.append(float(np.mean(vals)))
            errs.append(_ci95_margin(vals))

        if not xs or all(np.isnan(ys)):
            continue

        ys_arr = np.array(ys)
        errs_arr = np.array(errs)

        (line,) = ax.plot(
            xs,
            ys,
            marker=MARKERS[agg],
            color=COLORS[agg],
            linewidth=2,
            markersize=6,
            label=AGG_LABELS[agg],
        )
        valid = ~np.isnan(ys_arr)
        xs_v = np.array(xs)[valid]
        ys_v = ys_arr[valid]
        errs_v = errs_arr[valid]
        ax.fill_between(
            xs_v,
            ys_v - errs_v,
            ys_v + errs_v,
            color=COLORS[agg],
            alpha=0.15,
        )
        handles.append(line)

    ax.set_title(dataset_label, fontweight="bold")
    ax.set_xlabel("Adversary Fraction (%)")
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    return handles


def generate_plot02(df: pd.DataFrame, output_dir: Path) -> None:
    """Attack resilience: F1 vs adversary fraction, one panel per dataset."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Attack Resilience: Macro F1 vs Adversary Fraction (alpha=0.5, 95% CI)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    legend_handles: list | None = None
    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        handles = _plot_resilience_panel(ax, df[df["dataset"] == ds_key], ds_label, ALPHA_ATTACK)
        if handles and legend_handles is None:
            legend_handles = handles

    axes[0].set_ylabel("Macro F1")
    for ax in axes[1:]:
        ax.set_ylabel("")

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(AGGREGATORS),
            bbox_to_anchor=(0.5, -0.05),
            frameon=True,
        )

    plt.tight_layout()
    out = output_dir / "plot02_attack_resilience.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}")
    generate_plot01(df, OUTPUT_DIR)
    generate_plot02(df, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
