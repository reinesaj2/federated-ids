#!/usr/bin/env python3
"""
Objective 1: Robust Aggregation Methods vs Adversarial Attacks

Generates thesis-quality plots comparing FedAvg, Bulyan, Krum, and Median
aggregators under adversarial conditions (0%, 10%, 30% Byzantine clients)
for both IIoT and CIC datasets at alpha=0.5 (moderate heterogeneity).

Metrics plotted:
- Macro F1 Score over rounds (primary performance metric)
- Benign False Positive Rate over rounds
- Final round accuracy comparison (bar chart)

Usage:
    python plot_objective1_robust_aggregation.py
"""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COLORBLIND_PALETTE = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Purple
]

AGGREGATOR_LABELS = {
    "fedavg": "FedAvg",
    "bulyan": "Bulyan",
    "krum": "Krum",
    "median": "Median",
}

AGGREGATOR_ORDER = ["fedavg", "bulyan", "krum", "median"]

ADVERSARIAL_LEVELS = [0, 10, 30]

ADVERSARIAL_LABELS = {
    0: "No Attack (0%)",
    10: "10% Byzantine",
    30: "30% Byzantine",
}

SEEDS = [42, 43, 44, 45, 46]

DATASET_SUFFIXES = {
    "iiot": "",
    "cic": "_datasetcic",
}

DATASET_LABELS = {
    "iiot": "Edge-IIoTset",
    "cic": "CIC-IDS2017",
}

BASE_DIR = Path(__file__).parent / "cluster-runs"
OUTPUT_DIR = Path(__file__).parent / "thesis_plots" / "objective1"


def parse_experiment_params(dirname: str) -> Optional[dict]:
    pattern = (
        r"comp_(?P<agg>\w+)_alpha(?P<alpha>[\d.]+)_adv(?P<adv>\d+)_"
        r"dp(?P<dp>\d+)_pers(?P<pers>\d+)_mu(?P<mu>[\d.]+)_seed(?P<seed>\d+)"
        r"(?P<suffix>_datasetcic)?"
    )
    match = re.match(pattern, dirname)
    if not match:
        return None

    return {
        "aggregator": match.group("agg"),
        "alpha": float(match.group("alpha")),
        "adv_pct": int(match.group("adv")),
        "dp": int(match.group("dp")),
        "pers": int(match.group("pers")),
        "mu": float(match.group("mu")),
        "seed": int(match.group("seed")),
        "dataset": "cic" if match.group("suffix") else "iiot",
    }


def load_client_metrics(exp_dir: Path) -> Optional[pd.DataFrame]:
    client_files = list(exp_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None

    dfs = []
    for f in client_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def aggregate_client_metrics_per_round(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        return pd.Series(dtype=float)

    return df.groupby("round")[metric].mean()


def collect_experiment_data(
    aggregator: str, adv_pct: int, dataset: str, seeds: list[int]
) -> dict:
    suffix = DATASET_SUFFIXES[dataset]
    data = {"macro_f1": [], "benign_fpr": [], "acc_after": [], "rounds": None}

    for seed in seeds:
        dirname = f"comp_{aggregator}_alpha0.5_adv{adv_pct}_dp0_pers0_mu0.0_seed{seed}{suffix}"
        exp_dir = BASE_DIR / dirname

        if not exp_dir.exists():
            continue

        client_df = load_client_metrics(exp_dir)
        if client_df is None:
            continue

        f1_series = aggregate_client_metrics_per_round(client_df, "macro_f1_after")
        fpr_series = aggregate_client_metrics_per_round(client_df, "benign_fpr_argmax")
        acc_series = aggregate_client_metrics_per_round(client_df, "acc_after")

        if len(f1_series) > 0:
            data["macro_f1"].append(f1_series.values)
            if data["rounds"] is None:
                data["rounds"] = f1_series.index.values

        if len(fpr_series) > 0:
            data["benign_fpr"].append(fpr_series.values)

        if len(acc_series) > 0:
            data["acc_after"].append(acc_series.values)

    return data


def compute_mean_and_ci(data_list: list[np.ndarray], confidence: float = 0.95) -> dict:
    if not data_list:
        return {"mean": None, "lower": None, "upper": None}

    valid_data = [d for d in data_list if len(d) > 0]
    if not valid_data:
        return {"mean": None, "lower": None, "upper": None}

    min_len = min(len(d) for d in valid_data)
    if min_len == 0:
        return {"mean": None, "lower": None, "upper": None}

    data_array = np.array([d[:min_len] for d in valid_data])

    mean = np.mean(data_array, axis=0)
    n = len(data_array)

    if n < 2:
        return {"mean": mean, "lower": mean, "upper": mean}

    se = stats.sem(data_array, axis=0)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return {"mean": mean, "lower": mean - ci, "upper": mean + ci}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_metric_over_rounds(
    ax,
    metric_name: str,
    dataset: str,
    adv_pct: int,
    ylabel: str,
    ylim: Optional[tuple] = None,
):
    colors = {agg: COLORBLIND_PALETTE[i] for i, agg in enumerate(AGGREGATOR_ORDER)}

    for aggregator in AGGREGATOR_ORDER:
        data = collect_experiment_data(aggregator, adv_pct, dataset, SEEDS)

        if metric_name not in data or not data[metric_name]:
            continue

        stats_data = compute_mean_and_ci(data[metric_name])
        if stats_data["mean"] is None:
            continue

        mean_arr = np.atleast_1d(stats_data["mean"])
        lower_arr = np.atleast_1d(stats_data["lower"])
        upper_arr = np.atleast_1d(stats_data["upper"])

        rounds = data["rounds"]
        if rounds is None:
            rounds = np.arange(1, len(mean_arr) + 1)
        else:
            rounds = rounds[: len(mean_arr)]

        ax.plot(
            rounds,
            mean_arr,
            label=AGGREGATOR_LABELS[aggregator],
            color=colors[aggregator],
            linewidth=2.0,
            marker="o",
            markersize=4,
            markevery=max(1, len(rounds) // 10),
        )

        ax.fill_between(
            rounds,
            lower_arr,
            upper_arr,
            color=colors[aggregator],
            alpha=0.2,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{DATASET_LABELS[dataset]} - {ADVERSARIAL_LABELS[adv_pct]}",
        fontweight="bold",
    )

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(loc="lower right", framealpha=0.9)


def plot_f1_comparison():
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Macro F1 Score: Robust Aggregation Methods Under Adversarial Attacks",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for row, dataset in enumerate(["iiot", "cic"]):
        for col, adv_pct in enumerate(ADVERSARIAL_LEVELS):
            ax = axes[row, col]
            plot_metric_over_rounds(
                ax, "macro_f1", dataset, adv_pct, "Macro F1 Score", ylim=(0.5, 1.02)
            )

    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj1_macro_f1_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_fpr_comparison():
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Benign False Positive Rate: Robust Aggregation Methods Under Adversarial Attacks",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for row, dataset in enumerate(["iiot", "cic"]):
        for col, adv_pct in enumerate(ADVERSARIAL_LEVELS):
            ax = axes[row, col]
            plot_metric_over_rounds(
                ax,
                "benign_fpr",
                dataset,
                adv_pct,
                "Benign FPR (argmax)",
                ylim=(-0.02, 0.5),
            )

    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj1_benign_fpr_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_final_round_accuracy():
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Final Round Accuracy: Aggregators Under Adversarial Attacks",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    bar_width = 0.2
    x = np.arange(len(ADVERSARIAL_LEVELS))

    for ax_idx, dataset in enumerate(["iiot", "cic"]):
        ax = axes[ax_idx]

        for i, aggregator in enumerate(AGGREGATOR_ORDER):
            means = []
            errors = []

            for adv_pct in ADVERSARIAL_LEVELS:
                data = collect_experiment_data(aggregator, adv_pct, dataset, SEEDS)
                if data["acc_after"]:
                    final_accs = [d[-1] for d in data["acc_after"] if len(d) > 0]
                    if final_accs:
                        means.append(np.mean(final_accs))
                        n = len(final_accs)
                        if n > 1:
                            se = stats.sem(final_accs)
                            ci = se * stats.t.ppf(0.975, n - 1)
                            errors.append(ci)
                        else:
                            errors.append(0)
                    else:
                        means.append(0)
                        errors.append(0)
                else:
                    means.append(0)
                    errors.append(0)

            positions = x + (i - 1.5) * bar_width
            ax.bar(
                positions,
                means,
                bar_width,
                label=AGGREGATOR_LABELS[aggregator],
                color=COLORBLIND_PALETTE[i],
                yerr=errors,
                capsize=3,
                alpha=0.85,
            )

        ax.set_xlabel("Adversarial Setting")
        ax.set_ylabel("Final Round Accuracy")
        ax.set_title(f"{DATASET_LABELS[dataset]}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([ADVERSARIAL_LABELS[adv] for adv in ADVERSARIAL_LEVELS])
        ax.set_ylim(0.5, 1.05)
        ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj1_final_accuracy_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined_2x2_f1():
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Macro F1 Score: Aggregator Robustness Comparison\n(alpha=0.5, seeds 42-46)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    configs = [
        ("iiot", 0, "Edge-IIoTset - No Attack"),
        ("iiot", 30, "Edge-IIoTset - 30% Byzantine"),
        ("cic", 0, "CIC-IDS2017 - No Attack"),
        ("cic", 30, "CIC-IDS2017 - 30% Byzantine"),
    ]

    colors = {agg: COLORBLIND_PALETTE[i] for i, agg in enumerate(AGGREGATOR_ORDER)}

    for idx, (dataset, adv_pct, title) in enumerate(configs):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for aggregator in AGGREGATOR_ORDER:
            data = collect_experiment_data(aggregator, adv_pct, dataset, SEEDS)

            if not data["macro_f1"]:
                continue

            stats_data = compute_mean_and_ci(data["macro_f1"])
            if stats_data["mean"] is None:
                continue

            mean_arr = np.atleast_1d(stats_data["mean"])
            lower_arr = np.atleast_1d(stats_data["lower"])
            upper_arr = np.atleast_1d(stats_data["upper"])

            rounds = data["rounds"]
            if rounds is None:
                rounds = np.arange(1, len(mean_arr) + 1)
            else:
                rounds = rounds[: len(mean_arr)]

            ax.plot(
                rounds,
                mean_arr,
                label=AGGREGATOR_LABELS[aggregator],
                color=colors[aggregator],
                linewidth=2.0,
                marker="o",
                markersize=4,
                markevery=max(1, len(rounds) // 10),
            )

            ax.fill_between(
                rounds,
                lower_arr,
                upper_arr,
                color=colors[aggregator],
                alpha=0.2,
            )

        ax.set_xlabel("Round")
        ax.set_ylabel("Macro F1 Score")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.5, 1.02)
        ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj1_f1_2x2_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_degradation_heatmap():
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "F1 Performance Degradation Under Attack (%)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for ax_idx, dataset in enumerate(["iiot", "cic"]):
        ax = axes[ax_idx]

        matrix = np.zeros((len(AGGREGATOR_ORDER), len(ADVERSARIAL_LEVELS)))

        for i, aggregator in enumerate(AGGREGATOR_ORDER):
            baseline_data = collect_experiment_data(aggregator, 0, dataset, SEEDS)
            if baseline_data["macro_f1"]:
                baseline_f1 = np.mean(
                    [d[-1] for d in baseline_data["macro_f1"] if len(d) > 0]
                )
            else:
                baseline_f1 = 1.0

            for j, adv_pct in enumerate(ADVERSARIAL_LEVELS):
                data = collect_experiment_data(aggregator, adv_pct, dataset, SEEDS)
                if data["macro_f1"]:
                    final_f1 = np.mean([d[-1] for d in data["macro_f1"] if len(d) > 0])
                    degradation = ((baseline_f1 - final_f1) / baseline_f1) * 100
                    matrix[i, j] = degradation
                else:
                    matrix[i, j] = np.nan

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-5, vmax=50)

        ax.set_xticks(np.arange(len(ADVERSARIAL_LEVELS)))
        ax.set_yticks(np.arange(len(AGGREGATOR_ORDER)))
        ax.set_xticklabels([f"{adv}%" for adv in ADVERSARIAL_LEVELS])
        ax.set_yticklabels([AGGREGATOR_LABELS[agg] for agg in AGGREGATOR_ORDER])

        ax.set_xlabel("Byzantine Clients (%)")
        ax.set_ylabel("Aggregation Method")
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")

        for i in range(len(AGGREGATOR_ORDER)):
            for j in range(len(ADVERSARIAL_LEVELS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > 25 else "black"
                    ax.text(
                        j, i, f"{val:.1f}%", ha="center", va="center", color=text_color
                    )

    fig.colorbar(im, ax=axes, shrink=0.6, label="F1 Degradation (%)")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "obj1_degradation_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_summary_table():
    summary_data = []

    for dataset in ["iiot", "cic"]:
        for aggregator in AGGREGATOR_ORDER:
            for adv_pct in ADVERSARIAL_LEVELS:
                data = collect_experiment_data(aggregator, adv_pct, dataset, SEEDS)

                if data["macro_f1"]:
                    final_f1s = [d[-1] for d in data["macro_f1"] if len(d) > 0]
                    f1_mean = np.mean(final_f1s) if final_f1s else np.nan
                    f1_std = np.std(final_f1s) if len(final_f1s) > 1 else 0
                else:
                    f1_mean, f1_std = np.nan, np.nan

                if data["benign_fpr"]:
                    final_fprs = [d[-1] for d in data["benign_fpr"] if len(d) > 0]
                    fpr_mean = np.mean(final_fprs) if final_fprs else np.nan
                    fpr_std = np.std(final_fprs) if len(final_fprs) > 1 else 0
                else:
                    fpr_mean, fpr_std = np.nan, np.nan

                if data["acc_after"]:
                    final_accs = [d[-1] for d in data["acc_after"] if len(d) > 0]
                    acc_mean = np.mean(final_accs) if final_accs else np.nan
                    acc_std = np.std(final_accs) if len(final_accs) > 1 else 0
                else:
                    acc_mean, acc_std = np.nan, np.nan

                summary_data.append(
                    {
                        "Dataset": DATASET_LABELS[dataset],
                        "Aggregator": AGGREGATOR_LABELS[aggregator],
                        "Byzantine (%)": adv_pct,
                        "F1 Mean": f1_mean,
                        "F1 Std": f1_std,
                        "FPR Mean": fpr_mean,
                        "FPR Std": fpr_std,
                        "Acc Mean": acc_mean,
                        "Acc Std": acc_std,
                        "Seeds": len(data.get("macro_f1", [])),
                    }
                )

    df = pd.DataFrame(summary_data)
    output_path = OUTPUT_DIR / "obj1_summary_table.csv"
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved: {output_path}")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Objective 1: Robust Aggregation Methods vs Adversarial Attacks")
    print("=" * 60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Aggregators: {', '.join(AGGREGATOR_LABELS.values())}")
    print(f"Adversarial levels: {ADVERSARIAL_LEVELS}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {', '.join(DATASET_LABELS.values())}")
    print()

    print("Generating Macro F1 comparison (2x3 grid)...")
    plot_f1_comparison()

    print("Generating Benign FPR comparison (2x3 grid)...")
    plot_fpr_comparison()

    print("Generating Final Round Accuracy comparison...")
    plot_final_round_accuracy()

    print("Generating 2x2 F1 comparison (main figure)...")
    plot_combined_2x2_f1()

    print("Generating Degradation Heatmap...")
    plot_degradation_heatmap()

    print("Generating Summary Table...")
    summary_df = generate_summary_table()

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
