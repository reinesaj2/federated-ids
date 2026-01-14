#!/usr/bin/env python3
"""
OBJECTIVE 2: FedProx Effectiveness under Non-IID Data

This script generates publication-quality plots analyzing how the FedProx proximal term (mu)
affects model performance under varying levels of data heterogeneity (alpha).

Plots generated:
1. Convergence curves: macro_f1_after vs rounds for different mu values
2. Final F1 vs alpha: heterogeneity impact on performance
3. Optimal mu heatmap: mu vs alpha with color = final F1
4. Training stability: variance across clients
5. FedProx benefit over vanilla FedAvg

Data structure:
- Directory pattern: comp_fedavg_alpha{alpha}_adv0_dp0_pers0_mu{mu}_seed{seed}[_datasetcic]
- IIoT dataset: no suffix or datasetedge-iiotset-full
- CIC dataset: _datasetcic suffix
- Seeds: 42-46
- Alpha values: 0.02, 0.05, 0.1, 0.2, 0.5, 1.0
- Mu values: 0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

COLORBLIND_PALETTE = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#CC78BC",  # purple
    "#ECE133",  # yellow
    "#56B4E9",  # light blue
    "#D55E00",  # red-orange
    "#F0E442",  # light yellow
    "#999999",  # gray
]

MU_VALUES = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2]
ALPHA_VALUES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEEDS = [42, 43, 44, 45, 46]


class ExperimentConfig(NamedTuple):
    alpha: float
    mu: float
    seed: int
    dataset: str


def apply_thesis_style():
    """Apply publication-quality styling."""
    sns.set_theme(
        context="paper", style="whitegrid", palette="colorblind", font="serif"
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def parse_directory_name(dirname: str) -> ExperimentConfig | None:
    """Extract experiment configuration from directory name."""
    if "comp_fedavg" not in dirname:
        return None
    if "adv0_dp0_pers0" not in dirname:
        return None

    alpha_match = re.search(r"alpha([0-9.]+)", dirname)
    mu_match = re.search(r"mu([0-9.]+)", dirname)
    seed_match = re.search(r"seed(\d+)", dirname)

    if not all([alpha_match, mu_match, seed_match]):
        return None

    alpha = float(alpha_match.group(1))
    mu = float(mu_match.group(1))
    seed = int(seed_match.group(1))

    if seed not in SEEDS:
        return None
    if alpha not in ALPHA_VALUES:
        return None
    if mu not in MU_VALUES:
        return None

    dataset = "CIC" if "_datasetcic" in dirname else "IIoT"

    return ExperimentConfig(alpha=alpha, mu=mu, seed=seed, dataset=dataset)


def load_client_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load and aggregate client metrics from a run directory."""
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None

    dfs = []
    for cf in client_files:
        try:
            df = pd.read_csv(cf)
            client_id_match = re.search(r"client_(\d+)", cf.name)
            if client_id_match:
                df["client_id"] = int(client_id_match.group(1))
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def aggregate_across_clients(client_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across clients for each round."""
    agg_df = (
        client_df.groupby("round")
        .agg(
            {
                "macro_f1_after": ["mean", "std"],
                "loss_after": ["mean", "std"],
            }
        )
        .reset_index()
    )

    agg_df.columns = ["round", "macro_f1_mean", "macro_f1_std", "loss_mean", "loss_std"]
    return agg_df


def compute_confidence_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0

    mean = np.mean(values)
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - ci, mean + ci


def load_all_experiments(runs_dir: Path) -> pd.DataFrame:
    """Load all FedAvg experiments with varying mu and alpha."""
    records = []

    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue

        config = parse_directory_name(run_path.name)
        if config is None:
            continue

        client_df = load_client_metrics(run_path)
        if client_df is None:
            continue

        if "macro_f1_after" not in client_df.columns:
            continue

        for round_num in client_df["round"].unique():
            round_data = client_df[client_df["round"] == round_num]

            f1_values = round_data["macro_f1_after"].dropna().values
            loss_values = (
                round_data["loss_after"].dropna().values
                if "loss_after" in round_data.columns
                else []
            )

            if len(f1_values) == 0:
                continue

            records.append(
                {
                    "alpha": config.alpha,
                    "mu": config.mu,
                    "seed": config.seed,
                    "dataset": config.dataset,
                    "round": round_num,
                    "macro_f1_mean": np.mean(f1_values),
                    "macro_f1_std": np.std(f1_values),
                    "macro_f1_min": np.min(f1_values),
                    "macro_f1_max": np.max(f1_values),
                    "n_clients": len(f1_values),
                    "loss_mean": (
                        np.mean(loss_values) if len(loss_values) > 0 else np.nan
                    ),
                    "loss_std": np.std(loss_values) if len(loss_values) > 0 else np.nan,
                }
            )

    return pd.DataFrame(records)


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across seeds, computing mean and CI."""
    grouped = df.groupby(["alpha", "mu", "dataset", "round"])

    records = []
    for (alpha, mu, dataset, round_num), group in grouped:
        f1_values = group["macro_f1_mean"].values
        std_values = group["macro_f1_std"].values

        if len(f1_values) == 0:
            continue

        mean_f1 = np.mean(f1_values)
        ci_low, ci_high = compute_confidence_interval(f1_values)

        mean_client_std = np.mean(std_values) if len(std_values) > 0 else np.nan

        records.append(
            {
                "alpha": alpha,
                "mu": mu,
                "dataset": dataset,
                "round": round_num,
                "macro_f1": mean_f1,
                "macro_f1_ci_low": ci_low,
                "macro_f1_ci_high": ci_high,
                "macro_f1_client_std": mean_client_std,
                "n_seeds": len(f1_values),
            }
        )

    return pd.DataFrame(records)


def get_final_round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract metrics from the final round for each configuration."""
    final_records = []

    for (alpha, mu, dataset), group in df.groupby(["alpha", "mu", "dataset"]):
        max_round = group["round"].max()
        final_data = group[group["round"] == max_round]

        if len(final_data) == 0:
            continue

        final_records.append(
            {
                "alpha": alpha,
                "mu": mu,
                "dataset": dataset,
                "final_round": max_round,
                "final_f1": final_data["macro_f1"].values[0],
                "final_f1_ci_low": final_data["macro_f1_ci_low"].values[0],
                "final_f1_ci_high": final_data["macro_f1_ci_high"].values[0],
                "final_client_std": final_data["macro_f1_client_std"].values[0],
            }
        )

    return pd.DataFrame(final_records)


def plot_convergence_curves(
    agg_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    selected_alphas: list[float] | None = None,
):
    """Plot convergence curves showing macro_f1 vs rounds for different mu values."""
    subset = agg_df[agg_df["dataset"] == dataset].copy()

    if selected_alphas is None:
        selected_alphas = [0.02, 0.1, 0.5]

    n_plots = len(selected_alphas)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)

    if n_plots == 1:
        axes = [axes]

    mu_colors = {
        mu: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        for i, mu in enumerate(MU_VALUES)
    }

    for ax, alpha in zip(axes, selected_alphas):
        alpha_data = subset[subset["alpha"] == alpha]

        for mu in MU_VALUES:
            mu_data = alpha_data[alpha_data["mu"] == mu].sort_values("round")

            if len(mu_data) == 0:
                continue

            label = f"mu={mu}" if mu > 0 else "FedAvg (mu=0)"
            linewidth = 2.5 if mu == 0 else 1.5
            linestyle = "-" if mu == 0 else "--"

            ax.plot(
                mu_data["round"],
                mu_data["macro_f1"],
                label=label,
                color=mu_colors[mu],
                linewidth=linewidth,
                linestyle=linestyle,
            )

            ax.fill_between(
                mu_data["round"],
                mu_data["macro_f1_ci_low"],
                mu_data["macro_f1_ci_high"],
                alpha=0.15,
                color=mu_colors[mu],
            )

        ax.set_xlabel("Communication Round")
        ax.set_title(f"alpha = {alpha}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.02)

    axes[0].set_ylabel("Macro F1 Score")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        title="FedProx mu",
    )

    fig.suptitle(
        f"FedProx Convergence under Non-IID Data ({dataset} Dataset)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / f"convergence_curves_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_final_f1_vs_alpha(final_df: pd.DataFrame, dataset: str, output_dir: Path):
    """Plot final F1 score vs alpha (heterogeneity) for different mu values."""
    subset = final_df[final_df["dataset"] == dataset].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    mu_colors = {
        mu: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        for i, mu in enumerate(MU_VALUES)
    }
    mu_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]

    for i, mu in enumerate(MU_VALUES):
        mu_data = subset[subset["mu"] == mu].sort_values("alpha")

        if len(mu_data) == 0:
            continue

        label = f"mu={mu}" if mu > 0 else "FedAvg (mu=0)"
        marker = mu_markers[i % len(mu_markers)]
        linewidth = 2.5 if mu == 0 else 1.5

        ax.errorbar(
            mu_data["alpha"],
            mu_data["final_f1"],
            yerr=[
                mu_data["final_f1"] - mu_data["final_f1_ci_low"],
                mu_data["final_f1_ci_high"] - mu_data["final_f1"],
            ],
            label=label,
            color=mu_colors[mu],
            marker=marker,
            linewidth=linewidth,
            markersize=8,
            capsize=3,
        )

    ax.set_xlabel("Heterogeneity (alpha, lower = more heterogeneous)")
    ax.set_ylabel("Final Macro F1 Score")
    ax.set_title(f"Impact of Heterogeneity on FedProx Performance ({dataset})")
    ax.set_xscale("log")
    ax.set_ylim(0.7, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="FedProx mu", loc="lower right")

    plt.tight_layout()

    output_path = output_dir / f"final_f1_vs_alpha_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_optimal_mu_heatmap(final_df: pd.DataFrame, dataset: str, output_dir: Path):
    """Plot heatmap of final F1 for mu vs alpha combinations."""
    subset = final_df[final_df["dataset"] == dataset].copy()

    pivot_table = subset.pivot_table(
        values="final_f1",
        index="mu",
        columns="alpha",
        aggfunc="mean",
    )

    pivot_table = pivot_table.reindex(index=MU_VALUES)
    pivot_table = pivot_table.reindex(columns=sorted(ALPHA_VALUES))

    fig, ax = plt.subplots(figsize=(10, 6))

    mask = pivot_table.isna()

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        mask=mask,
        ax=ax,
        vmin=0.7,
        vmax=1.0,
        cbar_kws={"label": "Final Macro F1 Score"},
        annot_kws={"size": 9},
    )

    best_mu_per_alpha = {}
    for alpha in pivot_table.columns:
        col_data = pivot_table[alpha].dropna()
        if len(col_data) > 0:
            best_mu = col_data.idxmax()
            best_mu_per_alpha[alpha] = best_mu

    for alpha, best_mu in best_mu_per_alpha.items():
        col_idx = list(pivot_table.columns).index(alpha)
        row_idx = list(pivot_table.index).index(best_mu)
        ax.add_patch(
            plt.Rectangle(
                (col_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=3,
            )
        )

    ax.set_xlabel("Heterogeneity (alpha)")
    ax.set_ylabel("FedProx Proximal Term (mu)")
    ax.set_title(
        f"Optimal FedProx Configuration ({dataset})\nBlack boxes indicate best mu per alpha"
    )

    plt.tight_layout()

    output_path = output_dir / f"optimal_mu_heatmap_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_training_stability(final_df: pd.DataFrame, dataset: str, output_dir: Path):
    """Plot training stability (client variance) across configurations."""
    subset = final_df[final_df["dataset"] == dataset].copy()

    pivot_table = subset.pivot_table(
        values="final_client_std",
        index="mu",
        columns="alpha",
        aggfunc="mean",
    )

    pivot_table = pivot_table.reindex(index=MU_VALUES)
    pivot_table = pivot_table.reindex(columns=sorted(ALPHA_VALUES))

    fig, ax = plt.subplots(figsize=(10, 6))

    mask = pivot_table.isna()

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        mask=mask,
        ax=ax,
        cbar_kws={"label": "Client F1 Std Dev (lower = more stable)"},
        annot_kws={"size": 9},
    )

    ax.set_xlabel("Heterogeneity (alpha)")
    ax.set_ylabel("FedProx Proximal Term (mu)")
    ax.set_title(
        f"Training Stability Across Clients ({dataset})\n"
        "Lower values indicate more consistent performance"
    )

    plt.tight_layout()

    output_path = output_dir / f"training_stability_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_fedprox_benefit(final_df: pd.DataFrame, dataset: str, output_dir: Path):
    """Plot FedProx improvement over vanilla FedAvg (mu=0)."""
    subset = final_df[final_df["dataset"] == dataset].copy()

    baseline_df = subset[subset["mu"] == 0.0][["alpha", "final_f1"]].rename(
        columns={"final_f1": "baseline_f1"}
    )

    merged = subset.merge(baseline_df, on="alpha", how="left")
    merged["improvement"] = merged["final_f1"] - merged["baseline_f1"]
    merged["improvement_pct"] = (merged["improvement"] / merged["baseline_f1"]) * 100

    fedprox_only = merged[merged["mu"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    pivot_abs = fedprox_only.pivot_table(
        values="improvement",
        index="mu",
        columns="alpha",
        aggfunc="mean",
    )
    pivot_abs = pivot_abs.reindex(index=[m for m in MU_VALUES if m > 0])
    pivot_abs = pivot_abs.reindex(columns=sorted(ALPHA_VALUES))

    mask = pivot_abs.isna()
    sns.heatmap(
        pivot_abs,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        mask=mask,
        ax=ax1,
        center=0,
        cbar_kws={"label": "F1 Improvement over FedAvg"},
        annot_kws={"size": 9},
    )
    ax1.set_xlabel("Heterogeneity (alpha)")
    ax1.set_ylabel("FedProx Proximal Term (mu)")
    ax1.set_title("Absolute F1 Improvement")

    ax2 = axes[1]
    pivot_pct = fedprox_only.pivot_table(
        values="improvement_pct",
        index="mu",
        columns="alpha",
        aggfunc="mean",
    )
    pivot_pct = pivot_pct.reindex(index=[m for m in MU_VALUES if m > 0])
    pivot_pct = pivot_pct.reindex(columns=sorted(ALPHA_VALUES))

    mask = pivot_pct.isna()
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        mask=mask,
        ax=ax2,
        center=0,
        cbar_kws={"label": "% Improvement over FedAvg"},
        annot_kws={"size": 9},
    )
    ax2.set_xlabel("Heterogeneity (alpha)")
    ax2.set_ylabel("FedProx Proximal Term (mu)")
    ax2.set_title("Percentage F1 Improvement")

    fig.suptitle(
        f"FedProx Benefit over Vanilla FedAvg ({dataset})", fontsize=14, y=1.02
    )
    plt.tight_layout()

    output_path = output_dir / f"fedprox_benefit_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_mu_sensitivity_by_alpha(agg_df: pd.DataFrame, dataset: str, output_dir: Path):
    """Plot mu sensitivity for each alpha level."""
    subset = agg_df[agg_df["dataset"] == dataset].copy()

    max_rounds = subset.groupby(["alpha", "mu"])["round"].max().reset_index()
    max_rounds = max_rounds.rename(columns={"round": "max_round"})

    subset = subset.merge(max_rounds, on=["alpha", "mu"])
    final_data = subset[subset["round"] == subset["max_round"]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, alpha in enumerate(sorted(ALPHA_VALUES)):
        ax = axes[i]
        alpha_data = final_data[final_data["alpha"] == alpha].sort_values("mu")

        if len(alpha_data) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"alpha = {alpha}")
            continue

        bars = ax.bar(
            range(len(alpha_data)),
            alpha_data["macro_f1"],
            yerr=[
                alpha_data["macro_f1"] - alpha_data["macro_f1_ci_low"],
                alpha_data["macro_f1_ci_high"] - alpha_data["macro_f1"],
            ],
            capsize=3,
            color=[
                COLORBLIND_PALETTE[j % len(COLORBLIND_PALETTE)]
                for j in range(len(alpha_data))
            ],
            edgecolor="black",
            linewidth=0.5,
        )

        best_idx = alpha_data["macro_f1"].idxmax()
        best_pos = list(alpha_data.index).index(best_idx)
        bars[best_pos].set_edgecolor("red")
        bars[best_pos].set_linewidth(3)

        ax.set_xticks(range(len(alpha_data)))
        ax.set_xticklabels(
            [f"{mu:.3f}" if mu < 0.1 else f"{mu:.2f}" for mu in alpha_data["mu"]],
            rotation=45,
        )
        ax.set_xlabel("mu")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"alpha = {alpha}")
        ax.set_ylim(0.7, 1.02)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"FedProx mu Sensitivity Analysis ({dataset})\nRed border indicates optimal mu",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / f"mu_sensitivity_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_combined_datasets_comparison(final_df: pd.DataFrame, output_dir: Path):
    """Create a side-by-side comparison of both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dataset in zip(axes, ["IIoT", "CIC"]):
        subset = final_df[final_df["dataset"] == dataset].copy()

        pivot_table = subset.pivot_table(
            values="final_f1",
            index="mu",
            columns="alpha",
            aggfunc="mean",
        )

        pivot_table = pivot_table.reindex(index=MU_VALUES)
        pivot_table = pivot_table.reindex(columns=sorted(ALPHA_VALUES))

        mask = pivot_table.isna()

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            mask=mask,
            ax=ax,
            vmin=0.7,
            vmax=1.0,
            cbar_kws={"label": "Final Macro F1"},
            annot_kws={"size": 8},
        )

        ax.set_xlabel("Heterogeneity (alpha)")
        ax.set_ylabel("FedProx mu")
        ax.set_title(f"{dataset} Dataset")

    fig.suptitle(
        "FedProx Performance Comparison: IIoT vs CIC Datasets", fontsize=14, y=1.02
    )
    plt.tight_layout()

    output_path = output_dir / "datasets_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def generate_summary_statistics(final_df: pd.DataFrame, output_dir: Path):
    """Generate and save summary statistics."""
    summary_lines = [
        "OBJECTIVE 2: FedProx Effectiveness under Non-IID Data",
        "=" * 60,
        "",
    ]

    for dataset in ["IIoT", "CIC"]:
        subset = final_df[final_df["dataset"] == dataset]

        if len(subset) == 0:
            continue

        summary_lines.append(f"\n{dataset} Dataset Summary")
        summary_lines.append("-" * 40)

        baseline = subset[subset["mu"] == 0.0]
        if len(baseline) > 0:
            summary_lines.append("\nVanilla FedAvg (mu=0) Performance:")
            for _, row in baseline.iterrows():
                summary_lines.append(
                    f"  alpha={row['alpha']:.2f}: F1={row['final_f1']:.4f} "
                    f"(CI: {row['final_f1_ci_low']:.4f}-{row['final_f1_ci_high']:.4f})"
                )

        summary_lines.append("\nOptimal mu per alpha level:")
        for alpha in sorted(ALPHA_VALUES):
            alpha_data = subset[subset["alpha"] == alpha]
            if len(alpha_data) == 0:
                continue

            best_row = alpha_data.loc[alpha_data["final_f1"].idxmax()]
            baseline_row = alpha_data[alpha_data["mu"] == 0.0]

            if len(baseline_row) > 0:
                baseline_f1 = baseline_row["final_f1"].values[0]
                improvement = best_row["final_f1"] - baseline_f1
                improvement_pct = (improvement / baseline_f1) * 100
                summary_lines.append(
                    f"  alpha={alpha:.2f}: best mu={best_row['mu']:.3f}, "
                    f"F1={best_row['final_f1']:.4f} "
                    f"(+{improvement:.4f}, +{improvement_pct:.2f}% vs FedAvg)"
                )
            else:
                summary_lines.append(
                    f"  alpha={alpha:.2f}: best mu={best_row['mu']:.3f}, "
                    f"F1={best_row['final_f1']:.4f}"
                )

        overall_best = subset.loc[subset["final_f1"].idxmax()]
        summary_lines.append(
            f"\nOverall best: alpha={overall_best['alpha']:.2f}, "
            f"mu={overall_best['mu']:.3f}, F1={overall_best['final_f1']:.4f}"
        )

    summary_text = "\n".join(summary_lines)

    summary_path = output_dir / "objective2_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\nSaved summary: {summary_path}")
    print("\n" + summary_text)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate OBJECTIVE 2 thesis plots: FedProx Effectiveness under Non-IID Data"
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="cluster-experiments/cluster-runs",
        help="Path to cluster runs directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cluster-experiments/thesis_plots/objective2",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    if not runs_dir.exists():
        runs_dir = Path("/Users/abrahamreines/Documents/Thesis") / args.runs_dir

    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    apply_thesis_style()

    print("Loading experiment data...")
    raw_df = load_all_experiments(runs_dir)

    if len(raw_df) == 0:
        print("No experiment data found!")
        return

    print(f"Loaded {len(raw_df)} data points from experiments")
    print(f"Datasets: {raw_df['dataset'].unique()}")
    print(f"Alpha values: {sorted(raw_df['alpha'].unique())}")
    print(f"Mu values: {sorted(raw_df['mu'].unique())}")
    print(f"Seeds: {sorted(raw_df['seed'].unique())}")

    print("\nAggregating across seeds...")
    agg_df = aggregate_across_seeds(raw_df)

    print("Extracting final round metrics...")
    final_df = get_final_round_metrics(agg_df)

    print(f"\nGenerating plots for {len(final_df)} configurations...")

    for dataset in ["IIoT", "CIC"]:
        if dataset not in agg_df["dataset"].values:
            print(f"Skipping {dataset} - no data found")
            continue

        print(f"\n--- {dataset} Dataset ---")

        plot_convergence_curves(agg_df, dataset, output_dir)
        plot_final_f1_vs_alpha(final_df, dataset, output_dir)
        plot_optimal_mu_heatmap(final_df, dataset, output_dir)
        plot_training_stability(final_df, dataset, output_dir)
        plot_fedprox_benefit(final_df, dataset, output_dir)
        plot_mu_sensitivity_by_alpha(agg_df, dataset, output_dir)

    if len(final_df["dataset"].unique()) > 1:
        plot_combined_datasets_comparison(final_df, output_dir)

    generate_summary_statistics(final_df, output_dir)

    print(f"\nAll plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
