#!/usr/bin/env python3
"""
Comprehensive Thesis Plots for Edge-IIoTset-Full Dataset

Generates publication-ready figures for all thesis objectives using
EXCLUSIVELY the edge-iiotset-full experimental results.

Data source: cluster-experiments/cluster-runs/dsedge-iiotset-full_*
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

CLUSTER_RUNS_DIR = Path(__file__).resolve().parents[2] / "cluster-experiments" / "cluster-runs"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "full_iiot_thesis"


def parse_run_config(run_name: str) -> dict:
    """Extract configuration from run directory name."""
    config = {
        "aggregation": "unknown",
        "alpha": None,
        "adversary": 0,
        "seed": 0,
        "mu": 0.0,
        "dp": 0,
        "pers": 0,
    }

    if "comp_fedavg" in run_name or "_fedavg_" in run_name:
        config["aggregation"] = "FedAvg"
    elif "comp_krum" in run_name or "_krum_" in run_name:
        config["aggregation"] = "Krum"
    elif "comp_bulyan" in run_name or "_bulyan_" in run_name:
        config["aggregation"] = "Bulyan"
    elif "comp_median" in run_name or "_median_" in run_name:
        config["aggregation"] = "Median"
    elif "fedprox" in run_name.lower():
        config["aggregation"] = "FedProx"

    alpha_match = re.search(r"alpha([\d.]+|inf)", run_name)
    if alpha_match:
        val = alpha_match.group(1)
        config["alpha"] = float("inf") if val == "inf" else float(val)

    adv_match = re.search(r"adv(\d+)", run_name)
    if adv_match:
        config["adversary"] = int(adv_match.group(1))

    seed_match = re.search(r"seed(\d+)", run_name)
    if seed_match:
        config["seed"] = int(seed_match.group(1))

    mu_match = re.search(r"mu([\d.]+)", run_name)
    if mu_match:
        config["mu"] = float(mu_match.group(1))

    dp_match = re.search(r"dp(\d)", run_name)
    if dp_match:
        config["dp"] = int(dp_match.group(1))

    pers_match = re.search(r"pers(\d+)", run_name)
    if pers_match:
        config["pers"] = int(pers_match.group(1))

    return config


def load_client_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load and combine all client metrics."""
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None

    dfs = []
    for cf in client_files:
        try:
            df = pd.read_csv(cf)
            client_match = re.search(r"client_(\d+)", cf.name)
            if client_match:
                df["client_id"] = int(client_match.group(1))
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_server_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load server metrics."""
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        return None
    try:
        return pd.read_csv(metrics_file)
    except Exception:
        return None


def load_full_iiot_data() -> pd.DataFrame:
    """Load all edge-iiotset-full experimental data."""
    all_data = []

    full_iiot_dirs = [d for d in CLUSTER_RUNS_DIR.iterdir() if d.is_dir() and "edge-iiotset-full" in d.name]

    print(f"Found {len(full_iiot_dirs)} edge-iiotset-full experiment directories")

    for run_dir in full_iiot_dirs:
        config = parse_run_config(run_dir.name)
        if config["alpha"] is None:
            continue

        client_df = load_client_metrics(run_dir)
        server_df = load_server_metrics(run_dir)

        if client_df is not None and "macro_f1_after" in client_df.columns:
            for rnd in client_df["round"].unique():
                round_data = client_df[client_df["round"] == rnd]
                f1_mean = round_data["macro_f1_after"].mean()
                f1_std = round_data["macro_f1_after"].std()

                record = {
                    **config,
                    "round": rnd,
                    "f1_mean": f1_mean,
                    "f1_std": f1_std,
                    "n_clients": len(round_data),
                    "run_dir": run_dir.name,
                }

                if server_df is not None:
                    server_round = server_df[server_df["round"] == rnd]
                    if not server_round.empty:
                        if "l2_dispersion_mean" in server_df.columns:
                            record["l2_dispersion"] = server_round["l2_dispersion_mean"].iloc[0]
                        if "t_aggregate_ms" in server_df.columns:
                            record["t_aggregate_ms"] = server_round["t_aggregate_ms"].iloc[0]

                all_data.append(record)

    return pd.DataFrame(all_data)


def compute_ci(values, confidence=0.95):
    """Compute mean and confidence interval."""
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return values.iloc[0], values.iloc[0], values.iloc[0]

    mean = values.mean()
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - ci, mean + ci


def plot_obj2_heterogeneity(df: pd.DataFrame, output_dir: Path):
    """
    Objective 2: Handling Data Heterogeneity
    6-panel figure showing performance across Dirichlet alpha values.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Objective 2: Handling Data Heterogeneity in Edge-IIoTset-Full",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, :])

    colors = {
        "FedAvg": "#1f77b4",
        "Krum": "#ff7f0e",
        "Bulyan": "#2ca02c",
        "Median": "#d62728",
        "FedProx": "#9467bd",
    }
    markers = {"FedAvg": "o", "Krum": "s", "Bulyan": "D", "Median": "^", "FedProx": "P"}

    benign_df = df[df["adversary"] == 0].copy()
    final_df = benign_df[benign_df["round"] == benign_df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    alpha_order = sorted([a for a in final_df["alpha"].unique() if pd.notna(a) and 0 < a < 100])

    # Panel 1: Performance vs Heterogeneity
    for agg in ["FedAvg", "FedProx"]:
        agg_data = final_df[final_df["aggregation"] == agg]
        if agg_data.empty:
            continue

        summary = []
        for alpha in alpha_order:
            alpha_data = agg_data[np.isclose(agg_data["alpha"], alpha, rtol=0.1)]
            if len(alpha_data) > 0:
                mean, ci_lo, ci_hi = compute_ci(alpha_data["f1_mean"])
                summary.append({"alpha": alpha, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi})

        if not summary:
            continue

        sum_df = pd.DataFrame(summary)
        ax1.plot(
            sum_df["alpha"],
            sum_df["mean"],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            linewidth=2,
            markersize=8,
        )
        ax1.fill_between(
            sum_df["alpha"],
            sum_df["ci_lo"],
            sum_df["ci_hi"],
            color=colors.get(agg, "#333"),
            alpha=0.2,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Dirichlet Alpha (lower = more heterogeneous)")
    ax1.set_ylabel("Final Macro F1 Score")
    ax1.set_title("A) Performance vs Data Heterogeneity", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="IID boundary")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Convergence Speed (rounds to 90% final F1)
    def rounds_to_threshold(group_df, threshold_pct=0.90):
        final_f1 = group_df["f1_mean"].max()
        target = final_f1 * threshold_pct
        for _, row in group_df.sort_values("round").iterrows():
            if row["f1_mean"] >= target:
                return row["round"]
        return group_df["round"].max()

    conv_data = []
    for (agg, alpha, seed), group in benign_df.groupby(["aggregation", "alpha", "seed"]):
        if pd.isna(alpha) or alpha > 100 or alpha < 0.01:
            continue
        r90 = rounds_to_threshold(group, 0.90)
        conv_data.append({"aggregation": agg, "alpha": alpha, "seed": seed, "rounds_to_90": r90})

    conv_df = pd.DataFrame(conv_data)

    for agg in ["FedAvg", "FedProx"]:
        agg_conv = conv_df[conv_df["aggregation"] == agg]
        if agg_conv.empty:
            continue

        summary = []
        for alpha in alpha_order:
            alpha_data = agg_conv[np.isclose(agg_conv["alpha"], alpha, rtol=0.1)]
            if len(alpha_data) > 0:
                mean, ci_lo, ci_hi = compute_ci(alpha_data["rounds_to_90"])
                summary.append({"alpha": alpha, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi})

        if not summary:
            continue

        sum_df = pd.DataFrame(summary)
        ax2.errorbar(
            sum_df["alpha"],
            sum_df["mean"],
            yerr=[sum_df["mean"] - sum_df["ci_lo"], sum_df["ci_hi"] - sum_df["mean"]],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            linewidth=2,
            capsize=3,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Dirichlet Alpha")
    ax2.set_ylabel("Rounds to 90% Final F1")
    ax2.set_title("B) Convergence Speed", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Client Model Drift (L2 Dispersion)
    drift_df = final_df[final_df["l2_dispersion"].notna()].copy()

    for agg in ["FedAvg", "FedProx"]:
        agg_drift = drift_df[drift_df["aggregation"] == agg]
        if agg_drift.empty:
            continue

        summary = []
        for alpha in alpha_order:
            alpha_data = agg_drift[np.isclose(agg_drift["alpha"], alpha, rtol=0.1)]
            if len(alpha_data) > 0:
                mean, ci_lo, ci_hi = compute_ci(alpha_data["l2_dispersion"])
                summary.append({"alpha": alpha, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi})

        if not summary:
            continue

        sum_df = pd.DataFrame(summary)
        ax3.plot(
            sum_df["alpha"],
            sum_df["mean"],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            linewidth=2,
        )

    ax3.set_xscale("log")
    ax3.set_xlabel("Dirichlet Alpha")
    ax3.set_ylabel("Final L2 Dispersion")
    ax3.set_title("C) Client Model Drift", fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Heterogeneity Impact Heatmap
    heatmap_alphas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    heatmap_aggs = ["FedAvg", "FedProx"]

    heatmap_data = np.full((len(heatmap_aggs), len(heatmap_alphas)), np.nan)

    for i, agg in enumerate(heatmap_aggs):
        for j, alpha in enumerate(heatmap_alphas):
            subset = final_df[(final_df["aggregation"] == agg) & (np.isclose(final_df["alpha"], alpha, rtol=0.1))]
            if not subset.empty:
                heatmap_data[i, j] = subset["f1_mean"].mean()

    if not np.all(np.isnan(heatmap_data)):
        sns.heatmap(
            heatmap_data,
            ax=ax4,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            xticklabels=[f"{a}" for a in heatmap_alphas],
            yticklabels=heatmap_aggs,
            vmin=np.nanmin(heatmap_data) - 0.05,
            vmax=np.nanmax(heatmap_data) + 0.02,
            cbar_kws={"label": "Final F1"},
        )
    ax4.set_xlabel("Dirichlet Alpha")
    ax4.set_title("D) Heterogeneity Impact Matrix", fontweight="bold")

    # Panel 5: IID vs Non-IID Performance Bar Chart
    iid_data = final_df[np.isclose(final_df["alpha"], 1.0, rtol=0.1)]
    noniid_data = final_df[(final_df["alpha"] >= 0.02) & (final_df["alpha"] <= 0.1)]

    bar_aggs = ["FedAvg", "FedProx"]
    bar_data = []
    for agg in bar_aggs:
        iid_vals = iid_data[iid_data["aggregation"] == agg]["f1_mean"]
        noniid_vals = noniid_data[noniid_data["aggregation"] == agg]["f1_mean"]
        bar_data.append(
            {
                "Aggregator": agg,
                "IID": iid_vals.mean() if len(iid_vals) > 0 else np.nan,
                "IID_sem": iid_vals.sem() if len(iid_vals) > 1 else 0,
                "Non-IID": noniid_vals.mean() if len(noniid_vals) > 0 else np.nan,
                "Non-IID_sem": noniid_vals.sem() if len(noniid_vals) > 1 else 0,
            }
        )

    bar_df = pd.DataFrame(bar_data)
    x = np.arange(len(bar_aggs))
    width = 0.35

    if not bar_df.empty and not bar_df["IID"].isna().all():
        ax5.bar(x - width / 2, bar_df["IID"], width, yerr=1.96 * bar_df["IID_sem"], label="IID (alpha=1.0)", color="#1f77b4", capsize=3)
        ax5.bar(
            x + width / 2,
            bar_df["Non-IID"],
            width,
            yerr=1.96 * bar_df["Non-IID_sem"],
            label="Non-IID (alpha<=0.1)",
            color="#ff7f0e",
            capsize=3,
        )
        ax5.set_xticks(x)
        ax5.set_xticklabels(bar_aggs)
        ax5.set_ylabel("Macro F1 Score")
        ax5.set_title("E) IID vs Non-IID Performance", fontweight="bold")
        ax5.legend(loc="upper right", fontsize=9)
        ax5.grid(True, alpha=0.3, axis="y")

    # Panel 6: FedAvg Convergence Curves by Alpha
    fedavg_df = benign_df[benign_df["aggregation"] == "FedAvg"].copy()
    conv_alphas = [0.02, 0.1, 0.5, 1.0]
    alpha_colors = {0.02: "#1f77b4", 0.1: "#ff7f0e", 0.5: "#2ca02c", 1.0: "#d62728"}

    for alpha in conv_alphas:
        subset = fedavg_df[np.isclose(fedavg_df["alpha"], alpha, rtol=0.1)]
        if subset.empty:
            continue

        round_f1 = subset.groupby("round")["f1_mean"].agg(["mean", "sem"]).reset_index()

        ax6.plot(
            round_f1["round"],
            round_f1["mean"],
            marker="o",
            color=alpha_colors.get(alpha, "#333"),
            label=f"alpha={alpha}",
            linewidth=2,
            markersize=5,
        )
        ax6.fill_between(
            round_f1["round"],
            round_f1["mean"] - 1.96 * round_f1["sem"],
            round_f1["mean"] + 1.96 * round_f1["sem"],
            color=alpha_colors.get(alpha, "#333"),
            alpha=0.2,
        )

    ax6.set_xlabel("Communication Round")
    ax6.set_ylabel("Macro F1 Score")
    ax6.set_title("F) FedAvg Convergence Under Different Heterogeneity Levels", fontweight="bold")
    ax6.legend(loc="lower right")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "obj2_heterogeneity_full_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_fedprox_mu_analysis(df: pd.DataFrame, output_dir: Path):
    """
    FedProx Mu Sensitivity Analysis
    Shows how proximal term strength affects performance across heterogeneity.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "FedProx Mu Sensitivity Analysis (Edge-IIoTset-Full)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    fedprox_df = df[(df["aggregation"] == "FedProx") & (df["adversary"] == 0)].copy()
    final_df = fedprox_df[fedprox_df["round"] == fedprox_df.groupby(["alpha", "mu", "seed"])["round"].transform("max")]

    mu_values = sorted(final_df["mu"].unique())
    alpha_values = sorted([a for a in final_df["alpha"].unique() if pd.notna(a) and 0 < a < 100])

    mu_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(mu_values)))

    # Panel A: F1 vs Mu for each Alpha
    ax1 = axes[0, 0]
    alpha_subset = [0.02, 0.1, 0.5, 1.0]
    alpha_markers = {0.02: "o", 0.1: "s", 0.5: "D", 1.0: "^"}
    alpha_colors = {0.02: "#1f77b4", 0.1: "#ff7f0e", 0.5: "#2ca02c", 1.0: "#d62728"}

    for alpha in alpha_subset:
        alpha_data = final_df[np.isclose(final_df["alpha"], alpha, rtol=0.1)]
        if alpha_data.empty:
            continue

        summary = (
            alpha_data.groupby("mu")
            .agg(
                f1_mean=("f1_mean", "mean"),
                f1_sem=("f1_mean", "sem"),
            )
            .reset_index()
        )

        ax1.errorbar(
            summary["mu"],
            summary["f1_mean"],
            yerr=1.96 * summary["f1_sem"],
            marker=alpha_markers.get(alpha, "o"),
            color=alpha_colors.get(alpha, "#333"),
            label=f"alpha={alpha}",
            linewidth=2,
            capsize=3,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("FedProx Mu (Proximal Term Strength)")
    ax1.set_ylabel("Final Macro F1 Score")
    ax1.set_title("A) Performance vs Mu by Heterogeneity Level", fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Heatmap of Mu x Alpha
    ax2 = axes[0, 1]
    heatmap_mus = [0.002, 0.01, 0.05, 0.1, 0.2]
    heatmap_alphas = [0.02, 0.1, 0.5, 1.0]

    heatmap_data = np.full((len(heatmap_mus), len(heatmap_alphas)), np.nan)

    for i, mu in enumerate(heatmap_mus):
        for j, alpha in enumerate(heatmap_alphas):
            subset = final_df[(np.isclose(final_df["mu"], mu, rtol=0.1)) & (np.isclose(final_df["alpha"], alpha, rtol=0.1))]
            if not subset.empty:
                heatmap_data[i, j] = subset["f1_mean"].mean()

    if not np.all(np.isnan(heatmap_data)):
        sns.heatmap(
            heatmap_data,
            ax=ax2,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            xticklabels=[f"{a}" for a in heatmap_alphas],
            yticklabels=[f"{m}" for m in heatmap_mus],
            cbar_kws={"label": "Final F1"},
        )
    ax2.set_xlabel("Dirichlet Alpha")
    ax2.set_ylabel("FedProx Mu")
    ax2.set_title("B) Mu x Alpha Interaction Heatmap", fontweight="bold")

    # Panel C: Optimal Mu per Alpha
    ax3 = axes[1, 0]
    optimal_data = []

    for alpha in alpha_values:
        alpha_data = final_df[np.isclose(final_df["alpha"], alpha, rtol=0.1)]
        if alpha_data.empty:
            continue

        mu_perf = alpha_data.groupby("mu")["f1_mean"].mean()
        if not mu_perf.empty:
            best_mu = mu_perf.idxmax()
            best_f1 = mu_perf.max()
            optimal_data.append({"alpha": alpha, "best_mu": best_mu, "best_f1": best_f1})

    if optimal_data:
        opt_df = pd.DataFrame(optimal_data)
        ax3.scatter(opt_df["alpha"], opt_df["best_mu"], s=100, c=opt_df["best_f1"], cmap="RdYlGn", edgecolor="black")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("Dirichlet Alpha")
        ax3.set_ylabel("Optimal Mu")
        ax3.set_title("C) Optimal Mu per Heterogeneity Level", fontweight="bold")
        ax3.grid(True, alpha=0.3)

    # Panel D: FedProx vs FedAvg comparison
    ax4 = axes[1, 1]
    fedavg_df = df[(df["aggregation"] == "FedAvg") & (df["adversary"] == 0)].copy()
    fedavg_final = fedavg_df[fedavg_df["round"] == fedavg_df.groupby(["alpha", "seed"])["round"].transform("max")]

    comparison_data = []
    for alpha in alpha_values:
        fedavg_vals = fedavg_final[np.isclose(fedavg_final["alpha"], alpha, rtol=0.1)]["f1_mean"]
        fedprox_vals = final_df[np.isclose(final_df["alpha"], alpha, rtol=0.1)]["f1_mean"]

        if len(fedavg_vals) > 0 and len(fedprox_vals) > 0:
            comparison_data.append(
                {
                    "alpha": alpha,
                    "FedAvg": fedavg_vals.mean(),
                    "FedAvg_sem": fedavg_vals.sem() if len(fedavg_vals) > 1 else 0,
                    "FedProx_best": fedprox_vals.max(),
                    "FedProx_mean": fedprox_vals.mean(),
                }
            )

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x = np.arange(len(comp_df))
        width = 0.35

        ax4.bar(x - width / 2, comp_df["FedAvg"], width, label="FedAvg", color="#1f77b4")
        ax4.bar(x + width / 2, comp_df["FedProx_mean"], width, label="FedProx (mean)", color="#9467bd")

        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{a}" for a in comp_df["alpha"]])
        ax4.set_xlabel("Dirichlet Alpha")
        ax4.set_ylabel("Final Macro F1 Score")
        ax4.set_title("D) FedAvg vs FedProx Comparison", fontweight="bold")
        ax4.legend(loc="lower right", fontsize=9)
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "fedprox_mu_analysis_full_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_convergence_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Convergence Analysis
    Shows learning curves and convergence characteristics.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Convergence Analysis (Edge-IIoTset-Full)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    benign_df = df[df["adversary"] == 0].copy()

    # Panel A: FedAvg convergence by alpha
    ax1 = axes[0, 0]
    fedavg_df = benign_df[benign_df["aggregation"] == "FedAvg"]
    conv_alphas = [0.02, 0.05, 0.1, 0.5, 1.0]
    alpha_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(conv_alphas)))

    for idx, alpha in enumerate(conv_alphas):
        subset = fedavg_df[np.isclose(fedavg_df["alpha"], alpha, rtol=0.1)]
        if subset.empty:
            continue

        round_f1 = subset.groupby("round")["f1_mean"].agg(["mean", "sem"]).reset_index()

        ax1.plot(
            round_f1["round"],
            round_f1["mean"],
            color=alpha_colors[idx],
            label=f"alpha={alpha}",
            linewidth=2,
        )
        ax1.fill_between(
            round_f1["round"],
            round_f1["mean"] - 1.96 * round_f1["sem"],
            round_f1["mean"] + 1.96 * round_f1["sem"],
            color=alpha_colors[idx],
            alpha=0.2,
        )

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("A) FedAvg Convergence by Heterogeneity", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: FedProx convergence by mu (alpha=0.1)
    ax2 = axes[0, 1]
    fedprox_df = benign_df[(benign_df["aggregation"] == "FedProx") & (np.isclose(benign_df["alpha"], 0.1, rtol=0.1))]
    mu_values = [0.01, 0.05, 0.1, 0.2]
    mu_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(mu_values)))

    for idx, mu in enumerate(mu_values):
        subset = fedprox_df[np.isclose(fedprox_df["mu"], mu, rtol=0.1)]
        if subset.empty:
            continue

        round_f1 = subset.groupby("round")["f1_mean"].agg(["mean", "sem"]).reset_index()

        ax2.plot(
            round_f1["round"],
            round_f1["mean"],
            color=mu_colors[idx],
            label=f"mu={mu}",
            linewidth=2,
        )

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("B) FedProx Convergence by Mu (alpha=0.1)", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel C: First-round performance distribution
    ax3 = axes[1, 0]
    first_round = benign_df[benign_df["round"] == 1]
    first_by_alpha = first_round.groupby(["aggregation", "alpha"])["f1_mean"].mean().reset_index()

    for agg in ["FedAvg", "FedProx"]:
        agg_data = first_by_alpha[first_by_alpha["aggregation"] == agg]
        if not agg_data.empty:
            ax3.scatter(
                agg_data["alpha"],
                agg_data["f1_mean"],
                label=agg,
                s=80,
                alpha=0.7,
            )

    ax3.set_xscale("log")
    ax3.set_xlabel("Dirichlet Alpha")
    ax3.set_ylabel("Round 1 Macro F1")
    ax3.set_title("C) Initial Performance by Heterogeneity", fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel D: Final vs Initial F1 scatter
    ax4 = axes[1, 1]

    initial_final = []
    for (agg, alpha, seed), group in benign_df.groupby(["aggregation", "alpha", "seed"]):
        group_sorted = group.sort_values("round")
        if len(group_sorted) >= 2:
            initial_f1 = group_sorted["f1_mean"].iloc[0]
            final_f1 = group_sorted["f1_mean"].iloc[-1]
            initial_final.append(
                {
                    "aggregation": agg,
                    "alpha": alpha,
                    "initial_f1": initial_f1,
                    "final_f1": final_f1,
                    "gain": final_f1 - initial_f1,
                }
            )

    if initial_final:
        if_df = pd.DataFrame(initial_final)

        for agg in ["FedAvg", "FedProx"]:
            agg_data = if_df[if_df["aggregation"] == agg]
            if not agg_data.empty:
                ax4.scatter(
                    agg_data["initial_f1"],
                    agg_data["final_f1"],
                    label=agg,
                    s=50,
                    alpha=0.6,
                )

        ax4.plot([0, 1], [0, 1], "k--", alpha=0.5, label="No improvement")
        ax4.set_xlabel("Initial F1 (Round 1)")
        ax4.set_ylabel("Final F1")
        ax4.set_title("D) Training Improvement", fontweight="bold")
        ax4.legend(loc="lower right", fontsize=9)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "convergence_analysis_full_iiot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def generate_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics CSV."""
    benign_df = df[df["adversary"] == 0].copy()
    final_df = benign_df[benign_df["round"] == benign_df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    summary = (
        final_df.groupby(["aggregation", "alpha"])
        .agg(
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            f1_min=("f1_mean", "min"),
            f1_max=("f1_mean", "max"),
            n_seeds=("seed", "nunique"),
            n_records=("f1_mean", "count"),
        )
        .reset_index()
    )

    output_path = output_dir / "summary_stats_full_iiot.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(summary.to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL IIOT THESIS PLOTTING")
    print("=" * 70)
    print(f"Data source: {CLUSTER_RUNS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print("Loading data...")
    df = load_full_iiot_data()

    if df.empty:
        print("ERROR: No data found!")
        return 1

    print(f"Loaded {len(df)} records")
    print(f"Unique seeds: {df['seed'].nunique()}")
    print(f"Alpha values: {sorted(df['alpha'].dropna().unique())}")
    print(f"Aggregations: {df['aggregation'].value_counts().to_dict()}")
    print(f"Adversary levels: {df['adversary'].unique()}")

    has_attack = (df["adversary"] > 0).any()
    has_dp = (df["dp"] > 0).any()
    has_pers = (df["pers"] > 0).any()

    print()
    print("=" * 70)
    print("DATA AVAILABILITY CHECK")
    print("=" * 70)
    print(f"Attack experiments (Obj 1): {'YES' if has_attack else 'NO - benign only'}")
    print(f"DP experiments (Obj 4): {'YES' if has_dp else 'NO'}")
    print(f"Personalization experiments (Obj 3): {'YES' if has_pers else 'NO'}")
    print()

    print("=" * 70)
    print("GENERATING OBJECTIVE 2: HETEROGENEITY PLOTS")
    print("=" * 70)
    plot_obj2_heterogeneity(df, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("GENERATING FEDPROX MU ANALYSIS")
    print("=" * 70)
    plot_fedprox_mu_analysis(df, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("GENERATING CONVERGENCE ANALYSIS")
    print("=" * 70)
    plot_convergence_analysis(df, OUTPUT_DIR)

    print()
    generate_summary_stats(df, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"All plots saved to: {OUTPUT_DIR}")

    if not has_attack:
        print("\nNOTE: Objective 1 (Attack Resilience) plots not generated - no adversary experiments in full IIoT data")
    if not has_pers:
        print("NOTE: Objective 3 (Personalization) plots not generated - no personalization experiments in full IIoT data")
    if not has_dp:
        print("NOTE: Objective 4 (Privacy) plots not generated - no DP experiments in full IIoT data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
