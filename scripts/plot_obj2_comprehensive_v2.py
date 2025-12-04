#!/usr/bin/env python3
"""
Comprehensive Objective 2 Plot V2: Handling Data Heterogeneity in IIoT Federated IDS.

Generates a publication-ready 6-panel figure showing:
1. Performance vs Data Heterogeneity (FedAvg focus, full alpha range)
2. Convergence Speed by Alpha
3. Client Model Drift
4. Heterogeneity Impact Matrix (heatmap)
5. IID vs Non-IID Performance (bar chart)
6. FedAvg Convergence Under Different Heterogeneity Levels
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("thesis_plots_iiot")


def parse_run_config(run_name: str) -> dict:
    """Extract configuration from run directory name."""
    config = {
        "aggregation": "unknown",
        "alpha": None,
        "adversary": 0,
        "seed": 0,
        "mu": 0.0,
    }

    if "comp_fedavg" in run_name:
        config["aggregation"] = "FedAvg"
    elif "comp_krum" in run_name:
        config["aggregation"] = "Krum"
    elif "comp_bulyan" in run_name:
        config["aggregation"] = "Bulyan"
    elif "comp_median" in run_name:
        config["aggregation"] = "Median"
    elif "fedprox" in run_name.lower():
        config["aggregation"] = "FedProx"

    alpha_match = re.search(r"_alpha([\d.]+|inf)_", run_name)
    if alpha_match:
        val = alpha_match.group(1)
        config["alpha"] = float("inf") if val == "inf" else float(val)

    adv_match = re.search(r"_adv(\d+)_", run_name)
    if adv_match:
        config["adversary"] = int(adv_match.group(1))

    seed_match = re.search(r"_seed(\d+)", run_name)
    if seed_match:
        config["seed"] = int(seed_match.group(1))

    mu_match = re.search(r"_mu([\d.]+)_", run_name)
    if mu_match:
        config["mu"] = float(mu_match.group(1))

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


def collect_all_data():
    """Collect data from all runs."""
    all_data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = parse_run_config(run_dir.name)
        if config["alpha"] is None:
            continue

        if config["adversary"] != 0:
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
                }

                if server_df is not None and "l2_dispersion_mean" in server_df.columns:
                    server_round = server_df[server_df["round"] == rnd]
                    if not server_round.empty:
                        record["l2_dispersion"] = server_round["l2_dispersion_mean"].iloc[0]

                all_data.append(record)

    return pd.DataFrame(all_data)


def plot_comprehensive_obj2(df: pd.DataFrame, output_path: Path):
    """Generate the comprehensive 6-panel Objective 2 figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Objective 2: Handling Data Heterogeneity in IIoT Federated IDS",
        fontsize=16,
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

    final_df = df[df["round"] == df.groupby(["aggregation", "alpha", "seed"])["round"].transform("max")]

    alpha_order = sorted([a for a in final_df["alpha"].unique() if 0 < a < 100])

    agg_alpha_f1 = (
        final_df.groupby(["aggregation", "alpha"])
        .agg(f1_mean=("f1_mean", "mean"), f1_sem=("f1_mean", "sem"), n=("f1_mean", "count"))
        .reset_index()
    )

    # Panel 1: Performance vs Heterogeneity - Focus on FedAvg and FedProx (full coverage)
    for agg in ["FedAvg", "FedProx"]:
        subset = agg_alpha_f1[agg_alpha_f1["aggregation"] == agg].copy()
        subset = subset[subset["alpha"].isin(alpha_order)].sort_values("alpha")

        if subset.empty or len(subset) < 2:
            continue

        ax1.plot(
            subset["alpha"],
            subset["f1_mean"],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            linewidth=2,
            markersize=8,
        )
        ax1.fill_between(
            subset["alpha"],
            subset["f1_mean"] - 1.96 * subset["f1_sem"],
            subset["f1_mean"] + 1.96 * subset["f1_sem"],
            color=colors.get(agg, "#333"),
            alpha=0.2,
        )

    # Add robust aggregators only at their reliable data points (alpha >= 0.5)
    for agg in ["Krum", "Bulyan", "Median"]:
        subset = agg_alpha_f1[agg_alpha_f1["aggregation"] == agg].copy()
        subset = subset[(subset["alpha"] >= 0.5) & (subset["alpha"].isin(alpha_order))].sort_values("alpha")

        if subset.empty:
            continue

        ax1.scatter(
            subset["alpha"],
            subset["f1_mean"],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            s=100,
            zorder=5,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Dirichlet $\\alpha$ (Heterogeneity: 0=high, 1=low)")
    ax1.set_ylabel("Final Macro F1 Score")
    ax1.set_title("Performance vs Data Heterogeneity (Benign)", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.55, 0.80)

    # Panel 2: Convergence Speed
    def rounds_to_threshold(group_df, threshold_pct=0.90):
        final_f1 = group_df["f1_mean"].max()
        target = final_f1 * threshold_pct
        for _, row in group_df.sort_values("round").iterrows():
            if row["f1_mean"] >= target:
                return row["round"]
        return group_df["round"].max()

    conv_data = []
    for (agg, alpha, seed), group in df.groupby(["aggregation", "alpha", "seed"]):
        if alpha > 100:
            continue
        r90 = rounds_to_threshold(group, 0.90)
        conv_data.append({"aggregation": agg, "alpha": alpha, "seed": seed, "rounds_to_90": r90})

    conv_df = pd.DataFrame(conv_data)
    conv_summary = conv_df.groupby(["aggregation", "alpha"]).agg(
        r90_mean=("rounds_to_90", "mean"), r90_sem=("rounds_to_90", "sem")
    ).reset_index()

    for agg in ["FedAvg", "FedProx"]:
        subset = conv_summary[conv_summary["aggregation"] == agg].copy()
        subset = subset[subset["alpha"].isin(alpha_order)].sort_values("alpha")
        if subset.empty or len(subset) < 2:
            continue

        ax2.errorbar(
            subset["alpha"],
            subset["r90_mean"],
            yerr=1.96 * subset["r90_sem"],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            linewidth=2,
            capsize=3,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Dirichlet $\\alpha$")
    ax2.set_ylabel("Rounds to 90% Final F1")
    ax2.set_title("Convergence Speed", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Client Model Drift
    drift_df = final_df[final_df["l2_dispersion"].notna()].copy()
    drift_summary = drift_df.groupby(["aggregation", "alpha"]).agg(
        drift_mean=("l2_dispersion", "mean"), drift_sem=("l2_dispersion", "sem")
    ).reset_index()

    for agg in ["FedAvg", "FedProx"]:
        subset = drift_summary[drift_summary["aggregation"] == agg].copy()
        subset = subset[subset["alpha"].isin(alpha_order)].sort_values("alpha")
        if subset.empty or len(subset) < 2:
            continue

        ax3.plot(
            subset["alpha"],
            subset["drift_mean"],
            marker=markers.get(agg, "o"),
            color=colors.get(agg, "#333"),
            label=agg,
            linewidth=2,
        )

    ax3.set_xscale("log")
    ax3.set_xlabel("Dirichlet $\\alpha$")
    ax3.set_ylabel("Final L2 Dispersion")
    ax3.set_title("Client Model Drift", fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Heterogeneity Impact Matrix (FedAvg and FedProx)
    heatmap_alphas = [0.005, 0.02, 0.1, 0.5, 1.0]
    heatmap_aggs = ["FedAvg", "FedProx"]

    heatmap_data = np.full((len(heatmap_aggs), len(heatmap_alphas)), np.nan)

    for i, agg in enumerate(heatmap_aggs):
        for j, alpha in enumerate(heatmap_alphas):
            subset = final_df[
                (final_df["aggregation"] == agg) & 
                (np.isclose(final_df["alpha"], alpha, rtol=0.1))
            ]
            if not subset.empty:
                heatmap_data[i, j] = subset["f1_mean"].mean()

    sns.heatmap(
        heatmap_data,
        ax=ax4,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=[f"$\\alpha$={a}" for a in heatmap_alphas],
        yticklabels=heatmap_aggs,
        vmin=0.65,
        vmax=0.75,
        cbar_kws={"label": "Final F1"},
    )
    ax4.set_title("Heterogeneity Impact Matrix", fontweight="bold")

    # Panel 5: IID vs Non-IID Performance
    iid_data = final_df[np.isclose(final_df["alpha"], 1.0, rtol=0.1)]
    noniid_data = final_df[final_df["alpha"] <= 0.02]  # Extreme non-IID (0.005, 0.01, 0.02)

    bar_aggs = ["FedAvg", "FedProx"]
    bar_data = []
    for agg in bar_aggs:
        iid_vals = iid_data[iid_data["aggregation"] == agg]["f1_mean"]
        noniid_vals = noniid_data[noniid_data["aggregation"] == agg]["f1_mean"]
        bar_data.append({
            "Aggregator": agg,
            "IID": iid_vals.mean() if len(iid_vals) > 0 else np.nan,
            "IID_sem": iid_vals.sem() if len(iid_vals) > 1 else 0,
            "Non-IID": noniid_vals.mean() if len(noniid_vals) > 0 else np.nan,
            "Non-IID_sem": noniid_vals.sem() if len(noniid_vals) > 1 else 0,
        })

    bar_df = pd.DataFrame(bar_data)
    x = np.arange(len(bar_aggs))
    width = 0.35

    if not bar_df.empty:
        ax5.bar(
            x - width / 2, bar_df["IID"], width,
            yerr=1.96 * bar_df["IID_sem"],
            label="IID ($\\alpha$=1.0)", color="#1f77b4", capsize=3
        )
        ax5.bar(
            x + width / 2, bar_df["Non-IID"], width,
            yerr=1.96 * bar_df["Non-IID_sem"],
            label="Non-IID ($\\alpha$$\\leq$0.02)", color="#ff7f0e", capsize=3
        )
        ax5.set_xticks(x)
        ax5.set_xticklabels(bar_aggs)
        ax5.set_ylabel("F1 Score")
        ax5.set_title("IID vs Non-IID Performance", fontweight="bold")
        ax5.legend(loc="upper right", fontsize=8)
        ax5.set_ylim(0.60, 0.80)
        ax5.grid(True, alpha=0.3, axis="y")

    # Panel 6: FedAvg Convergence Curves
    fedavg_df = df[df["aggregation"] == "FedAvg"].copy()
    conv_alphas = [0.005, 0.02, 0.1, 0.5, 1.0]
    alpha_colors = {0.005: "#9467bd", 0.02: "#1f77b4", 0.1: "#ff7f0e", 0.5: "#2ca02c", 1.0: "#d62728"}

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
            label=f"$\\alpha$={alpha}",
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
    ax6.set_title("FedAvg Convergence Under Different Heterogeneity Levels", fontweight="bold")
    ax6.legend(loc="lower right")
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting data from runs...")
    df = collect_all_data()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} records from {df['seed'].nunique()} unique seeds")
    print(f"Alpha values: {sorted(df['alpha'].unique())}")
    print(f"Aggregations: {df['aggregation'].unique()}")

    output_path = OUTPUT_DIR / "obj2_heterogeneity_comprehensive.png"
    plot_comprehensive_obj2(df, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
