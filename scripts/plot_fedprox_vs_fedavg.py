#!/usr/bin/env python3
"""
FedProx vs FedAvg Comparison Plots (Edge-IIoTset-Full)

Creates focused visualizations showing when FedProx outperforms or underperforms
FedAvg across different heterogeneity levels and mu values.

Excludes 5 smallest classes (FINGERPRINTING, MITM, DDOS_TCP, XSS, RANSOMWARE)
to compute a more representative "Top-10 Class Macro F1" metric.
"""

import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

CLUSTER_RUNS_DIR = Path(__file__).resolve().parents[2] / "cluster-experiments" / "cluster-runs"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "fedprox_vs_fedavg"

EXCLUDE_CLASS_INDICES = {1, 10, 11, 13, 14}
EXCLUDE_CLASS_NAMES = {"FINGERPRINTING", "MITM", "DDOS_TCP", "XSS", "RANSOMWARE"}


def parse_run_config(run_name: str) -> dict:
    """Extract configuration from run directory name."""
    config = {
        "aggregation": "unknown",
        "alpha": None,
        "seed": 0,
        "mu": 0.0,
    }

    if "comp_fedavg" in run_name or "_fedavg_" in run_name:
        config["aggregation"] = "FedAvg"
    elif "fedprox" in run_name.lower():
        config["aggregation"] = "FedProx"
    else:
        return config

    alpha_match = re.search(r"alpha([\d.]+|inf)", run_name)
    if alpha_match:
        val = alpha_match.group(1)
        config["alpha"] = float("inf") if val == "inf" else float(val)

    seed_match = re.search(r"seed(\d+)", run_name)
    if seed_match:
        config["seed"] = int(seed_match.group(1))

    mu_match = re.search(r"mu([\d.]+)", run_name)
    if mu_match:
        config["mu"] = float(mu_match.group(1))

    return config


def compute_top10_f1(f1_per_class_str: str) -> float:
    """Compute macro F1 excluding 5 smallest classes."""
    try:
        f1_dict = ast.literal_eval(f1_per_class_str)
        f1_values = []
        for cls_idx, f1_val in f1_dict.items():
            if int(cls_idx) not in EXCLUDE_CLASS_INDICES:
                f1_values.append(f1_val)
        return np.mean(f1_values) if f1_values else np.nan
    except:
        return np.nan


def load_experiment_data(run_dir: Path) -> dict | None:
    """Load final round metrics from an experiment."""
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None

    all_f1 = []
    top10_f1 = []

    for cf in client_files:
        try:
            df = pd.read_csv(cf)
            if df.empty:
                continue

            final_row = df.iloc[-1]

            if "macro_f1_after" in df.columns:
                all_f1.append(final_row["macro_f1_after"])

            if "f1_per_class_after" in df.columns:
                t10 = compute_top10_f1(final_row["f1_per_class_after"])
                if not np.isnan(t10):
                    top10_f1.append(t10)
        except:
            continue

    if not all_f1:
        return None

    return {
        "macro_f1_all": np.mean(all_f1),
        "macro_f1_top10": np.mean(top10_f1) if top10_f1 else np.nan,
        "n_clients": len(all_f1),
    }


def load_all_data() -> pd.DataFrame:
    """Load all FedAvg and FedProx experiments from full IIoT."""
    records = []

    full_iiot_dirs = [d for d in CLUSTER_RUNS_DIR.iterdir() if d.is_dir() and "edge-iiotset-full" in d.name]

    print(f"Found {len(full_iiot_dirs)} edge-iiotset-full directories")

    for run_dir in full_iiot_dirs:
        config = parse_run_config(run_dir.name)

        if config["aggregation"] not in ("FedAvg", "FedProx"):
            continue
        if config["alpha"] is None:
            continue

        metrics = load_experiment_data(run_dir)
        if metrics is None:
            continue

        records.append(
            {
                **config,
                **metrics,
                "run_dir": run_dir.name,
            }
        )

    return pd.DataFrame(records)


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


def plot_fedprox_vs_fedavg_comparison(df: pd.DataFrame, output_dir: Path):
    """Main comparison figure: 6-panel FedProx vs FedAvg analysis."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "FedProx vs FedAvg: When Does Proximal Regularization Help?\n" "(Edge-IIoTset-Full, Top-10 Classes Macro F1)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    fedavg_df = df[df["aggregation"] == "FedAvg"].copy()
    fedprox_df = df[df["aggregation"] == "FedProx"].copy()

    alpha_order = sorted([a for a in df["alpha"].unique() if pd.notna(a) and 0 < a < 100])

    colors = {"FedAvg": "#1f77b4", "FedProx": "#9467bd", "better": "#2ca02c", "worse": "#d62728"}

    # =========================================================================
    # Panel A: Top-10 F1 vs Alpha (FedAvg vs FedProx best mu)
    # =========================================================================
    fedavg_by_alpha = []
    fedprox_best_by_alpha = []

    for alpha in alpha_order:
        fa = fedavg_df[np.isclose(fedavg_df["alpha"], alpha, rtol=0.1)]
        if not fa.empty:
            mean, ci_lo, ci_hi = compute_ci(fa["macro_f1_top10"])
            fedavg_by_alpha.append({"alpha": alpha, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi})

        fp = fedprox_df[np.isclose(fedprox_df["alpha"], alpha, rtol=0.1)]
        if not fp.empty:
            best_mu_perf = fp.groupby("mu")["macro_f1_top10"].mean()
            best_mu = best_mu_perf.idxmax()
            fp_best = fp[np.isclose(fp["mu"], best_mu, rtol=0.1)]
            mean, ci_lo, ci_hi = compute_ci(fp_best["macro_f1_top10"])
            fedprox_best_by_alpha.append({"alpha": alpha, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "best_mu": best_mu})

    if fedavg_by_alpha:
        fa_df = pd.DataFrame(fedavg_by_alpha)
        ax1.plot(fa_df["alpha"], fa_df["mean"], "o-", color=colors["FedAvg"], label="FedAvg", linewidth=2, markersize=8)
        ax1.fill_between(fa_df["alpha"], fa_df["ci_lo"], fa_df["ci_hi"], color=colors["FedAvg"], alpha=0.2)

    if fedprox_best_by_alpha:
        fp_df = pd.DataFrame(fedprox_best_by_alpha)
        ax1.plot(fp_df["alpha"], fp_df["mean"], "s-", color=colors["FedProx"], label="FedProx (best mu)", linewidth=2, markersize=8)
        ax1.fill_between(fp_df["alpha"], fp_df["ci_lo"], fp_df["ci_hi"], color=colors["FedProx"], alpha=0.2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Dirichlet Alpha (lower = more heterogeneous)")
    ax1.set_ylabel("Top-10 Class Macro F1")
    ax1.set_title("A) Performance vs Heterogeneity", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    # =========================================================================
    # Panel B: Delta F1 (FedProx - FedAvg) by Alpha
    # =========================================================================
    delta_data = []
    for fa_row in fedavg_by_alpha:
        alpha = fa_row["alpha"]
        fp_row = next((x for x in fedprox_best_by_alpha if np.isclose(x["alpha"], alpha, rtol=0.1)), None)
        if fp_row:
            delta = fp_row["mean"] - fa_row["mean"]
            delta_data.append(
                {
                    "alpha": alpha,
                    "delta": delta,
                    "best_mu": fp_row["best_mu"],
                    "fedavg": fa_row["mean"],
                    "fedprox": fp_row["mean"],
                }
            )

    if delta_data:
        delta_df = pd.DataFrame(delta_data)
        bar_colors = [colors["better"] if d > 0 else colors["worse"] for d in delta_df["delta"]]

        bars = ax2.bar(range(len(delta_df)), delta_df["delta"] * 100, color=bar_colors, edgecolor="black")
        ax2.axhline(0, color="black", linewidth=1)

        ax2.set_xticks(range(len(delta_df)))
        ax2.set_xticklabels([f"a={a}" for a in delta_df["alpha"]], rotation=45, ha="right")
        ax2.set_ylabel("Delta F1 (FedProx - FedAvg) [%]")
        ax2.set_title("B) FedProx Advantage/Disadvantage by Alpha", fontweight="bold")

        for i, (_, row) in enumerate(delta_df.iterrows()):
            label = f"mu={row['best_mu']}"
            y_pos = row["delta"] * 100
            va = "bottom" if y_pos >= 0 else "top"
            ax2.annotate(label, (i, y_pos), ha="center", va=va, fontsize=8, rotation=90)

        ax2.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # Panel C: FedProx F1 by Mu (heatmap across alpha)
    # =========================================================================
    mu_values = sorted(fedprox_df["mu"].unique())
    heatmap_data = np.full((len(mu_values), len(alpha_order)), np.nan)

    for i, mu in enumerate(mu_values):
        for j, alpha in enumerate(alpha_order):
            subset = fedprox_df[(np.isclose(fedprox_df["mu"], mu, rtol=0.1)) & (np.isclose(fedprox_df["alpha"], alpha, rtol=0.1))]
            if not subset.empty:
                heatmap_data[i, j] = subset["macro_f1_top10"].mean()

    if not np.all(np.isnan(heatmap_data)):
        sns.heatmap(
            heatmap_data,
            ax=ax3,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            xticklabels=[f"{a}" for a in alpha_order],
            yticklabels=[f"{m}" for m in mu_values],
            cbar_kws={"label": "Top-10 F1"},
            vmin=np.nanmin(heatmap_data) - 0.02,
            vmax=np.nanmax(heatmap_data) + 0.02,
        )
    ax3.set_xlabel("Dirichlet Alpha")
    ax3.set_ylabel("FedProx Mu")
    ax3.set_title("C) FedProx Performance: Mu x Alpha", fontweight="bold")

    # =========================================================================
    # Panel D: FedProx vs FedAvg Delta Heatmap
    # =========================================================================
    delta_heatmap = np.full((len(mu_values), len(alpha_order)), np.nan)

    for i, mu in enumerate(mu_values):
        for j, alpha in enumerate(alpha_order):
            fp_subset = fedprox_df[(np.isclose(fedprox_df["mu"], mu, rtol=0.1)) & (np.isclose(fedprox_df["alpha"], alpha, rtol=0.1))]
            fa_subset = fedavg_df[np.isclose(fedavg_df["alpha"], alpha, rtol=0.1)]

            if not fp_subset.empty and not fa_subset.empty:
                delta_heatmap[i, j] = (fp_subset["macro_f1_top10"].mean() - fa_subset["macro_f1_top10"].mean()) * 100

    if not np.all(np.isnan(delta_heatmap)):
        max_abs = np.nanmax(np.abs(delta_heatmap))
        sns.heatmap(
            delta_heatmap,
            ax=ax4,
            annot=True,
            fmt=".1f",
            cmap="RdBu_r",
            center=0,
            xticklabels=[f"{a}" for a in alpha_order],
            yticklabels=[f"{m}" for m in mu_values],
            cbar_kws={"label": "Delta F1 [%]"},
            vmin=-max_abs,
            vmax=max_abs,
        )
    ax4.set_xlabel("Dirichlet Alpha")
    ax4.set_ylabel("FedProx Mu")
    ax4.set_title("D) FedProx - FedAvg Delta [%] (Green=FedProx wins)", fontweight="bold")

    # =========================================================================
    # Panel E: Detailed comparison with error bars (all mu values at alpha=0.1)
    # =========================================================================
    target_alpha = 0.1
    fa_at_alpha = fedavg_df[np.isclose(fedavg_df["alpha"], target_alpha, rtol=0.1)]
    fp_at_alpha = fedprox_df[np.isclose(fedprox_df["alpha"], target_alpha, rtol=0.1)]

    if not fa_at_alpha.empty and not fp_at_alpha.empty:
        fa_mean, fa_lo, fa_hi = compute_ci(fa_at_alpha["macro_f1_top10"])

        comparison_data = [{"method": "FedAvg\n(mu=0)", "mean": fa_mean, "ci_lo": fa_lo, "ci_hi": fa_hi}]

        for mu in sorted(fp_at_alpha["mu"].unique()):
            fp_mu = fp_at_alpha[np.isclose(fp_at_alpha["mu"], mu, rtol=0.1)]
            if not fp_mu.empty:
                mean, ci_lo, ci_hi = compute_ci(fp_mu["macro_f1_top10"])
                comparison_data.append(
                    {
                        "method": f"FedProx\nmu={mu}",
                        "mean": mean,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )

        comp_df = pd.DataFrame(comparison_data)
        x = np.arange(len(comp_df))

        bar_colors = [colors["FedAvg"]] + [
            colors["better"] if comp_df.iloc[i]["mean"] > fa_mean else colors["worse"] for i in range(1, len(comp_df))
        ]

        ax5.bar(x, comp_df["mean"], color=bar_colors, edgecolor="black", alpha=0.8)
        ax5.errorbar(
            x,
            comp_df["mean"],
            yerr=[comp_df["mean"] - comp_df["ci_lo"], comp_df["ci_hi"] - comp_df["mean"]],
            fmt="none",
            color="black",
            capsize=4,
        )

        ax5.axhline(fa_mean, color=colors["FedAvg"], linestyle="--", linewidth=2, label=f"FedAvg baseline ({fa_mean:.3f})")

        ax5.set_xticks(x)
        ax5.set_xticklabels(comp_df["method"], fontsize=9)
        ax5.set_ylabel("Top-10 Class Macro F1")
        ax5.set_title(f"E) Detailed Comparison at alpha={target_alpha} (High Non-IID)", fontweight="bold")
        ax5.legend(loc="upper right")
        ax5.grid(True, alpha=0.3, axis="y")

        wins = sum(1 for i in range(1, len(comp_df)) if comp_df.iloc[i]["mean"] > fa_mean)
        losses = len(comp_df) - 1 - wins
        ax5.annotate(
            f"FedProx wins: {wins}/{len(comp_df)-1}\nFedProx loses: {losses}/{len(comp_df)-1}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "fedprox_vs_fedavg_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_win_loss_summary(df: pd.DataFrame, output_dir: Path):
    """Summary plot showing win/loss/tie counts across all configurations."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "FedProx vs FedAvg Win/Loss Analysis (Top-10 Class Macro F1)",
        fontsize=14,
        fontweight="bold",
    )

    fedavg_df = df[df["aggregation"] == "FedAvg"].copy()
    fedprox_df = df[df["aggregation"] == "FedProx"].copy()

    alpha_order = sorted([a for a in df["alpha"].unique() if pd.notna(a) and 0 < a < 100])
    mu_values = sorted(fedprox_df["mu"].unique())

    # Compute win/loss for each (alpha, mu) pair
    results = []
    for alpha in alpha_order:
        fa = fedavg_df[np.isclose(fedavg_df["alpha"], alpha, rtol=0.1)]
        fa_mean = fa["macro_f1_top10"].mean() if not fa.empty else np.nan

        for mu in mu_values:
            fp = fedprox_df[(np.isclose(fedprox_df["alpha"], alpha, rtol=0.1)) & (np.isclose(fedprox_df["mu"], mu, rtol=0.1))]
            if not fp.empty and not np.isnan(fa_mean):
                fp_mean = fp["macro_f1_top10"].mean()
                delta = fp_mean - fa_mean
                results.append(
                    {
                        "alpha": alpha,
                        "mu": mu,
                        "fedavg": fa_mean,
                        "fedprox": fp_mean,
                        "delta": delta,
                        "winner": "FedProx" if delta > 0.001 else ("FedAvg" if delta < -0.001 else "Tie"),
                    }
                )

    results_df = pd.DataFrame(results)

    # Panel 1: Win counts by alpha
    ax1 = axes[0]
    win_by_alpha = results_df.groupby(["alpha", "winner"]).size().unstack(fill_value=0)
    win_by_alpha = win_by_alpha.reindex(columns=["FedProx", "Tie", "FedAvg"], fill_value=0)

    win_by_alpha.plot(kind="bar", ax=ax1, color=["#2ca02c", "#7f7f7f", "#d62728"], edgecolor="black")
    ax1.set_xlabel("Dirichlet Alpha")
    ax1.set_ylabel("Count")
    ax1.set_title("Win/Loss by Heterogeneity Level", fontweight="bold")
    ax1.legend(title="Winner")
    ax1.tick_params(axis="x", rotation=45)

    # Panel 2: Win counts by mu
    ax2 = axes[1]
    win_by_mu = results_df.groupby(["mu", "winner"]).size().unstack(fill_value=0)
    win_by_mu = win_by_mu.reindex(columns=["FedProx", "Tie", "FedAvg"], fill_value=0)

    win_by_mu.plot(kind="bar", ax=ax2, color=["#2ca02c", "#7f7f7f", "#d62728"], edgecolor="black")
    ax2.set_xlabel("FedProx Mu")
    ax2.set_ylabel("Count")
    ax2.set_title("Win/Loss by Mu Value", fontweight="bold")
    ax2.legend(title="Winner")
    ax2.tick_params(axis="x", rotation=45)

    # Panel 3: Overall summary pie chart
    ax3 = axes[2]
    overall = results_df["winner"].value_counts()
    colors_pie = {"FedProx": "#2ca02c", "Tie": "#7f7f7f", "FedAvg": "#d62728"}
    pie_colors = [colors_pie.get(w, "#333") for w in overall.index]

    wedges, texts, autotexts = ax3.pie(
        overall.values,
        labels=overall.index,
        autopct=lambda p: f"{int(p * len(results_df) / 100)}\n({p:.1f}%)",
        colors=pie_colors,
        explode=[0.05 if w == "FedProx" else 0 for w in overall.index],
        startangle=90,
    )
    ax3.set_title(f"Overall: {len(results_df)} Comparisons", fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "fedprox_vs_fedavg_winloss.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    return results_df


def generate_comparison_table(df: pd.DataFrame, output_dir: Path):
    """Generate a comparison table matching the temporal validation protocol format."""
    fedavg_df = df[df["aggregation"] == "FedAvg"].copy()
    fedprox_df = df[df["aggregation"] == "FedProx"].copy()

    alpha_order = sorted([a for a in df["alpha"].unique() if pd.notna(a) and 0 < a < 100])

    rows = []
    for alpha in alpha_order:
        fa = fedavg_df[np.isclose(fedavg_df["alpha"], alpha, rtol=0.1)]
        fp = fedprox_df[np.isclose(fedprox_df["alpha"], alpha, rtol=0.1)]

        if fa.empty:
            continue

        fa_mean, fa_lo, fa_hi = compute_ci(fa["macro_f1_top10"])

        if not fp.empty:
            best_mu_perf = fp.groupby("mu")["macro_f1_top10"].mean()
            best_mu = best_mu_perf.idxmax()
            fp_best = fp[np.isclose(fp["mu"], best_mu, rtol=0.1)]
            fp_mean, fp_lo, fp_hi = compute_ci(fp_best["macro_f1_top10"])

            delta = fp_mean - fa_mean

            fa_vals = fa["macro_f1_top10"].values
            fp_vals = fp_best["macro_f1_top10"].values
            if len(fa_vals) > 1 and len(fp_vals) > 1:
                t_stat, p_val = stats.ttest_ind(fp_vals, fa_vals)
                pooled_std = np.sqrt((fa_vals.std() ** 2 + fp_vals.std() ** 2) / 2)
                cohens_d = delta / pooled_std if pooled_std > 0 else 0
            else:
                p_val = np.nan
                cohens_d = np.nan
        else:
            fp_mean, fp_lo, fp_hi = np.nan, np.nan, np.nan
            best_mu = np.nan
            delta = np.nan
            p_val = np.nan
            cohens_d = np.nan

        rows.append(
            {
                "alpha": alpha,
                "fedavg_mean": fa_mean,
                "fedavg_ci": f"({fa_lo:.3f}, {fa_hi:.3f})",
                "fedprox_mean": fp_mean,
                "fedprox_ci": f"({fp_lo:.3f}, {fp_hi:.3f})" if not np.isnan(fp_mean) else "N/A",
                "best_mu": best_mu,
                "delta": delta,
                "p_value": p_val,
                "cohens_d": cohens_d,
                "winner": "FedProx" if delta > 0.001 else ("FedAvg" if delta < -0.001 else "Tie"),
            }
        )

    table_df = pd.DataFrame(rows)

    output_path = output_dir / "fedprox_vs_fedavg_table.csv"
    table_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    print("\n" + "=" * 90)
    print("FEDPROX vs FEDAVG COMPARISON TABLE (Top-10 Class Macro F1)")
    print("=" * 90)
    print(f"{'Alpha':<8} {'FedAvg':<20} {'FedProx (best mu)':<20} {'mu*':<6} {'Delta':<8} {'p-val':<8} {'d':<6} {'Winner'}")
    print("-" * 90)
    for _, row in table_df.iterrows():
        print(
            f"{row['alpha']:<8.2f} {row['fedavg_mean']:.3f} {row['fedavg_ci']:<11} "
            f"{row['fedprox_mean']:.3f} {row['fedprox_ci']:<11} "
            f"{row['best_mu']:<6.3f} {row['delta']:+.3f}  {row['p_value']:<8.3f} "
            f"{row['cohens_d']:<6.2f} {row['winner']}"
        )

    return table_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEDPROX vs FEDAVG COMPARISON (Top-10 Classes)")
    print("=" * 70)
    print(f"Excluding classes: {EXCLUDE_CLASS_NAMES}")
    print()

    print("Loading data...")
    df = load_all_data()

    if df.empty:
        print("ERROR: No data found!")
        return 1

    print(f"Loaded {len(df)} experiment records")
    print(f"  FedAvg: {len(df[df['aggregation'] == 'FedAvg'])}")
    print(f"  FedProx: {len(df[df['aggregation'] == 'FedProx'])}")
    print()

    print("=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    plot_fedprox_vs_fedavg_comparison(df, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("GENERATING WIN/LOSS SUMMARY")
    print("=" * 70)
    results_df = plot_win_loss_summary(df, OUTPUT_DIR)

    print()
    table_df = generate_comparison_table(df, OUTPUT_DIR)

    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"All plots saved to: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
