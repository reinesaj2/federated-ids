#!/usr/bin/env python3
"""
Objective 2 statistical visuals: significance tests and per-class F1.

Outputs:
- obj2_fedprox_seed_significance.png/pdf
- obj2_fedprox_perclass_heatmap.png/pdf
- obj2_fedprox_stats.csv (Welch t-tests, Cohen's d)
"""

import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("results/thesis_plots_package/objective2_heterogeneity")


def parse_run_config(run_name: str) -> Dict:
    config: Dict = {
        "aggregation": "unknown",
        "alpha": None,
        "adversary": 0,
        "seed": 0,
        "mu": 0.0,
    }

    if "comp_fedavg" in run_name:
        config["aggregation"] = "FedAvg"
    elif "fedprox" in run_name.lower():
        config["aggregation"] = "FedProx"
    elif "comp_bulyan" in run_name:
        config["aggregation"] = "Bulyan"
    elif "comp_krum" in run_name:
        config["aggregation"] = "Krum"
    elif "comp_median" in run_name:
        config["aggregation"] = "Median"

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
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None

    dfs: List[pd.DataFrame] = []
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


def parse_class_metrics(row: pd.Series) -> Tuple[List[str], np.ndarray]:
    names_raw = row.get("confusion_matrix_class_names")
    if isinstance(names_raw, str):
        try:
            names = json.loads(names_raw)
        except json.JSONDecodeError:
            names = ast.literal_eval(names_raw)
    else:
        names = []

    f1_raw = row.get("f1_per_class_after")
    f1_vals: Sequence[float] = []
    if isinstance(f1_raw, str):
        try:
            obj = json.loads(f1_raw)
        except json.JSONDecodeError:
            obj = ast.literal_eval(f1_raw)
        if isinstance(obj, dict):
            f1_vals = [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x))]
        elif isinstance(obj, list):
            f1_vals = obj
    return names, np.array(f1_vals, dtype=float)


def collect_data() -> pd.DataFrame:
    records: List[Dict] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        config = parse_run_config(run_dir.name)
        if config["alpha"] is None or config["adversary"] != 0:
            continue

        client_df = load_client_metrics(run_dir)
        if client_df is None or "macro_f1_after" not in client_df.columns:
            continue

        final_round = client_df["round"].max()
        final = client_df[client_df["round"] == final_round].copy()

        for _, row in final.iterrows():
            class_names, f1_per_class = parse_class_metrics(row)
            records.append(
                {
                    "aggregation": config["aggregation"],
                    "alpha": config["alpha"],
                    "mu": config["mu"],
                    "seed": config["seed"],
                    "client_id": row.get("client_id", -1),
                    "f1": float(row["macro_f1_after"]),
                    "f1_per_class": f1_per_class,
                    "class_names": class_names,
                }
            )

    return pd.DataFrame(records)


def aggregate_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    grouped = df.groupby(["aggregation", "alpha", "mu", "seed"])
    for (agg, alpha, mu, seed), sub in grouped:
        arrays = [a for a in sub["f1_per_class"] if a.size > 0]
        f1_class_mean = np.array([])
        names: List[str] = []
        if arrays:
            lengths = [len(a) for a in arrays if len(a) > 0]
            if lengths:
                # keep the most common length to avoid shape mismatches
                target_len = max(set(lengths), key=lengths.count)
                filtered = [a for a in arrays if len(a) == target_len]
                if filtered:
                    stacked = np.stack(filtered)
                    f1_class_mean = stacked.mean(axis=0)
                    names_series = sub["class_names"].dropna()
                    if len(names_series):
                        candidate = names_series.iloc[0]
                        names = candidate if isinstance(candidate, list) else []
        rows.append(
            {
                "aggregation": agg,
                "alpha": alpha,
                "mu": mu,
                "seed": seed,
                "f1_mean": sub["f1"].mean(),
                "f1_per_class_mean": f1_class_mean,
                "class_names": names,
            }
        )
    return pd.DataFrame(rows)


def welch_and_cohens_d(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    t_res = stats.ttest_ind(a, b, equal_var=False)
    d_num = a.mean() - b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    d = d_num / pooled if pooled > 0 else float("nan")
    return float(t_res.pvalue), float(d)


def compute_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for alpha, alpha_df in per_seed.groupby("alpha"):
        baseline = alpha_df[alpha_df["aggregation"] == "FedAvg"]["f1_mean"].to_numpy()
        if baseline.size == 0:
            continue
        for mu in [0.01, 0.05, 0.1]:
            prox_vals = alpha_df[(alpha_df["aggregation"] == "FedProx") & np.isclose(alpha_df["mu"], mu)]["f1_mean"].to_numpy()
            if prox_vals.size == 0:
                continue
            p, d = welch_and_cohens_d(prox_vals, baseline)
            rows.append(
                {
                    "alpha": alpha,
                    "comparison": f"FedProx_mu_{mu}",
                    "agg_variant": "FedProx",
                    "mu": mu,
                    "n_baseline": baseline.size,
                    "n_variant": prox_vals.size,
                    "p_value": p,
                    "cohens_d": d,
                    "mean_baseline": baseline.mean(),
                    "mean_variant": prox_vals.mean(),
                }
            )
        bulyan_vals = alpha_df[alpha_df["aggregation"] == "Bulyan"]["f1_mean"].to_numpy()
        if bulyan_vals.size > 0:
            p, d = welch_and_cohens_d(bulyan_vals, baseline)
            rows.append(
                {
                    "alpha": alpha,
                    "comparison": "Bulyan_vs_FedAvg",
                    "agg_variant": "Bulyan",
                    "mu": 0.0,
                    "n_baseline": baseline.size,
                    "n_variant": bulyan_vals.size,
                    "p_value": p,
                    "cohens_d": d,
                    "mean_baseline": baseline.mean(),
                    "mean_variant": bulyan_vals.mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_seed_significance(per_seed: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"

    focus_alphas = [0.05, 0.1, 0.2]
    algo_order = ["FedAvg", "FedProx(0.01)", "FedProx(0.05)", "FedProx(0.1)", "Bulyan"]
    color_map = {
        "FedAvg": "#1f77b4",
        "FedProx(0.01)": "#2ca02c",
        "FedProx(0.05)": "#ff7f0e",
        "FedProx(0.1)": "#d62728",
        "Bulyan": "#9467bd",
    }

    plot_rows: List[Dict] = []
    for _, row in per_seed.iterrows():
        label = row["aggregation"]
        if label == "FedProx":
            label = f"FedProx({row['mu']})"
        plot_rows.append(
            {
                "alpha": row["alpha"],
                "algo": label,
                "f1": row["f1_mean"],
            }
        )
    plot_df = pd.DataFrame(plot_rows)
    plot_df = plot_df[plot_df["alpha"].isin(focus_alphas)]

    fig, axes = plt.subplots(1, len(focus_alphas), figsize=(4.2 * len(focus_alphas), 5), sharey=True)
    if len(focus_alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, focus_alphas):
        sub = plot_df[plot_df["alpha"] == alpha]
        sns.stripplot(
            data=sub,
            x="algo",
            y="f1",
            order=algo_order,
            palette=color_map,
            jitter=0.15,
            alpha=0.6,
            size=6,
            ax=ax,
        )
        summary = (
            sub.groupby("algo")["f1"]
            .agg(["mean", "sem"])
            .reindex(algo_order)
            .reset_index()
        )
        ax.errorbar(
            x=np.arange(len(algo_order)),
            y=summary["mean"],
            yerr=1.96 * summary["sem"],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            markersize=6,
        )
        ax.set_title(f"α = {alpha}", fontweight="bold", fontsize=12)
        ax.set_xlabel("")
        ax.set_xticklabels(algo_order, rotation=30, ha="right")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Final Macro-F1")
    fig.suptitle("Seed-level Macro-F1 with FedProx/Bulyan Baselines", fontsize=16, fontweight="bold", y=1.02)

    handles = [
        Line2D([], [], marker="o", color=color_map[a], linestyle="", markersize=8, label=a)
        for a in algo_order
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(algo_order), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "obj2_fedprox_seed_significance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_bulyan_vs_fedavg(per_seed: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"

    focus_alphas = [0.05, 0.1, 0.2]
    color_map = {
        "FedAvg": "#1f77b4",
        "Bulyan": "#9467bd",
    }

    plot_rows: List[Dict] = []
    for _, row in per_seed.iterrows():
        if row["aggregation"] not in ("FedAvg", "Bulyan"):
            continue
        plot_rows.append(
            {"alpha": row["alpha"], "algo": row["aggregation"], "f1": row["f1_mean"]}
        )
    plot_df = pd.DataFrame(plot_rows)
    plot_df = plot_df[plot_df["alpha"].isin(focus_alphas)]

    fig, axes = plt.subplots(1, len(focus_alphas), figsize=(3.6 * len(focus_alphas), 4.5), sharey=True)
    if len(focus_alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, focus_alphas):
        sub = plot_df[plot_df["alpha"] == alpha]
        sns.stripplot(
            data=sub,
            x="algo",
            y="f1",
            order=["FedAvg", "Bulyan"],
            palette=color_map,
            jitter=0.15,
            alpha=0.65,
            size=6,
            ax=ax,
        )
        summary = (
            sub.groupby("algo")["f1"]
            .agg(["mean", "sem"])
            .reindex(["FedAvg", "Bulyan"])
            .reset_index()
        )
        ax.errorbar(
            x=np.arange(2),
            y=summary["mean"],
            yerr=1.96 * summary["sem"],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            markersize=6,
        )
        ax.set_title(f"α = {alpha}", fontweight="bold", fontsize=12)
        ax.set_xlabel("")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Final Macro-F1")
    fig.suptitle("Bulyan vs FedAvg (Seed-level Macro-F1)", fontsize=15, fontweight="bold", y=1.04)

    handles = [
        Line2D([], [], marker="o", color=color_map[a], linestyle="", markersize=8, label=a)
        for a in ["FedAvg", "Bulyan"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "obj2_bulyan_vs_fedavg.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_per_class_heatmap(per_seed: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"

    target_alpha = 0.05
    targets = [
        ("FedAvg", 0.0),
        ("FedProx", 0.05),
        ("FedProx", 0.1),
        ("Bulyan", 0.0),
    ]

    records: List[Dict] = []
    class_names: List[str] = []
    for agg, mu in targets:
        subset = per_seed[(per_seed["aggregation"] == agg) & np.isclose(per_seed["mu"], mu) & np.isclose(per_seed["alpha"], target_alpha)]
        if subset.empty:
            continue
        # average per-class across seeds
        stacked = np.stack(subset["f1_per_class_mean"].to_list())
        mean_vec = stacked.mean(axis=0)
        if not class_names:
            names_candidate = subset["class_names"].dropna().iloc[0]
            class_names = names_candidate if isinstance(names_candidate, list) else []
        records.append({"algo": f"{agg} μ={mu}", "values": mean_vec})

    if not records or not class_names:
        return

    plot_rows: List[Dict] = []
    for rec in records:
        for cls, val in zip(class_names, rec["values"]):
            plot_rows.append({"Algorithm": rec["algo"], "Class": cls, "Macro-F1": val})

    heat_df = pd.DataFrame(plot_rows)
    pivot = heat_df.pivot(index="Algorithm", columns="Class", values="Macro-F1")

    fig, ax = plt.subplots(figsize=(1.2 * len(class_names), 3.5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Macro-F1"},
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_title("Per-class Macro-F1 at α=0.05 (Bulyan vs FedAvg vs FedProx)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "obj2_fedprox_perclass_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from runs/ ...")
    df = collect_data()
    if df.empty:
        print("No data found.")
        return

    per_seed = aggregate_per_seed(df)
    print(f"Aggregated seeds: {len(per_seed)} rows")

    stats_df = compute_stats(per_seed)
    stats_path = OUTPUT_DIR / "obj2_fedprox_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats: {stats_path}")

    print("Plotting seed significance...")
    plot_seed_significance(per_seed)

    print("Plotting Bulyan vs FedAvg...")
    plot_bulyan_vs_fedavg(per_seed)

    print("Plotting per-class heatmap...")
    plot_per_class_heatmap(per_seed)

    print("Done.")


if __name__ == "__main__":
    main()
