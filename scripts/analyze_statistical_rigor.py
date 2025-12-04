#!/usr/bin/env python3
"""Analyze statistical rigor of heterogeneity experiments for NeurIPS submission."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUNS_DIR = Path("runs")


def collect_final_f1_data():
    """Collect final F1 scores per configuration."""
    data = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or "adv0" not in run_dir.name:
            continue

        name = run_dir.name

        alpha_match = re.search(r"_alpha([\d.]+|inf)_", name)
        mu_match = re.search(r"_mu([\d.]+)_", name)
        seed_match = re.search(r"_seed(\d+)", name)

        if not (alpha_match and mu_match and seed_match):
            continue

        alpha = float("inf") if alpha_match.group(1) == "inf" else float(alpha_match.group(1))
        mu = float(mu_match.group(1))
        seed = int(seed_match.group(1))

        if alpha > 100:
            continue

        client_files = list(run_dir.glob("client_*_metrics.csv"))
        if not client_files:
            continue

        try:
            dfs = []
            for cf in client_files:
                df = pd.read_csv(cf)
                dfs.append(df)
            combined = pd.concat(dfs, ignore_index=True)

            final_round = combined["round"].max()
            final_data = combined[combined["round"] == final_round]

            f1_mean = final_data["macro_f1_after"].mean()

            data.append({"alpha": alpha, "mu": mu, "seed": seed, "f1": f1_mean})
        except Exception:
            continue

    return pd.DataFrame(data)


def cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def main():
    df = collect_final_f1_data()

    print("\n" + "=" * 70)
    print("STATISTICAL RIGOR ANALYSIS FOR NEURIPS")
    print("=" * 70)

    # Key comparisons
    fedavg_005 = df[(df["alpha"] == 0.05) & (df["mu"] == 0.0)]["f1"].values
    fedavg_010 = df[(df["alpha"] == 0.1) & (df["mu"] == 0.0)]["f1"].values
    fedavg_100 = df[(df["alpha"] == 1.0) & (df["mu"] == 0.0)]["f1"].values

    fedprox_005_mu01 = df[(df["alpha"] == 0.05) & (df["mu"] == 0.01)]["f1"].values
    fedprox_005_mu05 = df[(df["alpha"] == 0.05) & (df["mu"] == 0.05)]["f1"].values
    fedprox_005_mu10 = df[(df["alpha"] == 0.05) & (df["mu"] == 0.1)]["f1"].values

    fedprox_100_mu01 = df[(df["alpha"] == 1.0) & (df["mu"] == 0.01)]["f1"].values
    fedprox_100_mu05 = df[(df["alpha"] == 1.0) & (df["mu"] == 0.05)]["f1"].values
    fedprox_100_mu10 = df[(df["alpha"] == 1.0) & (df["mu"] == 0.1)]["f1"].values

    print("\n1. STATISTICAL SIGNIFICANCE TESTS (Welch's t-test):")
    print("-" * 70)

    # Test at α=0.05
    if len(fedavg_005) > 0 and len(fedprox_005_mu05) > 0:
        t_stat, p_val = stats.ttest_ind(fedavg_005, fedprox_005_mu05, equal_var=False)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"FedAvg vs FedProx(μ=0.05) at α=0.05:")
        print(f"  FedAvg:         {fedavg_005.mean():.3f} ± {fedavg_005.std():.3f} (n={len(fedavg_005)})")
        print(f"  FedProx μ=0.05: {fedprox_005_mu05.mean():.3f} ± {fedprox_005_mu05.std():.3f} (n={len(fedprox_005_mu05)})")
        print(f"  t={t_stat:.3f}, p={p_val:.6f} {sig}")

    if len(fedavg_005) > 0 and len(fedprox_005_mu10) > 0:
        t_stat, p_val = stats.ttest_ind(fedavg_005, fedprox_005_mu10, equal_var=False)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"\nFedAvg vs FedProx(μ=0.1) at α=0.05:")
        print(f"  FedAvg:        {fedavg_005.mean():.3f} ± {fedavg_005.std():.3f} (n={len(fedavg_005)})")
        print(f"  FedProx μ=0.1: {fedprox_005_mu10.mean():.3f} ± {fedprox_005_mu10.std():.3f} (n={len(fedprox_005_mu10)})")
        print(f"  t={t_stat:.3f}, p={p_val:.6f} {sig}")

    # Test at α=1.0 (IID)
    if len(fedavg_100) > 0 and len(fedprox_100_mu10) > 0:
        t_stat, p_val = stats.ttest_ind(fedavg_100, fedprox_100_mu10, equal_var=False)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"\nFedAvg vs FedProx(μ=0.1) at α=1.0 (IID):")
        print(f"  FedAvg:        {fedavg_100.mean():.3f} ± {fedavg_100.std():.3f} (n={len(fedavg_100)})")
        print(f"  FedProx μ=0.1: {fedprox_100_mu10.mean():.3f} ± {fedprox_100_mu10.std():.3f} (n={len(fedprox_100_mu10)})")
        print(f"  t={t_stat:.3f}, p={p_val:.6f} {sig}")

    print("\n2. EFFECT SIZES (Cohen's d):")
    print("-" * 70)

    if len(fedavg_005) > 0 and len(fedprox_005_mu05) > 0:
        d = cohens_d(fedavg_005, fedprox_005_mu05)
        magnitude = "huge" if abs(d) > 1.2 else ("very large" if abs(d) > 0.8 else ("large" if abs(d) > 0.5 else "medium"))
        print(f"FedAvg vs FedProx(μ=0.05) at α=0.05: d={d:.3f} ({magnitude})")

    if len(fedavg_005) > 0 and len(fedprox_005_mu10) > 0:
        d = cohens_d(fedavg_005, fedprox_005_mu10)
        magnitude = "huge" if abs(d) > 1.2 else ("very large" if abs(d) > 0.8 else ("large" if abs(d) > 0.5 else "medium"))
        print(f"FedAvg vs FedProx(μ=0.1) at α=0.05:  d={d:.3f} ({magnitude})")

    if len(fedavg_100) > 0 and len(fedprox_100_mu10) > 0:
        d = cohens_d(fedavg_100, fedprox_100_mu10)
        magnitude = "huge" if abs(d) > 1.2 else ("very large" if abs(d) > 0.8 else ("large" if abs(d) > 0.5 else "medium"))
        print(f"FedAvg vs FedProx(μ=0.1) at α=1.0:   d={d:.3f} ({magnitude})")

    print("\n3. SAMPLE SIZE SUMMARY:")
    print("-" * 70)
    summary = (
        df.groupby(["alpha", "mu"])
        .agg(n=("f1", "count"), mean=("f1", "mean"), std=("f1", "std"), sem=("f1", lambda x: x.sem()))
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n4. CONFIDENCE INTERVAL COVERAGE:")
    print("-" * 70)
    for _, row in summary.iterrows():
        ci_width = 1.96 * row["sem"]
        ci_pct = (ci_width / row["mean"]) * 100 if row["mean"] > 0 else 0
        print(f"α={row['alpha']:.2f}, μ={row['mu']:.2f}: 95% CI = ±{ci_width:.3f} ({ci_pct:.1f}% of mean)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
