#!/usr/bin/env python3
"""
Comprehensive analysis of full Edge-IIoTset experimental results.
Maps findings to thesis objectives and identifies configurations achieving >95% macro F1.
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_experiment_name(dirname):
    """Parse experiment directory name to extract hyperparameters."""
    parts = dirname.replace("dsedge-iiotset-full_comp_", "").split("_")

    config = {
        "aggregator": None,
        "alpha": None,
        "adv_percent": None,
        "mu": None,
        "seed": None,
        "attack_mode": "none"
    }

    for i, part in enumerate(parts):
        if part in ["fedavg", "krum", "bulyan", "median", "fedprox"]:
            config["aggregator"] = part
        elif part.startswith("alpha"):
            config["alpha"] = part.replace("alpha", "")
        elif part.startswith("adv"):
            config["adv_percent"] = int(part.replace("adv", ""))
        elif part.startswith("mu"):
            config["mu"] = float(part.replace("mu", ""))
        elif part.startswith("seed"):
            config["seed"] = int(part.replace("seed", ""))
        elif part.startswith("mode") and i < len(parts) - 1:
            mode = part.replace("mode", "")
            if mode != "none":
                config["attack_mode"] = mode

    return config


def load_final_metrics(exp_dir):
    """Load final round metrics from experiment directory."""
    metrics_file = os.path.join(exp_dir, "metrics.csv")

    if not os.path.exists(metrics_file):
        return None

    try:
        df = pd.read_csv(metrics_file)
        if df.empty:
            return None

        final_row = df.iloc[-1]

        metrics = {
            "round": int(final_row.get("round", -1)),
            "macro_f1": float(final_row.get("global_macro_f1_test", 0.0)),
            "macro_f1_val": float(final_row.get("global_macro_f1_val", 0.0)),
            "n_test_total": int(final_row.get("n_test_total", 0))
        }

        for col in df.columns:
            if col.startswith("f1_class_"):
                class_name = col.replace("f1_class_", "")
                metrics[f"f1_{class_name}"] = float(final_row[col])

        return metrics
    except Exception as e:
        print(f"Error loading {metrics_file}: {e}")
        return None


def analyze_full_iiot_experiments(runs_dir="runs"):
    """Analyze all full IIoT experiments and organize by thesis objectives."""

    full_iiot_dirs = []
    for dirname in os.listdir(runs_dir):
        if "iiotset-full" in dirname and dirname.startswith("dsedge"):
            full_path = os.path.join(runs_dir, dirname)
            if os.path.isdir(full_path):
                full_iiot_dirs.append((dirname, full_path))

    print(f"Found {len(full_iiot_dirs)} full IIoT experiment directories")

    results = []
    for dirname, exp_path in full_iiot_dirs:
        config = parse_experiment_name(dirname)
        metrics = load_final_metrics(exp_path)

        if metrics is not None and config["aggregator"] is not None:
            results.append({
                "exp_dir": dirname,
                **config,
                **metrics
            })

    df = pd.DataFrame(results)

    if df.empty:
        print("No valid results found!")
        return

    print(f"Loaded {len(df)} valid experiments")

    print("\n" + "="*80)
    print("THESIS OBJECTIVE 1: ROBUST AGGREGATION METHODS")
    print("="*80)

    high_performers = df[df["macro_f1"] > 0.95]

    print(f"\n>>> CONFIGURATIONS ACHIEVING >95% MACRO F1: {len(high_performers)}")

    if not high_performers.empty:
        by_agg = high_performers.groupby("aggregator")

        for agg_name, group in by_agg:
            print(f"\n{agg_name.upper()}:")
            print(f"  Total configs >95%: {len(group)}")
            print(f"  Best macro F1: {group['macro_f1'].max():.4f}")
            print(f"  Mean macro F1: {group['macro_f1'].mean():.4f}")

            best_configs = group.nlargest(5, "macro_f1")
            print(f"\n  Top 5 configurations:")
            for idx, row in best_configs.iterrows():
                print(f"    - alpha={row['alpha']}, adv={row['adv_percent']}%, "
                      f"mu={row['mu']}, seed={row['seed']}, "
                      f"attack={row['attack_mode']}")
                print(f"      Test F1: {row['macro_f1']:.4f}, Val F1: {row['macro_f1_val']:.4f}")

    print("\n>>> ROBUSTNESS UNDER BYZANTINE ATTACKS:")

    for adv_level in sorted(df["adv_percent"].unique()):
        subset = df[df["adv_percent"] == adv_level]
        print(f"\n  {adv_level}% Adversarial Clients:")

        agg_stats = subset.groupby("aggregator")["macro_f1"].agg(["mean", "std", "max", "count"])
        print(agg_stats.to_string())

        if adv_level > 0:
            robust_configs = subset[subset["macro_f1"] > 0.95]
            if not robust_configs.empty:
                print(f"\n  Aggregators maintaining >95% F1 under {adv_level}% attack:")
                for agg in robust_configs["aggregator"].unique():
                    count = len(robust_configs[robust_configs["aggregator"] == agg])
                    print(f"    - {agg}: {count} configurations")

    print("\n" + "="*80)
    print("THESIS OBJECTIVE 2: DATA HETEROGENEITY (NON-IID)")
    print("="*80)

    print("\n>>> DIRICHLET ALPHA IMPACT (Non-IIDness):")

    alpha_values = sorted([a for a in df["alpha"].unique() if a != "inf"])

    for alpha in alpha_values:
        subset = df[df["alpha"] == alpha]
        print(f"\n  Alpha={alpha} (lower = more non-IID):")

        print(f"    Mean macro F1: {subset['macro_f1'].mean():.4f} ± {subset['macro_f1'].std():.4f}")
        print(f"    Max macro F1: {subset['macro_f1'].max():.4f}")
        print(f"    Configs >95%: {len(subset[subset['macro_f1'] > 0.95])}")

        fedprox_subset = subset[subset["aggregator"] == "fedprox"]
        others_subset = subset[subset["aggregator"] != "fedprox"]

        if not fedprox_subset.empty and not others_subset.empty:
            print(f"    FedProx mean F1: {fedprox_subset['macro_f1'].mean():.4f}")
            print(f"    Other aggs mean F1: {others_subset['macro_f1'].mean():.4f}")

    print("\n>>> FEDPROX MU PARAMETER IMPACT:")
    fedprox_results = df[df["aggregator"] == "fedprox"]

    if not fedprox_results.empty:
        mu_stats = fedprox_results.groupby("mu")["macro_f1"].agg(["mean", "std", "max", "count"])
        print(mu_stats.to_string())

        fedprox_high = fedprox_results[fedprox_results["macro_f1"] > 0.95]
        if not fedprox_high.empty:
            print(f"\n  FedProx configs achieving >95% F1: {len(fedprox_high)}")
            best_mu = fedprox_high.groupby("mu")["macro_f1"].agg(["mean", "count"])
            print("\n  Best mu values:")
            print(best_mu.to_string())

    print("\n" + "="*80)
    print("THESIS OBJECTIVE 3: EMPIRICAL VALIDATION ON EDGE-IIOTSET")
    print("="*80)

    print("\n>>> OVERALL DATASET PERFORMANCE:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Mean macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
    print(f"  Max macro F1: {df['macro_f1'].max():.4f}")
    print(f"  Experiments >95% F1: {len(df[df['macro_f1'] > 0.95])} ({len(df[df['macro_f1'] > 0.95])/len(df)*100:.1f}%)")
    print(f"  Experiments >90% F1: {len(df[df['macro_f1'] > 0.90])} ({len(df[df['macro_f1'] > 0.90])/len(df)*100:.1f}%)")

    f1_class_cols = [col for col in df.columns if col.startswith("f1_")]

    if f1_class_cols:
        print("\n>>> PER-CLASS F1 SCORES (from high-performing configs):")
        high_perf = df[df["macro_f1"] > 0.95]

        if not high_perf.empty:
            for col in sorted(f1_class_cols):
                class_name = col.replace("f1_", "")
                mean_f1 = high_perf[col].mean()
                std_f1 = high_perf[col].std()
                print(f"  {class_name}: {mean_f1:.4f} ± {std_f1:.4f}")

    print("\n" + "="*80)
    print("NOTABLE DISCOVERIES & INSIGHTS")
    print("="*80)

    best_exp = df.nlargest(1, "macro_f1").iloc[0]
    print(f"\n1. BEST OVERALL CONFIGURATION:")
    print(f"   Aggregator: {best_exp['aggregator']}")
    print(f"   Alpha: {best_exp['alpha']} (data heterogeneity)")
    print(f"   Adversarial: {best_exp['adv_percent']}%")
    print(f"   Mu: {best_exp['mu']}")
    print(f"   Attack mode: {best_exp['attack_mode']}")
    print(f"   Test Macro F1: {best_exp['macro_f1']:.4f}")
    print(f"   Val Macro F1: {best_exp['macro_f1_val']:.4f}")

    under_attack = df[df["adv_percent"] >= 20]
    if not under_attack.empty:
        robust_stats = under_attack.groupby("aggregator")["macro_f1"].mean().sort_values(ascending=False)
        print(f"\n2. MOST ROBUST AGGREGATORS (≥20% adversarial clients):")
        for agg, score in robust_stats.head(3).items():
            print(f"   {agg}: {score:.4f} mean macro F1")

    highly_noniid = df[df["alpha"].isin(["0.02", "0.05", "0.1"])]
    if not highly_noniid.empty:
        best_noniid = highly_noniid.nlargest(1, "macro_f1").iloc[0]
        print(f"\n3. BEST FOR HIGHLY NON-IID DATA (alpha ≤ 0.1):")
        print(f"   Aggregator: {best_noniid['aggregator']}")
        print(f"   Alpha: {best_noniid['alpha']}")
        print(f"   Macro F1: {best_noniid['macro_f1']:.4f}")

    print(f"\n4. ATTACK MODE IMPACT:")
    attack_modes = df.groupby("attack_mode")["macro_f1"].agg(["mean", "count"])
    print(attack_modes.to_string())

    print(f"\n5. STATISTICAL STABILITY ACROSS SEEDS:")
    df["config_key"] = df.apply(
        lambda row: f"{row['aggregator']}_alpha{row['alpha']}_adv{row['adv_percent']}_mu{row['mu']}",
        axis=1
    )

    config_stats = df.groupby("config_key").agg({
        "macro_f1": ["mean", "std", "count"],
        "seed": "count"
    })

    multi_seed_configs = config_stats[config_stats[("seed", "count")] >= 5]
    if not multi_seed_configs.empty:
        most_stable = multi_seed_configs.nsmallest(3, ("macro_f1", "std"))
        print("\n   Most stable configurations (≥5 seeds, lowest std dev):")
        for config_key in most_stable.index:
            mean_f1 = most_stable.loc[config_key, ("macro_f1", "mean")]
            std_f1 = most_stable.loc[config_key, ("macro_f1", "std")]
            count = int(most_stable.loc[config_key, ("macro_f1", "count")])
            print(f"   {config_key}: {mean_f1:.4f} ± {std_f1:.4f} (n={count})")

    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)

    high_performers_file = "full_iiot_high_performers.csv"
    high_performers.to_csv(high_performers_file, index=False)
    print(f"\nHigh performers (>95% F1) exported to: {high_performers_file}")

    full_results_file = "full_iiot_all_results.csv"
    df.to_csv(full_results_file, index=False)
    print(f"All results exported to: {full_results_file}")

    summary = {
        "total_experiments": len(df),
        "high_performers_95": len(df[df["macro_f1"] > 0.95]),
        "high_performers_90": len(df[df["macro_f1"] > 0.90]),
        "best_macro_f1": float(df["macro_f1"].max()),
        "mean_macro_f1": float(df["macro_f1"].mean()),
        "std_macro_f1": float(df["macro_f1"].std()),
        "aggregators": {
            agg: {
                "count": int(group["macro_f1"].count()),
                "mean_f1": float(group["macro_f1"].mean()),
                "max_f1": float(group["macro_f1"].max()),
                "high_performers_95": int((group["macro_f1"] > 0.95).sum())
            }
            for agg, group in df.groupby("aggregator")
        }
    }

    summary_file = "full_iiot_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics exported to: {summary_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    analyze_full_iiot_experiments()
