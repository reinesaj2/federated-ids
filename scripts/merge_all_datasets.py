#!/usr/bin/env python3
"""
Merge CIC/UNSW summary_clean.csv with Edge-IIoTset full_iiot_all_results.csv

Creates a unified summary_all_clean.csv with consistent schema:
    aggregator, alpha, adv_pct, mu, seed, dataset, macro_f1, adversary_mode

Author: Abraham Reines
"""

from pathlib import Path

import pandas as pd


def normalize_attack_mode(mode: str) -> str:
    """Normalize attack mode names to consistent format."""
    mode = str(mode).lower().strip()
    mode_map = {
        "targeted": "targeted_label",
        "label": "label_flip",
        "grad": "grad_ascent",
        "sign": "sign_flip_topk",
        "none": "none",
        "": "none",
        "nan": "none",
    }
    return mode_map.get(mode, mode)


def validate_mu_aggregator(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where mu > 0 only for FedProx."""
    mask = (df["mu"] == 0) | (df["aggregator"] == "fedprox")
    filtered = df[mask].copy()
    dropped = len(df) - len(filtered)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with mu > 0 for non-FedProx")
    return filtered


def main():
    base_path = Path(__file__).parent.parent

    cic_unsw_path = base_path / "results" / "summary_clean.csv"
    iiot_path = base_path / "full_iiot_all_results.csv"
    output_path = base_path / "results" / "summary_all_clean.csv"

    print("Loading CIC/UNSW data...")
    cic_unsw_df = pd.read_csv(cic_unsw_path)
    print(f"  Loaded {len(cic_unsw_df)} records from summary_clean.csv")

    cic_unsw_clean = cic_unsw_df[["aggregator", "alpha", "adv_pct", "mu", "seed", "dataset", "macro_f1", "adversary_mode"]].copy()

    print("\nLoading Edge-IIoTset data...")
    iiot_df = pd.read_csv(iiot_path)
    print(f"  Loaded {len(iiot_df)} records from full_iiot_all_results.csv")

    iiot_clean = pd.DataFrame(
        {
            "aggregator": iiot_df["aggregator"],
            "alpha": iiot_df["alpha"],
            "adv_pct": iiot_df["adv_percent"],
            "mu": iiot_df["mu"],
            "seed": iiot_df["seed"],
            "dataset": "iiot",
            "macro_f1": iiot_df["macro_f1"],
            "adversary_mode": iiot_df["attack_mode"].apply(normalize_attack_mode),
        }
    )

    print("\nValidating mu/aggregator constraints...")
    iiot_clean = validate_mu_aggregator(iiot_clean)

    print("\nMerging datasets...")
    merged = pd.concat([cic_unsw_clean, iiot_clean], ignore_index=True)

    merged_sorted = merged.sort_values(["dataset", "aggregator", "alpha", "adv_pct", "mu", "seed"]).reset_index(drop=True)

    merged_sorted.to_csv(output_path, index=False)
    print(f"\nSaved {len(merged_sorted)} records to {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for ds in ["cic", "unsw", "iiot"]:
        ds_df = merged_sorted[merged_sorted["dataset"] == ds]
        print(f"\n{ds.upper()} ({len(ds_df)} records):")

        print("  Aggregators:")
        for agg in sorted(ds_df["aggregator"].unique()):
            agg_df = ds_df[ds_df["aggregator"] == agg]
            print(f"    {agg}: {len(agg_df)}, mean F1={agg_df['macro_f1'].mean():.3f}")

        print("  Alpha values:", sorted(ds_df["alpha"].unique()))

        print("  Adversary modes:", ds_df["adversary_mode"].unique().tolist())

    print("\n" + "=" * 60)
    print("ALPHA=INF COVERAGE")
    print("=" * 60)

    inf_df = merged_sorted[merged_sorted["alpha"] == float("inf")]
    for ds in ["cic", "unsw", "iiot"]:
        ds_inf = inf_df[inf_df["dataset"] == ds]
        print(f"\n{ds.upper()} alpha=inf: {len(ds_inf)} records")
        for agg in sorted(ds_inf["aggregator"].unique()):
            agg_df = ds_inf[ds_inf["aggregator"] == agg]
            print(f"  {agg}: {len(agg_df)}, F1={agg_df['macro_f1'].mean():.3f}")


if __name__ == "__main__":
    main()
