#!/usr/bin/env python3
"""
Build Clean Summary Dataset from Raw Experimental Runs

This script extracts final-round macro-F1 scores from raw run directories,
enforcing data integrity rules:
- Exclude mixed_silo_3dataset experiments
- Enforce mu > 0 only for FedProx aggregator
- Include alpha=inf as explicit category
- Extract adversary_mode from config.json

Output: summary_clean.csv with columns:
    aggregator, alpha, adv_pct, mu, seed, dataset, macro_f1, adversary_mode

Author: Abraham Reines
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_run_directory_name(dirname: str) -> Optional[dict]:
    """
    Parse experiment parameters from directory name.

    Examples:
        cic_simple_xxx_comp_fedavg_alpha0.2_adv0_dp0_pers0_mu0.0_seed47
        unsw_simple_xxx_comp_krum_alphainf_adv30_dp0_pers0_mu0.0_seed54
    """
    if "mixed_silo" in dirname.lower():
        return None

    aggregator_match = re.search(r"_comp_(\w+)_alpha", dirname)
    alpha_match = re.search(r"_alpha(inf|[\d.]+)_", dirname)
    adv_match = re.search(r"_adv(\d+)_", dirname)
    mu_match = re.search(r"_mu([\d.]+)_", dirname)
    seed_match = re.search(r"_seed(\d+)", dirname)

    if not all([aggregator_match, alpha_match, adv_match, mu_match, seed_match]):
        return None

    aggregator = aggregator_match.group(1).lower()
    alpha_str = alpha_match.group(1)
    alpha = float("inf") if alpha_str == "inf" else float(alpha_str)
    adv_pct = int(adv_match.group(1))
    mu = float(mu_match.group(1))
    seed = int(seed_match.group(1))

    if dirname.startswith("cic_"):
        dataset = "cic"
    elif dirname.startswith("unsw_"):
        dataset = "unsw"
    elif dirname.startswith("iiot_") or dirname.startswith("dsedge-iiotset"):
        dataset = "iiot"
    else:
        dataset = "unknown"

    if "_src" in dirname:
        return None

    return {
        "aggregator": aggregator,
        "alpha": alpha,
        "adv_pct": adv_pct,
        "mu": mu,
        "seed": seed,
        "dataset": dataset,
    }


def extract_final_macro_f1(run_dir: Path) -> Optional[float]:
    """Extract final round macro-F1 from metrics.csv."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    try:
        df = pd.read_csv(metrics_path)
        if "global_macro_f1_test" in df.columns:
            final_row = df.iloc[-1]
            f1 = final_row["global_macro_f1_test"]
            if pd.notna(f1):
                return float(f1)
    except Exception:
        pass

    return None


def extract_adversary_mode(run_dir: Path) -> str:
    """Extract adversary_mode from config.json."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return "none"

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        mode = config.get("adversary_mode", "none")
        if mode is None or mode == "":
            return "none"
        return str(mode).lower()
    except Exception:
        return "none"


def validate_mu_aggregator(row: dict) -> bool:
    """
    Validate mu > 0 only for FedProx.

    Returns True if the row passes validation.
    """
    if row["mu"] > 0 and row["aggregator"] != "fedprox":
        return False
    return True


def build_summary_clean(runs_dir: Path, output_path: Path) -> pd.DataFrame:
    """Build clean summary from all run directories."""
    records = []

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "archive_20260107_185922"]

    print(f"Processing {len(run_dirs)} run directories...")

    for i, run_dir in enumerate(run_dirs):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(run_dirs)}")

        params = parse_run_directory_name(run_dir.name)
        if params is None:
            continue

        if not validate_mu_aggregator(params):
            continue

        macro_f1 = extract_final_macro_f1(run_dir)
        if macro_f1 is None:
            continue

        adversary_mode = extract_adversary_mode(run_dir)

        records.append(
            {
                "aggregator": params["aggregator"],
                "alpha": params["alpha"],
                "adv_pct": params["adv_pct"],
                "mu": params["mu"],
                "seed": params["seed"],
                "dataset": params["dataset"],
                "macro_f1": macro_f1,
                "adversary_mode": adversary_mode,
                "run_dir": run_dir.name,
            }
        )

    df = pd.DataFrame(records)

    df = df[df["dataset"].isin(["cic", "unsw"])]

    df_sorted = df.sort_values(["dataset", "aggregator", "alpha", "adv_pct", "mu", "seed"]).reset_index(drop=True)

    df_sorted.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_sorted)} records to {output_path}")

    print("\nSummary statistics:")
    print(f"  Datasets: {df_sorted['dataset'].unique().tolist()}")
    print(f"  Aggregators: {df_sorted['aggregator'].unique().tolist()}")
    print(f"  Alpha values: {sorted(df_sorted['alpha'].unique().tolist())}")
    print(f"  Adversary modes: {df_sorted['adversary_mode'].unique().tolist()}")

    return df_sorted


def main():
    base_path = Path(__file__).parent.parent
    runs_dir = base_path / "runs"
    output_path = base_path / "results" / "summary_clean.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_summary_clean(runs_dir, output_path)

    print("\n--- Dataset breakdown ---")
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        print(f"\n{ds.upper()}:")
        print(f"  Total experiments: {len(ds_df)}")
        print(f"  Seeds: {sorted(ds_df['seed'].unique())}")
        for agg in sorted(ds_df["aggregator"].unique()):
            agg_df = ds_df[ds_df["aggregator"] == agg]
            print(f"  {agg}: {len(agg_df)} runs, mean F1={agg_df['macro_f1'].mean():.3f}")


if __name__ == "__main__":
    main()
