#!/usr/bin/env python3
"""
Summarize Robust Aggregation Experiment Results

Collects metrics from robust aggregation experiments and generates summary statistics
for comparison across algorithms and adversary rates.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


RUN_NAME_PATTERN = re.compile(
    r"robust_agg_(?P<aggregation>\w+)_adv(?P<adv_fraction>[0-9.]+)_seed(?P<seed>\d+)"
)


def extract_run_config(run_dir: Path) -> Dict[str, str | float | None]:
    """Extract configuration from run directory name."""
    match = RUN_NAME_PATTERN.match(run_dir.name)
    if not match:
        return {
            "aggregation": None,
            "adv_fraction": None,
            "seed": None,
        }

    return {
        "aggregation": match.group("aggregation"),
        "adv_fraction": float(match.group("adv_fraction")),
        "seed": int(match.group("seed")),
    }


def collect_metrics_from_run(run_dir: Path) -> Dict:
    """Collect final metrics from a single run."""
    metrics = {
        "run_dir": run_dir.name,
        "macro_f1": None,
        "accuracy": None,
        "l2_distance": None,
        "cosine_similarity": None,
        "benign_clients": 0,
        "adversarial_clients": 0,
    }

    # Collect client metrics
    client_metrics_files = list(run_dir.glob("client_*_metrics.csv"))
    macro_f1_values = []
    acc_values = []

    for client_file in client_metrics_files:
        try:
            df = pd.read_csv(client_file)
            if len(df) == 0:
                continue

            last_row = df.iloc[-1]

            # Count benign vs adversarial clients
            if "adv" in client_file.name:
                metrics["adversarial_clients"] += 1
            else:
                metrics["benign_clients"] += 1

            # Collect final metrics
            if "macro_f1_after" in df.columns:
                macro_f1 = last_row["macro_f1_after"]
                if pd.notna(macro_f1):
                    macro_f1_values.append(float(macro_f1))

            if "acc_after" in df.columns:
                acc = last_row["acc_after"]
                if pd.notna(acc):
                    acc_values.append(float(acc))

        except Exception as e:
            print(f"Warning: Failed to read {client_file}: {e}")
            continue

    # Compute averages
    if macro_f1_values:
        metrics["macro_f1"] = sum(macro_f1_values) / len(macro_f1_values)
    if acc_values:
        metrics["accuracy"] = sum(acc_values) / len(acc_values)

    # Collect server metrics
    server_metrics_path = run_dir / "metrics.csv"
    if server_metrics_path.exists():
        try:
            df = pd.read_csv(server_metrics_path)
            if len(df) > 0:
                last_row = df.iloc[-1]

                if "l2_to_benign_mean" in df.columns:
                    l2 = last_row["l2_to_benign_mean"]
                    if pd.notna(l2):
                        metrics["l2_distance"] = float(l2)

                if "cos_to_benign_mean" in df.columns:
                    cos = last_row["cos_to_benign_mean"]
                    if pd.notna(cos):
                        metrics["cosine_similarity"] = float(cos)

        except Exception as e:
            print(f"Warning: Failed to read {server_metrics_path}: {e}")

    return metrics


def summarize_robust_agg_experiments(
    runs_dir: Path,
    aggregation: str | None = None,
    adv_fraction: float | None = None,
) -> pd.DataFrame:
    """Summarize all robust aggregation experiments.

    Args:
        runs_dir: Directory containing experiment runs
        aggregation: Filter by aggregation algorithm (optional)
        adv_fraction: Filter by adversary fraction (optional)

    Returns:
        DataFrame with summary statistics
    """
    results = []

    # Find all robust aggregation runs
    if not runs_dir.exists():
        print(f"Error: runs directory does not exist: {runs_dir}")
        return pd.DataFrame()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config = extract_run_config(run_dir)
        if config["aggregation"] is None:
            continue

        # Apply filters
        if aggregation and config["aggregation"] != aggregation:
            continue
        if adv_fraction is not None and abs(config["adv_fraction"] - adv_fraction) > 0.01:
            continue

        metrics = collect_metrics_from_run(run_dir)

        # Combine config and metrics
        result = {**config, **metrics}
        results.append(result)

    if not results:
        print("No robust aggregation runs found")
        return pd.DataFrame()

    return pd.DataFrame(results)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by algorithm and adversary rate."""
    if df.empty:
        return pd.DataFrame()

    # Group by aggregation and adv_fraction
    grouped = df.groupby(["aggregation", "adv_fraction"])

    summary = grouped.agg({
        "macro_f1": ["mean", "std", "min", "max", "count"],
        "accuracy": ["mean", "std"],
        "l2_distance": ["mean", "std"],
        "cosine_similarity": ["mean", "std"],
        "benign_clients": "first",
        "adversarial_clients": "first",
    }).reset_index()

    # Flatten column names
    summary.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in summary.columns
    ]

    return summary


def save_summary(
    df: pd.DataFrame,
    summary_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save summary results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_csv = output_dir / "robust_agg_detailed.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"Saved detailed results to {detailed_csv}")

    # Save summary statistics
    summary_csv = output_dir / "robust_agg_summary.csv"
    summary_stats.to_csv(summary_csv, index=False)
    print(f"Saved summary statistics to {summary_csv}")

    # Save as JSON for easier parsing
    summary_json = output_dir / "robust_agg_summary.json"
    summary_dict = summary_stats.to_dict(orient="records")
    with open(summary_json, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved summary JSON to {summary_json}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize robust aggregation experiment results"
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing experiment runs",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        help="Filter by aggregation algorithm (fedavg, krum, bulyan, median)",
    )
    parser.add_argument(
        "--adv_fraction",
        type=float,
        help="Filter by adversary fraction (0.0, 0.2, 0.4)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/robust_agg_weekly"),
        help="Output directory for summary files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Robust Aggregation Summary")
    print("=" * 60)

    # Collect results
    df = summarize_robust_agg_experiments(
        args.runs_dir,
        aggregation=args.aggregation,
        adv_fraction=args.adv_fraction,
    )

    if df.empty:
        print("No results to summarize")
        return

    print(f"\nCollected {len(df)} experiment runs")
    print(f"  Algorithms: {sorted(df['aggregation'].unique())}")
    print(f"  Adversary rates: {sorted(df['adv_fraction'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")

    # Compute summary statistics
    summary_stats = compute_summary_statistics(df)

    print("\nSummary Statistics:")
    print(summary_stats.to_string())

    # Save results
    save_summary(df, summary_stats, args.output_dir)

    print("\n" + "=" * 60)
    print("Summary generation completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
