#!/usr/bin/env python3
"""Select optimal mu* per alpha from temporal validation tuning results.

This script implements the mu selection rule from TEMPORAL_VALIDATION_PROTOCOL.md:
    mu*[alpha] = argmax_{mu} ( mean_{seed in {42,43,44}} global_macro_f1_val(alpha, mu, seed) )
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_run_metrics(run_dir: Path) -> dict | None:
    """Load metrics from a single run directory."""
    metrics_file = run_dir / "metrics.csv"
    config_file = run_dir / "config.json"

    if not metrics_file.exists() or not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            config = json.load(f)

        df = pd.read_csv(metrics_file)
        if df.empty:
            return None

        last_row = df.iloc[-1]

        global_f1_val = last_row.get("global_macro_f1_val")
        global_f1_test = last_row.get("global_macro_f1_test")
        macro_f1_holdout = last_row.get("macro_f1_global_holdout")

        if global_f1_val is None and macro_f1_holdout is not None:
            global_f1_val = macro_f1_holdout

        return {
            "alpha": config.get("alpha", config.get("heterogeneity_alpha")),
            "mu": config.get("fedprox_mu", 0.0),
            "seed": config.get("seed"),
            "global_macro_f1_val": global_f1_val,
            "global_macro_f1_test": global_f1_test,
            "run_dir": str(run_dir),
        }
    except Exception as e:
        print(f"Error loading {run_dir}: {e}")
        return None


def select_mu_star(runs_dir: Path, tuning_seeds: set[int] = {42, 43, 44}) -> dict[float, dict]:
    """Select optimal mu per alpha based on validation macro_f1.

    Args:
        runs_dir: Directory containing experiment runs
        tuning_seeds: Seeds to use for mu selection

    Returns:
        Dict mapping alpha -> {mu_star, mean_val_f1, all_results}
    """
    results_by_alpha_mu: dict[tuple[float, float], list[float]] = defaultdict(list)
    all_results: list[dict] = []

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metrics = load_run_metrics(run_dir)
        if metrics is None:
            continue

        if metrics["seed"] not in tuning_seeds:
            continue

        if metrics["global_macro_f1_val"] is None:
            continue

        alpha = metrics["alpha"]
        mu = metrics["mu"]
        val_f1 = float(metrics["global_macro_f1_val"])

        results_by_alpha_mu[(alpha, mu)].append(val_f1)
        all_results.append(metrics)

    mu_star_per_alpha: dict[float, dict] = {}
    alphas = sorted(set(alpha for alpha, _ in results_by_alpha_mu.keys()))

    for alpha in alphas:
        best_mu = None
        best_mean_f1 = -1.0

        mus = sorted(set(mu for a, mu in results_by_alpha_mu.keys() if a == alpha))

        for mu in mus:
            f1_values = results_by_alpha_mu[(alpha, mu)]
            if len(f1_values) < len(tuning_seeds):
                continue

            mean_f1 = sum(f1_values) / len(f1_values)
            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_mu = mu

        if best_mu is not None:
            mu_star_per_alpha[alpha] = {
                "mu_star": best_mu,
                "mean_val_f1": best_mean_f1,
                "n_seeds": len(results_by_alpha_mu[(alpha, best_mu)]),
            }

    return mu_star_per_alpha


def main():
    parser = argparse.ArgumentParser(description="Select optimal mu* per alpha from tuning results")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing experiment runs")
    parser.add_argument("--output", type=str, default="mu_star_selection.json", help="Output file for mu* selections")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    print("Analyzing tuning results...")
    print(f"Runs directory: {runs_dir}")
    print("")

    mu_star = select_mu_star(runs_dir)

    print("=" * 60)
    print("Mu* Selection Results (Tuning Seeds: 42, 43, 44)")
    print("=" * 60)
    print(f"{'Alpha':<10} {'mu*':<10} {'Mean Val F1':<15} {'Seeds':<10}")
    print("-" * 60)

    for alpha in sorted(mu_star.keys(), key=lambda x: (x == float("inf"), x)):
        info = mu_star[alpha]
        alpha_str = "inf" if alpha == float("inf") else f"{alpha:.2f}"
        print(f"{alpha_str:<10} {info['mu_star']:<10.3f} {info['mean_val_f1']:<15.4f} {info['n_seeds']:<10}")

    print("=" * 60)
    print("")

    output_file = Path(args.output)
    serializable = {}
    for alpha, info in mu_star.items():
        key = "inf" if alpha == float("inf") else str(alpha)
        serializable[key] = info

    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved to: {output_file}")
    print("")
    print("Next steps:")
    print("  1. Review the mu* selections above")
    print("  2. Run evaluation phase with seeds 45-49:")
    print("     sbatch --array=0-69%17 scripts/slurm/temporal_validation_eval.sbatch")


if __name__ == "__main__":
    main()
