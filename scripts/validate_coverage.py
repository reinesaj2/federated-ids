#!/usr/bin/env python3
"""
Experiment Coverage Validation Script

Validates that all required experiments for Issue #44 exist with proper metrics.
Checks against expected experiment matrix for both CIC and UNSW datasets.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from comparative_analysis import ComparisonMatrix


def validate_coverage(runs_dir: Path, dataset: str, expected_seeds: List[int]) -> Tuple[Dict, List[str]]:
    """
    Validate experiment coverage for a dataset.

    Returns:
        (coverage_stats, missing_experiments)
    """
    matrix = ComparisonMatrix(dataset=dataset)

    # Expected counts per dimension
    expected = {
        "aggregation": len(matrix.aggregation_methods) * len(expected_seeds),
        "attack": len(matrix.aggregation_methods) * len(matrix.adversary_fractions) * len(expected_seeds),
        "heterogeneity": len(matrix.alpha_values) * len(expected_seeds),
        "privacy": len(matrix.dp_configs) * len(expected_seeds),
        "personalization": len(matrix.personalization_epochs) * len(expected_seeds),
    }

    # Count actual experiments
    actual = defaultdict(lambda: defaultdict(set))
    missing = []

    for cfg_path in runs_dir.glob("comp_*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("dataset") != dataset:
                continue

            # Check metrics exist
            metrics_path = cfg_path.parent / "metrics.csv"
            if not metrics_path.exists():
                missing.append(f"{cfg_path.parent.name}: missing metrics.csv")
                continue

            # Infer dimension
            seed = cfg.get("seed")
            if float(cfg.get("adversary_fraction", 0)) > 0:
                dim = "attack"
            elif cfg.get("dp_enabled"):
                dim = "privacy"
            elif int(cfg.get("personalization_epochs", 0)) > 0:
                dim = "personalization"
            elif cfg.get("aggregation") != "fedavg":
                dim = "aggregation"
            elif cfg.get("alpha") != 1.0:
                dim = "heterogeneity"
            else:
                continue  # Skip baseline configs

            actual[dim][seed].add(cfg_path.parent.name)

        except Exception as e:
            missing.append(f"{cfg_path}: error reading - {e}")

    # Build coverage stats
    stats = {}
    for dim in expected.keys():
        seed_counts = {seed: len(actual[dim][seed]) for seed in expected_seeds}
        total = sum(seed_counts.values())
        stats[dim] = {
            "expected": expected[dim],
            "actual": total,
            "coverage_pct": (total / expected[dim] * 100) if expected[dim] > 0 else 0,
            "seeds": seed_counts,
        }

    return stats, missing


def main():
    runs_dir = Path("runs")
    seeds = [42, 43, 44, 45, 46]

    print("=" * 70)
    print("EXPERIMENT COVERAGE VALIDATION (Issue #44)")
    print("=" * 70)

    for dataset in ["cic", "unsw"]:
        print(f"\n### {dataset.upper()} Dataset ###\n")
        stats, missing = validate_coverage(runs_dir, dataset, seeds)

        total_expected = sum(s["expected"] for s in stats.values())
        total_actual = sum(s["actual"] for s in stats.values())
        overall_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0

        print(f"Overall: {total_actual}/{total_expected} ({overall_pct:.1f}%)\n")

        for dim in sorted(stats.keys()):
            s = stats[dim]
            print(f"{dim:20} {s['actual']:3}/{s['expected']:3} ({s['coverage_pct']:5.1f}%)")
            seed_summary = ", ".join(f"s{k}:{v}" for k, v in sorted(s["seeds"].items()))
            print(f"{'':20} seeds: {seed_summary}")

        if missing:
            print(f"\nMissing/Errors ({len(missing)}):")
            for msg in missing[:10]:  # Show first 10
                print(f"  - {msg}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

    print("\n" + "=" * 70)

    # Exit with error if coverage < 95%
    all_stats = {}
    for dataset in ["cic", "unsw"]:
        stats, _ = validate_coverage(runs_dir, dataset, seeds)
        all_stats[dataset] = stats

    cic_total = sum(s["actual"] for s in all_stats["cic"].values())
    cic_expected = sum(s["expected"] for s in all_stats["cic"].values())
    unsw_total = sum(s["actual"] for s in all_stats["unsw"].values())
    unsw_expected = sum(s["expected"] for s in all_stats["unsw"].values())

    grand_total = cic_total + unsw_total
    grand_expected = cic_expected + unsw_expected
    grand_pct = (grand_total / grand_expected * 100) if grand_expected > 0 else 0

    print(f"\nGRAND TOTAL: {grand_total}/{grand_expected} ({grand_pct:.1f}%)")

    if grand_pct < 95.0:
        print(f"\n [FAIL] FAILED: Coverage {grand_pct:.1f}% below required 95%")
        return 1
    else:
        print(f"\n [PASS] PASSED: Coverage {grand_pct:.1f}% meets 95% threshold")
        return 0


if __name__ == "__main__":
    sys.exit(main())
