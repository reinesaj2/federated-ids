#!/usr/bin/env python3
"""
Comprehensive analysis of data gaps for all thesis plots.
Cross-references plot requirements with existing experiment runs.
"""
import json
from collections import defaultdict
from pathlib import Path

SEEDS = list(range(42, 52))  # 10 seeds for statistical power
RUNS_DIR = Path("runs")


def parse_run_name(run_name: str) -> dict:
    """Extract experiment parameters from run directory name."""
    if "p0b1bacd1" not in run_name and "nightly" not in run_name:
        return None  # Skip old/invalid runs

    parts = run_name.split("_")
    result = {}

    for i, part in enumerate(parts):
        if part.startswith("comp") and i + 1 < len(parts):
            result["aggregation"] = parts[i + 1]
        elif part.startswith("alpha"):
            alpha_str = part.replace("alpha", "")
            result["alpha"] = float('inf') if alpha_str == "inf" else float(alpha_str)
        elif part.startswith("adv"):
            result["adv_pct"] = int(part.replace("adv", ""))
        elif part.startswith("seed"):
            result["seed"] = int(part.replace("seed", ""))
        elif part.startswith("pers"):
            result["pers_epochs"] = int(part.replace("pers", ""))

    return result if "aggregation" in result and "seed" in result else None


def scan_existing_runs():
    """Scan runs directory and categorize existing experiments."""
    existing = {
        "attack": defaultdict(set),  # (agg, adv) -> set of seeds
        "heterogeneity": defaultdict(set),  # (agg, alpha) -> set of seeds
        "personalization": defaultdict(set),  # (agg, pers_epochs) -> set of seeds
    }

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        params = parse_run_name(run_dir.name)
        if not params:
            continue

        agg = params.get("aggregation")
        alpha = params.get("alpha")
        adv = params.get("adv_pct")
        seed = params.get("seed")
        pers = params.get("pers_epochs", 0)

        if seed not in SEEDS:
            continue

        # Attack experiments: alpha=0.5, varying adv
        if alpha == 0.5 and adv is not None:
            existing["attack"][(agg, adv)].add(seed)

        # Heterogeneity experiments: adv=0, varying alpha
        if adv == 0 and alpha is not None:
            existing["heterogeneity"][(agg, alpha)].add(seed)

        # Personalization experiments
        if pers > 0:
            existing["personalization"][(agg, pers)].add(seed)

    return existing


def analyze_objective1_gaps(existing):
    """Objective 1: Robustness - Attack experiments at alpha=0.5."""
    print("=" * 80)
    print("OBJECTIVE 1: ROBUSTNESS (Attack Experiments)")
    print("=" * 80)
    print("Requirements: alpha=0.5, attack levels [0,10,20,30], all aggregators")
    print()

    attack_requirements = {
        "fedavg": [0, 10, 20, 30],
        "krum": [0, 10, 20, 30],
        "bulyan": [0, 10, 20],  # Limited by n>=4f+3
        "median": [0, 10, 20, 30],
    }

    missing = []
    for agg, adv_levels in attack_requirements.items():
        for adv in adv_levels:
            have = existing["attack"].get((agg, adv), set())
            missing_seeds = sorted(set(SEEDS) - have)
            if missing_seeds:
                print(f"{agg:8s} adv={adv:2d}%: {len(missing_seeds):2d} missing "
                      f"(have {len(have):2d}/10) - seeds {missing_seeds}")
                for seed in missing_seeds:
                    missing.append({
                        "objective": 1,
                        "aggregation": agg,
                        "alpha": 0.5,
                        "adv_pct": adv,
                        "seed": seed,
                        "dimension": "attack",
                    })

    print(f"\nTotal Objective 1 gaps: {len(missing)}")
    return missing


def analyze_objective2_gaps(existing):
    """Objective 2: Heterogeneity - Alpha sweep at adv=0."""
    print("\n" + "=" * 80)
    print("OBJECTIVE 2: HETEROGENEITY (Alpha Sweep)")
    print("=" * 80)
    print("Requirements: adv=0%, alpha values [0.02,0.05,0.1,0.2,0.5,1.0,inf]")
    print()

    alphas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf')]
    aggregators = ["fedavg", "krum", "bulyan", "median"]

    missing = []
    for agg in aggregators:
        print(f"\n{agg.upper()}:")
        for alpha in alphas:
            have = existing["heterogeneity"].get((agg, alpha), set())
            missing_seeds = sorted(set(SEEDS) - have)
            alpha_str = "inf" if alpha == float('inf') else f"{alpha:.2f}"
            if missing_seeds:
                print(f"  alpha={alpha_str:5s}: {len(missing_seeds):2d} missing "
                      f"(have {len(have):2d}/10) - seeds {missing_seeds}")
                for seed in missing_seeds:
                    missing.append({
                        "objective": 2,
                        "aggregation": agg,
                        "alpha": alpha,
                        "adv_pct": 0,
                        "seed": seed,
                        "dimension": "heterogeneity",
                    })
            else:
                print(f"  alpha={alpha_str:5s}: COMPLETE (10/10)")

    print(f"\nTotal Objective 2 gaps: {len(missing)}")
    return missing


def analyze_objective3_gaps(existing):
    """Objective 3: Personalization."""
    print("\n" + "=" * 80)
    print("OBJECTIVE 3: PERSONALIZATION")
    print("=" * 80)
    print("Requirements: pers_epochs [3,5], various conditions")
    print()

    # Personalization needs experiments with pers_epochs > 0
    # Typically at alpha=0.5, adv=0, but also with attacks
    pers_epochs_list = [3, 5]
    aggregators = ["fedavg", "krum", "bulyan", "median"]

    missing = []
    for agg in aggregators:
        for pers in pers_epochs_list:
            have = existing["personalization"].get((agg, pers), set())
            missing_seeds = sorted(set(SEEDS) - have)
            if missing_seeds:
                print(f"{agg:8s} pers={pers}: {len(missing_seeds):2d} missing "
                      f"(have {len(have):2d}/10) - seeds {missing_seeds}")
                # Note: Personalization experiments also need alpha/adv specs
                # For now, assume alpha=0.5, adv=0 as baseline
                for seed in missing_seeds:
                    missing.append({
                        "objective": 3,
                        "aggregation": agg,
                        "alpha": 0.5,
                        "adv_pct": 0,
                        "pers_epochs": pers,
                        "seed": seed,
                        "dimension": "personalization",
                    })

    print(f"\nTotal Objective 3 gaps: {len(missing)}")
    return missing


def main():
    print("COMPREHENSIVE THESIS PLOT DATA GAP ANALYSIS")
    print("=" * 80)
    print("Scanning existing runs...")
    existing = scan_existing_runs()
    print(f"Found {sum(len(seeds) for seeds in existing['attack'].values())} attack experiments")
    print(f"Found {sum(len(seeds) for seeds in existing['heterogeneity'].values())} heterogeneity experiments")
    print(f"Found {sum(len(seeds) for seeds in existing['personalization'].values())} personalization experiments")
    print()

    # Analyze gaps for each objective
    obj1_gaps = analyze_objective1_gaps(existing)
    obj2_gaps = analyze_objective2_gaps(existing)
    obj3_gaps = analyze_objective3_gaps(existing)

    # Combine and prioritize
    all_gaps = obj1_gaps + obj2_gaps + obj3_gaps

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Objective 1 (Robustness):      {len(obj1_gaps):3d} missing experiments")
    print(f"Objective 2 (Heterogeneity):   {len(obj2_gaps):3d} missing experiments")
    print(f"Objective 3 (Personalization): {len(obj3_gaps):3d} missing experiments")
    print(f"{'':33s}{'─'*15}")
    print(f"TOTAL:                         {len(all_gaps):3d} experiments")
    print()

    # Save queue
    queue_file = Path("complete_experiment_queue.json")
    with open(queue_file, "w") as f:
        json.dump(all_gaps, f, indent=2)

    print(f"Saved complete queue to: {queue_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
