#!/usr/bin/env python3
"""
Analyze missing experiments needed for complete thesis plots.
"""
import json
from pathlib import Path
from collections import defaultdict

# Expected experiment matrix
SEEDS = list(range(42, 52))  # 42-51 (10 seeds)
AGGREGATORS = ["fedavg", "krum", "bulyan", "median"]

# Attack experiments (alpha=0.5 only)
ATTACK_ALPHA = 0.5
ATTACK_LEVELS = {
    "fedavg": [0, 10, 20, 30],
    "krum": [0, 10, 20, 30],
    "bulyan": [0, 10, 20],  # Limited by n>=4f+3 constraint
    "median": [0, 10, 20, 30],
}

# Heterogeneity experiments (adv=0% only)
HETEROGENEITY_ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf')]

def parse_run_name(run_name: str) -> dict:
    """Extract experiment parameters from run directory name."""
    if "p0b1bacd1" not in run_name:
        return None  # Skip old runs without proper hash

    parts = run_name.split("_")
    result = {}

    for i, part in enumerate(parts):
        if part.startswith("comp_"):
            result["aggregation"] = part.replace("comp_", "")
        elif part == "comp" and i + 1 < len(parts):
            result["aggregation"] = parts[i + 1]
        elif part.startswith("alpha"):
            result["alpha"] = float(part.replace("alpha", "").replace("inf", "inf"))
        elif part.startswith("adv"):
            result["adv_pct"] = int(part.replace("adv", ""))
        elif part.startswith("seed"):
            result["seed"] = int(part.replace("seed", ""))

    return result if "aggregation" in result and "seed" in result else None


def main():
    runs_dir = Path("runs")

    # Track existing runs
    existing = defaultdict(set)

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        params = parse_run_name(run_dir.name)
        if not params:
            continue

        agg = params.get("aggregation")
        alpha = params.get("alpha")
        adv = params.get("adv_pct")
        seed = params.get("seed")

        if agg in AGGREGATORS and alpha == 0.5 and adv is not None and seed in SEEDS:
            key = (agg, adv)
            existing[key].add(seed)

    # Identify missing experiments
    missing = []

    for agg in AGGREGATORS:
        for adv in ATTACK_LEVELS[agg]:
            for seed in SEEDS:
                key = (agg, adv)
                if seed not in existing.get(key, set()):
                    missing.append({
                        "aggregation": agg,
                        "alpha": 0.5,
                        "adv_pct": adv,
                        "seed": seed,
                    })

    # Print summary
    print("=" * 80)
    print("MISSING EXPERIMENTS ANALYSIS")
    print("=" * 80)
    print()

    # Group by (agg, adv)
    grouped = defaultdict(list)
    for exp in missing:
        key = (exp["aggregation"], exp["adv_pct"])
        grouped[key].append(exp["seed"])

    print(f"Total missing experiments: {len(missing)}")
    print()

    for (agg, adv), seeds in sorted(grouped.items()):
        existing_seeds = existing.get((agg, adv), set())
        print(f"{agg:8s} adv={adv:2d}%: {len(seeds):2d} missing (have {len(existing_seeds):2d}/10) - seeds {sorted(seeds)}")

    print()
    print("=" * 80)
    print("EXPERIMENT QUEUE (Sequential Execution)")
    print("=" * 80)
    print()

    # Sort by priority: Bulyan adv20 first, then complete other gaps
    priority_order = []

    # Priority 1: Bulyan adv20 (needed for constraint validation)
    bulyan_adv20 = [exp for exp in missing if exp["aggregation"] == "bulyan" and exp["adv_pct"] == 20]
    priority_order.extend(sorted(bulyan_adv20, key=lambda x: x["seed"]))

    # Priority 2: Other adv20 experiments (for heatmap 20% column)
    other_adv20 = [exp for exp in missing if exp["adv_pct"] == 20 and exp["aggregation"] != "bulyan"]
    priority_order.extend(sorted(other_adv20, key=lambda x: (x["aggregation"], x["seed"])))

    # Priority 3: adv30 experiments (for heatmap 30% column)
    adv30 = [exp for exp in missing if exp["adv_pct"] == 30]
    priority_order.extend(sorted(adv30, key=lambda x: (x["aggregation"], x["seed"])))

    # Priority 4: Remaining adv10 and adv0
    remaining = [exp for exp in missing if exp not in priority_order]
    priority_order.extend(sorted(remaining, key=lambda x: (x["adv_pct"], x["aggregation"], x["seed"])))

    # Write queue file
    queue_file = Path("experiment_queue.json")
    with open(queue_file, "w") as f:
        json.dump(priority_order, f, indent=2)

    print(f"Saved {len(priority_order)} experiments to: {queue_file}")
    print()

    # Print first 20 for preview
    print("First 20 experiments in queue:")
    for i, exp in enumerate(priority_order[:20], 1):
        print(f"{i:3d}. {exp['aggregation']:8s} alpha={exp['alpha']:.1f} adv={exp['adv_pct']:2d}% seed={exp['seed']}")

    if len(priority_order) > 20:
        print(f"... and {len(priority_order) - 20} more")

    print()
    print("=" * 80)
    print("To run these experiments sequentially:")
    print("  python scripts/run_experiment_queue.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
