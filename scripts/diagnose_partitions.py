#!/usr/bin/env python3
"""Diagnostic tool for inspecting data partition quality.

Usage:
    python scripts/diagnose_partitions.py \\
        --dataset data/edge-iiotset/edge_iiotset_full.csv \\
        --alpha 0.02 --num-clients 6 --seed 42

Outputs:
- Per-client class distribution
- Heterogeneity metrics (coefficient of variation, KL divergence)
- Warnings about constraint violations
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import (
    MIN_SAMPLES_PER_CLASS,
    dirichlet_partition,
    load_cic_ids2017,
    load_edge_iiotset,
    load_unsw_nb15,
)


def calculate_heterogeneity_metrics(labels, shards):
    """Calculate heterogeneity metrics for partition quality.

    Args:
        labels: Full label array
        shards: List of index lists per client

    Returns:
        dict with metrics
    """
    num_classes = int(labels.max()) + 1

    # Global distribution
    global_dist = np.array([np.sum(labels == c) / len(labels) for c in range(num_classes)])

    # Per-client distributions
    client_dists = []
    for shard in shards:
        shard_labels = labels[shard]
        dist = np.array([np.sum(shard_labels == c) / len(shard_labels) for c in range(num_classes)])
        client_dists.append(dist)

    # Calculate coefficient of variation for each class
    cvs = []
    for class_idx in range(num_classes):
        counts = [np.sum(labels[shard] == class_idx) for shard in shards]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv = std_count / mean_count if mean_count > 0 else 0
        cvs.append(cv)

    # Calculate average KL divergence from global distribution
    kl_divs = []
    for client_dist in client_dists:
        # KL(client || global)
        kl = np.sum(client_dist * np.log((client_dist + 1e-10) / (global_dist + 1e-10)))
        kl_divs.append(kl)

    return {
        "global_distribution": global_dist,
        "client_distributions": client_dists,
        "coefficient_variation_per_class": cvs,
        "avg_coefficient_variation": np.mean(cvs),
        "kl_divergences": kl_divs,
        "avg_kl_divergence": np.mean(kl_divs),
        "max_kl_divergence": np.max(kl_divs),
    }


def diagnose_partition(
    dataset: str,
    data_path: str,
    alpha: float,
    num_clients: int,
    seed: int,
    min_samples_per_class: int,
):
    """Diagnose partition quality for given configuration."""
    print(f"{'=' * 80}")
    print(f"PARTITION DIAGNOSTIC REPORT")
    print(f"{'=' * 80}\n")

    # Load dataset
    print(f"Loading dataset: {dataset} from {data_path}")
    if dataset == "unsw":
        df, label_col, _ = load_unsw_nb15(data_path)
    elif dataset == "cic":
        df, label_col, _ = load_cic_ids2017(data_path)
    elif dataset.startswith("edge-iiotset"):
        df, label_col, _ = load_edge_iiotset(data_path, use_multiclass=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Encode labels
    from data_preprocessing import _encode_labels_to_ints

    labels = _encode_labels_to_ints(df[label_col])
    num_classes = int(labels.max()) + 1

    print(f"Dataset: {len(labels)} samples, {num_classes} classes")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print()

    # Configuration
    print(f"CONFIGURATION:")
    print(f"  Alpha: {alpha}")
    print(f"  Num clients: {num_clients}")
    print(f"  Seed: {seed}")
    print(f"  Min samples per class: {min_samples_per_class}")
    print()

    # Attempt partitioning
    try:
        shards = dirichlet_partition(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_class=min_samples_per_class,
        )
        print(f"SUCCESS: Partitioning completed\n")
    except ValueError as e:
        print(f"FAILURE: Partitioning failed")
        print(f"Error: {e}\n")
        return

    # Per-client analysis
    print(f"{'=' * 80}")
    print(f"PER-CLIENT DISTRIBUTION")
    print(f"{'=' * 80}\n")

    violations = []
    for client_id, shard in enumerate(shards):
        shard_labels = labels[shard]
        counts = [np.sum(shard_labels == c) for c in range(num_classes)]
        total = len(shard_labels)

        print(f"Client {client_id}:")
        print(f"  Total samples: {total}")
        print(f"  Per-class counts: {counts}")

        # Check violations
        for class_idx, count in enumerate(counts):
            if count < min_samples_per_class:
                violations.append((client_id, class_idx, count))
                print(f"  WARNING: Class {class_idx} has {count} < {min_samples_per_class}")

        # Calculate proportions
        props = [c / total for c in counts]
        print(f"  Proportions: {[f'{p:.3f}' for p in props]}")
        print()

    # Heterogeneity metrics
    print(f"{'=' * 80}")
    print(f"HETEROGENEITY METRICS")
    print(f"{'=' * 80}\n")

    metrics = calculate_heterogeneity_metrics(labels, shards)

    print(f"Global distribution: {[f'{p:.3f}' for p in metrics['global_distribution']]}")
    print(f"Average CV (coefficient of variation): {metrics['avg_coefficient_variation']:.3f}")
    print(f"Average KL divergence: {metrics['avg_kl_divergence']:.3f}")
    print(f"Max KL divergence: {metrics['max_kl_divergence']:.3f}")
    print()

    print(f"Interpretation:")
    if metrics["avg_coefficient_variation"] > 0.5:
        print(f"  - HIGH heterogeneity (CV > 0.5)")
    elif metrics["avg_coefficient_variation"] > 0.3:
        print(f"  - MODERATE heterogeneity (0.3 < CV < 0.5)")
    else:
        print(f"  - LOW heterogeneity (CV < 0.3, approaching IID)")
    print()

    # Summary
    print(f"{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}\n")

    if violations:
        print(f"STATUS: FAILED - {len(violations)} constraint violations")
        for client_id, class_idx, count in violations:
            print(f"  Client {client_id}, class {class_idx}: {count} < {min_samples_per_class}")
    else:
        print(f"STATUS: PASSED - All constraints satisfied")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose data partition quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose low-alpha partition (extreme heterogeneity)
  python scripts/diagnose_partitions.py \\
      --dataset edge-iiotset-full \\
      --data-path data/edge-iiotset/edge_iiotset_full.csv \\
      --alpha 0.02 --num-clients 6

  # Diagnose IID partition
  python scripts/diagnose_partitions.py \\
      --dataset unsw \\
      --data-path data/unsw/UNSW_NB15_training-set.csv \\
      --alpha inf --num-clients 10
""",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["unsw", "cic", "edge-iiotset-quick", "edge-iiotset-nightly", "edge-iiotset-full"],
        help="Dataset type",
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--alpha", type=float, required=True, help="Dirichlet alpha parameter")
    parser.add_argument("--num-clients", type=int, required=True, help="Number of clients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=MIN_SAMPLES_PER_CLASS,
        help=f"Minimum samples per class (default: {MIN_SAMPLES_PER_CLASS})",
    )

    args = parser.parse_args()

    # Validate data path exists
    if not Path(args.data_path).exists():
        print(f"ERROR: Data path does not exist: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    diagnose_partition(
        dataset=args.dataset,
        data_path=args.data_path,
        alpha=args.alpha,
        num_clients=args.num_clients,
        seed=args.seed,
        min_samples_per_class=args.min_samples_per_class,
    )


if __name__ == "__main__":
    main()
