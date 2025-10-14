#!/usr/bin/env python3
"""
Analyze train/test data splits to diagnose personalization zero-gain.

This script investigates whether train and test data have different enough
distributions to benefit from personalization.

Usage:
    python scripts/analyze_data_splits.py --dataset unsw \\
        --data_path data/unsw/unsw_nb15_sample.csv --num_clients 5 \\
        --alpha 0.1
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    fit_preprocessor_global,
    load_unsw_nb15,
    load_cic_ids2017,
    prepare_partitions_from_dataframe,
)


def analyze_client_data_split(
    X_client: np.ndarray,
    y_client: np.ndarray,
    client_id: int,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """Analyze a single client's train/test split."""
    print(f"\n{'=' * 80}")
    print(f"Client {client_id} Data Analysis")
    print(f"{'=' * 80}")

    # Perform the same split as numpy_to_loaders
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_client,
            y_client,
            test_size=test_size,
            random_state=seed,
            stratify=y_client,
        )
    except ValueError:
        # Fall back to non-stratified if necessary
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=test_size, random_state=seed, stratify=None
        )

    # Class distributions
    print(f"\nTotal samples: {len(y_client)}")
    print(f"Train samples: {len(y_train)} ({len(y_train) / len(y_client) * 100:.1f}%)")
    print(f"Test samples: {len(y_test)} ({len(y_test) / len(y_client) * 100:.1f}%)")

    # Class distribution comparison
    print("\nClass Distribution:")
    print(f"{'Class':<10} {'Train %':<12} {'Test %':<12} {'Difference':<12}")
    print("-" * 50)

    unique_classes = np.unique(y_client)
    max_diff = 0.0
    for cls in unique_classes:
        train_pct = (y_train == cls).sum() / len(y_train) * 100
        test_pct = (y_test == cls).sum() / len(y_test) * 100
        diff = abs(train_pct - test_pct)
        max_diff = max(max_diff, diff)
        print(f"{cls:<10} {train_pct:<12.2f} {test_pct:<12.2f} {diff:<12.4f}")

    print(f"\nMax class distribution difference: {max_diff:.4f}%")

    if max_diff < 1.0:
        print(
            "[WARNING]  WARNING: Train and test have nearly identical " "class distributions!"
        )
        print(
            "   This is expected for stratified splits but limits "
            "personalization benefit."
        )
        print(
            "   Personalization helps most when train/test "
            "distributions differ significantly."
        )

    # Feature statistics
    print("\nFeature Statistics:")
    train_mean = X_train.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    mean_diff = np.abs(train_mean - test_mean).mean()
    print(f"Average feature mean difference: {mean_diff:.6f}")

    train_std = X_train.std(axis=0)
    test_std = X_test.std(axis=0)
    std_diff = np.abs(train_std - test_std).mean()
    print(f"Average feature std difference: {std_diff:.6f}")

    if mean_diff < 0.01 and std_diff < 0.01:
        print("[WARNING]  WARNING: Train and test feature distributions are " "very similar!")

    # Recommendation
    print("\nPersonalization Likelihood:")
    if max_diff < 1.0 and mean_diff < 0.01:
        print(
            "[LOW] - Train/test distributions are nearly identical "
            "(stratified split)"
        )
        print(
            "   Personalization unlikely to help unless model is "
            "underfitting globally."
        )
    elif max_diff < 5.0:
        print(
            "[WARNING]  MODERATE - Some distribution difference exists but "
            "limited benefit expected"
        )
    else:
        print(
            "[HIGH] - Significant distribution differences; "
            "personalization should help"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze data splits for personalization"
    )
    parser.add_argument("--dataset", type=str, default="unsw", choices=["unsw", "cic"])
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/unsw/unsw_nb15_sample.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet alpha (lower = more heterogeneous)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print("Data Split Analysis for Personalization Diagnosis")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Num clients: {args.num_clients}")
    print(f"Dirichlet alpha: {args.alpha}")
    print(f"Seed: {args.seed}")

    # Load dataset
    if args.dataset == "unsw":
        df = load_unsw_nb15(args.data_path)
        label_col = "attack_cat"
    elif args.dataset == "cic":
        df = load_cic_ids2017(args.data_path)
        label_col = "Label"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"\nTotal samples in dataset: {len(df)}")
    print(f"Classes: {df[label_col].nunique()}")
    print("Class distribution:")
    print(df[label_col].value_counts())

    # Preprocess globally
    pre, X, y = fit_preprocessor_global(df, label_col)
    print(f"\nPreprocessed features: {X.shape[1]}")

    # Partition data across clients
    partitions = prepare_partitions_from_dataframe(
        df,
        label_col,
        num_clients=args.num_clients,
        strategy="dirichlet",
        alpha=args.alpha,
        seed=args.seed,
    )

    # Analyze each client
    for client_id in range(args.num_clients):
        indices = partitions[client_id]
        if len(indices) == 0:
            print(f"\n[WARNING]  Client {client_id}: No data (empty partition)")
            continue

        X_client = X[indices]
        y_client = y[indices]

        analyze_client_data_split(
            X_client, y_client, client_id, test_size=0.2, seed=args.seed
        )

    # Overall summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 80}\n")

    print("Key Findings:")
    print(
        "1. Stratified train_test_split maintains class proportions in train and test"
    )
    print("2. This is good for unbiased evaluation but reduces personalization benefit")
    print("3. Personalization helps most when:")
    print("   - Train and test have different class distributions")
    print("   - Clients have heterogeneous data (low alpha)")
    print("   - Model has not fully converged globally")

    print("\nRecommendations to increase personalization gains:")
    print("1. Use very low alpha (0.01-0.1) to ensure high client heterogeneity")
    print("2. Increase personalization_epochs (5-10 instead of 2-3)")
    print("3. Consider using validation set for personalization instead of train set")
    print("4. Use higher learning rate for personalization (e.g., 0.01 or 0.02)")
    print(
        "5. If gains remain zero, this may indicate the global model "
        "already performs optimally"
    )


if __name__ == "__main__":
    main()
