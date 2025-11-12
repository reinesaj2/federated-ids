#!/usr/bin/env python3
"""
Create stratified samples from Edge-IIoTset for tiered testing.

This script generates three sample sizes:
- Quick (50k): For rapid CI validation on PRs
- Nightly (500k): For comprehensive nightly experiments
- Full (2M): For publication-quality thesis results (90% of dataset)

Each sample maintains the original attack distribution to preserve
dataset characteristics for valid federated learning experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


DATASET_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    DATASET_ROOT / "datasets" / "edge-iiotset" / "Edge-IIoTset dataset" / "Selected dataset for ML and DL" / "DNN-EdgeIIoT-dataset.csv"
)
OUTPUT_DIR = DATASET_ROOT / "data" / "edge-iiotset"

SAMPLE_CONFIGS: dict[str, dict[str, int | str]] = {
    "quick": {"size": 50_000, "description": "Quick CI validation sample"},
    "nightly": {"size": 500_000, "description": "Comprehensive nightly experiment sample"},
    "full": {"size": 2_000_000, "description": "Full-scale thesis sample (90% of dataset)"},
}


def create_stratified_sample(
    source_csv: Path,
    output_csv: Path,
    target_size: int,
    seed: int = 42,
    label_col: str = "Attack_type",
) -> None:
    """
    Create stratified sample maintaining attack distribution.

    Args:
        source_csv: Path to full Edge-IIoTset CSV
        output_csv: Path to write sample
        target_size: Target number of samples
        seed: Random seed for reproducibility
        label_col: Label column to stratify on
    """
    print(f"Loading source dataset from {source_csv}")
    df = pd.read_csv(source_csv, low_memory=False)

    total_samples = len(df)
    print(f"Total samples in source: {total_samples:,}")

    if target_size >= total_samples:
        print(f"Target size {target_size:,} >= total samples, using full dataset")
        df_sample = df
    else:
        print(f"Creating stratified sample of {target_size:,} samples")

        # Get class distribution
        class_counts = df[label_col].value_counts()
        print("\nOriginal class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count:,} ({count / total_samples * 100:.1f}%)")

        # Calculate per-class sample sizes
        sampling_fraction = target_size / total_samples
        samples_per_class = (class_counts * sampling_fraction).round().astype(int)

        # Adjust to hit exact target size
        diff = target_size - samples_per_class.sum()
        if diff > 0:
            # Add to largest classes
            largest_classes = samples_per_class.nlargest(abs(diff)).index
            for cls in largest_classes:
                samples_per_class[cls] += 1
        elif diff < 0:
            # Remove from largest classes
            largest_classes = samples_per_class.nlargest(abs(diff)).index
            for cls in largest_classes:
                samples_per_class[cls] -= 1

        # Sample from each class
        dfs = []
        rng = np.random.default_rng(seed)
        for cls, n_samples in samples_per_class.items():
            class_df = df[df[label_col] == cls]
            if n_samples >= len(class_df):
                sampled = class_df
            else:
                indices = rng.choice(len(class_df), size=n_samples, replace=False)
                sampled = class_df.iloc[indices]
            dfs.append(sampled)

        df_sample = pd.concat(dfs, ignore_index=True)

        # Shuffle
        df_sample = df_sample.sample(frac=1, random_state=seed).reset_index(drop=True)

        print("\nSampled class distribution:")
        sampled_counts = df_sample[label_col].value_counts()
        for cls, count in sampled_counts.items():
            orig_pct = class_counts[cls] / total_samples * 100
            samp_pct = count / len(df_sample) * 100
            print(f"  {cls}: {count:,} ({samp_pct:.1f}% vs {orig_pct:.1f}% original)")

    # Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_csv, index=False)
    print(f"\nWrote {len(df_sample):,} samples to {output_csv}")
    print(f"File size: {output_csv.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified samples from Edge-IIoTset dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=["quick", "nightly", "full", "all"],
        default="all",
        help="Which sample tier to generate (default: all)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_PATH,
        help=f"Path to source Edge-IIoTset CSV (default: {SOURCE_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for samples (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source dataset not found: {args.source}\n" f"Please ensure Edge-IIoTset is extracted to datasets/edge-iiotset/")

    tiers_to_generate = [args.tier] if args.tier != "all" else ["quick", "nightly", "full"]

    for tier in tiers_to_generate:
        config = SAMPLE_CONFIGS[tier]
        output_path = args.output_dir / f"edge_iiotset_{tier}.csv"

        print("\n" + "=" * 70)
        print(f"Generating {tier.upper()} tier sample")
        print(f"Description: {config['description']}")
        print(f"Target size: {config['size']:,} samples")
        print("=" * 70)

        create_stratified_sample(
            source_csv=args.source,
            output_csv=output_path,
            target_size=cast(int, config["size"]),
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("Sample generation complete!")
    print(f"Samples written to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
