#!/usr/bin/env python3
"""Create a stratified multi-class sample from CIC-IDS2017 full dataset.

This script creates a ~10k sample with representative attack type distribution
to enable multi-class intrusion detection experiments for the thesis.

Target distribution (issue #41):
    BENIGN:          ~40% (4000 samples)
    DoS Hulk:        ~15% (1500 samples)
    PortScan:        ~15% (1500 samples)
    DDoS:            ~10% (1000 samples)
    DoS GoldenEye:    ~8%  (800 samples)
    FTP-Patator:      ~5%  (500 samples)
    SSH-Patator:      ~4%  (400 samples)
    Web Attack*:      ~3%  (300 samples)
    ─────────────────────────────────────
    Total:          ~100% (10,000 samples)

    * Web Attack = Web Attack – Brute Force + XSS + SQL Injection combined
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Target sample counts per label (total = 10,000)
TARGET_DISTRIBUTION: dict[str, int] = {
    "BENIGN": 4000,
    "DoS Hulk": 1500,
    "PortScan": 1500,
    "DDoS": 1000,
    "DoS GoldenEye": 800,
    "FTP-Patator": 500,
    "SSH-Patator": 400,
    # Web Attack types combined = 300
    "Web Attack � Brute Force": 150,
    "Web Attack � XSS": 100,
    "Web Attack � Sql Injection": 50,
}


def load_all_csvs(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CIC-IDS2017 CSV files."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {raw_dir}")

    print(f"Found {len(csv_files)} CSV files:")
    dfs = []
    for csv_file in csv_files:
        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined):,}")
    return combined


def stratified_sample(df: pd.DataFrame, label_col: str, targets: dict[str, int], seed: int = 42) -> pd.DataFrame:
    """Create stratified sample based on target counts per label."""
    rng = np.random.default_rng(seed)
    sampled_parts = []

    print("\nSampling by label:")
    for label, target_count in targets.items():
        label_df = df[df[label_col] == label]
        available = len(label_df)

        if available == 0:
            print(f"  [WARNING]  {label}: 0 available (skipping)")
            continue

        # If we don't have enough samples, take all available
        n_sample = min(target_count, available)
        if n_sample < target_count:
            print(f"  [WARNING]  {label}: {n_sample}/{target_count} (insufficient data)")
        else:
            print(f"  [PASS]  {label}: {n_sample}/{target_count}")

        # Random sample without replacement
        indices = rng.choice(available, size=n_sample, replace=False)
        sampled_parts.append(label_df.iloc[indices])

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # Shuffle final dataset
    indices = rng.permutation(len(sampled))
    sampled = sampled.iloc[indices].reset_index(drop=True)

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified multi-class CIC-IDS2017 sample")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with raw CIC-IDS2017 CSVs (MachineLearningCVE/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cic/cic_ids2017_multiclass.csv",
        help="Output CSV path (default: data/cic/cic_ids2017_multiclass.csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    df = load_all_csvs(input_dir)

    # Detect label column (CIC-IDS2017 uses " Label" with leading space)
    label_col = " Label" if " Label" in df.columns else "Label"
    if label_col not in df.columns:
        raise SystemExit("Could not find label column. Tried: ' Label', 'Label'")

    print(f"\nLabel column: '{label_col}'")
    print(f"Unique labels ({df[label_col].nunique()}):")
    for label in sorted(df[label_col].unique()):
        count = (df[label_col] == label).sum()
        print(f"  {label}: {count:,}")

    sampled = stratified_sample(df, label_col, TARGET_DISTRIBUTION, args.seed)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    sampled.to_csv(output_path, index=False)
    print(f"\n[PASS] Wrote {len(sampled):,} samples to {output_path}")

    # Verify final distribution
    print("\nFinal label distribution:")
    for label in sorted(sampled[label_col].unique()):
        count = (sampled[label_col] == label).sum()
        pct = 100.0 * count / len(sampled)
        print(f"  {label}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
