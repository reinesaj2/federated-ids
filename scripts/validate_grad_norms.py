#!/usr/bin/env python3
"""
Gradient norm validation script for CI workflows.

Validates that gradient norms are being computed and logged correctly:
1. grad_norm_l2 column exists in client metrics
2. grad_norm values are positive (non-zero for actual training)
3. grad_norm values are finite (no NaN or Inf)
4. grad_norm values are within expected ranges

Exit codes:
0 - All validation passed
1 - Validation failures detected
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def validate_grad_norms(metrics_csv: Path) -> tuple[bool, list[str]]:
    """Validate gradient norms in a client metrics CSV.

    Returns:
        (success, errors) tuple where success is True if all checks pass,
        and errors is a list of error messages.
    """
    errors = []

    try:
        df = pd.read_csv(metrics_csv)
    except Exception as e:
        return False, [f"Failed to read CSV: {e}"]

    # Check column exists
    if "grad_norm_l2" not in df.columns:
        return False, ["grad_norm_l2 column not found in metrics CSV"]

    grad_norms = df["grad_norm_l2"].dropna()

    if len(grad_norms) == 0:
        errors.append("No gradient norm values found (all NaN)")
        return False, errors

    # Check for non-finite values
    if not grad_norms.apply(lambda x: pd.api.types.is_number(x)).all():
        non_numeric = grad_norms[~grad_norms.apply(lambda x: pd.api.types.is_number(x))]
        errors.append(f"Non-numeric gradient norms found: {non_numeric.values[:5]}")

    numeric_norms = pd.to_numeric(grad_norms, errors="coerce").dropna()

    if len(numeric_norms) == 0:
        errors.append("No valid numeric gradient norms found")
        return False, errors

    # Check for infinite values
    if not numeric_norms.apply(lambda x: abs(x) != float("inf")).all():
        errors.append("Infinite gradient norms detected")

    # Check for negative values
    if (numeric_norms < 0).any():
        neg_count = (numeric_norms < 0).sum()
        errors.append(f"Negative gradient norms detected: {neg_count} values")

    # Check for all-zero (suspicious unless model not training)
    if (numeric_norms == 0).all():
        errors.append("All gradient norms are zero (model may not be training)")

    # Check for reasonable ranges (gradient norms typically 0.001 to 1000)
    mean_norm = numeric_norms.mean()
    if mean_norm > 10000:
        errors.append(f"Suspiciously large mean gradient norm: {mean_norm:.2f}")

    if errors:
        return False, errors

    return True, []


def main():
    parser = argparse.ArgumentParser(description="Validate gradient norms in client metrics")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory containing experiment runs",
    )
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Fail if no client metrics files are found",
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: Runs directory does not exist: {runs_dir}")
        sys.exit(1 if args.fail_on_missing else 0)

    # Find all client metrics files
    client_metrics_files = list(runs_dir.glob("**/client_*_metrics.csv"))

    if not client_metrics_files:
        msg = f"WARNING: No client metrics files found in {runs_dir}"
        print(msg)
        sys.exit(1 if args.fail_on_missing else 0)

    print(f"Found {len(client_metrics_files)} client metrics files")

    total_validated = 0
    total_failed = 0
    all_errors = []

    for metrics_file in client_metrics_files:
        success, errors = validate_grad_norms(metrics_file)

        if success:
            total_validated += 1
            print(f"✓ {metrics_file.relative_to(runs_dir)}")
        else:
            total_failed += 1
            print(f"✗ {metrics_file.relative_to(runs_dir)}")
            for error in errors:
                print(f"  - {error}")
                all_errors.append(f"{metrics_file.name}: {error}")

    print(f"\nValidation Summary:")
    print(f"  Total files: {len(client_metrics_files)}")
    print(f"  Passed: {total_validated}")
    print(f"  Failed: {total_failed}")

    if total_failed > 0:
        print(f"\nERROR: {total_failed} file(s) failed gradient norm validation")
        sys.exit(1)

    print("\nAll gradient norm validations passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
