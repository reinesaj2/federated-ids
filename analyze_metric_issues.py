#!/usr/bin/env python3
"""Analyze metric issues from Issue #78."""

from pathlib import Path

import pandas as pd


def analyze_metrics_file(csv_path: Path) -> dict:
    """Analyze a single metrics CSV file for issues."""
    try:
        df = pd.read_csv(csv_path)

        results = {"file": str(csv_path), "total_rows": len(df), "cosine_issues": [], "l2_issues": [], "f1_issues": []}

        # Check cosine similarity issues
        if "cos_to_benign_mean" in df.columns:
            cosine_col = df["cos_to_benign_mean"].dropna()
            if len(cosine_col) > 0:
                cos_min = cosine_col.min()
                cos_max = cosine_col.max()

                if cos_min < 0.5:
                    results["cosine_issues"].append(f"Min cosine {cos_min:.6f} < 0.5")
                if cos_max == 1.0 and cosine_col.std() < 1e-6:
                    results["cosine_issues"].append("All cosine values = 1.0 (no variance)")
                if cos_min < 0.9:
                    results["cosine_issues"].append(f"Min cosine {cos_min:.6f} < 0.9 (suspicious)")

        # Check L2 distance issues
        if "l2_to_benign_mean" in df.columns:
            l2_col = df["l2_to_benign_mean"].dropna()
            if len(l2_col) > 0:
                l2_zeros = (l2_col == 0.0).sum()
                if l2_zeros > 0:
                    results["l2_issues"].append(f"{l2_zeros} L2 values = 0.0 exactly")
                if l2_zeros > 1:
                    results["l2_issues"].append("Multiple L2=0 values (possible bug)")

        # Check F1 ceiling effect
        if "macro_f1" in df.columns:
            f1_col = df["macro_f1"].dropna()
            if len(f1_col) > 0:
                f1_perfect = (f1_col >= 0.999).sum()
                if f1_perfect > len(f1_col) * 0.8:
                    results["f1_issues"].append(f"{f1_perfect}/{len(f1_col)} F1 values ≥ 0.999 (ceiling effect)")

        return results

    except Exception as e:
        return {"file": str(csv_path), "error": str(e)}


def main():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("No runs directory found")
        return

    print("Analyzing metrics files for Issue #78 issues...")
    print("=" * 60)

    issues_found = 0
    total_files = 0

    for csv_file in runs_dir.rglob("metrics.csv"):
        total_files += 1
        results = analyze_metrics_file(csv_file)

        if results.get("error"):
            print(f"ERROR: {results['file']} - {results['error']}")
            continue

        has_issues = (
            len(results.get("cosine_issues", [])) > 0 or len(results.get("l2_issues", [])) > 0 or len(results.get("f1_issues", [])) > 0
        )

        if has_issues:
            issues_found += 1
            print(f"\nISSUES FOUND: {results['file']}")
            print(f"  Rows: {results['total_rows']}")

            for issue in results.get("cosine_issues", []):
                print(f"  COSINE: {issue}")
            for issue in results.get("l2_issues", []):
                print(f"  L2: {issue}")
            for issue in results.get("f1_issues", []):
                print(f"  F1: {issue}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: Found issues in {issues_found}/{total_files} files")

    if issues_found == 0:
        print("No metric issues found - Issue #78 may already be resolved!")
        print("Current metrics appear to be within expected ranges.")
    else:
        print(f"Found {issues_found} files with metric issues that need investigation.")


if __name__ == "__main__":
    main()
