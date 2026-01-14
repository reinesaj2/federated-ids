#!/usr/bin/env python3
"""
Extract metrics from all experiments to a single CSV file.
Memory-efficient batch processing approach.
"""

import csv
import os
import re
from pathlib import Path


def identify_dataset(dirname: str) -> str:
    dirname_lower = dirname.lower()
    if "unsw" in dirname_lower:
        return "unsw"
    if "_datasetcic" in dirname_lower or dirname_lower.startswith("cic_"):
        return "cic"
    if "datasetedge" in dirname_lower or dirname_lower.startswith("dsedge"):
        return "iiot"
    if dirname_lower.startswith("comp_") and "cic" not in dirname_lower and "unsw" not in dirname_lower:
        return "iiot"
    return "unknown"


def parse_params(dirname: str) -> dict:
    params = {"dataset": identify_dataset(dirname)}

    patterns = {
        "aggregator": r"comp_([a-z]+)_",
        "alpha": r"alpha([\d.]+|inf)",
        "adv_pct": r"adv(\d+)",
        "dp": r"dp(\d+)",
        "pers": r"pers(\d+)",
        "mu": r"mu([\d.]+)",
        "seed": r"seed(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            val = match.group(1)
            if key == "alpha":
                params[key] = float("inf") if val == "inf" else float(val)
            elif key in ["adv_pct", "dp", "pers", "seed"]:
                params[key] = int(val)
            elif key == "mu":
                params[key] = float(val)
            else:
                params[key] = val

    return params


def extract_final_metrics(exp_dir: Path) -> dict:
    metrics = {}
    client_files = list(exp_dir.glob("client_*_metrics.csv"))

    if not client_files:
        return metrics

    final_f1s = []
    final_accs = []
    final_fprs = []

    for cf in client_files:
        try:
            with open(cf, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue

                max_round = max(int(r.get("round", 0)) for r in rows)
                final_rows = [r for r in rows if int(r.get("round", 0)) == max_round]

                for r in final_rows:
                    if "macro_f1_after" in r and r["macro_f1_after"]:
                        try:
                            final_f1s.append(float(r["macro_f1_after"]))
                        except ValueError:
                            pass
                    if "acc_after" in r and r["acc_after"]:
                        try:
                            final_accs.append(float(r["acc_after"]))
                        except ValueError:
                            pass
                    if "benign_fpr_argmax" in r and r["benign_fpr_argmax"]:
                        try:
                            final_fprs.append(float(r["benign_fpr_argmax"]))
                        except ValueError:
                            pass
        except Exception:
            continue

    if final_f1s:
        metrics["macro_f1"] = sum(final_f1s) / len(final_f1s)
    if final_accs:
        metrics["accuracy"] = sum(final_accs) / len(final_accs)
    if final_fprs:
        metrics["benign_fpr"] = sum(final_fprs) / len(final_fprs)

    return metrics


def main():
    base_path = Path("/Users/abrahamreines/Documents/Thesis")
    data_dir = base_path / "all_experiment_results"
    output_file = base_path / "cluster-experiments" / "all_experiments_summary.csv"

    print("=" * 70)
    print("EXTRACTING METRICS FROM ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Source: {data_dir}")
    print(f"Output: {output_file}")

    fieldnames = [
        "dir_name", "dataset", "aggregator", "alpha", "adv_pct",
        "dp", "pers", "mu", "seed", "macro_f1", "accuracy", "benign_fpr"
    ]

    dirs = [d for d in data_dir.iterdir() if d.is_symlink() or d.is_dir()]
    total = len(dirs)
    print(f"\nTotal directories to process: {total}")

    processed = 0
    loaded = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, exp_dir in enumerate(dirs):
            if i % 1000 == 0:
                print(f"  Processing {i}/{total}...")

            real_path = exp_dir.resolve() if exp_dir.is_symlink() else exp_dir
            if not real_path.is_dir():
                continue

            params = parse_params(exp_dir.name)
            if params.get("dataset") == "unknown":
                continue

            metrics = extract_final_metrics(real_path)
            if not metrics:
                continue

            row = {
                "dir_name": exp_dir.name,
                **params,
                **metrics,
            }

            for field in fieldnames:
                if field not in row:
                    row[field] = ""

            writer.writerow(row)
            loaded += 1
            processed += 1

    print(f"\nCompleted!")
    print(f"  Processed: {processed}")
    print(f"  Loaded with metrics: {loaded}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
