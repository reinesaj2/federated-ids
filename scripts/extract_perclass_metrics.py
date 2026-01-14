#!/usr/bin/env python3
"""
Extract per-class F1 metrics from all experiments.
Creates a comprehensive CSV for per-class analysis across all 3 datasets.
"""

import ast
import csv
import json
import os
import re
from pathlib import Path


IIOT_CLASSES = [
    "BENIGN", "DDOS_TCP", "DDOS_ICMP", "VULNERABILITY_SCANNER", "DDOS_UDP",
    "SQL_INJECTION", "DDOS_HTTP", "PORT_SCANNING", "PASSWORD", "UPLOADING",
    "XSS", "RANSOMWARE", "BACKDOOR", "MITM", "FINGERPRINTING"
]

CIC_CLASSES = [
    "BENIGN", "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
    "DoS Hulk", "DoS GoldenEye", "Heartbleed", "Web Attack Brute Force",
    "Web Attack XSS", "Web Attack Sql Injection", "Infiltration", "Bot",
    "PortScan", "DDoS"
]

UNSW_CLASSES = [
    "BENIGN", "EXPLOITS", "GENERIC", "FUZZERS", "RECONNAISSANCE",
    "DOS", "ANALYSIS", "BACKDOOR", "SHELLCODE", "WORMS"
]


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
        "mu": r"mu([\d.]+)",
        "seed": r"seed(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            val = match.group(1)
            if key == "alpha":
                params[key] = float("inf") if val == "inf" else float(val)
            elif key in ["adv_pct", "seed"]:
                params[key] = int(val)
            elif key == "mu":
                params[key] = float(val)
            else:
                params[key] = val

    return params


def get_class_names(dataset: str) -> list:
    if dataset == "iiot":
        return IIOT_CLASSES
    elif dataset == "cic":
        return CIC_CLASSES
    elif dataset == "unsw":
        return UNSW_CLASSES
    return []


def extract_perclass_metrics(exp_dir: Path, dataset: str) -> list:
    records = []
    client_files = list(exp_dir.glob("client_*_metrics.csv"))

    if not client_files:
        return records

    class_names = get_class_names(dataset)

    for cf in client_files:
        try:
            with open(cf, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                continue

            client_match = re.search(r"client_(\d+)", cf.name)
            client_id = int(client_match.group(1)) if client_match else 0

            for row in rows:
                round_num = int(row.get("round", 0))
                f1_str = row.get("f1_per_class_after", "{}")

                try:
                    f1_dict = ast.literal_eval(f1_str) if f1_str and f1_str != "nan" else {}
                except:
                    f1_dict = {}

                for class_idx, f1_val in f1_dict.items():
                    class_idx_int = int(class_idx)
                    class_name = class_names[class_idx_int] if class_idx_int < len(class_names) else f"Class_{class_idx}"

                    records.append({
                        "client_id": client_id,
                        "round": round_num,
                        "class_idx": class_idx_int,
                        "class_name": class_name,
                        "f1": float(f1_val) if f1_val else 0.0,
                    })

        except Exception as e:
            continue

    return records


def main():
    base_path = Path("/Users/abrahamreines/Documents/Thesis")
    data_dir = base_path / "all_experiment_results"
    output_file = base_path / "cluster-experiments" / "all_perclass_metrics.csv"

    print("=" * 70)
    print("EXTRACTING PER-CLASS F1 METRICS FROM ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Source: {data_dir}")
    print(f"Output: {output_file}")

    fieldnames = [
        "dir_name", "dataset", "aggregator", "alpha", "adv_pct", "mu", "seed",
        "client_id", "round", "class_idx", "class_name", "f1"
    ]

    dirs = [d for d in data_dir.iterdir() if d.is_symlink() or d.is_dir()]
    total = len(dirs)
    print(f"\nTotal directories to process: {total}")

    processed = 0
    records_written = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, exp_dir in enumerate(dirs):
            if i % 500 == 0:
                print(f"  Processing {i}/{total}... ({records_written} records)")

            real_path = exp_dir.resolve() if exp_dir.is_symlink() else exp_dir
            if not real_path.is_dir():
                continue

            params = parse_params(exp_dir.name)
            if params.get("dataset") == "unknown":
                continue

            perclass_records = extract_perclass_metrics(real_path, params["dataset"])
            if not perclass_records:
                continue

            for rec in perclass_records:
                row = {
                    "dir_name": exp_dir.name,
                    **params,
                    **rec,
                }
                for field in fieldnames:
                    if field not in row:
                        row[field] = ""
                writer.writerow(row)
                records_written += 1

            processed += 1

    print(f"\nCompleted!")
    print(f"  Experiments processed: {processed}")
    print(f"  Per-class records written: {records_written}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
