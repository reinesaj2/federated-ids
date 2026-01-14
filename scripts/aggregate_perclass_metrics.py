#!/usr/bin/env python3
"""
Aggregate per-class metrics from the large CSV into summary statistics.
This creates manageable files for plotting.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    base_path = Path("/Users/abrahamreines/Documents/Thesis")
    input_file = base_path / "cluster-experiments" / "all_perclass_metrics.csv"
    output_dir = base_path / "cluster-experiments"

    print("=" * 70)
    print("AGGREGATING PER-CLASS METRICS")
    print("=" * 70)

    final_round_stats = defaultdict(lambda: {"f1_values": [], "seeds": set(), "max_round": 0})
    convergence_stats = defaultdict(lambda: {"f1_values": []})

    print("\nReading CSV in streaming mode...")
    
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        
        row_count = 0
        for row in reader:
            row_count += 1
            if row_count % 5000000 == 0:
                print(f"  Processed {row_count:,} rows...")

            dataset = row["dataset"]
            aggregator = row["aggregator"]
            alpha = row["alpha"]
            adv_pct = row["adv_pct"]
            mu = row["mu"]
            seed = row["seed"]
            round_num = int(row["round"])
            class_name = row["class_name"]
            f1 = float(row["f1"]) if row["f1"] else 0.0

            key = (dataset, aggregator, alpha, adv_pct, mu, class_name)
            conv_key = (dataset, aggregator, alpha, adv_pct, mu, class_name, round_num)
            
            stats = final_round_stats[key]
            stats["f1_values"].append(f1)
            stats["seeds"].add(seed)
            stats["max_round"] = max(stats["max_round"], round_num)
            
            convergence_stats[conv_key]["f1_values"].append(f1)

    print(f"\nTotal rows processed: {row_count:,}")

    print("\nComputing final round aggregates...")
    final_output = output_dir / "perclass_final_summary.csv"
    
    with open(final_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "aggregator", "alpha", "adv_pct", "mu", "class_name",
            "f1_mean", "f1_std", "f1_min", "f1_max", "n_seeds", "n_samples"
        ])

        for key, stats in final_round_stats.items():
            dataset, aggregator, alpha, adv_pct, mu, class_name = key
            f1_values = stats["f1_values"]
            
            if not f1_values:
                continue
            
            mean_f1 = statistics.mean(f1_values)
            std_f1 = statistics.stdev(f1_values) if len(f1_values) > 1 else 0
            min_f1 = min(f1_values)
            max_f1 = max(f1_values)
            n_seeds = len(stats["seeds"])
            n_samples = len(f1_values)
            
            writer.writerow([
                dataset, aggregator, alpha, adv_pct, mu, class_name,
                f"{mean_f1:.6f}", f"{std_f1:.6f}", f"{min_f1:.6f}", f"{max_f1:.6f}",
                n_seeds, n_samples
            ])

    print(f"  Saved: {final_output}")

    print("\nComputing convergence aggregates...")
    conv_output = output_dir / "perclass_convergence_summary.csv"
    
    with open(conv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "aggregator", "alpha", "adv_pct", "mu", "class_name", "round",
            "f1_mean", "f1_std", "n_samples"
        ])

        for key, stats in convergence_stats.items():
            dataset, aggregator, alpha, adv_pct, mu, class_name, round_num = key
            f1_values = stats["f1_values"]
            
            if not f1_values:
                continue
            
            mean_f1 = statistics.mean(f1_values)
            std_f1 = statistics.stdev(f1_values) if len(f1_values) > 1 else 0
            n_samples = len(f1_values)
            
            writer.writerow([
                dataset, aggregator, alpha, adv_pct, mu, class_name, round_num,
                f"{mean_f1:.6f}", f"{std_f1:.6f}", n_samples
            ])

    print(f"  Saved: {conv_output}")

    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
