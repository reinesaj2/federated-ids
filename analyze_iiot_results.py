#!/usr/bin/env python3
"""Analyze IIoT experiment results to identify performance patterns and issues."""

import csv
import json
from pathlib import Path
import re
from collections import defaultdict
import statistics

def parse_experiment_name(exp_dir):
    """Extract experiment parameters from directory name."""
    name = exp_dir.name
    params = {}

    # Extract aggregation method
    if '_comp_fedavg_' in name:
        params['aggregation'] = 'fedavg'
    elif '_comp_bulyan_' in name:
        params['aggregation'] = 'bulyan'
    elif '_comp_krum_' in name:
        params['aggregation'] = 'krum'
    elif '_comp_median_' in name:
        params['aggregation'] = 'median'
    else:
        params['aggregation'] = 'unknown'

    # Extract alpha
    alpha_match = re.search(r'alpha([\d.]+)', name)
    if alpha_match:
        params['alpha'] = float(alpha_match.group(1))

    # Extract seed
    seed_match = re.search(r'seed(\d+)', name)
    if seed_match:
        params['seed'] = int(seed_match.group(1))

    # Extract mu (FedProx parameter)
    mu_match = re.search(r'mu([\d.]+)', name)
    if mu_match:
        params['mu'] = float(mu_match.group(1))

    return params

def safe_float(val):
    """Safely convert to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def analyze_iiot_results():
    """Analyze all IIoT experiment results."""

    runs_dir = Path('runs')

    # Find all IIoT experiments
    iiot_experiments = [d for d in runs_dir.iterdir()
                       if d.is_dir() and 'iiot' in d.name.lower()]

    print(f"Found {len(iiot_experiments)} IIoT experiment directories\n")

    results = []

    for exp_dir in iiot_experiments:
        # Parse experiment parameters
        params = parse_experiment_name(exp_dir)

        # Try to load config
        config_file = exp_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    params.update({
                        'dataset': config.get('dataset', 'unknown'),
                        'num_clients': config.get('num_clients', 0),
                        'num_rounds': config.get('num_rounds', 0),
                    })
            except:
                pass

        # Load client metrics
        final_f1_scores = []
        final_acc_scores = []
        max_round = 0

        for client_file in exp_dir.glob('client_*_metrics.csv'):
            try:
                with open(client_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if len(rows) > 0:
                        # Get last row (final round)
                        last_row = rows[-1]
                        round_num = int(last_row.get('round', 0))
                        max_round = max(max_round, round_num)

                        f1_val = safe_float(last_row.get('macro_f1_after'))
                        acc_val = safe_float(last_row.get('acc_after'))

                        if f1_val is not None:
                            final_f1_scores.append(f1_val)
                        if acc_val is not None:
                            final_acc_scores.append(acc_val)
            except Exception as e:
                continue

        if not final_f1_scores:
            continue

        # Calculate statistics
        result = {
            'experiment': exp_dir.name,
            **params,
            'final_round': max_round,
            'num_clients_reporting': len(final_f1_scores),
            'macro_f1_mean': statistics.mean(final_f1_scores),
            'macro_f1_std': statistics.stdev(final_f1_scores) if len(final_f1_scores) > 1 else 0,
            'macro_f1_min': min(final_f1_scores),
            'macro_f1_max': max(final_f1_scores),
        }

        if final_acc_scores:
            result['acc_mean'] = statistics.mean(final_acc_scores)
            result['acc_std'] = statistics.stdev(final_acc_scores) if len(final_acc_scores) > 1 else 0

        results.append(result)

    if len(results) == 0:
        print("No valid results found!")
        return

    print("=" * 80)
    print("IIOT EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # Group by aggregation method
    print("\n1. PERFORMANCE BY AGGREGATION METHOD:")
    print("-" * 80)
    agg_groups = defaultdict(list)
    for r in results:
        agg_groups[r.get('aggregation', 'unknown')].append(r)

    for agg, agg_results in sorted(agg_groups.items()):
        f1_scores = [r['macro_f1_mean'] for r in agg_results]
        acc_scores = [r.get('acc_mean', 0) for r in agg_results if 'acc_mean' in r]

        print(f"\n{agg.upper()}:")
        print(f"  Number of experiments: {len(agg_results)}")
        print(f"  Mean Macro F1: {statistics.mean(f1_scores):.4f} ± {statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0:.4f}")
        print(f"  Best Macro F1: {max(f1_scores):.4f}")
        print(f"  Worst Macro F1: {min(f1_scores):.4f}")
        if acc_scores:
            print(f"  Mean Accuracy: {statistics.mean(acc_scores):.4f} ± {statistics.stdev(acc_scores) if len(acc_scores) > 1 else 0:.4f}")

    # Group by alpha (heterogeneity)
    print("\n\n2. PERFORMANCE BY DATA HETEROGENEITY (alpha):")
    print("-" * 80)
    alpha_groups = defaultdict(list)
    for r in results:
        if 'alpha' in r:
            alpha_groups[r['alpha']].append(r)

    for alpha in sorted(alpha_groups.keys()):
        alpha_results = alpha_groups[alpha]
        f1_scores = [r['macro_f1_mean'] for r in alpha_results]
        acc_scores = [r.get('acc_mean', 0) for r in alpha_results if 'acc_mean' in r]

        print(f"\nAlpha = {alpha}:")
        print(f"  Number of experiments: {len(alpha_results)}")
        print(f"  Mean Macro F1: {statistics.mean(f1_scores):.4f} ± {statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0:.4f}")
        if acc_scores:
            print(f"  Mean Accuracy: {statistics.mean(acc_scores):.4f} ± {statistics.stdev(acc_scores) if len(acc_scores) > 1 else 0:.4f}")

    # Best performing configurations
    print("\n\n3. TOP 10 BEST PERFORMING CONFIGURATIONS:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x['macro_f1_mean'], reverse=True)
    for row in sorted_results[:10]:
        print(f"\nMacro F1: {row['macro_f1_mean']:.4f}")
        print(f"  Aggregation: {row.get('aggregation', 'unknown')}")
        print(f"  Alpha: {row.get('alpha', 'N/A')}")
        print(f"  Mu: {row.get('mu', 'N/A')}")
        print(f"  Dataset: {row.get('dataset', 'unknown')}")
        print(f"  Seed: {row.get('seed', 'N/A')}")

    # Worst performing configurations
    print("\n\n4. BOTTOM 10 WORST PERFORMING CONFIGURATIONS:")
    print("-" * 80)
    for row in sorted_results[-10:]:
        print(f"\nMacro F1: {row['macro_f1_mean']:.4f}")
        print(f"  Aggregation: {row.get('aggregation', 'unknown')}")
        print(f"  Alpha: {row.get('alpha', 'N/A')}")
        print(f"  Mu: {row.get('mu', 'N/A')}")
        print(f"  Dataset: {row.get('dataset', 'unknown')}")
        print(f"  Seed: {row.get('seed', 'N/A')}")

    # Overall statistics
    print("\n\n5. OVERALL STATISTICS:")
    print("-" * 80)
    all_f1 = [r['macro_f1_mean'] for r in results]
    all_acc = [r.get('acc_mean', 0) for r in results if 'acc_mean' in r]

    print(f"Total experiments analyzed: {len(results)}")
    print(f"Overall mean Macro F1: {statistics.mean(all_f1):.4f} ± {statistics.stdev(all_f1) if len(all_f1) > 1 else 0:.4f}")
    if all_acc:
        print(f"Overall mean Accuracy: {statistics.mean(all_acc):.4f} ± {statistics.stdev(all_acc) if len(all_acc) > 1 else 0:.4f}")
    print(f"Best Macro F1 achieved: {max(all_f1):.4f}")
    print(f"Worst Macro F1 achieved: {min(all_f1):.4f}")

    # Save detailed results to CSV
    output_file = 'iiot_results_summary.csv'
    with open(output_file, 'w', newline='') as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n\nDetailed results saved to: {output_file}")

    return results

if __name__ == '__main__':
    analyze_iiot_results()
