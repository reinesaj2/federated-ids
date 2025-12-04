#!/usr/bin/env python3
"""Quick analysis and visualization of IIoT experimental results"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
from pathlib import Path

# Set up matplotlib for display
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def load_experiment_results():
    """Load all IIoT experiment results"""
    results = []

    # Find all IIoT experiments
    exp_dirs = sorted(glob.glob('runs/dsedge-iiotset-full_comp_*'))

    for exp_dir in exp_dirs:
        config_path = Path(exp_dir) / 'config.json'
        metrics_path = Path(exp_dir) / 'metrics.csv'

        if not (config_path.exists() and metrics_path.exists()):
            continue

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Load metrics
        metrics = pd.read_csv(metrics_path)

        # Get client metrics for final round
        client_files = sorted(glob.glob(f'{exp_dir}/client_*_metrics.csv'))
        if client_files:
            client_dfs = [pd.read_csv(f) for f in client_files]
            # Filter out empty dataframes
            client_dfs = [df for df in client_dfs if len(df) > 0]

            if client_dfs:
                final_client_metrics = [df.iloc[-1] for df in client_dfs]
                avg_acc = np.mean([m['acc_after'] for m in final_client_metrics])
                avg_f1 = np.mean([m['macro_f1_after'] for m in final_client_metrics])
                std_acc = np.std([m['acc_after'] for m in final_client_metrics])
                std_f1 = np.std([m['macro_f1_after'] for m in final_client_metrics])
            else:
                avg_acc = avg_f1 = std_acc = std_f1 = 0
        else:
            avg_acc = avg_f1 = std_acc = std_f1 = 0

        results.append({
            'exp_dir': exp_dir,
            'strategy': config['aggregation'],
            'alpha': config['alpha'],
            'seed': config['seed'],
            'num_rounds': config['num_rounds'],
            'metrics': metrics,
            'final_avg_acc': avg_acc,
            'final_avg_f1': avg_f1,
            'std_acc': std_acc,
            'std_f1': std_f1
        })

    return results

# Load results
print("Loading experimental results...")
results = load_experiment_results()
print(f"Loaded {len(results)} experiments")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Edge-IIoT Dataset: Robust Aggregation Strategy Comparison',
             fontsize=16, fontweight='bold')

# Group by strategy
strategies = {}
for r in results:
    strat = r['strategy']
    if strat not in strategies:
        strategies[strat] = []
    strategies[strat].append(r)

# Define colors
colors = {
    'fedavg': '#2E86AB',
    'krum': '#A23B72',
    'bulyan': '#F18F01',
    'median': '#C73E1D'
}

# 1. Final Accuracy by Strategy (alpha=1.0, IID)
ax = axes[0, 0]
iid_results = [r for r in results if r['alpha'] == 1.0]
strategy_names = []
accuracies = []
f1_scores = []

for strat in sorted(set(r['strategy'] for r in iid_results)):
    strat_results = [r for r in iid_results if r['strategy'] == strat]
    if strat_results:
        strategy_names.append(strat.upper())
        accs = [r['final_avg_acc'] for r in strat_results]
        f1s = [r['final_avg_f1'] for r in strat_results]
        accuracies.append(np.mean(accs))
        f1_scores.append(np.mean(f1s))

x = np.arange(len(strategy_names))
width = 0.35
ax.bar(x - width/2, [a*100 for a in accuracies], width,
       label='Accuracy', alpha=0.8, color='#2E86AB')
ax.bar(x + width/2, [f*100 for f in f1_scores], width,
       label='Macro F1', alpha=0.8, color='#F18F01')
ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Final Performance: IID Data (α=1.0)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategy_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([85, 100])

# 2. Accuracy vs Data Heterogeneity (FedAvg)
ax = axes[0, 1]
fedavg_results = [r for r in results if r['strategy'] == 'fedavg']
alphas = sorted(set(r['alpha'] for r in fedavg_results if isinstance(r['alpha'], (int, float))))

alpha_accs = []
alpha_labels = []
for alpha in alphas:
    alpha_results = [r for r in fedavg_results if r['alpha'] == alpha]
    if alpha_results:
        mean_acc = np.mean([r['final_avg_acc'] for r in alpha_results])
        std_acc = np.std([r['final_avg_acc'] for r in alpha_results])
        alpha_accs.append((mean_acc, std_acc))
        alpha_labels.append(f'{alpha}')

if alpha_accs:
    means = [a[0]*100 for a in alpha_accs]
    stds = [a[1]*100 for a in alpha_accs]
    ax.errorbar(alpha_labels, means, yerr=stds, marker='o', linewidth=2,
                markersize=8, capsize=5, color='#2E86AB',
                label='FedAvg Accuracy')
    ax.set_xlabel('α (Dirichlet Parameter)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Data Heterogeneity Impact on FedAvg', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

# 3. Training Convergence Comparison
ax = axes[0, 2]
for strat_name, strat_results in strategies.items():
    # Get alpha=1.0, seed=42 run
    target_run = [r for r in strat_results if r['alpha'] == 1.0 and r['seed'] == 42]
    if target_run:
        metrics = target_run[0]['metrics']
        if 'round' in metrics.columns:
            # This is server metrics - skip
            continue
        # Try to get learning curve from client metrics
        client_file = glob.glob(f"{target_run[0]['exp_dir']}/client_0_metrics.csv")
        if client_file:
            client_df = pd.read_csv(client_file[0])
            rounds = client_df['round'].values
            acc = client_df['acc_after'].values * 100
            ax.plot(rounds, acc, marker='o', linewidth=2, markersize=4,
                   label=strat_name.upper(), color=colors.get(strat_name, '#888'))

ax.set_xlabel('Round', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Training Convergence (α=1.0, seed=42)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 4. F1-Score Comparison Box Plot
ax = axes[1, 0]
f1_data = []
labels = []
for strat in sorted(strategies.keys()):
    strat_results = [r for r in strategies[strat] if r['alpha'] == 1.0]
    if strat_results:
        f1_vals = [r['final_avg_f1'] * 100 for r in strat_results]
        f1_data.append(f1_vals)
        labels.append(strat.upper())

if f1_data:
    bp = ax.boxplot(f1_data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.6),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Macro F1-Score (%)', fontweight='bold')
    ax.set_title('F1-Score Distribution Across Seeds (α=1.0)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# 5. Per-Client Variance
ax = axes[1, 1]
for strat_name in sorted(strategies.keys()):
    strat_results = [r for r in strategies[strat_name] if r['alpha'] == 1.0]
    if strat_results:
        mean_stds = np.mean([r['std_acc'] * 100 for r in strat_results])
        sem_stds = np.std([r['std_acc'] * 100 for r in strat_results])
        ax.bar(strat_name.upper(), mean_stds, alpha=0.8,
               color=colors.get(strat_name, '#888'),
               yerr=sem_stds, capsize=5)

ax.set_ylabel('Std Dev of Client Accuracies (%)', fontweight='bold')
ax.set_title('Model Fairness: Cross-Client Variance (α=1.0)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 6. Summary Statistics Table
ax = axes[1, 2]
ax.axis('off')

# Create summary table
summary_data = []
for strat in sorted(strategies.keys()):
    strat_results = [r for r in strategies[strat] if r['alpha'] == 1.0]
    if strat_results:
        n = len(strat_results)
        mean_acc = np.mean([r['final_avg_acc'] * 100 for r in strat_results])
        mean_f1 = np.mean([r['final_avg_f1'] * 100 for r in strat_results])
        summary_data.append([
            strat.upper(),
            f'{n}',
            f'{mean_acc:.2f}%',
            f'{mean_f1:.2f}%'
        ])

table = ax.table(cellText=summary_data,
                colLabels=['Strategy', 'Runs', 'Acc', 'F1'],
                cellLoc='center',
                loc='center',
                bbox=[0.1, 0.3, 0.8, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Summary Statistics (IID, α=1.0)', fontweight='bold', pad=20)

plt.tight_layout()

# Save figure
output_path = 'iiot_analysis_dashboard.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

# Also create a detailed statistical summary
print("\n" + "="*80)
print("DETAILED EXPERIMENTAL SUMMARY")
print("="*80)

for strat in sorted(strategies.keys()):
    print(f"\n{strat.upper()}:")
    strat_results = strategies[strat]

    # Group by alpha
    alpha_groups = {}
    for r in strat_results:
        alpha = r['alpha']
        if alpha not in alpha_groups:
            alpha_groups[alpha] = []
        alpha_groups[alpha].append(r)

    for alpha in sorted(alpha_groups.keys()):
        runs = alpha_groups[alpha]
        accs = [r['final_avg_acc'] * 100 for r in runs]
        f1s = [r['final_avg_f1'] * 100 for r in runs]

        print(f"  α={alpha}: {len(runs)} runs")
        print(f"    Accuracy:  {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"    Macro F1:  {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")

print("\n" + "="*80)
print("Opening visualization...")

# Open the image
import subprocess
subprocess.run(['open', output_path])
