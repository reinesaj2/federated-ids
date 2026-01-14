#!/usr/bin/env python3
"""
Generate Chapter 4 plots one at a time to avoid memory issues.
Each plot is saved individually and the figure is closed after saving.

DEPRECATED: Use `python -m plots chapter4 --data <CSV>` instead.
"""

import sys
import warnings

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
try:
    from plots.deprecation import check_and_warn
    check_and_warn()
except ImportError:
    warnings.warn(
        "DEPRECATED: generate_chapter4_plots.py is deprecated. "
        "Use: python -m plots chapter4 --data <CSV>",
        DeprecationWarning,
        stacklevel=1,
    )

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DATASETS = ['iiot', 'cic', 'unsw']
LABELS = {'iiot': 'Edge-IIoTset', 'cic': 'CIC-IDS2017', 'unsw': 'UNSW-NB15'}
COLORS = {'iiot': '#0173B2', 'cic': '#DE8F05', 'unsw': '#029E73'}
MARKERS = {'iiot': 'o', 'cic': 's', 'unsw': '^'}

AGG_ORDER = ['fedavg', 'bulyan', 'krum', 'median']
AGG_LABELS = {'fedavg': 'FedAvg', 'bulyan': 'Bulyan', 'krum': 'Krum', 'median': 'Median'}
AGG_COLORS = {'fedavg': '#0173B2', 'bulyan': '#DE8F05', 'krum': '#029E73', 'median': '#CC78BC'}

ADV_LEVELS = [0, 10, 20, 30]
ALPHA_VALUES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
MU_VALUES = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 11, 'figure.dpi': 100, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3
})


def save_plot(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path.name}')


def plot_fig41a(df, output_dir):
    benign = df[(df['adv_pct'] == 0) & (df['mu'] == 0.0)]
    means = [benign[benign['dataset']==d]['macro_f1'].mean() for d in DATASETS]
    stds = [benign[benign['dataset']==d]['macro_f1'].std() for d in DATASETS]
    counts = [len(benign[benign['dataset']==d]) for d in DATASETS]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=stds, color=[COLORS[d] for d in DATASETS], capsize=5, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in DATASETS], rotation=15, ha='right')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('(a) Mean Macro F1 (Benign)', fontweight='bold')
    ax.set_ylim(0, 1.05)
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.annotate(f'n={cnt}', xy=(bar.get_x()+bar.get_width()/2, means[i]+stds[i]+0.02), ha='center', fontsize=8)
    plt.tight_layout()
    save_plot(fig, output_dir / 'fig4.1a_mean_f1_benign.png')


def plot_fig41b(df, output_dir):
    benign = df[(df['adv_pct'] == 0) & (df['mu'] == 0.0)]
    box_data = [benign[benign['dataset']==d]['macro_f1'].dropna().values for d in DATASETS]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(box_data, patch_artist=True, labels=[LABELS[d] for d in DATASETS])
    for patch, d in zip(bp['boxes'], DATASETS):
        patch.set_facecolor(COLORS[d])
        patch.set_alpha(0.7)
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('(b) F1 Distribution (Benign)', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    save_plot(fig, output_dir / 'fig4.1b_f1_distribution.png')


def plot_fig41c(df, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for ds in DATASETS:
        ds_data = df[(df['dataset'] == ds) & (df['adv_pct'] == 0)]
        alpha_means = ds_data.groupby('alpha')['macro_f1'].mean()
        valid = [(a, alpha_means[a]) for a in ALPHA_VALUES if a in alpha_means.index]
        if valid:
            alphas, means = zip(*valid)
            ax.plot(alphas, means, marker=MARKERS[ds], label=LABELS[ds], color=COLORS[ds], linewidth=2)
    ax.set_xlabel('Dirichlet Alpha')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('(c) Heterogeneity Impact', fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_plot(fig, output_dir / 'fig4.1c_heterogeneity.png')


def plot_fig41d(df, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    x = np.arange(len(ADV_LEVELS))
    for i, ds in enumerate(DATASETS):
        means = [df[(df['dataset']==ds) & (df['adv_pct']==adv)]['macro_f1'].mean() for adv in ADV_LEVELS]
        ax.bar(x + i*width, means, width, label=LABELS[ds], color=COLORS[ds])
    ax.set_xlabel('Byzantine Clients (%)')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('(d) Attack Impact', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{a}%' for a in ADV_LEVELS])
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_plot(fig, output_dir / 'fig4.1d_attack_impact.png')


def plot_fig42_per_dataset(df, output_dir):
    for idx, ds in enumerate(DATASETS):
        ds_benign = df[(df['dataset']==ds) & (df['adv_pct']==0) & (df['mu']==0.0)]
        agg_stats = []
        for agg in AGG_ORDER:
            agg_data = ds_benign[ds_benign['aggregator']==agg]['macro_f1'].dropna()
            if len(agg_data) > 0:
                agg_stats.append((agg, agg_data.mean(), agg_data.std()))

        if agg_stats:
            fig, ax = plt.subplots(figsize=(6, 4))
            aggs, means, stds = zip(*agg_stats)
            x = np.arange(len(aggs))
            ax.bar(x, means, yerr=stds, color=[AGG_COLORS[a] for a in aggs], edgecolor='black', capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([AGG_LABELS[a] for a in aggs], rotation=45, ha='right')
            ax.set_ylabel('Macro F1 Score')
            ax.set_title(f'({chr(97+idx)}) {LABELS[ds]} - Aggregators', fontweight='bold')
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            save_plot(fig, output_dir / f'fig4.2{chr(97+idx)}_{ds}_aggregators.png')


def plot_fig42d(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.2
    x = np.arange(len(DATASETS))
    for i, agg in enumerate(AGG_ORDER):
        means = [df[(df['dataset']==ds) & (df['aggregator']==agg) & (df['adv_pct']==0)]['macro_f1'].mean() for ds in DATASETS]
        ax.bar(x + i*width, means, width, label=AGG_LABELS[agg], color=AGG_COLORS[agg])
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('(d) Cross-Dataset Aggregator Comparison', fontweight='bold')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([LABELS[d] for d in DATASETS], rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_plot(fig, output_dir / 'fig4.2d_cross_dataset.png')


def plot_fig43_per_dataset(df, output_dir):
    for idx, ds in enumerate(DATASETS):
        fig, ax = plt.subplots(figsize=(6, 4))
        for agg in AGG_ORDER:
            means = [df[(df['dataset']==ds) & (df['aggregator']==agg) & (df['adv_pct']==adv)]['macro_f1'].mean() for adv in ADV_LEVELS]
            valid = [(ADV_LEVELS[i], means[i]) for i in range(len(means)) if not np.isnan(means[i])]
            if valid:
                advs, ms = zip(*valid)
                ax.plot(advs, ms, marker='o', label=AGG_LABELS[agg], color=AGG_COLORS[agg], linewidth=2)
        ax.set_xlabel('Byzantine Clients (%)')
        ax.set_ylabel('Macro F1 Score')
        ax.set_title(f'({chr(97+idx)}) {LABELS[ds]} - Attack Resilience', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        save_plot(fig, output_dir / f'fig4.3{chr(97+idx)}_{ds}_resilience.png')


def plot_fig43d_retention(df, output_dir):
    ret_data = []
    for ds in DATASETS:
        for agg in AGG_ORDER:
            benign = df[(df['dataset']==ds) & (df['aggregator']==agg) & (df['adv_pct']==0)]['macro_f1'].mean()
            attack30 = df[(df['dataset']==ds) & (df['aggregator']==agg) & (df['adv_pct']==30)]['macro_f1'].mean()
            if benign > 0 and not np.isnan(attack30):
                ret_data.append({'ds': ds, 'agg': agg, 'ret': (attack30/benign)*100})

    if ret_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ret_df = pd.DataFrame(ret_data)
        pivot = ret_df.pivot(index='agg', columns='ds', values='ret')
        pivot = pivot.reindex(AGG_ORDER).reindex(columns=DATASETS)
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_yticks(range(len(AGG_ORDER)))
        ax.set_xticklabels([LABELS[d] for d in DATASETS], rotation=15, ha='right')
        ax.set_yticklabels([AGG_LABELS[a] for a in AGG_ORDER])
        for i in range(len(AGG_ORDER)):
            for j in range(len(DATASETS)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Retention %')
        ax.set_title('(d) F1 Retention at 30% Attack', fontweight='bold')
        plt.tight_layout()
        save_plot(fig, output_dir / 'fig4.3d_retention_heatmap.png')


def plot_fig44_per_dataset(df, output_dir):
    for idx, ds in enumerate(DATASETS):
        ds_data = df[(df['dataset']==ds) & (df['adv_pct']==0) & (df['mu']==0.0)]
        alpha_stats = [(a, ds_data[ds_data['alpha']==a]['macro_f1'].mean(), ds_data[ds_data['alpha']==a]['macro_f1'].std())
                       for a in ALPHA_VALUES if len(ds_data[ds_data['alpha']==a]) > 0]

        if alpha_stats:
            fig, ax = plt.subplots(figsize=(6, 4))
            alphas, means, stds = zip(*alpha_stats)
            ax.errorbar(alphas, means, yerr=stds, marker=MARKERS[ds], color=COLORS[ds], linewidth=2, capsize=3)
            ax.set_xscale('log')
            ax.set_xlabel('Dirichlet Alpha (log)')
            ax.set_ylabel('Macro F1 Score')
            ax.set_title(f'({chr(97+idx)}) {LABELS[ds]} - Heterogeneity', fontweight='bold')
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            save_plot(fig, output_dir / f'fig4.4{chr(97+idx)}_{ds}_alpha.png')


def plot_fig44d_comparison(df, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for ds in DATASETS:
        ds_data = df[(df['dataset']==ds) & (df['adv_pct']==0) & (df['mu']==0.0)]
        alpha_means = [(a, ds_data[ds_data['alpha']==a]['macro_f1'].mean()) for a in ALPHA_VALUES
                       if len(ds_data[ds_data['alpha']==a]) > 0]
        if alpha_means:
            alphas, means = zip(*alpha_means)
            ax.plot(alphas, means, marker=MARKERS[ds], label=LABELS[ds], color=COLORS[ds], linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Dirichlet Alpha (log)')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('(d) Cross-Dataset Heterogeneity', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_plot(fig, output_dir / 'fig4.4d_cross_dataset_alpha.png')


def plot_fig45_per_dataset(df, output_dir):
    for idx, ds in enumerate(DATASETS):
        ds_data = df[(df['dataset']==ds) & (df['adv_pct']==0)]
        mu_stats = [(m, ds_data[ds_data['mu']==m]['macro_f1'].mean(), ds_data[ds_data['mu']==m]['macro_f1'].std())
                    for m in MU_VALUES if len(ds_data[ds_data['mu']==m]) > 0]

        if mu_stats:
            fig, ax = plt.subplots(figsize=(6, 4))
            mus, means, stds = zip(*mu_stats)
            x = np.arange(len(mus))
            ax.bar(x, means, yerr=stds, color=COLORS[ds], edgecolor='black', capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{m}' for m in mus], rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('FedProx mu')
            ax.set_ylabel('Macro F1 Score')
            ax.set_title(f'({chr(97+idx)}) {LABELS[ds]} - FedProx', fontweight='bold')
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            save_plot(fig, output_dir / f'fig4.5{chr(97+idx)}_{ds}_mu.png')


def plot_fig45d_improvement(df, output_dir):
    imp_data = []
    for ds in DATASETS:
        ds_data = df[(df['dataset']==ds) & (df['adv_pct']==0)]
        baseline = ds_data[ds_data['mu']==0.0]['macro_f1'].mean()
        if np.isnan(baseline) or baseline == 0:
            continue
        for mu in MU_VALUES:
            if mu == 0:
                continue
            mu_mean = ds_data[ds_data['mu']==mu]['macro_f1'].mean()
            if not np.isnan(mu_mean):
                imp = ((mu_mean - baseline) / baseline) * 100
                imp_data.append({'ds': ds, 'mu': mu, 'imp': imp})

    if imp_data:
        fig, ax = plt.subplots(figsize=(6, 5))
        imp_df = pd.DataFrame(imp_data)
        valid_mus = sorted([m for m in imp_df['mu'].unique() if m > 0])
        pivot = imp_df.pivot(index='mu', columns='ds', values='imp')
        pivot = pivot.reindex(valid_mus).reindex(columns=DATASETS)
        im = ax.imshow(pivot.values, cmap='RdBu', aspect='auto', vmin=-30, vmax=30)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_yticks(range(len(valid_mus)))
        ax.set_xticklabels([LABELS[d] for d in DATASETS], rotation=15, ha='right')
        ax.set_yticklabels([f'{m}' for m in valid_mus], fontsize=7)
        for i in range(len(valid_mus)):
            for j in range(len(DATASETS)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8, label='% vs FedAvg')
        ax.set_title('(d) FedProx vs FedAvg Improvement', fontweight='bold')
        plt.tight_layout()
        save_plot(fig, output_dir / 'fig4.5d_fedprox_improvement.png')


def generate_summary_csv(df, output_dir):
    summary = []
    for ds in DATASETS:
        for agg in AGG_ORDER:
            for adv in ADV_LEVELS:
                data = df[(df['dataset']==ds) & (df['aggregator']==agg) & (df['adv_pct']==adv)]['macro_f1']
                if len(data) > 0:
                    summary.append({
                        'Dataset': LABELS[ds], 'Aggregator': AGG_LABELS[agg], 'Byzantine_%': adv,
                        'F1_Mean': data.mean(), 'F1_Std': data.std(), 'N_Runs': len(data)
                    })
    pd.DataFrame(summary).to_csv(output_dir / 'chapter4_summary_table.csv', index=False, float_format='%.4f')
    print('  Saved: chapter4_summary_table.csv')


def main():
    base = Path('/Users/abrahamreines/Documents/Thesis')
    csv_path = base / 'cluster-experiments' / 'all_experiments_summary.csv'
    output_dir = base / 'cluster-experiments' / 'thesis_plots' / 'chapter4'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('CHAPTER 4 RESULTS PLOTTING')
    print('=' * 70)

    print(f'\nLoading {csv_path}...')
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} experiments')
    for ds in DATASETS:
        print(f'  {LABELS[ds]}: {len(df[df["dataset"]==ds])}')

    print('\n--- Figure 4.1: Overall Performance ---')
    plot_fig41a(df, output_dir)
    plot_fig41b(df, output_dir)
    plot_fig41c(df, output_dir)
    plot_fig41d(df, output_dir)

    print('\n--- Figure 4.2: Aggregation Comparison ---')
    plot_fig42_per_dataset(df, output_dir)
    plot_fig42d(df, output_dir)

    print('\n--- Figure 4.3: Adversarial Resilience ---')
    plot_fig43_per_dataset(df, output_dir)
    plot_fig43d_retention(df, output_dir)

    print('\n--- Figure 4.4: Heterogeneity Analysis ---')
    plot_fig44_per_dataset(df, output_dir)
    plot_fig44d_comparison(df, output_dir)

    print('\n--- Figure 4.5: FedProx Analysis ---')
    plot_fig45_per_dataset(df, output_dir)
    plot_fig45d_improvement(df, output_dir)

    print('\n--- Summary Table ---')
    generate_summary_csv(df, output_dir)

    print('\n' + '=' * 70)
    print('COMPLETE')
    print('=' * 70)
    print(f'\nOutput: {output_dir}')


if __name__ == '__main__':
    main()
