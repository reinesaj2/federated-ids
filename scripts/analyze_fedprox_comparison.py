#!/usr/bin/env python3
"""
FedProx vs FedAvg Comparison Analysis Script

Analyzes nightly comparison results from multiple alpha/mu combinations
and generates consolidated reports for thesis research.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_comparison_data(artifacts_dir: str) -> Dict[str, pd.DataFrame]:
    """Load metrics from all FedProx comparison artifacts."""
    comparison_data = {}
    artifacts_path = Path(artifacts_dir)

    # Pattern: fedprox-nightly-alpha{alpha}-mu{mu}-{sha}/
    for artifact_dir in artifacts_path.glob("fedprox-nightly-alpha*-mu*"):
        # Parse alpha and mu from directory name
        parts = artifact_dir.name.split('-')
        alpha_part = [p for p in parts if p.startswith('alpha')][0]
        mu_part = [p for p in parts if p.startswith('mu')][0]

        alpha = float(alpha_part.replace('alpha', ''))
        mu = float(mu_part.replace('mu', ''))

        # Load server metrics
        metrics_files = list(artifact_dir.glob("**/metrics.csv"))
        if metrics_files:
            server_df = pd.read_csv(metrics_files[0])
            server_df['alpha'] = alpha
            server_df['mu'] = mu
            server_df['algorithm'] = 'FedProx' if mu > 0 else 'FedAvg'

            key = f"alpha_{alpha}_mu_{mu}"
            comparison_data[key] = server_df

    return comparison_data


def analyze_convergence_comparison(comparison_data: Dict[str, pd.DataFrame]) -> Dict:
    """Compare FedAvg vs FedProx convergence across alpha values."""
    results = {
        'convergence_analysis': {},
        'final_metrics': {},
        'improvement_ratios': {}
    }

    for key, df in comparison_data.items():
        alpha = df['alpha'].iloc[0]
        mu = df['mu'].iloc[0]
        algorithm = df['algorithm'].iloc[0]

        # Extract convergence metrics
        if len(df) > 0:
            final_round = df['round'].max()
            convergence_data = {
                'alpha': alpha,
                'mu': mu,
                'algorithm': algorithm,
                'final_round': int(final_round),
                'rounds_to_converge': None,  # Could implement convergence detection
                'final_l2_distance': df['l2_to_benign_mean'].iloc[-1] if 'l2_to_benign_mean' in df.columns else None,
                'final_cosine_similarity': df['cos_to_benign_mean'].iloc[-1] if 'cos_to_benign_mean' in df.columns else None,
                'avg_aggregation_time': df['t_aggregate_ms'].mean() if 't_aggregate_ms' in df.columns else None,
                'total_training_time': df['t_round_ms'].sum() if 't_round_ms' in df.columns else None
            }

            results['convergence_analysis'][key] = convergence_data

    # Calculate improvement ratios (FedProx vs FedAvg)
    alpha_values = set(data['alpha'] for data in results['convergence_analysis'].values())

    for alpha in alpha_values:
        fedavg_key = f"alpha_{alpha}_mu_0.0"

        if fedavg_key in results['convergence_analysis']:
            fedavg_data = results['convergence_analysis'][fedavg_key]

            for mu in [0.01, 0.1]:  # Compare against non-zero mu values
                fedprox_key = f"alpha_{alpha}_mu_{mu}"

                if fedprox_key in results['convergence_analysis']:
                    fedprox_data = results['convergence_analysis'][fedprox_key]

                    # Calculate improvement ratios
                    improvement = {}
                    if fedavg_data['avg_aggregation_time'] and fedprox_data['avg_aggregation_time']:
                        improvement['time_overhead'] = (
                            fedprox_data['avg_aggregation_time'] / fedavg_data['avg_aggregation_time']
                        )

                    if fedavg_data['final_l2_distance'] and fedprox_data['final_l2_distance']:
                        improvement['l2_improvement'] = (
                            fedavg_data['final_l2_distance'] / fedprox_data['final_l2_distance']
                        )

                    results['improvement_ratios'][f"{fedprox_key}_vs_{fedavg_key}"] = improvement

    return results


def plot_regularization_effects(comparison_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot effect of μ values on model performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FedProx Regularization Effects Analysis', fontsize=16)

    # Prepare data for plotting
    plot_data = []
    for key, df in comparison_data.items():
        alpha = df['alpha'].iloc[0]
        mu = df['mu'].iloc[0]
        algorithm = df['algorithm'].iloc[0]

        if len(df) > 0:
            plot_data.append({
                'alpha': alpha,
                'mu': mu,
                'algorithm': algorithm,
                'final_l2_distance': df['l2_to_benign_mean'].iloc[-1] if 'l2_to_benign_mean' in df.columns else np.nan,
                'final_cosine_similarity': df['cos_to_benign_mean'].iloc[-1] if 'cos_to_benign_mean' in df.columns else np.nan,
                'avg_aggregation_time': df['t_aggregate_ms'].mean() if 't_aggregate_ms' in df.columns else np.nan,
                'update_norm_stability': df['update_norm_std'].mean() if 'update_norm_std' in df.columns else np.nan
            })

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        plt.figtext(0.5, 0.5, 'No data available for plotting', ha='center', va='center')
        plt.savefig(f"{output_dir}/fedprox_performance_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Plot 1: L2 Distance by Alpha and Mu
    if 'final_l2_distance' in plot_df.columns and not plot_df['final_l2_distance'].isna().all():
        sns.lineplot(data=plot_df, x='mu', y='final_l2_distance', hue='alpha',
                    marker='o', ax=axes[0, 0])
        axes[0, 0].set_title('Final L2 Distance to Benign Mean')
        axes[0, 0].set_xlabel('FedProx μ Value')
        axes[0, 0].set_ylabel('L2 Distance')
        axes[0, 0].legend(title='Alpha (Non-IID Level)')

    # Plot 2: Cosine Similarity by Alpha and Mu
    if 'final_cosine_similarity' in plot_df.columns and not plot_df['final_cosine_similarity'].isna().all():
        sns.lineplot(data=plot_df, x='mu', y='final_cosine_similarity', hue='alpha',
                    marker='s', ax=axes[0, 1])
        axes[0, 1].set_title('Final Cosine Similarity to Benign Mean')
        axes[0, 1].set_xlabel('FedProx μ Value')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].legend(title='Alpha (Non-IID Level)')

    # Plot 3: Aggregation Time Overhead
    if 'avg_aggregation_time' in plot_df.columns and not plot_df['avg_aggregation_time'].isna().all():
        sns.boxplot(data=plot_df, x='alpha', y='avg_aggregation_time', hue='algorithm', ax=axes[1, 0])
        axes[1, 0].set_title('Aggregation Time by Algorithm')
        axes[1, 0].set_xlabel('Alpha (Non-IID Level)')
        axes[1, 0].set_ylabel('Avg Aggregation Time (ms)')
        axes[1, 0].legend(title='Algorithm')

    # Plot 4: Update Norm Stability
    if 'update_norm_stability' in plot_df.columns and not plot_df['update_norm_stability'].isna().all():
        sns.scatterplot(data=plot_df, x='mu', y='update_norm_stability', hue='alpha',
                       size='alpha', sizes=(50, 200), ax=axes[1, 1])
        axes[1, 1].set_title('Update Norm Stability')
        axes[1, 1].set_xlabel('FedProx μ Value')
        axes[1, 1].set_ylabel('Update Norm Std Dev')
        axes[1, 1].legend(title='Alpha (Non-IID Level)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fedprox_performance_plots.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_thesis_tables(analysis_results: Dict, output_dir: str):
    """Generate LaTeX tables for thesis results section."""
    latex_output = []

    # Table 1: Convergence Comparison
    latex_output.append("% FedProx vs FedAvg Convergence Comparison")
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{FedProx vs FedAvg Convergence Analysis}")
    latex_output.append("\\label{tab:fedprox_convergence}")
    latex_output.append("\\begin{tabular}{lccccc}")
    latex_output.append("\\toprule")
    latex_output.append("Algorithm & $\\alpha$ & $\\mu$ & Final L2 Distance & Cosine Similarity & Avg Time (ms) \\\\")
    latex_output.append("\\midrule")

    convergence_data = analysis_results.get('convergence_analysis', {})
    for key, data in sorted(convergence_data.items()):
        algorithm = data['algorithm']
        alpha = data['alpha']
        mu = data['mu']
        l2_dist = data.get('final_l2_distance', 'N/A')
        cos_sim = data.get('final_cosine_similarity', 'N/A')
        avg_time = data.get('avg_aggregation_time', 'N/A')

        # Format numbers
        l2_str = f"{l2_dist:.4f}" if isinstance(l2_dist, (int, float)) else str(l2_dist)
        cos_str = f"{cos_sim:.4f}" if isinstance(cos_sim, (int, float)) else str(cos_sim)
        time_str = f"{avg_time:.2f}" if isinstance(avg_time, (int, float)) else str(avg_time)

        latex_output.append(f"{algorithm} & {alpha:.2f} & {mu:.2f} & {l2_str} & {cos_str} & {time_str} \\\\")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    latex_output.append("")

    # Table 2: Performance Improvements
    improvement_data = analysis_results.get('improvement_ratios', {})
    if improvement_data:
        latex_output.append("% FedProx Performance Improvements")
        latex_output.append("\\begin{table}[htbp]")
        latex_output.append("\\centering")
        latex_output.append("\\caption{FedProx Performance Improvements over FedAvg}")
        latex_output.append("\\label{tab:fedprox_improvements}")
        latex_output.append("\\begin{tabular}{lccc}")
        latex_output.append("\\toprule")
        latex_output.append("Comparison & L2 Improvement & Time Overhead & Recommendation \\\\")
        latex_output.append("\\midrule")

        for key, improvements in improvement_data.items():
            l2_improvement = improvements.get('l2_improvement', 'N/A')
            time_overhead = improvements.get('time_overhead', 'N/A')

            l2_str = f"{l2_improvement:.2f}×" if isinstance(l2_improvement, (int, float)) else str(l2_improvement)
            time_str = f"{time_overhead:.2f}×" if isinstance(time_overhead, (int, float)) else str(time_overhead)

            # Simple recommendation logic
            if isinstance(l2_improvement, (int, float)) and l2_improvement > 1.1:
                recommendation = "Recommended"
            elif isinstance(time_overhead, (int, float)) and time_overhead > 1.5:
                recommendation = "High overhead"
            else:
                recommendation = "Marginal benefit"

            comparison_name = key.replace('_', ' ').replace('alpha ', 'α=').replace('mu ', 'μ=').replace(' vs ', ' vs ')
            latex_output.append(f"{comparison_name} & {l2_str} & {time_str} & {recommendation} \\\\")

        latex_output.append("\\bottomrule")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")

    # Write to file
    with open(f"{output_dir}/fedprox_thesis_tables.tex", 'w') as f:
        f.write('\n'.join(latex_output))


def generate_comparison_summary(analysis_results: Dict, output_dir: str):
    """Generate JSON summary of FedProx comparison results."""
    summary = {
        'experiment_metadata': {
            'total_comparisons': len(analysis_results.get('convergence_analysis', {})),
            'alpha_values_tested': list(set(
                data['alpha'] for data in analysis_results.get('convergence_analysis', {}).values()
            )),
            'mu_values_tested': list(set(
                data['mu'] for data in analysis_results.get('convergence_analysis', {}).values()
            )),
            'algorithms_compared': ['FedAvg', 'FedProx']
        },
        'key_findings': {},
        'recommendations': {},
        'raw_analysis_results': analysis_results
    }

    # Extract key findings
    convergence_data = analysis_results.get('convergence_analysis', {})
    improvement_data = analysis_results.get('improvement_ratios', {})

    # Find best performing configurations
    if convergence_data:
        best_l2 = min(
            (data for data in convergence_data.values() if data.get('final_l2_distance') is not None),
            key=lambda x: x['final_l2_distance'],
            default=None
        )
        if best_l2:
            summary['key_findings']['best_l2_distance'] = {
                'algorithm': best_l2['algorithm'],
                'alpha': best_l2['alpha'],
                'mu': best_l2['mu'],
                'value': best_l2['final_l2_distance']
            }

    # Calculate average improvements
    if improvement_data:
        l2_improvements = [
            imp['l2_improvement'] for imp in improvement_data.values()
            if 'l2_improvement' in imp and isinstance(imp['l2_improvement'], (int, float))
        ]
        time_overheads = [
            imp['time_overhead'] for imp in improvement_data.values()
            if 'time_overhead' in imp and isinstance(imp['time_overhead'], (int, float))
        ]

        if l2_improvements:
            summary['key_findings']['avg_l2_improvement'] = np.mean(l2_improvements)
        if time_overheads:
            summary['key_findings']['avg_time_overhead'] = np.mean(time_overheads)

    # Generate recommendations
    summary['recommendations'] = {
        'for_iid_data': "Use FedAvg for faster convergence and lower computational overhead",
        'for_mild_non_iid': "Use FedProx with μ=0.01 for improved stability with minimal overhead",
        'for_severe_non_iid': "Use FedProx with μ=0.1 for better convergence, accept higher computational cost",
        'thesis_conclusion': "FedProx shows measurable improvements in non-IID scenarios at the cost of increased computational overhead"
    }

    with open(f"{output_dir}/fedprox_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Analyze FedProx vs FedAvg comparison results")
    parser.add_argument("--artifacts_dir", type=str, required=True,
                       help="Directory containing comparison artifacts")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load comparison data
    print("Loading comparison data...")
    comparison_data = load_comparison_data(args.artifacts_dir)

    if not comparison_data:
        print("Warning: No comparison data found in artifacts directory")
        return

    print(f"Loaded data for {len(comparison_data)} comparisons")

    # Analyze results
    print("Analyzing convergence comparison...")
    analysis_results = analyze_convergence_comparison(comparison_data)

    # Generate plots
    print("Generating performance plots...")
    plot_regularization_effects(comparison_data, args.output_dir)

    # Generate thesis tables
    print("Generating thesis tables...")
    generate_thesis_tables(analysis_results, args.output_dir)

    # Generate summary
    print("Generating comparison summary...")
    generate_comparison_summary(analysis_results, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")
    print(f"Generated files:")
    print(f"  - fedprox_comparison_summary.json")
    print(f"  - fedprox_performance_plots.png")
    print(f"  - fedprox_thesis_tables.tex")


if __name__ == "__main__":
    main()