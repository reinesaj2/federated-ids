#!/usr/bin/env python3
"""
Simple visualization script for federated learning metrics.
Generates sample plots from the metrics CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def plot_server_metrics(metrics_path: str, output_path: str):
    """Plot server aggregation metrics."""
    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Federated Learning Server Metrics - D2 Demo', fontsize=16)

    # Plot 1: Aggregation timing
    axes[0, 0].plot(df['round'], df['t_aggregate_ms'], 'o-', color='blue', label='Aggregation Time')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Aggregation Time per Round')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Robustness metrics
    axes[0, 1].plot(df['round'], df['l2_to_benign_mean'], 'o-', color='red', label='L2 to Benign Mean')
    axes[0, 1].plot(df['round'], df['cos_to_benign_mean'], 'o-', color='green', label='Cosine Similarity')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_title('Robustness Metrics')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Update norms
    axes[1, 0].plot(df['round'], df['update_norm_mean'], 'o-', color='purple', label='Mean Update Norm')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Norm Value')
    axes[1, 0].set_title('Update Norms')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Coordinate agreement
    axes[1, 1].plot(df['round'], df['coord_median_agree_pct'], 'o-', color='orange', label='Coordinate Agreement %')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Agreement (%)')
    axes[1, 1].set_title('Coordinate-wise Median Agreement')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_client_metrics(client_metrics_paths: list, output_path: str):
    """Plot client training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Federated Learning Client Metrics - D2 Demo', fontsize=16)

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, client_path in enumerate(client_metrics_paths):
        df = pd.read_csv(client_path)
        client_id = df['client_id'].iloc[0]
        color = colors[i % len(colors)]

        # Plot 1: Training loss progression
        axes[0, 0].plot(df['round'], df['loss_after'], 'o-', color=color, label=f'Client {client_id}')

        # Plot 2: Accuracy progression
        axes[0, 1].plot(df['round'], df['acc_after'], 'o-', color=color, label=f'Client {client_id}')

        # Plot 3: Weight norms
        axes[1, 0].plot(df['round'], df['weight_norm_after'], 'o-', color=color, label=f'Client {client_id}')

        # Plot 4: Training time
        axes[1, 1].plot(df['round'], df['t_fit_ms'], 'o-', color=color, label=f'Client {client_id}')

    # Configure subplots
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss After Fit')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy After Fit')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Weight Norm')
    axes[1, 0].set_title('Model Weight Norms')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].set_title('Training Time per Round')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fedprox_comparison(comparison_dir: str, output_path: str):
    """Plot FedProx vs FedAvg comparison metrics."""
    fedavg_path = Path(comparison_dir) / "fedavg" / "metrics.csv"
    fedprox_path = Path(comparison_dir) / "fedprox" / "metrics.csv"

    if not (fedavg_path.exists() and fedprox_path.exists()):
        print(f"Missing comparison data: {fedavg_path} or {fedprox_path}")
        return

    fedavg_df = pd.read_csv(fedavg_path)
    fedprox_df = pd.read_csv(fedprox_path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FedProx vs FedAvg Comparison on Non-IID Data', fontsize=16)

    # Plot 1: Convergence comparison - L2 distance to benign mean
    axes[0, 0].plot(fedavg_df['round'], fedavg_df['l2_to_benign_mean'], 'o-',
                   color='blue', linewidth=2, markersize=8, label='FedAvg')
    axes[0, 0].plot(fedprox_df['round'], fedprox_df['l2_to_benign_mean'], 's-',
                   color='red', linewidth=2, markersize=8, label='FedProx')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('L2 Distance to Benign Mean')
    axes[0, 0].set_title('Model Drift (Lower = Better)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Cosine similarity comparison
    axes[0, 1].plot(fedavg_df['round'], fedavg_df['cos_to_benign_mean'], 'o-',
                   color='blue', linewidth=2, markersize=8, label='FedAvg')
    axes[0, 1].plot(fedprox_df['round'], fedprox_df['cos_to_benign_mean'], 's-',
                   color='red', linewidth=2, markersize=8, label='FedProx')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cosine Similarity to Benign Mean')
    axes[0, 1].set_title('Model Alignment (Higher = Better)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Update norm stability
    axes[1, 0].plot(fedavg_df['round'], fedavg_df['update_norm_mean'], 'o-',
                   color='blue', linewidth=2, markersize=8, label='FedAvg')
    axes[1, 0].plot(fedprox_df['round'], fedprox_df['update_norm_mean'], 's-',
                   color='red', linewidth=2, markersize=8, label='FedProx')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Mean Update Norm')
    axes[1, 0].set_title('Update Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Aggregation efficiency
    axes[1, 1].plot(fedavg_df['round'], fedavg_df['t_aggregate_ms'], 'o-',
                   color='blue', linewidth=2, markersize=8, label='FedAvg')
    axes[1, 1].plot(fedprox_df['round'], fedprox_df['t_aggregate_ms'], 's-',
                   color='red', linewidth=2, markersize=8, label='FedProx')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Aggregation Time (ms)')
    axes[1, 1].set_title('Computational Overhead')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot federated learning metrics")
    parser.add_argument("--run_dir", type=str, default=None, help="Directory containing metrics CSV files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--fedprox_comparison", action="store_true", help="Plot FedProx vs FedAvg comparison")
    parser.add_argument("--logdir", type=str, default=None, help="Log directory for comparison (used with --fedprox_comparison)")
    args = parser.parse_args()

    if args.fedprox_comparison:
        # Handle comparison plotting
        comparison_dir = args.logdir or "./comparison_logs"
        output_dir = Path(args.output_dir) if args.output_dir else Path(comparison_dir)
        output_dir.mkdir(exist_ok=True)

        comparison_plot_path = output_dir / "fedprox_vs_fedavg_comparison.png"
        plot_fedprox_comparison(comparison_dir, str(comparison_plot_path))
        print(f"FedProx vs FedAvg comparison plot saved to: {comparison_plot_path}")
        return

    # Original functionality
    if not args.run_dir:
        parser.error("--run_dir is required when not using --fedprox_comparison")

    run_path = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_path
    output_dir.mkdir(exist_ok=True)

    # Plot server metrics
    server_metrics_path = run_path / "metrics.csv"
    if server_metrics_path.exists():
        server_plot_path = output_dir / "server_metrics_plot.png"
        plot_server_metrics(str(server_metrics_path), str(server_plot_path))
        print(f"Server metrics plot saved to: {server_plot_path}")

    # Plot client metrics
    client_metrics_paths = list(run_path.glob("client_*_metrics.csv"))
    if client_metrics_paths:
        client_plot_path = output_dir / "client_metrics_plot.png"
        plot_client_metrics([str(p) for p in client_metrics_paths], str(client_plot_path))
        print(f"Client metrics plot saved to: {client_plot_path}")

    print(f"Visualization complete. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()