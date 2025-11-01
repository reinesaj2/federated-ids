#!/usr/bin/env python3
"""CLI entry point for federated learning metric visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from plot_config import PlotStyle, create_default_config
from plot_metrics_client import (
    _render_client_accuracy,
    _render_client_f1_overlay,
    _render_client_fpr,
    _render_client_loss,
    _render_client_norms,
    _render_client_tau,
    plot_client_metrics,
)
from plot_metrics_server import (
    _render_server_dispersion,
    _render_server_norms,
    _render_server_robustness,
    _render_server_timing,
    plot_server_metrics,
)
from plot_metrics_utils import first_present, render_mu_scatter

__all__ = [
    "plot_server_metrics",
    "plot_client_metrics",
    "plot_fedprox_comparison",
    "_render_server_timing",
    "_render_server_robustness",
    "_render_server_norms",
    "_render_server_dispersion",
    "_render_client_loss",
    "_render_client_accuracy",
    "_render_client_norms",
    "_render_client_f1_overlay",
    "_render_client_tau",
    "_render_client_fpr",
    "render_mu_scatter",
    "_render_mu_scatter",
]


_render_mu_scatter = render_mu_scatter


def plot_fedprox_comparison(comparison_dir: str, output_path: str, config: dict | None = None) -> None:
    if config is None:
        config = create_default_config()

    style = config.get("style", PlotStyle())
    title = config.get("title", "FedProx vs FedAvg Comparison on Non-IID Data")

    style.apply()

    fedavg_path = Path(comparison_dir) / "fedavg" / "metrics.csv"
    fedprox_path = Path(comparison_dir) / "fedprox" / "metrics.csv"

    if not (fedavg_path.exists() and fedprox_path.exists()):
        print(f"Missing comparison data: {fedavg_path} or {fedprox_path}")
        return

    fedavg_df = pd.read_csv(fedavg_path)
    fedprox_df = pd.read_csv(fedprox_path)

    fig, axes = plt.subplots(2, 2, figsize=style.figsize)
    fig.suptitle(title, fontsize=style.title_size)

    axes[0, 0].plot(
        fedavg_df["round"],
        fedavg_df["l2_to_benign_mean"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
        label="FedAvg",
    )
    axes[0, 0].plot(
        fedprox_df["round"],
        fedprox_df["l2_to_benign_mean"],
        "s-",
        color="red",
        linewidth=2,
        markersize=8,
        label="FedProx",
    )
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("L2 Distance to Benign Mean")
    axes[0, 0].set_title("Model Drift (Lower = Better)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        fedavg_df["round"],
        fedavg_df["cos_to_benign_mean"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
        label="FedAvg",
    )
    axes[0, 1].plot(
        fedprox_df["round"],
        fedprox_df["cos_to_benign_mean"],
        "s-",
        color="red",
        linewidth=2,
        markersize=8,
        label="FedProx",
    )
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Cosine Similarity to Benign Mean")
    axes[0, 1].set_title("Model Alignment (Higher = Better)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    norm_series_avg = first_present(fedavg_df, ["update_norm_mean"])
    norm_series_prox = first_present(fedprox_df, ["update_norm_mean"])
    if norm_series_avg is not None and norm_series_prox is not None:
        axes[1, 0].plot(
            fedavg_df["round"],
            norm_series_avg,
            "o-",
            color="blue",
            linewidth=2,
            markersize=8,
            label="FedAvg",
        )
        axes[1, 0].plot(
            fedprox_df["round"],
            norm_series_prox,
            "s-",
            color="red",
            linewidth=2,
            markersize=8,
            label="FedProx",
        )
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Mean Update Norm")
        axes[1, 0].set_title("Update Magnitude")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].axis("off")

    agg_series_avg = first_present(fedavg_df, ["t_aggregate_ms"])
    agg_series_prox = first_present(fedprox_df, ["t_aggregate_ms"])
    if agg_series_avg is not None and agg_series_prox is not None:
        axes[1, 1].plot(
            fedavg_df["round"],
            agg_series_avg,
            "o-",
            color="blue",
            linewidth=2,
            markersize=8,
            label="FedAvg",
        )
        axes[1, 1].plot(
            fedprox_df["round"],
            agg_series_prox,
            "s-",
            color="red",
            linewidth=2,
            markersize=8,
            label="FedProx",
        )
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Aggregation Time (ms)")
        axes[1, 1].set_title("Computational Overhead")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot federated learning metrics")
    parser.add_argument("--run_dir", type=str, help="Directory containing metrics CSV files")
    parser.add_argument("--output_dir", type=str, help="Output directory for plots")
    parser.add_argument("--fedprox_comparison", action="store_true", help="Plot FedProx vs FedAvg comparison")
    parser.add_argument("--logdir", type=str, help="Log directory for comparison (used with --fedprox_comparison)")
    parser.add_argument("--title", type=str, help="Custom plot title")
    parser.add_argument("--palette", type=str, default="colorblind")
    parser.add_argument("--style", type=str, default="whitegrid")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])

    args = parser.parse_args()

    config = create_default_config(title=args.title or "Federated Learning Metrics")
    config["style"].palette = args.palette
    config["style"].theme = args.style
    config["style"].dpi = args.dpi

    if args.fedprox_comparison:
        comparison_dir = args.logdir or "./comparison_logs"
        output_dir = Path(args.output_dir) if args.output_dir else Path(comparison_dir)
        output_dir.mkdir(exist_ok=True)
        comparison_plot_path = output_dir / f"fedprox_vs_fedavg_comparison.{args.format}"
        plot_fedprox_comparison(comparison_dir, str(comparison_plot_path), config)
        print(f"FedProx vs FedAvg comparison plot saved to: {comparison_plot_path}")
        return

    if not args.run_dir:
        parser.error("--run_dir is required when not using --fedprox_comparison")

    run_path = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_path
    output_dir.mkdir(exist_ok=True)

    server_metrics_path = run_path / "metrics.csv"
    if server_metrics_path.exists():
        server_plot_path = output_dir / f"server_metrics_plot.{args.format}"
        plot_server_metrics(str(server_metrics_path), str(server_plot_path), config)
        print(f"Server metrics plot saved to: {server_plot_path}")

    client_metrics_paths = [str(p) for p in run_path.glob("client_*_metrics.csv")]
    if client_metrics_paths:
        client_plot_path = output_dir / f"client_metrics_plot.{args.format}"
        plot_client_metrics(client_metrics_paths, str(client_plot_path), config)
        print(f"Client metrics plot saved to: {client_plot_path}")


if __name__ == "__main__":
    main()
