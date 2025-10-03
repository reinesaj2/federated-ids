#!/usr/bin/env python3
"""
Simple visualization script for federated learning metrics.
Generates sample plots from the metrics CSV files.
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_server_metrics(metrics_path: str, output_path: str):
    """Plot server aggregation metrics."""
    df = pd.read_csv(metrics_path)

    if df.empty or "round" not in df.columns:
        print(f"Skipping server metrics plot for {metrics_path} (empty/missing round)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Federated Learning Server Metrics - D2 Demo", fontsize=16)

    plotted_time = plotted_robust = plotted_norm = plotted_dispersion = False

    # Plot 1: Aggregation timing
    time_series = _first_present(df, ["t_aggregate_ms", "aggregation_time_ms"])
    if time_series is not None and not time_series.isna().all():
        axes[0, 0].plot(
            df["round"], time_series, "o-", color="blue", label="Aggregation Time (ms)"
        )
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Time (ms)")
        axes[0, 0].set_title("Aggregation Time per Round")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        plotted_time = True
        try:
            agg_s = time_series / 1000.0
            mean_s = agg_s.mean()
            std_s = agg_s.std()
            fig.text(
                0.5, 0.02, f"Aggregation time: {mean_s:.3f}±{std_s:.3f} s", ha="center"
            )
        except Exception:
            pass

    # Plot 2: Robustness metrics
    if "l2_to_benign_mean" in df.columns and "cos_to_benign_mean" in df.columns:
        axes[0, 1].plot(
            df["round"],
            df["l2_to_benign_mean"],
            "o-",
            color="red",
            label="L2 to Benign Mean",
        )
        axes[0, 1].plot(
            df["round"],
            df["cos_to_benign_mean"],
            "o-",
            color="green",
            label="Cosine Similarity",
        )
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Metric Value")
        axes[0, 1].set_title("Robustness Metrics")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        plotted_robust = True

    # Plot 3: Update norms
    if (
        "update_norm_mean" in df.columns
        and not pd.to_numeric(df["update_norm_mean"], errors="coerce").isna().all()
    ):
        axes[1, 0].plot(
            df["round"],
            df["update_norm_mean"],
            "o-",
            color="purple",
            label="Mean Update Norm",
        )
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Norm Value")
        axes[1, 0].set_title("Update Norms")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        plotted_norm = True

    # Plot 4: Pairwise dispersion (if present)
    if "pairwise_cosine_mean" in df.columns and "l2_dispersion_mean" in df.columns:
        axes[1, 1].plot(
            df["round"],
            df["pairwise_cosine_mean"],
            "o-",
            color="orange",
            label="Pairwise Cosine Mean",
        )
        axes[1, 1].plot(
            df["round"],
            df["l2_dispersion_mean"],
            "s-",
            color="brown",
            label="L2 Dispersion Mean",
        )
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].set_title("Pairwise Dispersion")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        plotted_dispersion = True

    if not any([plotted_time, plotted_robust, plotted_norm, plotted_dispersion]):
        plt.close(fig)
        print(f"No server metrics columns available to plot for {metrics_path}")
        return

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _first_present(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    for name in columns:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def _client_label(path: Path, df: pd.DataFrame) -> str:
    column = df.get("client_id")
    if column is not None and not column.empty and pd.notna(column.iloc[0]):
        return str(column.iloc[0])
    match = re.search(r"client_(\w+)", path.stem)
    return match.group(1) if match else path.stem


def plot_client_metrics(client_metrics_paths: list, output_path: str):
    """Plot client training metrics with argmax vs binary@tau overlays if present."""
    if not client_metrics_paths:
        print("No client metrics found; skipping client plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Federated Learning Client Metrics - D2 Demo", fontsize=16)

    colors = ["blue", "red", "green", "purple", "orange"]
    plotted_loss = plotted_acc = plotted_norm = plotted_overlay = plotted_tau = (
        plotted_fpr
    ) = False

    for i, client_path in enumerate(client_metrics_paths):
        try:
            df = pd.read_csv(client_path)
        except Exception as exc:
            print(f"Failed to read {client_path}: {exc}")
            continue

        if df.empty or "round" not in df.columns:
            print(f"Skipping {client_path} (empty or missing round column)")
            continue

        rounds = pd.to_numeric(df["round"], errors="coerce")
        color = colors[i % len(colors)]
        client_id = _client_label(Path(client_path), df)

        loss_series = _first_present(df, ["loss_after", "local_loss"])
        if loss_series is not None and not loss_series.isna().all():
            axes[0, 0].plot(
                rounds, loss_series, "o-", color=color, label=f"Client {client_id}"
            )
            plotted_loss = True

        acc_series = _first_present(df, ["acc_after", "local_accuracy"])
        if acc_series is not None and not acc_series.isna().all():
            axes[0, 1].plot(
                rounds, acc_series, "o-", color=color, label=f"Client {client_id}"
            )
            if not plotted_acc:
                plotted_acc = True

        norm_series = _first_present(df, ["weight_norm_after", "weight_norm"])
        if norm_series is not None and not norm_series.isna().all():
            axes[1, 0].plot(
                rounds, norm_series, "o-", color=color, label=f"Client {client_id}"
            )
            plotted_norm = True

        if {"macro_f1_argmax", "f1_bin_tau"}.issubset(df.columns):
            series_a = pd.to_numeric(df["macro_f1_argmax"], errors="coerce")
            series_b = pd.to_numeric(df["f1_bin_tau"], errors="coerce")
            axes[1, 1].plot(
                rounds,
                series_a,
                "o-",
                color=color,
                alpha=0.7,
                label=f"Argmax F1 C{client_id}",
            )
            axes[1, 1].plot(
                rounds,
                series_b,
                "x--",
                color=color,
                alpha=0.7,
                label=f"Bin@τ F1 C{client_id}",
            )
            plotted_overlay = True
        elif acc_series is not None and not acc_series.isna().all():
            axes[1, 1].plot(
                rounds,
                acc_series,
                "o-",
                color=color,
                alpha=0.7,
                label=f"Acc C{client_id}",
            )
            plotted_overlay = True

        # Plot tau_bin values over rounds
        tau_series = _first_present(df, ["tau_bin", "threshold_tau"])
        if tau_series is not None and not tau_series.isna().all():
            axes[0, 2].plot(
                rounds, tau_series, "o-", color=color, label=f"Client {client_id}"
            )
            plotted_tau = True

        # Plot benign FPR at tau over rounds
        fpr_series = _first_present(df, ["benign_fpr_bin_tau", "fpr_after"])
        if fpr_series is not None and not fpr_series.isna().all():
            axes[1, 2].plot(
                rounds, fpr_series, "o-", color=color, label=f"Client {client_id}"
            )
            plotted_fpr = True

    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss After Fit")
    axes[0, 0].grid(True, alpha=0.3)
    if plotted_loss:
        axes[0, 0].legend()

    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy After Fit")
    axes[0, 1].grid(True, alpha=0.3)
    if plotted_acc:
        axes[0, 1].legend()

    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Weight Norm")
    axes[1, 0].set_title("Model Weight Norms")
    axes[1, 0].grid(True, alpha=0.3)
    if plotted_norm:
        axes[1, 0].legend()

    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Argmax vs Binary@τ (if available)")
    axes[1, 1].grid(True, alpha=0.3)
    if plotted_overlay:
        axes[1, 1].legend()

    axes[0, 2].set_xlabel("Round")
    axes[0, 2].set_ylabel("Threshold τ")
    axes[0, 2].set_title("Threshold τ Selection per Round")
    axes[0, 2].grid(True, alpha=0.3)
    if plotted_tau:
        axes[0, 2].legend()

    axes[1, 2].set_xlabel("Round")
    axes[1, 2].set_ylabel("Benign FPR")
    axes[1, 2].set_title("False Positive Rate at τ")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(
        y=0.10, color="r", linestyle="--", alpha=0.5, label="Target FPR=0.10"
    )
    if plotted_fpr:
        axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    fig.suptitle("FedProx vs FedAvg Comparison on Non-IID Data", fontsize=16)

    # Plot 1: Convergence comparison - L2 distance to benign mean
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

    # Plot 2: Cosine similarity comparison
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

    # Plot 3: Update norm stability
    axes[1, 0].plot(
        fedavg_df["round"],
        fedavg_df["update_norm_mean"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
        label="FedAvg",
    )
    axes[1, 0].plot(
        fedprox_df["round"],
        fedprox_df["update_norm_mean"],
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

    # Plot 4: Aggregation efficiency
    axes[1, 1].plot(
        fedavg_df["round"],
        fedavg_df["t_aggregate_ms"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
        label="FedAvg",
    )
    axes[1, 1].plot(
        fedprox_df["round"],
        fedprox_df["t_aggregate_ms"],
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot federated learning metrics")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory containing metrics CSV files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for plots"
    )
    parser.add_argument(
        "--fedprox_comparison",
        action="store_true",
        help="Plot FedProx vs FedAvg comparison",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Log directory for comparison (used with --fedprox_comparison)",
    )
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
        plot_client_metrics(
            [str(p) for p in client_metrics_paths], str(client_plot_path)
        )
        print(f"Client metrics plot saved to: {client_plot_path}")

    print(f"Visualization complete. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
