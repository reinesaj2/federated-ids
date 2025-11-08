#!/usr/bin/env python3
"""
Generate Publication-Ready Thesis Plots

Creates comprehensive visualizations for all 5 comparison dimensions:
1. Aggregation methods comparison
2. Data heterogeneity (IID vs Non-IID)
3. Attack resilience
4. Privacy-utility tradeoff
5. Personalization benefit

Includes statistical significance testing and confidence intervals.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# Add scripts directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from plot_metrics_utils import compute_confidence_interval  # noqa: E402
from privacy_accounting import compute_epsilon  # noqa: E402
from metric_validation import MetricValidator  # noqa: E402

logger = logging.getLogger(__name__)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, normalize: bool = False) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.

    Args:
        y_true: Ground truth labels (1D array)
        y_pred: Predicted labels (1D array)
        num_classes: Number of classes
        normalize: If True, normalize by row (true class counts)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        cm[i, j] = count of samples with true label i predicted as j
    """
    if len(y_true) == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    cm = sk_confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)).astype(np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    return cm


def render_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Render confusion matrix as heatmap and save to file.

    Args:
        cm: Confusion matrix (num_classes x num_classes)
        class_names: List of class names for axis labels
        output_path: Path to save PNG file
        normalize: If True, display percentages instead of counts
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if normalize and not np.allclose(cm.sum(axis=1), 1.0):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_normalized = cm / row_sums
    else:
        cm_normalized = cm

    vmin = 0.0
    vmax = 1.0 if normalize else cm.max()

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage" if normalize else "Count"},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_confusion_matrices(cms: List[np.ndarray]) -> np.ndarray:
    """
    Aggregate multiple confusion matrices by summing.

    Args:
        cms: List of confusion matrices (same shape)

    Returns:
        Aggregated confusion matrix (sum of all inputs)

    Raises:
        ValueError: If cms is empty or matrices have mismatched shapes
    """
    if len(cms) == 0:
        raise ValueError("Cannot aggregate empty list of confusion matrices")

    first_shape = cms[0].shape
    for i, cm in enumerate(cms[1:], start=1):
        if cm.shape != first_shape:
            raise ValueError(f"Confusion matrix shape mismatch: cms[0].shape={first_shape}, cms[{i}].shape={cm.shape}")
    return np.sum(cms, axis=0)


def _prepare_client_scatter_data(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Prepare per-client scatter data for FedProx mu sweep visualization.

    Extracts last available metrics for each (client, seed, mu) combination,
    adds jitter for x-axis visibility, and filters missing values.
    """
    if df.empty:
        return pd.DataFrame(columns=["fedprox_mu", "client_id", "seed", metric, "jitter"])

    required_cols = ["round", "client_id", "fedprox_mu", "seed", metric]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns for scatter data: {missing_cols}")
        return pd.DataFrame(columns=["fedprox_mu", "client_id", "seed", metric, "jitter"])

    group_cols = ["client_id", "seed", "fedprox_mu"]
    if "alpha" in df.columns:
        group_cols.append("alpha")

    final_rounds = df.groupby(group_cols).tail(1).copy()
    final_rounds = final_rounds.dropna(subset=[metric])

    if final_rounds.empty:
        return pd.DataFrame(columns=["fedprox_mu", "client_id", "seed", metric, "jitter"])

    np.random.seed(42)
    jitter_amplitude = 0.02
    final_rounds["jitter"] = final_rounds["fedprox_mu"].apply(
        lambda mu: np.random.uniform(-jitter_amplitude * mu, jitter_amplitude * mu)
    )

    return final_rounds


def _compute_global_mean_by_mu(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute global mean and 95% CI aggregated across clients and seeds for each mu.
    """
    if df.empty:
        return pd.DataFrame(columns=["mu", "mean", "ci_lower", "ci_upper", "n"])

    if metric not in df.columns:
        logger.warning(f"Metric {metric} not in DataFrame")
        return pd.DataFrame(columns=["mu", "mean", "ci_lower", "ci_upper", "n"])

    stats_rows = []
    for mu in sorted(df["fedprox_mu"].unique()):
        mu_data = df[df["fedprox_mu"] == mu][metric].dropna().values
        if len(mu_data) == 0:
            continue

        if len(mu_data) >= 2:
            mean, ci_lower, ci_upper = compute_confidence_interval(mu_data)
        else:
            mean = ci_lower = ci_upper = float(mu_data[0])

        stats_rows.append({"mu": mu, "mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper, "n": len(mu_data)})

    return pd.DataFrame(stats_rows)


def _render_client_scatter_mu_plot(scatter_df: pd.DataFrame, metric: str, output_path: Path, use_log_y: bool = False) -> None:
    """
    Render per-client scatter plot with global mean overlay for FedProx mu sweep.
    """
    if scatter_df.empty:
        raise ValueError("Cannot render scatter plot: empty scatter data")

    fig, ax = plt.subplots(figsize=(10, 6))

    for (client_id, seed), group in scatter_df.groupby(["client_id", "seed"]):
        x_positions = group["fedprox_mu"] + group["jitter"]
        ax.scatter(x_positions, group[metric], alpha=0.3, s=20, color="gray", zorder=1)

    global_stats = _compute_global_mean_by_mu(scatter_df, metric)

    if not global_stats.empty:
        ax.plot(
            global_stats["mu"],
            global_stats["mean"],
            marker="o",
            linewidth=2,
            color="red",
            label="Global Mean",
            zorder=3,
        )
        ax.fill_between(
            global_stats["mu"],
            global_stats["ci_lower"],
            global_stats["ci_upper"],
            alpha=0.2,
            color="red",
            zorder=2,
        )

    if use_log_y:
        ax.set_yscale("log")

    metric_label = metric.replace("_", " ").title()
    ax.set_xlabel("FedProx Proximal Term (mu)")
    ax.set_ylabel(f'{metric_label} {"(log scale)" if use_log_y else ""}')
    ax.set_title(f"Per-Client {metric_label} vs mu (95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved FedProx scatter plot: {output_path}")


def _prepare_personalization_delta_f1_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare personalization delta F1 data for analysis.
    """
    required_cols = ["client_id", "seed", "alpha", "round", "macro_f1_global", "macro_f1_personalized"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns for personalization delta F1: {missing_cols}")
        return pd.DataFrame(
            columns=[
                "client_id",
                "seed",
                "alpha",
                "round",
                "f1_global",
                "f1_personalized",
                "delta_f1",
            ]
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "client_id",
                "seed",
                "alpha",
                "round",
                "f1_global",
                "f1_personalized",
                "delta_f1",
            ]
        )

    final_rounds = df.groupby(["client_id", "seed", "alpha"]).tail(1).copy()
    final_rounds = final_rounds.dropna(subset=["macro_f1_global", "macro_f1_personalized"])

    if final_rounds.empty:
        return pd.DataFrame(
            columns=[
                "client_id",
                "seed",
                "alpha",
                "round",
                "f1_global",
                "f1_personalized",
                "delta_f1",
            ]
        )

    final_rounds["f1_global"] = final_rounds["macro_f1_global"]
    final_rounds["f1_personalized"] = final_rounds["macro_f1_personalized"]
    final_rounds["delta_f1"] = final_rounds["f1_personalized"] - final_rounds["f1_global"]

    return final_rounds[["client_id", "seed", "alpha", "round", "f1_global", "f1_personalized", "delta_f1"]]


def _analyze_delta_f1_by_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze personalization delta F1 statistics stratified by alpha.
    """
    if df.empty or "alpha" not in df.columns or "delta_f1" not in df.columns:
        return pd.DataFrame(columns=["alpha", "mean_delta", "median_delta", "ci_lower", "ci_upper", "pct_positive", "n"])

    stats_rows = []
    for alpha in sorted(df["alpha"].unique()):
        alpha_data = df[df["alpha"] == alpha]["delta_f1"].dropna().values
        if len(alpha_data) == 0:
            continue

        mean_delta = float(np.mean(alpha_data))
        median_delta = float(np.median(alpha_data))
        pct_positive = float((alpha_data > 0).sum() / len(alpha_data) * 100)

        if len(alpha_data) >= 2:
            _, ci_lower, ci_upper = compute_confidence_interval(alpha_data)
        else:
            ci_lower = ci_upper = mean_delta

        stats_rows.append(
            {
                "alpha": alpha,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pct_positive": pct_positive,
                "n": len(alpha_data),
            }
        )

    return pd.DataFrame(stats_rows)


def _render_personalization_delta_f1_plots(scatter_df: pd.DataFrame, stats_df: pd.DataFrame, output_path: Path) -> None:
    """
    Render 3-panel personalization delta F1 analysis figure.
    """
    if scatter_df.empty:
        raise ValueError("Cannot render personalization delta F1 plots: empty data")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Personalization Benefit Analysis: Delta F1 by Heterogeneity", fontsize=14, fontweight="bold")

    ax1, ax2, ax3 = axes

    sns.violinplot(data=scatter_df, x="alpha", y="delta_f1", ax=ax1)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=2, label="No benefit (y=0)")
    ax1.set_xlabel("Alpha (Dirichlet Parameter)")
    ax1.set_ylabel("Delta F1 (Personalized - Global)")
    ax1.set_title("Delta F1 Distribution by Heterogeneity")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.scatter(
        scatter_df["f1_global"],
        scatter_df["f1_personalized"],
        alpha=0.5,
        s=30,
        c=scatter_df["alpha"].astype("category").cat.codes,
        cmap="viridis",
    )
    lims = [
        min(scatter_df["f1_global"].min(), scatter_df["f1_personalized"].min()) - 0.02,
        max(scatter_df["f1_global"].max(), scatter_df["f1_personalized"].max()) + 0.02,
    ]
    ax2.plot(lims, lims, "r--", alpha=0.5, linewidth=2, label="y=x (no benefit)")
    ax2.set_xlabel("F1 Global")
    ax2.set_ylabel("F1 Personalized")
    ax2.set_title("Global vs Personalized Performance")
    ax2.legend()

    bars = ax3.bar(stats_df["alpha"], stats_df["pct_positive"], color="steelblue", alpha=0.7)
    ax3.set_xlabel("Alpha")
    ax3.set_ylabel("% Clients with ΔF1 > 0")
    ax3.set_title("Personalization Success Rate")
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, pct in zip(bars, stats_df["pct_positive"]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_run_dir(reference: str, runs_root: Path) -> Optional[Path]:
    if not reference:
        return None

    candidates = []
    ref_path = Path(reference)
    candidates.append(ref_path)
    if not ref_path.is_absolute():
        candidates.append(runs_root / ref_path)
    candidates.append(runs_root / ref_path.name)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _extract_macro_f1(row: pd.Series) -> Optional[float]:
    macro_columns = [
        "macro_f1_after",
        "macro_f1_global",
        "macro_f1_personalized",
        "macro_f1_argmax",
    ]

    for column in macro_columns:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return None


def _compute_epsilon_fallback(row: Dict, final_row: Optional[pd.Series]) -> Optional[float]:
    noise = None
    if final_row is not None and "dp_sigma" in final_row:
        noise = final_row.get("dp_sigma")
        if pd.isna(noise):
            noise = None

    if noise is None:
        noise = row.get("dp_noise_multiplier")

    try:
        noise = float(noise) if noise is not None else None
    except (TypeError, ValueError):
        noise = None

    if not noise or noise <= 0.0:
        return None

    delta = None
    if final_row is not None and "dp_delta" in final_row:
        delta = final_row.get("dp_delta")
        if pd.isna(delta):
            delta = None
    if delta is None:
        delta = row.get("dp_delta", 1e-5)

    try:
        delta = float(delta) if delta is not None else None
    except (TypeError, ValueError):
        delta = None

    if delta is None or delta <= 0.0:
        delta = 1e-5

    steps = row.get("round")
    if (steps is None or steps <= 0) and final_row is not None and "round" in final_row:
        steps = final_row.get("round")

    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = None

    if steps is None or steps <= 0:
        return None

    sample_rate = None
    if final_row is not None and "dp_sample_rate" in final_row:
        sample_rate = final_row.get("dp_sample_rate")
        if pd.isna(sample_rate):
            sample_rate = None
    if sample_rate is None:
        sample_rate = row.get("dp_sample_rate", 1.0)

    try:
        sample_rate = float(sample_rate)
    except (TypeError, ValueError):
        sample_rate = 1.0

    try:
        epsilon = compute_epsilon(
            noise_multiplier=noise,
            delta=delta,
            num_steps=steps,
            sample_rate=sample_rate,
        )
    except Exception:
        return None

    if not math.isfinite(epsilon):
        return None

    return float(epsilon)


def _prepare_privacy_curve_data(final_rounds: pd.DataFrame, runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "run_dir" not in final_rounds.columns:
        return pd.DataFrame(), pd.DataFrame()

    dp_records: List[Dict] = []
    baseline_records: List[Dict] = []

    for row in final_rounds.to_dict(orient="records"):
        run_path = _resolve_run_dir(str(row.get("run_dir", "")), runs_root)
        if run_path is None:
            continue

        client_files = sorted(run_path.glob("client_*_metrics.csv"))
        if not client_files:
            continue

        macro_values: List[float] = []
        epsilon_candidates: List[float] = []

        for client_file in client_files:
            try:
                client_df = pd.read_csv(client_file)
            except Exception:
                continue

            if client_df.empty:
                continue

            final_row = client_df.iloc[-1]
            macro = _extract_macro_f1(final_row)
            if macro is None:
                continue
            macro_values.append(macro)

            if row.get("dp_enabled"):
                epsilon_val = final_row.get("dp_epsilon")
                if pd.isna(epsilon_val):
                    epsilon_val = None
                if epsilon_val is None:
                    epsilon_val = _compute_epsilon_fallback(row, final_row)
                if epsilon_val is not None:
                    epsilon_candidates.append(float(epsilon_val))

        if not macro_values:
            continue

        macro_mean = float(np.mean(macro_values))
        base_record = {
            "macro_f1": macro_mean,
            "seed": row.get("seed"),
            "dp_noise_multiplier": row.get("dp_noise_multiplier"),
        }

        if row.get("dp_enabled"):
            epsilon = float(np.mean(epsilon_candidates)) if epsilon_candidates else _compute_epsilon_fallback(row, None)
            if epsilon is None:
                continue
            dp_records.append({**base_record, "epsilon": epsilon})
        else:
            baseline_records.append(base_record)

    return pd.DataFrame(dp_records), pd.DataFrame(baseline_records)


def _render_privacy_curve(dp_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path) -> None:
    if dp_df.empty:
        return

    summary_rows: List[Dict] = []

    for epsilon, subset in dp_df.groupby("epsilon"):
        macros = subset["macro_f1"].dropna()
        if macros.empty:
            continue

        n = len(macros)
        mean = float(macros.mean())
        ci_lower = mean
        ci_upper = mean
        if n >= 2:
            se = stats.sem(macros)
            margin = se * stats.t.ppf(0.975, n - 1)
            ci_lower = mean - margin
            ci_upper = mean + margin

        summary_rows.append(
            {
                "epsilon": float(epsilon),
                "macro_f1_mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n,
                "dp_noise_multiplier": subset["dp_noise_multiplier"].dropna().mean(),
                "is_baseline": 0,
            }
        )

    baseline_row: Optional[Dict] = None
    if not baseline_df.empty:
        baseline_macros = baseline_df["macro_f1"].dropna()
        if not baseline_macros.empty:
            n = len(baseline_macros)
            mean = float(baseline_macros.mean())
            ci_lower = mean
            ci_upper = mean
            if n >= 2:
                se = stats.sem(baseline_macros)
                margin = se * stats.t.ppf(0.975, n - 1)
                ci_lower = mean - margin
                ci_upper = mean + margin
            baseline_row = {
                "epsilon": float("nan"),
                "macro_f1_mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n,
                "dp_noise_multiplier": float("nan"),
                "is_baseline": 1,
            }
            summary_rows.append(baseline_row)

    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["is_baseline", "epsilon"], na_position="last")
    summary_path = output_dir / "privacy_utility_curve.csv"
    summary_df.to_csv(summary_path, index=False)

    dp_summary = summary_df[summary_df["is_baseline"] == 0]
    if dp_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    lower_errors = np.maximum(0.0, dp_summary["macro_f1_mean"] - dp_summary["ci_lower"])
    upper_errors = np.maximum(0.0, dp_summary["ci_upper"] - dp_summary["macro_f1_mean"])
    yerr = np.vstack([lower_errors.to_numpy(), upper_errors.to_numpy()])

    ax.errorbar(
        dp_summary["epsilon"],
        dp_summary["macro_f1_mean"],
        yerr=yerr,
        fmt="o-",
        capsize=5,
        label="DP Enabled",
    )

    if baseline_row is not None and baseline_row["n"] > 0:
        ax.axhline(
            baseline_row["macro_f1_mean"],
            color="gray",
            linestyle="--",
            label=f"No DP baseline (n={baseline_row['n']})",
        )

    ax.set_xlabel("ε (DP accountant)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Privacy–Utility Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "privacy_utility_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_server_macro_f1_from_clients(run_dir: Path, round_num: int) -> Optional[float]:
    """Compute server-level macro-F1 by averaging client macro-F1 scores for a given round."""
    client_f1_scores = []

    for client_csv in run_dir.glob("client_*_metrics.csv"):
        try:
            client_df = pd.read_csv(client_csv)
            # Filter for the specific round
            round_data = client_df[client_df["round"] == round_num]

            if not round_data.empty:
                # Try macro_f1_after first, fall back to macro_f1_argmax
                f1_value = None
                if "macro_f1_after" in round_data.columns:
                    f1_value = round_data["macro_f1_after"].iloc[0]
                elif "macro_f1_argmax" in round_data.columns:
                    f1_value = round_data["macro_f1_argmax"].iloc[0]

                if f1_value is not None and not pd.isna(f1_value):
                    client_f1_scores.append(float(f1_value))
        except Exception:
            continue

    if not client_f1_scores:
        return None

    return float(np.mean(client_f1_scores))


def load_experiment_results(runs_dir: Path) -> pd.DataFrame:
    """Load all experiment results for a given dimension."""
    all_data = []

    # Try both comp_* and d2_* patterns
    patterns = ["comp_*", "d2_*"]

    for pattern in patterns:
        for run_dir in runs_dir.glob(pattern):
            # Load config if available
            config = {}
            config_file = run_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

            # Load server metrics
            metrics_file = run_dir / "metrics.csv"
            if not metrics_file.exists():
                continue

            df = pd.read_csv(metrics_file)

            # Compute macro-F1 from client metrics for each round
            macro_f1_values = []
            for idx, row in df.iterrows():
                round_num = row.get("round", idx)
                macro_f1 = compute_server_macro_f1_from_clients(run_dir, round_num)
                macro_f1_values.append(macro_f1)

            df["macro_f1"] = macro_f1_values

            # Add config columns
            for key, value in config.items():
                df[key] = value

            # Extract aggregation method and seed from config
            if "aggregation" not in df.columns:
                df["aggregation"] = config.get("aggregation", "fedavg")
            if "seed" not in df.columns:
                df["seed"] = config.get("seed", 42)

            df["run_dir"] = str(run_dir)
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # Validate metrics before returning
    validator = MetricValidator()
    warnings = validator.validate_plot_metrics(combined_df, "experiment_data")

    if warnings:
        logger.warning(f"Metric validation warnings: {len(warnings)} issues found")
        for warning in warnings[:5]:  # Show first 5 warnings
            logger.warning(f"  {warning}")
        if len(warnings) > 5:
            logger.warning(f"  ... and {len(warnings) - 5} more warnings")

    return combined_df


def _check_value_precision_issues(values: np.ndarray) -> Dict:
    """Check if values exhibit precision artifacts or ceiling effects.

    Returns dict with:
    - is_identical: True if all values within floating-point tolerance
    - is_near_perfect: True if all values > 0.999 (ceiling effect)
    - precision_variance: Maximum difference between values
    - max_value: Maximum value
    - min_value: Minimum value
    """
    if len(values) == 0:
        return {
            "is_identical": True,
            "is_near_perfect": False,
            "precision_variance": 0.0,
            "max_value": None,
            "min_value": None,
        }

    # Floating-point tolerance for "identical" check
    fp_tolerance = 1e-10

    # Precision artifact threshold (values within 0.0001 of each other)
    precision_threshold = 0.0001

    # Ceiling effect threshold (all values very close to maximum possible)
    ceiling_threshold = 0.999

    max_val = float(np.max(values))
    min_val = float(np.min(values))
    variance = max_val - min_val

    return {
        "is_identical": variance <= fp_tolerance,
        "is_near_perfect": min_val > ceiling_threshold,
        "precision_variance": variance,
        "max_value": max_val,
        "min_value": min_val,
        "precision_artifact": variance <= precision_threshold and max_val > 0.99,
    }


def perform_statistical_tests(df: pd.DataFrame, group_col: str, metric_col: str) -> Dict:
    """Perform statistical significance tests between groups.

    Issue #77: Handles precision artifacts and ceiling effects where values
    appear identical but have tiny differences that trigger false-positive ANOVA.
    """
    groups = df[group_col].unique()

    # Perform ANOVA if more than 2 groups
    group_data = [df[df[group_col] == g][metric_col].dropna() for g in groups]
    group_data = [g for g in group_data if len(g) > 0]

    if len(group_data) < 2:
        return {"test": "insufficient_data", "p_value": None}

    # Check for precision issues across all groups
    all_values = np.concatenate(group_data) if group_data else np.array([])
    precision_check = _check_value_precision_issues(all_values)

    # If values are truly identical (within FP tolerance), skip statistical test
    if precision_check["is_identical"]:
        return {
            "test": "skipped_identical",
            "p_value": None,
            "reason": "All values identical within floating-point tolerance",
            "precision_info": precision_check,
        }

    if len(group_data) == 2:
        # t-test for 2 groups
        stat, p_value = stats.ttest_ind(group_data[0], group_data[1])

        bonferroni_corrected_p = min(1.0, p_value) if p_value is not None else None

        return {
            "test": "t_test",
            "statistic": float(stat),
            "p_value": float(p_value),
            "bonferroni_corrected_p": bonferroni_corrected_p,
            "precision_info": precision_check,
        }
    else:
        # ANOVA for >2 groups
        stat, p_value = stats.f_oneway(*group_data)

        # Number of pairwise comparisons: n choose 2 where n = number of groups
        num_groups = len(group_data)
        num_pairwise = (num_groups * (num_groups - 1)) // 2
        bonferroni_corrected_p = min(1.0, p_value * num_pairwise) if p_value is not None else None

        result = {
            "test": "anova",
            "statistic": float(stat),
            "p_value": float(p_value),
            "bonferroni_corrected_p": bonferroni_corrected_p,
            "num_comparisons": num_pairwise,
            "precision_info": precision_check,
        }

        # Post-hoc pairwise comparisons
        pairwise = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                data1 = df[df[group_col] == g1][metric_col].dropna()
                data2 = df[df[group_col] == g2][metric_col].dropna()
                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    pairwise[f"{g1}_vs_{g2}"] = float(p_val)

        result["pairwise"] = pairwise
        return result


def _render_statistical_annotation(
    ax,
    stats_result: Dict,
    precision_check: Dict,
    all_f1_values: np.ndarray,
    total_valid_points: int,
) -> None:
    """Render statistical annotation on plot based on ANOVA results and precision checks.

    Handles three scenarios:
    1. Identical values: Show "all methods identical" (no ANOVA)
    2. Precision artifacts: Show F1 with 6 decimal places + ceiling effect note
    3. Normal differences: Show ANOVA with Bonferroni-corrected p-values
    """
    if stats_result.get("test") == "skipped_identical":
        logger.info(
            f"ANOVA skipped: All macro-F1 values identical (variance={precision_check['precision_variance']:.2e}). "
            "No statistical comparison needed."
        )
        if precision_check["is_near_perfect"]:
            mean_f1 = float(np.mean(all_f1_values))
            ax.text(
                0.02,
                0.02,
                f"F1={mean_f1:.6f} (all methods identical)",
                transform=ax.transAxes,
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )
        else:
            ax.text(
                0.02,
                0.02,
                f"n={total_valid_points} (all methods identical)",
                transform=ax.transAxes,
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )
        return

    p_value = stats_result.get("p_value")
    if p_value is None or np.isnan(p_value):
        logger.warning(
            f"ANOVA computation failed or returned NaN for macro-F1 data ({total_valid_points} points). "
            "This may occur with small sample sizes or identical values across groups."
        )
        ax.text(
            0.02,
            0.02,
            f"n={total_valid_points} (ANOVA inconclusive)",
            transform=ax.transAxes,
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )
        return

    precision_info = stats_result.get("precision_info", {})
    bonferroni_p = stats_result.get("bonferroni_corrected_p")

    if precision_info.get("precision_artifact", False) or precision_info.get("is_near_perfect", False):
        mean_f1 = float(np.mean(all_f1_values))
        variance = precision_info.get("precision_variance", 0.0)
        use_bonferroni = bonferroni_p is not None and p_value < 0.05 and bonferroni_p >= 0.05

        if use_bonferroni:
            annotation_text = f"F1={mean_f1:.6f} ± {variance:.2e}\n" f"(ceiling effect: sub-0.01% differences)"
            bg_color = "lightyellow"
        else:
            annotation_text = f"ANOVA p={p_value:.4f}\n" f"F1={mean_f1:.6f} ± {variance:.2e}"
            bg_color = "wheat"

        ax.text(
            0.02,
            0.02,
            annotation_text,
            transform=ax.transAxes,
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor=bg_color, alpha=0.5),
        )
        logger.info(
            f"Precision artifact detected: F1 values differ by {variance:.2e} "
            f"but all > {precision_info.get('min_value', 0):.6f}. "
            f"Showing 6-decimal precision and noting ceiling effect."
        )
    else:
        if bonferroni_p is not None and p_value < 0.05:
            if bonferroni_p >= 0.05:
                annotation_text = f"ANOVA p={p_value:.4f} (ns after Bonferroni correction)"
                bg_color = "lightyellow"
            else:
                annotation_text = f"ANOVA p={p_value:.4f} (Bonferroni-corrected: {bonferroni_p:.4f})"
                bg_color = "wheat"
        else:
            annotation_text = f"ANOVA p={p_value:.4f}"
            bg_color = "wheat"

        ax.text(
            0.02,
            0.02,
            annotation_text,
            transform=ax.transAxes,
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor=bg_color, alpha=0.5),
        )


def _render_macro_f1_plot(ax, final_rounds: pd.DataFrame, available_methods: list) -> bool:
    """Render macro-F1 comparison plot with 95% CIs. Returns True if rendered."""
    if "macro_f1" not in final_rounds.columns:
        return False

    macro_f1_data = final_rounds[final_rounds["macro_f1"].notna()]
    if macro_f1_data.empty:
        return False

    summary_data = []
    for method in available_methods:
        method_data = macro_f1_data[macro_f1_data["aggregation"] == method]["macro_f1"].values
        if len(method_data) > 0:
            if len(method_data) >= 2:
                mean, lower, upper = compute_confidence_interval(method_data)
                ci_range = upper - lower
            else:
                mean = float(np.mean(method_data))
                ci_range = 0.0
            summary_data.append({"aggregation": method, "mean": mean, "ci": ci_range, "n": len(method_data)})

    if not summary_data:
        return False

    summary_df = pd.DataFrame(summary_data)
    x_pos = np.arange(len(summary_df))
    ax.bar(
        x_pos,
        summary_df["mean"],
        yerr=summary_df["ci"] / 2,
        capsize=5,
        alpha=0.7,
        color=sns.color_palette("colorblind", len(summary_df)),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in summary_df["aggregation"]])
    ax.set_title("Detection Performance (Macro-F1, 95% CI)")
    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis="y")

    for i, row in summary_df.iterrows():
        ax.text(i, row["mean"] + row["ci"] / 2 + 0.02, f"n={row['n']}", ha="center", va="bottom", fontsize=8)

    # Issue #77 fix: Handle precision artifacts and ceiling effects
    min_data_for_anova = 3
    total_valid_points = len(macro_f1_data)

    # Extract all F1 values for precision checking
    all_f1_values = macro_f1_data["macro_f1"].values
    precision_check = _check_value_precision_issues(all_f1_values)

    if total_valid_points >= min_data_for_anova:
        stats_result = perform_statistical_tests(macro_f1_data, "aggregation", "macro_f1")
        _render_statistical_annotation(ax, stats_result, precision_check, all_f1_values, total_valid_points)
    else:
        logger.warning(
            f"Insufficient macro-F1 data ({total_valid_points}/{len(final_rounds)}) "
            f"for ANOVA (minimum {min_data_for_anova} required). "
            "Skipping statistical annotation. Consider enabling D2_EXTENDED_METRICS=1 "
            "or running experiments with extended client metrics logging."
        )
        ax.text(
            0.02,
            0.02,
            f"n={total_valid_points} (insufficient for ANOVA)",
            transform=ax.transAxes,
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )

    return True


def _render_timing_plot(ax, df: pd.DataFrame, available_methods: list) -> bool:
    """Render aggregation timing plot with 95% CIs. Returns True if rendered."""
    if "t_aggregate_ms" not in df.columns:
        return False

    timing_data = df.groupby(["aggregation", "seed"])["t_aggregate_ms"].mean().reset_index()

    summary_data = []
    for method in available_methods:
        method_data = timing_data[timing_data["aggregation"] == method]["t_aggregate_ms"].values
        if len(method_data) > 0:
            if len(method_data) >= 2:
                mean, lower, upper = compute_confidence_interval(method_data)
                ci_range = upper - lower
            else:
                mean = float(np.mean(method_data))
                ci_range = 0.0
            summary_data.append({"aggregation": method, "mean": mean, "ci": ci_range, "n": len(method_data)})

    if not summary_data:
        return False

    summary_df = pd.DataFrame(summary_data)
    x_pos = np.arange(len(summary_df))
    ax.bar(
        x_pos,
        summary_df["mean"],
        yerr=summary_df["ci"] / 2,
        capsize=5,
        alpha=0.7,
        color=sns.color_palette("colorblind", len(summary_df)),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in summary_df["aggregation"]])
    ax.set_title("Aggregation Time (95% CI)")
    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, alpha=0.3, axis="y")

    for i, row in summary_df.iterrows():
        y_pos = row["mean"] + row["ci"] / 2 + row["mean"] * 0.05
        ax.text(i, y_pos, f"n={row['n']}", ha="center", va="bottom", fontsize=8)

    return True


def _render_l2_plot(ax, final_rounds: pd.DataFrame, available_methods: list) -> bool:
    """Render L2 distance boxplot. Returns True if rendered."""
    if "l2_to_benign_mean" not in final_rounds.columns:
        return False

    l2_data = final_rounds[final_rounds["l2_to_benign_mean"].notna()]
    if l2_data.empty or len(available_methods) == 0:
        return False

    l2_data_filtered = l2_data[l2_data["aggregation"].isin(available_methods)]
    if l2_data_filtered.empty:
        return False

    sns.boxplot(data=l2_data_filtered, x="aggregation", y="l2_to_benign_mean", ax=ax, order=available_methods)
    ax.set_title("Model Drift (L2 Distance)")
    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("L2 Distance to Benign Mean")
    ax.set_xticklabels([m.upper() for m in available_methods])

    y_min = l2_data_filtered["l2_to_benign_mean"].min()
    if y_min > 0:
        ax.set_ylim(bottom=y_min * 0.5)

    return True


def _render_cosine_plot(ax, final_rounds: pd.DataFrame, available_methods: list) -> bool:
    """Render cosine similarity violinplot. Returns True if rendered."""
    if "cos_to_benign_mean" not in final_rounds.columns:
        return False

    cos_data = final_rounds[final_rounds["cos_to_benign_mean"].notna()]
    if cos_data.empty or len(available_methods) == 0:
        return False

    cos_data_filtered = cos_data[cos_data["aggregation"].isin(available_methods)]
    if cos_data_filtered.empty:
        return False

    sns.violinplot(data=cos_data_filtered, x="aggregation", y="cos_to_benign_mean", ax=ax, order=available_methods)
    ax.set_title("Model Alignment (Cosine Similarity)")
    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xticklabels([m.upper() for m in available_methods])

    return True


def plot_aggregation_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison of aggregation methods with macro-F1 as primary metric."""
    if "aggregation" not in df.columns:
        return

    method_order = ["fedavg", "krum", "bulyan", "median"]
    available_methods = [m for m in method_order if m in df["aggregation"].unique()]

    final_rounds = df.groupby(["aggregation", "seed"]).tail(1)

    if "l2_to_benign_mean" in final_rounds.columns:
        l2_data = final_rounds["l2_to_benign_mean"].dropna()
        if len(l2_data) > 0:
            zero_count = (l2_data == 0.0).sum()
            if zero_count > len(l2_data) * 0.5:
                print(f"WARNING: {zero_count}/{len(l2_data)} L2 values are exactly zero - possible data issue")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Aggregation Method Comparison", fontsize=16, fontweight="bold")

    _render_macro_f1_plot(axes[0, 0], final_rounds, available_methods)
    _render_timing_plot(axes[0, 1], df, available_methods)
    _render_l2_plot(axes[1, 0], final_rounds, available_methods)
    _render_cosine_plot(axes[1, 1], final_rounds, available_methods)

    plt.tight_layout()

    # Save both PNG and PDF
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "aggregation_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "aggregation_comparison.pdf", bbox_inches="tight")

    # Save summary statistics to CSV
    if "macro_f1" in final_rounds.columns:
        stats_data = []
        for method in available_methods:
            method_data = final_rounds[(final_rounds["aggregation"] == method) & (final_rounds["macro_f1"].notna())]
            if not method_data.empty:
                f1_values = method_data["macro_f1"].values
                stats_data.append(
                    {
                        "aggregation_method": method,
                        "macro_f1_mean": float(np.mean(f1_values)),
                        "macro_f1_std": float(np.std(f1_values)),
                        "n_seeds": len(f1_values),
                    }
                )

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv(output_dir / "aggregation_comparison_stats.csv", index=False)

    plt.close()


def plot_fedprox_heterogeneity_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot FedProx effectiveness across heterogeneity levels."""
    if "alpha" not in df.columns or "fedprox_mu" not in df.columns:
        print("Warning: Missing alpha or fedprox_mu columns for FedProx heterogeneity plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("FedProx Heterogeneity Matrix Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Final L2 Distance by Alpha and Mu
    ax1 = axes[0, 0]
    alpha_mu_data = df.groupby(['alpha', 'fedprox_mu'])['l2_to_benign_mean'].agg(['mean', 'std', 'count']).reset_index()

    for alpha in sorted(df['alpha'].unique()):
        alpha_data = alpha_mu_data[alpha_mu_data['alpha'] == alpha]
        mu_values = alpha_data['fedprox_mu'].values
        means = alpha_data['mean'].values
        stds = alpha_data['std'].values

        ax1.errorbar(mu_values, means, yerr=stds, marker='o', label=f'Alpha={alpha}', linewidth=2, markersize=8)

    ax1.set_xlabel('FedProx Mu Value')
    ax1.set_ylabel('Final L2 Distance to Benign Model')
    ax1.set_title('L2 Distance vs FedProx Strength by Heterogeneity Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Final Cosine Similarity by Alpha and Mu
    ax2 = axes[0, 1]
    alpha_mu_cos_data = df.groupby(['alpha', 'fedprox_mu'])['cos_to_benign_mean'].agg(['mean', 'std', 'count']).reset_index()

    for alpha in sorted(df['alpha'].unique()):
        alpha_data = alpha_mu_cos_data[alpha_mu_cos_data['alpha'] == alpha]
        mu_values = alpha_data['fedprox_mu'].values
        means = alpha_data['mean'].values
        stds = alpha_data['std'].values

        ax2.errorbar(mu_values, means, yerr=stds, marker='s', label=f'Alpha={alpha}', linewidth=2, markersize=8)

    ax2.set_xlabel('FedProx Mu Value')
    ax2.set_ylabel('Final Cosine Similarity to Benign Model')
    ax2.set_title('Cosine Similarity vs FedProx Strength by Heterogeneity Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Plot 3: Convergence Curves for Different Mu Values (Alpha=0.1)
    ax3 = axes[1, 0]
    extreme_non_iid = df[df['alpha'] == 0.1]

    for mu in sorted(extreme_non_iid['fedprox_mu'].unique()):
        mu_data = extreme_non_iid[extreme_non_iid['fedprox_mu'] == mu]
        if 'round' in mu_data.columns and 'l2_to_benign_mean' in mu_data.columns:
            round_means = mu_data.groupby('round')['l2_to_benign_mean'].mean()
            ax3.plot(round_means.index, round_means.values, marker='o', label=f'Mu={mu}', linewidth=2)

    ax3.set_xlabel('Round')
    ax3.set_ylabel('L2 Distance to Benign Model')
    ax3.set_title('Convergence Curves: Extreme Non-IID (Alpha=0.1)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Heatmap of Final Performance
    ax4 = axes[1, 1]
    pivot_data = df.groupby(['alpha', 'fedprox_mu'])['l2_to_benign_mean'].mean().unstack()

    im = ax4.imshow(pivot_data.values, cmap='viridis', aspect='auto')
    ax4.set_xticks(range(len(pivot_data.columns)))
    ax4.set_xticklabels([f'{mu:.2f}' for mu in pivot_data.columns])
    ax4.set_yticks(range(len(pivot_data.index)))
    ax4.set_yticklabels([f'{alpha:.1f}' for alpha in pivot_data.index])
    ax4.set_xlabel('FedProx Mu Value')
    ax4.set_ylabel('Alpha (Heterogeneity Level)')
    ax4.set_title('L2 Distance Heatmap: Alpha vs Mu')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Final L2 Distance')

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "fedprox_heterogeneity_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved FedProx heterogeneity plot: {output_file}")
    plt.close()


def plot_heterogeneity_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot IID vs Non-IID performance with 95% CIs."""
    if "alpha" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Data Heterogeneity Impact (IID vs Non-IID)", fontsize=16, fontweight="bold")

    # Plot 1: Convergence over rounds with 95% CIs
    if "l2_to_benign_mean" in df.columns and "seed" in df.columns:
        ax = axes[0]
        colors = sns.color_palette("colorblind", len(df["alpha"].unique()))

        for idx, alpha in enumerate(sorted(df["alpha"].unique())):
            alpha_data = df[df["alpha"] == alpha]
            all_rounds = sorted(alpha_data["round"].unique())

            rounds_with_data = []
            means_list = []
            ci_lower_list = []
            ci_upper_list = []

            for round_num in all_rounds:
                round_data = alpha_data[alpha_data["round"] == round_num]["l2_to_benign_mean"].dropna().values
                if len(round_data) == 0:
                    continue  # Skip rounds with no data

                rounds_with_data.append(round_num)

                if len(round_data) >= 2:
                    mean, ci_lower, ci_upper = compute_confidence_interval(round_data)
                    means_list.append(mean)
                    ci_lower_list.append(ci_lower)
                    ci_upper_list.append(ci_upper)
                elif len(round_data) == 1:
                    val = float(round_data[0])
                    means_list.append(val)
                    ci_lower_list.append(val)
                    ci_upper_list.append(val)

            if means_list:
                label = "IID" if alpha >= 1.0 else f"Non-IID (α={alpha})"
                ax.plot(rounds_with_data, means_list, marker="o", label=label, color=colors[idx], linewidth=2)
                ax.fill_between(rounds_with_data, ci_lower_list, ci_upper_list, alpha=0.2, color=colors[idx])

        ax.set_title("Convergence: L2 Distance Over Rounds (95% CI)")
        ax.set_xlabel("Round")
        ax.set_ylabel("L2 Distance to Benign Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Final performance by alpha with 95% CIs
    final_rounds = df.groupby(["alpha", "seed"]).tail(1)
    if "cos_to_benign_mean" in final_rounds.columns:
        ax = axes[1]

        alpha_values = sorted(final_rounds["alpha"].unique())
        summary_stats = []

        for alpha in alpha_values:
            alpha_data = final_rounds[final_rounds["alpha"] == alpha]["cos_to_benign_mean"].dropna().values
            if len(alpha_data) >= 2:
                mean, ci_lower, ci_upper = compute_confidence_interval(alpha_data)
                summary_stats.append(
                    {
                        "alpha": alpha,
                        "mean": mean,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "n": len(alpha_data),
                    }
                )
            elif len(alpha_data) == 1:
                val = float(alpha_data[0])
                summary_stats.append({"alpha": alpha, "mean": val, "ci_lower": val, "ci_upper": val, "n": 1})

        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            x_pos = np.arange(len(summary_df))
            yerr_lower = summary_df["mean"] - summary_df["ci_lower"]
            yerr_upper = summary_df["ci_upper"] - summary_df["mean"]

            ax.bar(
                x_pos,
                summary_df["mean"],
                yerr=[yerr_lower, yerr_upper],
                capsize=5,
                alpha=0.7,
                color=sns.color_palette("colorblind", len(summary_df)),
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{a:.2f}" for a in summary_df["alpha"]])
            ax.set_title("Final Cosine Similarity by α (95% CI)")
            ax.set_xlabel("Alpha (Dirichlet Parameter)")
            ax.set_ylabel("Cosine Similarity")

            for i, row in summary_df.iterrows():
                ax.text(
                    i,
                    row["mean"] + yerr_upper.iloc[i] + 0.02,
                    f"n={row['n']}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "heterogeneity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_attack_resilience(df: pd.DataFrame, output_dir: Path):
    """Plot attack resilience with macro-F1 as primary metric and bounded degradation."""
    if "adversary_fraction" not in df.columns:
        return

    # Extract threat model metadata from first config
    first_config = {}
    if "run_dir" in df.columns:
        for run_dir_str in df["run_dir"].dropna().unique():
            config_path = Path(run_dir_str) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    first_config = json.load(f)
                break

    dataset = first_config.get("dataset", "unknown").upper()
    num_clients = first_config.get("num_clients", 0)
    alpha = first_config.get("alpha", 0.5)
    num_seeds = len(df["seed"].unique()) if "seed" in df.columns else 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    subtitle = f"Dataset: {dataset} | Clients: {num_clients} | α={alpha} (Dirichlet) | " f"Attack: grad_ascent | Seeds: n={num_seeds}"
    fig.suptitle(f"Attack Resilience Comparison\n{subtitle}", fontsize=14, fontweight="bold")

    final_rounds = df.groupby(["aggregation", "adversary_fraction", "seed"]).tail(1)
    method_order = ["fedavg", "krum", "bulyan", "median"]
    available_methods = [m for m in method_order if m in final_rounds["aggregation"].unique()]

    # Plot 1: Macro-F1 vs Adversary Percentage with 95% CIs
    if "macro_f1" in final_rounds.columns:
        ax = axes[0, 0]
        colors = sns.color_palette("colorblind", len(available_methods))

        for idx, agg in enumerate(available_methods):
            agg_data = final_rounds[final_rounds["aggregation"] == agg]
            summary_stats = []

            for adv_frac in sorted(agg_data["adversary_fraction"].unique()):
                frac_data = agg_data[agg_data["adversary_fraction"] == adv_frac]["macro_f1"].dropna()
                if len(frac_data) >= 2:
                    mean, lower, upper = compute_confidence_interval(frac_data.values)
                    summary_stats.append(
                        {
                            "adversary_fraction": adv_frac * 100,
                            "mean": mean,
                            "ci_lower": lower,
                            "ci_upper": upper,
                        }
                    )
                elif len(frac_data) == 1:
                    val = float(frac_data.iloc[0])
                    summary_stats.append(
                        {
                            "adversary_fraction": adv_frac * 100,
                            "mean": val,
                            "ci_lower": val,
                            "ci_upper": val,
                        }
                    )

            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                ax.plot(
                    summary_df["adversary_fraction"],
                    summary_df["mean"],
                    marker="o",
                    label=agg.upper(),
                    color=colors[idx],
                    linewidth=2,
                )
                ax.fill_between(
                    summary_df["adversary_fraction"],
                    summary_df["ci_lower"],
                    summary_df["ci_upper"],
                    alpha=0.2,
                    color=colors[idx],
                )

        ax.set_title("Detection Performance vs Adversary Percentage (Macro-F1, 95% CI)")
        ax.set_xlabel("Adversary Percentage (%)")
        ax.set_ylabel("Macro-F1 Score")
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Degradation bars with 95% CIs (bounded [0, 100])
    if "macro_f1" in final_rounds.columns:
        ax = axes[1, 0]
        benign = final_rounds[final_rounds["adversary_fraction"] == 0.0]
        adversarial = final_rounds[final_rounds["adversary_fraction"] > 0.0]

        degradation_stats = []
        for agg in available_methods:
            benign_f1 = benign[benign["aggregation"] == agg]["macro_f1"].dropna()
            adv_f1 = adversarial[adversarial["aggregation"] == agg]["macro_f1"].dropna()

            if len(benign_f1) > 0 and len(adv_f1) > 0:
                benign_mean = float(benign_f1.mean())
                adv_mean = float(adv_f1.mean())

                degradation_pct = max(0.0, (benign_mean - adv_mean) / benign_mean * 100) if benign_mean > 0 else 0.0

                degradation_values = []
                for b_val in benign_f1:
                    for a_val in adv_f1:
                        deg = max(0.0, (b_val - a_val) / b_val * 100) if b_val > 0 else 0.0
                        degradation_values.append(deg)

                if len(degradation_values) >= 2:
                    _, ci_lower, ci_upper = compute_confidence_interval(np.array(degradation_values))
                    ci_lower = max(0.0, ci_lower)
                    ci_upper = min(100.0, ci_upper)
                else:
                    ci_lower = degradation_pct
                    ci_upper = degradation_pct

                degradation_stats.append(
                    {
                        "aggregation": agg,
                        "degradation_pct": degradation_pct,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )

        if degradation_stats:
            deg_df = pd.DataFrame(degradation_stats)
            x_pos = np.arange(len(deg_df))
            yerr_lower = np.maximum(0.0, deg_df["degradation_pct"] - deg_df["ci_lower"])
            yerr_upper = np.maximum(0.0, deg_df["ci_upper"] - deg_df["degradation_pct"])

            ax.bar(
                x_pos,
                deg_df["degradation_pct"],
                yerr=[yerr_lower, yerr_upper],
                capsize=5,
                alpha=0.7,
                color=colors[: len(deg_df)],
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([m.upper() for m in deg_df["aggregation"]])
            ax.set_title("Performance Degradation Under Attack (95% CI, Bounded [0,100])")  # noqa: E501
            ax.set_xlabel("Aggregation Method")
            ax.set_ylabel("Degradation (%)")
            ax.set_ylim([0, 110])
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Supplementary - L2 distance vs adversary fraction
    if "l2_to_benign_mean" in final_rounds.columns:
        ax = axes[0, 1]
        for idx, agg in enumerate(available_methods):
            agg_data = final_rounds[final_rounds["aggregation"] == agg]
            summary = agg_data.groupby("adversary_fraction")["l2_to_benign_mean"].agg(["mean", "std"])
            ax.errorbar(
                summary.index * 100,
                summary["mean"],
                yerr=summary["std"],
                marker="o",
                label=agg.upper(),
                capsize=5,
                color=colors[idx],
            )

        ax.set_title("Supplementary: Model Drift (L2 Distance)")
        ax.set_xlabel("Adversary Percentage (%)")
        ax.set_ylabel("L2 Distance to Benign Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Supplementary - Cosine similarity vs adversary fraction
    if "cos_to_benign_mean" in final_rounds.columns:
        ax = axes[1, 1]
        for idx, agg in enumerate(available_methods):
            agg_data = final_rounds[final_rounds["aggregation"] == agg]
            summary = agg_data.groupby("adversary_fraction")["cos_to_benign_mean"].agg(["mean", "std"])
            ax.errorbar(
                summary.index * 100,
                summary["mean"],
                yerr=summary["std"],
                marker="s",
                label=agg.upper(),
                capsize=5,
                color=colors[idx],
            )

        ax.set_title("Supplementary: Model Alignment (Cosine Similarity)")
        ax.set_xlabel("Adversary Percentage (%)")
        ax.set_ylabel("Cosine Similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save PNG and PDF
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "attack_resilience.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "attack_resilience.pdf", bbox_inches="tight")

    # Export CSV with macro-F1 stats and degradation
    if "macro_f1" in final_rounds.columns:
        csv_rows = []
        for agg in available_methods:
            for adv_frac in sorted(final_rounds["adversary_fraction"].unique()):
                mask = (final_rounds["aggregation"] == agg) & (final_rounds["adversary_fraction"] == adv_frac)
                frac_data = final_rounds[mask]["macro_f1"].dropna()

                if len(frac_data) >= 2:
                    mean, ci_lower, ci_upper = compute_confidence_interval(frac_data.values)
                elif len(frac_data) == 1:
                    mean = float(frac_data.iloc[0])
                    ci_lower = mean
                    ci_upper = mean
                else:
                    continue

                benign_f1 = benign[benign["aggregation"] == agg]["macro_f1"].dropna()
                benign_mean = float(benign_f1.mean()) if len(benign_f1) > 0 else 0.0

                degradation_pct = max(0.0, (benign_mean - mean) / benign_mean * 100) if benign_mean > 0 else 0.0

                csv_rows.append(
                    {
                        "aggregation": agg,
                        "adversary_fraction": adv_frac,
                        "macro_f1_mean": mean,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "n": len(frac_data),
                        "degradation_pct": degradation_pct,
                    }
                )

        if csv_rows:
            stats_df = pd.DataFrame(csv_rows)
            stats_df.to_csv(output_dir / "attack_resilience_stats.csv", index=False)

    plt.close()


def plot_privacy_utility(df: pd.DataFrame, output_dir: Path, runs_dir: Optional[Path] = None):
    """Plot privacy-utility tradeoff."""
    if "dp_enabled" not in df.columns:
        return

    final_rounds = df.groupby(["dp_enabled", "dp_noise_multiplier", "seed"]).tail(1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Privacy-Utility Tradeoff", fontsize=16, fontweight="bold")

    # Plot 1: L2 distance vs DP noise
    if "l2_to_benign_mean" in final_rounds.columns:
        ax = axes[0]
        dp_data = final_rounds[final_rounds["dp_enabled"]]
        if not dp_data.empty:
            summary = dp_data.groupby("dp_noise_multiplier")["l2_to_benign_mean"].agg(["mean", "std"])
            ax.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"],
                marker="o",
                capsize=5,
                label="DP Enabled",
            )

        # Add baseline without DP
        no_dp = final_rounds[~final_rounds["dp_enabled"]]["l2_to_benign_mean"].mean()
        ax.axhline(y=no_dp, color="green", linestyle="--", label="No DP (Baseline)")

        ax.set_title("Model Accuracy vs DP Noise")
        ax.set_xlabel("DP Noise Multiplier (σ)")
        ax.set_ylabel("L2 Distance to Benign Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Cosine similarity vs DP
    if "cos_to_benign_mean" in final_rounds.columns:
        ax = axes[1]
        comparison_data = []
        for enabled in [False, True]:
            subset = final_rounds[final_rounds["dp_enabled"] == enabled]
            if not subset.empty:
                comparison_data.append(
                    {
                        "DP": "Enabled" if enabled else "Disabled",
                        "Cosine Similarity": subset["cos_to_benign_mean"].values,
                    }
                )

        if comparison_data:
            rows = [{"DP": item["DP"], "Cosine Similarity": val} for item in comparison_data for val in item["Cosine Similarity"]]
            plot_df = pd.DataFrame(rows)
            sns.violinplot(data=plot_df, x="DP", y="Cosine Similarity", ax=ax)
            ax.set_title("Model Alignment with DP")

    plt.tight_layout()
    plt.savefig(output_dir / "privacy_utility.png", dpi=300, bbox_inches="tight")
    plt.close()

    runs_root = Path(runs_dir) if runs_dir is not None else Path("runs")
    dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_root)
    _render_privacy_curve(dp_df, baseline_df, output_dir)


def plot_personalization_benefit(df: pd.DataFrame, output_dir: Path):
    """Plot personalization benefit."""
    if "personalization_epochs" not in df.columns:
        return

    # Load client metrics for personalization data
    # This requires reading client metrics CSVs
    runs_dir = Path("runs")
    personalization_data = []

    for run_dir in runs_dir.glob("comp_*pers*"):
        config_file = run_dir / "config.json"
        if not config_file.exists():
            continue

        with open(config_file) as f:
            config = json.load(f)

        for client_metrics in run_dir.glob("client_*_metrics.csv"):
            client_df = pd.read_csv(client_metrics)
            if "macro_f1_global" in client_df.columns and "macro_f1_personalized" in client_df.columns:
                # Get last round
                if client_df.empty:
                    continue
                last_row = client_df.iloc[-1]
                personalization_data.append(
                    {
                        "personalization_epochs": config.get("personalization_epochs", 0),
                        "global_f1": last_row["macro_f1_global"],
                        "personalized_f1": last_row["macro_f1_personalized"],
                        "gain": last_row.get("personalization_gain", 0),
                    }
                )

    if not personalization_data:
        return

    pers_df = pd.DataFrame(personalization_data)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Personalization Benefit Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Global vs Personalized F1
    ax = axes[0]
    enabled = pers_df[pers_df["personalization_epochs"] > 0]
    if not enabled.empty:
        ax.scatter(enabled["global_f1"], enabled["personalized_f1"], alpha=0.6, s=100)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="No Improvement")
        ax.set_xlabel("Global Model F1")
        ax.set_ylabel("Personalized Model F1")
        ax.set_title("Global vs Personalized Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Personalization gain distribution with 95% CIs
    ax = axes[1]
    if not enabled.empty:
        # Compute CI for each epoch value
        epochs_unique = sorted(pers_df[pers_df["personalization_epochs"] > 0]["personalization_epochs"].unique())
        summary_stats = []

        for epoch in epochs_unique:
            epoch_data = pers_df[pers_df["personalization_epochs"] == epoch]["gain"].dropna().values
            if len(epoch_data) >= 2:
                mean, ci_lower, ci_upper = compute_confidence_interval(epoch_data)
                summary_stats.append(
                    {
                        "personalization_epochs": epoch,
                        "mean": mean,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "n": len(epoch_data),
                    }
                )
            elif len(epoch_data) == 1:
                val = float(epoch_data[0])
                summary_stats.append(
                    {
                        "personalization_epochs": epoch,
                        "mean": val,
                        "ci_lower": val,
                        "ci_upper": val,
                        "n": 1,
                    }
                )

        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            x_pos = np.arange(len(summary_df))
            yerr_lower = summary_df["mean"] - summary_df["ci_lower"]
            yerr_upper = summary_df["ci_upper"] - summary_df["mean"]

            ax.bar(
                x_pos,
                summary_df["mean"],
                yerr=[yerr_lower, yerr_upper],
                capsize=5,
                alpha=0.7,
                color=sns.color_palette("colorblind", len(summary_df)),
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(summary_df["personalization_epochs"].astype(int))
            ax.set_xlabel("Personalization Epochs")
            ax.set_ylabel("F1 Gain")
            ax.set_title("Personalization Gain by Epochs (95% CI)")

            # Add n annotations
            for i, row in summary_df.iterrows():
                y_pos = row["mean"] + yerr_upper.iloc[i] + 0.01
                ax.text(i, y_pos, f"n={row['n']}", ha="center", va="bottom", fontsize=8)

            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="No Gain")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "personalization_benefit.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_latex_summary(results_dir: Path, output_dir: Path):
    """Generate LaTeX summary tables."""
    latex_lines = []

    latex_lines.append("% Comparative Analysis Summary Tables")
    latex_lines.append("% Generated automatically")
    latex_lines.append("")

    # Table: Aggregation method comparison
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Aggregation Method Performance Comparison}")
    latex_lines.append("\\label{tab:aggregation_comparison}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Method & L2 Distance & Cosine Sim. & Time (ms) & Resilience \\\\")
    latex_lines.append("\\midrule")
    latex_lines.append("FedAvg & 0.XXX & 0.XXX & XX.X & Baseline \\\\")
    latex_lines.append("Krum & 0.XXX & 0.XXX & XX.X & Good \\\\")
    latex_lines.append("Bulyan & 0.XXX & 0.XXX & XX.X & Better \\\\")
    latex_lines.append("Median & 0.XXX & 0.XXX & XX.X & Best \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")

    # Save
    with open(output_dir / "thesis_tables.tex", "w") as f:
        f.write("\n".join(latex_lines))


def main():
    parser = argparse.ArgumentParser(description="Generate thesis plots and analysis")
    parser.add_argument(
        "--dimension",
        type=str,
        choices=[
            "aggregation",
            "heterogeneity",
            "heterogeneity_fedprox",
            "attack",
            "privacy",
            "personalization",
            "all",
        ],
        default="all",
        help="Which dimension to plot",
    )
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory with experiment runs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparative_analysis",
        help="Output directory",
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("Loading experiment results...")
    df = load_experiment_results(runs_dir)

    if df.empty:
        print("No experiment data found!")
        return

    print(f"Loaded {len(df)} rows from {len(df['run_dir'].unique())} experiments")

    # Generate plots based on dimension
    if args.dimension in ["aggregation", "all"]:
        print("Generating aggregation comparison plots...")
        plot_aggregation_comparison(df, output_dir)

    if args.dimension in ["heterogeneity", "all"]:
        print("Generating heterogeneity comparison plots...")
        plot_heterogeneity_comparison(df, output_dir)

    if args.dimension in ["attack", "all"]:
        print("Generating attack resilience plots...")
        plot_attack_resilience(df, output_dir)

    if args.dimension in ["privacy", "all"]:
        print("Generating privacy-utility plots...")
        plot_privacy_utility(df, output_dir, runs_dir=runs_dir)

    if args.dimension in ["personalization", "all"]:
        print("Generating personalization benefit plots...")
        plot_personalization_benefit(df, output_dir)

    if args.dimension in ["heterogeneity_fedprox", "all"]:
        print("Generating FedProx heterogeneity plots...")
        plot_fedprox_heterogeneity_comparison(df, output_dir)

    # Generate LaTeX tables
    print("Generating LaTeX summary tables...")
    generate_latex_summary(runs_dir, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
