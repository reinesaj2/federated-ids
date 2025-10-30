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

# Add scripts directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from plot_metrics_utils import compute_confidence_interval  # noqa: E402
from privacy_accounting import compute_epsilon  # noqa: E402
from metric_validation import MetricValidator  # noqa: E402

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    normalize: bool = False,
) -> np.ndarray:
    """Compute confusion matrix with optional row-wise normalization."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    cm = np.zeros((num_classes, num_classes), dtype=float)
    if y_true.size == 0:
        return cm

    for true_label, pred_label in zip(y_true, y_pred):
        try:
            cm[int(true_label), int(pred_label)] += 1.0
        except IndexError as err:
            raise ValueError("labels must be in range [0, num_classes)") from err

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.divide(cm, row_sums, where=row_sums != 0.0)
            cm[row_sums.squeeze() == 0.0] = 0.0

    return cm


def aggregate_confusion_matrices(confusion_matrices: List[np.ndarray]) -> np.ndarray:
    """Aggregate a list of same-shaped confusion matrices by summation."""
    if not confusion_matrices:
        raise ValueError("Cannot aggregate empty list of confusion matrices")

    first_shape = confusion_matrices[0].shape
    if any(cm.shape != first_shape for cm in confusion_matrices):
        raise ValueError("Confusion matrices have shape mismatch")

    stacked = np.stack(confusion_matrices, axis=0)
    return stacked.sum(axis=0)


def render_confusion_matrix_heatmap(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = False,
    cmap: str = "Blues",
) -> None:
    """Render and persist a confusion matrix heatmap."""
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")

    if len(class_names) != confusion_matrix.shape[0]:
        raise ValueError("class_names size must match matrix dimension")

    matrix = confusion_matrix.astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.divide(matrix, row_sums, where=row_sums != 0.0)
            matrix[row_sums.squeeze() == 0.0] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "g",
        cmap=cmap,
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_client_scatter_data(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Prepare per-client scatter data for FedProx comparison plots."""
    required_columns = {"client_id", "seed", "round", "fedprox_mu", metric}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame(columns=list(required_columns) + ["jitter"])

    working_df = df.dropna(subset=[metric, "fedprox_mu", "round"])
    if working_df.empty:
        return pd.DataFrame(columns=list(required_columns) + ["jitter"])

    group_cols = ["seed", "client_id", "fedprox_mu"]
    if "alpha" in working_df.columns:
        group_cols.append("alpha")

    working_df = working_df.sort_values(group_cols + ["round"])
    final_rows = (
        working_df.groupby(group_cols, as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )

    if final_rows.empty:
        return pd.DataFrame(columns=list(required_columns) + ["jitter"])

    # Deterministic jitter to avoid point overlap (±2% of mu)
    hashes = pd.util.hash_pandas_object(final_rows[group_cols], index=False).astype(np.int64)
    jitter_scale = final_rows["fedprox_mu"].astype(float) * 0.02
    jitter = ((hashes % 2001) / 1000.0 - 1.0) * jitter_scale
    final_rows = final_rows.assign(jitter=jitter)
    return final_rows


def _compute_global_mean_by_mu(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate scatter data to compute global means and confidence intervals per mu."""
    if df.empty or metric not in df.columns or "fedprox_mu" not in df.columns:
        return pd.DataFrame(columns=["mu", "mean", "ci_lower", "ci_upper", "n"])

    records: List[Dict] = []
    for mu_value, subset in df.groupby("fedprox_mu"):
        metric_values = subset[metric].dropna().astype(float)
        if metric_values.empty:
            continue
        n = len(metric_values)
        mean_val = float(metric_values.mean())
        ci_lower = mean_val
        ci_upper = mean_val
        if n >= 2:
            se = stats.sem(metric_values)
            margin = se * stats.t.ppf(0.975, n - 1)
            ci_lower = mean_val - margin
            ci_upper = mean_val + margin
        records.append(
            {
                "mu": float(mu_value),
                "mean": mean_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n,
            }
        )

    return pd.DataFrame(records).sort_values("mu").reset_index(drop=True)


def _render_client_scatter_mu_plot(
    scatter_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    use_log_y: bool = False,
) -> None:
    """Render scatter plot of client metrics stratified by FedProx mu with global overlay."""
    required_columns = {"fedprox_mu", metric, "jitter"}
    if scatter_df.empty or not required_columns.issubset(scatter_df.columns):
        raise ValueError("scatter_df is empty or missing required columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    x_positions = scatter_df["fedprox_mu"] + scatter_df["jitter"]
    ax.scatter(
        x_positions,
        scatter_df[metric],
        alpha=0.6,
        c=scatter_df.get("alpha", scatter_df.get("seed", 0)),
        cmap="viridis",
        label="Clients",
    )

    global_mean = _compute_global_mean_by_mu(scatter_df, metric)
    if not global_mean.empty:
        ax.errorbar(
            global_mean["mu"],
            global_mean["mean"],
            yerr=[
                global_mean["mean"] - global_mean["ci_lower"],
                global_mean["ci_upper"] - global_mean["mean"],
            ],
            fmt="-o",
            color="black",
            capsize=4,
            label="Global mean ±95% CI",
        )

    ax.set_xlabel("FedProx μ")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Per-client {metric.replace('_', ' ')} by FedProx μ")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if use_log_y:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_personalization_delta_f1_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-client personalization delta F1 for the final round."""
    required = {"client_id", "seed", "round", "macro_f1_global", "macro_f1_personalized"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=list(required) + ["alpha", "delta_f1", "f1_global", "f1_personalized"])

    working_df = df.dropna(subset=["macro_f1_global", "macro_f1_personalized"])
    if working_df.empty:
        return pd.DataFrame(columns=list(required) + ["alpha", "delta_f1", "f1_global", "f1_personalized"])

    working_df = working_df.sort_values("round")
    group_cols = ["seed", "client_id"]
    if "alpha" in working_df.columns:
        group_cols.append("alpha")
    final_rows = (
        working_df.groupby(group_cols, as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if final_rows.empty:
        return pd.DataFrame(columns=list(required) + ["alpha", "delta_f1", "f1_global", "f1_personalized"])

    delta_df = final_rows.copy()
    delta_df["delta_f1"] = delta_df["macro_f1_personalized"] - delta_df["macro_f1_global"]
    delta_df = delta_df.rename(
        columns={
            "macro_f1_global": "f1_global",
            "macro_f1_personalized": "f1_personalized",
        }
    )
    return delta_df


def _analyze_delta_f1_by_alpha(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize personalization delta F1 grouped by Dirichlet alpha."""
    if delta_df.empty or "alpha" not in delta_df.columns or "delta_f1" not in delta_df.columns:
        return pd.DataFrame(columns=["alpha", "mean_delta", "median_delta", "ci_lower", "ci_upper", "pct_positive", "n"])

    records: List[Dict] = []
    for alpha_value, subset in delta_df.groupby("alpha"):
        deltas = subset["delta_f1"].dropna().astype(float)
        if deltas.empty:
            continue
        n = len(deltas)
        mean_delta = float(deltas.mean())
        median_delta = float(deltas.median())
        ci_lower = mean_delta
        ci_upper = mean_delta
        if n >= 2:
            se = stats.sem(deltas)
            margin = se * stats.t.ppf(0.975, n - 1)
            ci_lower = mean_delta - margin
            ci_upper = mean_delta + margin
        pct_positive = float((deltas > 0).mean() * 100.0)
        records.append(
            {
                "alpha": float(alpha_value),
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pct_positive": pct_positive,
                "n": n,
            }
        )

    return pd.DataFrame(records).sort_values("alpha").reset_index(drop=True)


def _render_personalization_delta_f1_plots(
    scatter_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render personalization delta F1 analysis figure."""
    required_scatter = {"alpha", "delta_f1", "f1_global", "f1_personalized"}
    required_stats = {"alpha", "mean_delta", "pct_positive"}
    if scatter_df.empty or not required_scatter.issubset(scatter_df.columns):
        raise ValueError("scatter_df is empty or missing required columns")
    if stats_df.empty or not required_stats.issubset(stats_df.columns):
        raise ValueError("stats_df is empty or missing required columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter plot with baseline
    sns.scatterplot(
        data=scatter_df,
        x="f1_global",
        y="f1_personalized",
        hue="alpha",
        style="alpha",
        ax=ax1,
        palette="viridis",
    )
    min_val = min(scatter_df["f1_global"].min(), scatter_df["f1_personalized"].min())
    max_val = max(scatter_df["f1_global"].max(), scatter_df["f1_personalized"].max())
    ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="y = x baseline")
    ax1.set_xlabel("Global Macro-F1")
    ax1.set_ylabel("Personalized Macro-F1")
    ax1.set_title("Personalized vs Global Performance")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Violin plot of delta distributions
    sns.violinplot(
        data=scatter_df,
        x="alpha",
        y="delta_f1",
        ax=ax2,
        inner="quartile",
        palette="viridis",
    )
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Dirichlet α")
    ax2.set_ylabel("Δ Macro-F1 (Personalized - Global)")
    ax2.set_title("Distribution of Personalization Gains")

    # Bar plot of summary statistics
    stats_sorted = stats_df.sort_values("alpha")
    ax3.bar(
        stats_sorted["alpha"].astype(str),
        stats_sorted["mean_delta"],
        color=sns.color_palette("viridis", n_colors=len(stats_sorted)),
    )
    if {"ci_lower", "ci_upper"}.issubset(stats_sorted.columns):
        yerr = np.vstack(
            [
                stats_sorted["mean_delta"] - stats_sorted["ci_lower"],
                stats_sorted["ci_upper"] - stats_sorted["mean_delta"],
            ]
        )
        ax3.errorbar(
            stats_sorted["alpha"].astype(str),
            stats_sorted["mean_delta"],
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=4,
        )
    ax3.set_xlabel("Dirichlet α")
    ax3.set_ylabel("Mean Δ Macro-F1")
    ax3.set_title("Average Personalization Gain")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
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


def generate_privacy_utility_curve(df: pd.DataFrame, output_dir: Path, runs_dir: Path) -> None:
    """
    Generate privacy-utility curve visualization with formal epsilon accounting.

    Creates curve showing macro-F1 vs epsilon (privacy budget) for DP-enabled experiments.
    Aggregates multiple seeds with 95% confidence intervals.

    Args:
        df: Experiment results dataframe with final metrics per run
        output_dir: Directory to save plots and CSV summaries
        runs_dir: Root directory containing individual run outputs

    This function:
    1. Filters DP-enabled experiments from results
    2. Prepares privacy curve data (epsilon, macro-F1, seed)
    3. Aggregates across seeds with confidence intervals
    4. Renders epsilon-utility tradeoff visualization
    5. Saves summary CSV for thesis tables
    """
    if df.empty:
        return

    # Get final round metrics (groupby run, seed, take last row)
    final_rounds = df.groupby(["run_dir", "seed"]).tail(1).reset_index(drop=True)

    if final_rounds.empty:
        return

    # Prepare data for privacy curve (aggregates clients, computes epsilon)
    # Includes both DP-enabled and baseline experiments
    dp_df, baseline_df = _prepare_privacy_curve_data(final_rounds, runs_dir)

    if dp_df.empty and baseline_df.empty:
        return

    # Render curve with summary stats
    _render_privacy_curve(dp_df, baseline_df, output_dir)


def plot_privacy_utility(df: pd.DataFrame, output_dir: Path, runs_dir: Path) -> None:
    """
    Plot privacy-utility tradeoff for DP experiments.

    Wrapper for thesis dimension: "privacy".
    Generates formal privacy-utility curve showing macro-F1 vs epsilon.

    Args:
        df: Experiment results
        output_dir: Output directory
        runs_dir: Run directory root
    """
    generate_privacy_utility_curve(df, output_dir, runs_dir)


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
