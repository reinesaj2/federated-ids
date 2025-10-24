#!/usr/bin/env python3
"""
Aggregate FedProx nightlies into publication-ready summaries.

This script consolidates per-seed run artifacts produced by the FedProx nightly
workflow, computes aggregate statistics with 95% confidence intervals, and
emits plots/tables suitable for the thesis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ARTIFACT_DIR_PATTERN = re.compile(r"fedprox-nightly-alpha(?P<alpha>[0-9.]+)-mu(?P<mu>[0-9.]+)-")
RUN_DIR_PATTERN = re.compile(r"nightly_fedprox_alpha(?P<alpha>[0-9.]+)_mu(?P<mu>[0-9.]+)_seed(?P<seed>\d+)")


@dataclass(frozen=True)
class RunMetrics:
    alpha: float
    mu: float
    seed: int
    algorithm: str
    weighted_macro_f1: float
    mean_aggregation_time_ms: float
    rounds: int
    run_dir: Path


def _safe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_float(row: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in row:
            value = _safe_float(row[key])
            if value is not None and not math.isnan(value):
                return value
    return None


def compute_weighted_macro_f1(rows: Sequence[Mapping[str, object]]) -> float:
    """Compute dataset-size-weighted macro-F1 across clients."""
    weighted_sum = 0.0
    total_examples = 0.0

    for row in rows:
        dataset_size = _safe_float(row.get("dataset_size"))
        if dataset_size is None or dataset_size <= 0:
            continue

        macro_f1 = _extract_float(
            row,
            (
                "macro_f1_after",
                "macro_f1_argmax",
                "macro_f1_before",
            ),
        )
        if macro_f1 is None:
            continue

        weighted_sum += dataset_size * macro_f1
        total_examples += dataset_size

    if total_examples == 0:
        return float("nan")

    return weighted_sum / total_examples


def _parse_run_identifiers(path: Path) -> tuple[float, float, int] | None:
    match = ARTIFACT_DIR_PATTERN.search(path.name)
    if match:
        # Artifact names don't include seed, return None for seed to signal
        # that we should look inside for run directories
        return None
    match = RUN_DIR_PATTERN.search(path.name)
    if match:
        return (
            float(match.group("alpha")),
            float(match.group("mu")),
            int(match.group("seed")),
        )
    return None


def _load_client_summaries(run_dir: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for client_file in sorted(run_dir.glob("client_*_metrics.csv")):
        try:
            df = pd.read_csv(client_file)
        except Exception:
            continue
        if df.empty:
            continue
        last_row = df.iloc[-1].to_dict()
        summaries.append(last_row)
    return summaries


def _load_server_metrics(run_dir: Path) -> pd.DataFrame | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def _resolve_algorithm(mu: float, metadata_path: Path | None) -> str:
    if metadata_path and metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            algo = str(meta.get("algorithm", "")).strip()
            if algo:
                return algo
        except json.JSONDecodeError:
            pass
    return "FedProx" if mu > 0 else "FedAvg"


def collect_run_metrics(artifacts_dir: Path) -> list[RunMetrics]:
    run_metrics: list[RunMetrics] = []

    if not artifacts_dir.exists():
        return run_metrics

    for artifact_dir in sorted(artifacts_dir.iterdir()):
        identifiers = _parse_run_identifiers(artifact_dir)

        candidate_run_dirs: list[Path]
        if identifiers is not None:
            alpha, mu, seed = identifiers
            candidate_run_dirs = list(artifact_dir.glob(f"**/nightly_fedprox_alpha{alpha}_mu{mu}_seed{seed}")) or [artifact_dir]
        else:
            candidate_run_dirs = [p for p in artifact_dir.rglob("nightly_fedprox_alpha*_mu*_seed*") if p.is_dir()]

        for run_dir in candidate_run_dirs:
            run_identifiers = _parse_run_identifiers(run_dir)
            if run_identifiers is None:
                continue

            alpha, mu, seed = run_identifiers
            client_summaries = _load_client_summaries(run_dir)
            if not client_summaries:
                continue

            weighted_macro_f1 = compute_weighted_macro_f1(client_summaries)

            server_df = _load_server_metrics(run_dir)
            if server_df is not None and "t_aggregate_ms" in server_df.columns:
                mean_agg_time = float(server_df["t_aggregate_ms"].dropna().mean())
            else:
                mean_agg_time = float("nan")

            rounds = int(server_df["round"].max()) if server_df is not None else 0

            algorithm = _resolve_algorithm(mu, run_dir / "metadata.json")

            run_metrics.append(
                RunMetrics(
                    alpha=float(alpha),
                    mu=float(mu),
                    seed=int(seed),
                    algorithm=algorithm,
                    weighted_macro_f1=float(weighted_macro_f1),
                    mean_aggregation_time_ms=float(mean_agg_time),
                    rounds=rounds,
                    run_dir=run_dir,
                )
            )

    return run_metrics


def _mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean

    std = float(arr.std(ddof=1))
    if std == 0.0:
        return mean, mean, mean

    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=arr.size - 1))
    margin = t_crit * std / math.sqrt(arr.size)
    return mean, mean - margin, mean + margin


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size (mean difference / pooled std)
    """
    arr1 = np.array([v for v in group1 if not math.isnan(v)], dtype=float)
    arr2 = np.array([v for v in group2 if not math.isnan(v)], dtype=float)

    if arr1.size == 0 or arr2.size == 0:
        return float("nan")

    mean_diff = float(arr1.mean() - arr2.mean())
    n1, n2 = arr1.size, arr2.size

    if n1 == 1 and n2 == 1:
        return mean_diff

    var1 = float(arr1.var(ddof=1)) if n1 > 1 else 0.0
    var2 = float(arr2.var(ddof=1)) if n2 > 1 else 0.0

    pooled_std = math.sqrt((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2))

    if pooled_std == 0.0:
        return float("nan")

    return mean_diff / pooled_std


def aggregate_run_metrics(
    run_metrics: Sequence[RunMetrics],
    metrics: Sequence[str] = ("weighted_macro_f1", "mean_aggregation_time_ms"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not run_metrics:
        return pd.DataFrame(columns=["alpha", "mu", "algorithm", "metric", "mean", "ci_lower", "ci_upper", "n"])

    grouped: dict[tuple[float, float, str], list[RunMetrics]] = {}
    for metric in run_metrics:
        key = (metric.alpha, metric.mu, metric.algorithm)
        grouped.setdefault(key, []).append(metric)

    for (alpha, mu, algorithm), group_runs in grouped.items():
        for metric_name in metrics:
            values = [getattr(run, metric_name, float("nan")) for run in group_runs]
            mean, ci_lower, ci_upper = _mean_ci(values)
            n = sum(1 for v in values if not math.isnan(v))
            if n == 0:
                continue
            rows.append(
                {
                    "alpha": alpha,
                    "mu": mu,
                    "algorithm": algorithm,
                    "metric": metric_name,
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n": n,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical validation helpers


def ensure_minimum_samples(run_metrics: Sequence[RunMetrics], minimum: int = 5) -> None:
    """Ensure every configuration has at least ``minimum`` seeds available."""
    sample_counts: dict[tuple[float, float, str], set[int]] = defaultdict(set)
    for run in run_metrics:
        sample_counts[(run.alpha, run.mu, run.algorithm)].add(run.seed)

    violations = [(alpha, mu, algorithm, len(seeds)) for (alpha, mu, algorithm), seeds in sample_counts.items() if len(seeds) < minimum]

    if violations:
        alpha, mu, algorithm, observed = sorted(violations, key=lambda t: (t[0], t[1], t[2]))[0]
        raise ValueError(
            f"FedProx nightly runs for alpha={alpha} mu={mu} algorithm={algorithm} "
            f"have only {observed} seeds; require at least {minimum}."
        )


def compute_paired_statistics(
    run_metrics: Sequence[RunMetrics],
    metric_name: str = "weighted_macro_f1",
    baseline_algorithm: str = "FedAvg",
) -> list[dict[str, object]]:
    """Compute paired t-tests and effect sizes comparing FedProx to FedAvg."""
    baseline_values: dict[tuple[float, int], float] = {}
    candidate_map: dict[tuple[float, float], list[tuple[int, float, str]]] = defaultdict(list)

    for run in run_metrics:
        value = getattr(run, metric_name, float("nan"))
        if math.isnan(value):
            continue
        if run.algorithm == baseline_algorithm and math.isclose(run.mu, 0.0, abs_tol=1e-12):
            baseline_values[(run.alpha, run.seed)] = value
        else:
            candidate_map[(run.alpha, run.mu)].append((run.seed, value, run.algorithm))

    results: list[dict[str, object]] = []
    for (alpha, mu), entries in sorted(candidate_map.items()):
        diffs: list[float] = []
        prox_values: list[float] = []
        fedavg_values: list[float] = []
        algorithm_label = {label for _, _, label in entries}
        candidate_algorithm = next(iter(algorithm_label)) if algorithm_label else "FedProx"

        for seed, candidate_value, _ in entries:
            baseline_value = baseline_values.get((alpha, seed))
            if baseline_value is None:
                continue
            diffs.append(candidate_value - baseline_value)
            prox_values.append(candidate_value)
            fedavg_values.append(baseline_value)

        n = len(diffs)
        if n == 0:
            continue

        mean_diff, ci_lower, ci_upper = _mean_ci(diffs)
        if n > 1:
            effect_size = cohens_d(prox_values, fedavg_values)
            _, p_value = stats.ttest_rel(prox_values, fedavg_values)
            p_value = float(p_value)
        else:
            effect_size = float("nan")
            p_value = float("nan")

        results.append(
            {
                "alpha": alpha,
                "mu": mu,
                "metric": metric_name,
                "n": n,
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "effect_size": effect_size,
                "baseline_algorithm": baseline_algorithm,
                "candidate_algorithm": candidate_algorithm,
            }
        )

    return results


def plot_aggregated_metrics(aggregated: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FedAvg vs FedProx Nightly Aggregates", fontsize=16)

    def _plot_metric(ax, metric_name: str, ylabel: str) -> None:
        metric_df = aggregated[aggregated["metric"] == metric_name]
        if metric_df.empty:
            ax.set_visible(False)
            return

        for (alpha, algorithm), group in metric_df.groupby(["alpha", "algorithm"]):
            sorted_group = group.sort_values("mu")
            x = sorted_group["mu"].to_numpy()
            mean = sorted_group["mean"].to_numpy()
            lower = sorted_group["ci_lower"].to_numpy()
            upper = sorted_group["ci_upper"].to_numpy()
            label = f"{algorithm} (α={alpha})"
            ax.plot(x, mean, marker="o", label=label)
            ax.fill_between(x, lower, upper, alpha=0.2)

        ax.set_xlabel("FedProx μ")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    _plot_metric(axes[0], "weighted_macro_f1", "Macro-F1 (weighted)")
    _plot_metric(axes[1], "mean_aggregation_time_ms", "Aggregation time (ms)")

    fig.tight_layout()
    fig.savefig(output_dir / "fedprox_performance_plots.png", dpi=200)
    fig.savefig(output_dir / "fedprox_performance_plots.pdf", dpi=200)
    plt.close(fig)


def write_summary(
    run_metrics: Sequence[RunMetrics],
    aggregated: pd.DataFrame,
    significance: Sequence[dict[str, object]],
    output_dir: Path,
    run_timestamp: Optional[str] = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_timestamp is None:
        run_timestamp = datetime.now(timezone.utc).isoformat()

    runs_payload = [
        {
            "alpha": run.alpha,
            "mu": run.mu,
            "seed": run.seed,
            "algorithm": run.algorithm,
            "weighted_macro_f1": run.weighted_macro_f1,
            "mean_aggregation_time_ms": run.mean_aggregation_time_ms,
            "rounds": run.rounds,
            "run_dir": str(run.run_dir),
        }
        for run in run_metrics
    ]

    convergence_analysis = {}
    for run in run_metrics:
        key = f"alpha_{run.alpha}_mu_{run.mu}"
        if key not in convergence_analysis:
            convergence_analysis[key] = {
                "alpha": run.alpha,
                "mu": run.mu,
                "algorithm": run.algorithm,
                "final_l2_distance": float("nan"),
                "final_cosine_similarity": float("nan"),
                "avg_aggregation_time": run.mean_aggregation_time_ms,
            }

    summary = {
        "run_timestamp": run_timestamp,
        "runs": runs_payload,
        "aggregated": aggregated.to_dict(orient="records"),
        "significance": list(significance),
        "raw_analysis_results": {"convergence_analysis": convergence_analysis},
    }

    (output_dir / "fedprox_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    aggregated.to_csv(output_dir / "fedprox_comparison_summary.csv", index=False)

    return summary


def generate_thesis_tables(
    aggregated: pd.DataFrame,
    significance: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_df = aggregated[aggregated["metric"] == "weighted_macro_f1"].copy()
    if macro_df.empty:
        return

    macro_df = macro_df.sort_values(["alpha", "mu", "algorithm"])
    lines: list[str] = [
        "% Auto-generated by analyze_fedprox_comparison.py",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{FedAvg vs FedProx weighted Macro-F1 (95\\% CI)}",
        "\\label{tab:fedprox_macro_f1}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Algorithm & $\\alpha$ & $\\mu$ & Macro-F1 $\\pm$ CI \\\\",
        "\\midrule",
    ]

    for _, row in macro_df.iterrows():
        mean = row["mean"]
        lower = row["ci_lower"]
        upper = row["ci_upper"]
        ci_width = (upper - lower) / 2 if not math.isnan(upper) else 0.0
        lines.append(f"{row['algorithm']} & {row['alpha']} & {row['mu']} & " f"{mean:.4f} $\\pm$ {ci_width:.4f} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    diff_rows = [row for row in significance if row["metric"] == "weighted_macro_f1"]
    if diff_rows:
        lines.extend(
            [
                "",
                "\\begin{table}[ht]",
                "\\centering",
                "\\caption{FedProx vs FedAvg macro-F1 improvements (paired t-test)}",
                "\\label{tab:fedprox_macro_f1_diffs}",
                "\\begin{tabular}{lccccl}",
                "\\toprule",
                "$\\alpha$ & $\\mu$ & $n$ & $\\Delta$ Macro-F1 $\\pm$ CI & $p$-value & Cohen's $d$ \\\\",
                "\\midrule",
            ]
        )

        for row in diff_rows:
            mean_diff = row["mean_diff"]
            ci_lower = row["ci_lower"]
            ci_upper = row["ci_upper"]
            ci_width = (ci_upper - ci_lower) / 2 if not math.isnan(ci_upper) else 0.0
            p_value = row["p_value"]
            effect_size = row["effect_size"]
            lines.append(
                f"{row['alpha']} & {row['mu']} & {int(row['n'])} & "
                f"{mean_diff:.4f} $\\pm$ {ci_width:.4f} & {p_value:.3g} & {effect_size:.3f} \\\\"
            )

        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    (output_dir / "fedprox_thesis_tables.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FedProx nightly comparison artifacts.")
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
        help="Directory containing downloaded nightly artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where outputs (plots/tables) will be written.",
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        required=False,
        help="Directory containing historical baseline artifacts for regression detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_metrics = collect_run_metrics(args.artifacts_dir)
    if not run_metrics:
        print("No FedProx artifacts found; nothing to summarize.")
        return

    ensure_minimum_samples(run_metrics, minimum=5)

    aggregated = aggregate_run_metrics(run_metrics)
    if aggregated.empty:
        print("No valid metrics found across runs.")
        return

    significance_rows: list[dict[str, object]] = []
    for metric_name in ("weighted_macro_f1", "mean_aggregation_time_ms"):
        significance_rows.extend(compute_paired_statistics(run_metrics, metric_name=metric_name))

    summary = write_summary(run_metrics, aggregated, significance_rows, args.output_dir)
    plot_aggregated_metrics(aggregated, args.output_dir)
    generate_thesis_tables(aggregated, significance_rows, args.output_dir)

    if args.baseline_dir:
        try:
            from scripts.historical_tracking import (
                append_to_baseline,
                generate_regression_report,
                load_baseline_window,
                plot_metric_trend_90d,
                trim_baseline_to_window,
            )

            baseline_path = args.output_dir / "historical" / "baselines.csv"
            commit_sha = os.getenv("GITHUB_SHA", "local")

            append_to_baseline(summary, baseline_path, commit_sha)
            trim_baseline_to_window(baseline_path, window_days=90)

            baseline_df = load_baseline_window(baseline_path, window_days=90)

            if not baseline_df.empty and len(baseline_df) >= 5:
                regression_report = generate_regression_report(summary, baseline_df, threshold_std=2.0)

                regression_path = args.output_dir / "historical" / "regression_report.json"
                regression_path.write_text(json.dumps(regression_report, indent=2), encoding="utf-8")

                trend_dir = args.output_dir / "historical" / "trend_plots"
                trend_dir.mkdir(parents=True, exist_ok=True)

                metrics_to_plot = [
                    ("final_l2_distance", "L2 Distance to Benign Mean"),
                    ("final_cosine_similarity", "Cosine Similarity to Benign Mean"),
                    ("avg_aggregation_time_ms", "Aggregation Time (ms)"),
                ]

                for metric_name, metric_label in metrics_to_plot:
                    plot_path = trend_dir / f"{metric_name}_trend_90d.png"
                    plot_metric_trend_90d(baseline_df, None, metric_name, plot_path, metric_label)

                print(f"Historical tracking: baseline updated, {len(baseline_df)} records in 90-day window")
                if regression_report["any_regression_detected"]:
                    print("WARNING: Regression detected in one or more metrics")
            else:
                print("Historical tracking: insufficient baseline data for regression detection")

        except ImportError as e:
            print(f"Historical tracking disabled: {e}")
        except Exception as e:
            print(f"Historical tracking failed: {e}")


if __name__ == "__main__":
    main()
