from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    _CURRENT_FILE = Path(__file__).resolve()
    _REPO_ROOT = _CURRENT_FILE.parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from scripts.statistical_utils import paired_t_test


@dataclass
class MetricThresholds:
    """Thresholds for deciding when a metric regresses."""

    l2_ratio: float = 0.05
    cosine_delta: float = 0.002
    aggregation_ratio: float = 0.10
    norm_ratio: float = 0.10


@dataclass
class RegressionRecord:
    metric: str
    config_key: str
    baseline_value: float
    candidate_value: float
    threshold: float
    detail: str
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class ComparisonResult:
    regressions: List[RegressionRecord]
    total_compared: int
    baseline_missing: bool = False

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def _safe_ratio(baseline: float, candidate: float) -> float:
    if abs(baseline) < 1e-12:
        return float("inf") if candidate > baseline else 1.0
    return candidate / baseline


def _check_threshold_violation(
    metric: str,
    config_key: str,
    baseline_value: float,
    candidate_value: float,
    threshold_type: str,
    threshold_value: float,
) -> RegressionRecord | None:
    """Check if a metric exceeds threshold and return regression record if violated.

    Args:
        metric: Metric name (e.g., "final_l2_distance")
        config_key: Configuration key for grouping
        baseline_value: Baseline measurement
        candidate_value: Candidate measurement
        threshold_type: Either "ratio" or "delta" for comparison type
        threshold_value: Threshold value for the metric

    Returns:
        RegressionRecord if threshold violated, None otherwise.
    """
    if threshold_type == "ratio":
        ratio = _safe_ratio(baseline_value, candidate_value)
        threshold_exceeded = ratio > 1 + threshold_value
        detail = f"ratio {ratio:.3f} exceeded allowed {(1 + threshold_value):.3f}"
    elif threshold_type == "delta":
        delta = baseline_value - candidate_value
        threshold_exceeded = delta > threshold_value
        detail = f"drop {delta:.4f} exceeded allowed {threshold_value:.4f}"
    else:
        return None

    if threshold_exceeded:
        return RegressionRecord(
            metric=metric,
            config_key=config_key,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            threshold=threshold_value,
            detail=detail,
        )
    return None


def _compute_statistical_significance(
    baseline_values: List[float],
    candidate_values: List[float],
) -> tuple[float | None, float | None, float | None]:
    """Compute p-value and confidence intervals from paired observations.

    Args:
        baseline_values: Baseline measurements
        candidate_values: Candidate measurements

    Returns:
        Tuple of (p_value, ci_lower, ci_upper) or (None, None, None) if
        insufficient data.
    """
    if len(baseline_values) != len(candidate_values) or len(baseline_values) < 2:
        return None, None, None

    result = paired_t_test(baseline_values, candidate_values)
    p_value = result.get("p_value")
    if math.isnan(p_value):
        p_value = None

    return p_value, None, None


def _gather_convergence_rows(summary: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    results: Dict[str, Dict[str, Optional[float]]] = {}
    convergence = summary.get("raw_analysis_results", {}).get("convergence_analysis", {})
    for key, values in convergence.items():
        results[key] = {
            "final_l2_distance": values.get("final_l2_distance"),
            "final_cosine_similarity": values.get("final_cosine_similarity"),
            "avg_aggregation_time": values.get("avg_aggregation_time"),
            "update_norm_stability": values.get("update_norm_stability"),
        }
    return results


def compare_summaries(
    baseline_summary: Optional[Dict],
    candidate_summary: Dict,
    thresholds: MetricThresholds,
) -> ComparisonResult:
    if baseline_summary is None:
        return ComparisonResult(regressions=[], total_compared=0, baseline_missing=True)

    baseline_rows = _gather_convergence_rows(baseline_summary)
    candidate_rows = _gather_convergence_rows(candidate_summary)

    regressions: List[RegressionRecord] = []
    total_compared = 0

    for key, candidate_metrics in candidate_rows.items():
        baseline_metrics = baseline_rows.get(key)
        if not baseline_metrics:
            continue

        for metric, candidate_value in candidate_metrics.items():
            baseline_value = baseline_metrics.get(metric)
            if baseline_value is None or candidate_value is None:
                continue

            total_compared += 1

            violation = None
            if metric == "final_l2_distance":
                violation = _check_threshold_violation(
                    metric=metric,
                    config_key=key,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    threshold_type="ratio",
                    threshold_value=thresholds.l2_ratio,
                )
            elif metric == "final_cosine_similarity":
                violation = _check_threshold_violation(
                    metric=metric,
                    config_key=key,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    threshold_type="delta",
                    threshold_value=thresholds.cosine_delta,
                )
            elif metric == "avg_aggregation_time":
                violation = _check_threshold_violation(
                    metric=metric,
                    config_key=key,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    threshold_type="ratio",
                    threshold_value=thresholds.aggregation_ratio,
                )
            elif metric == "update_norm_stability":
                violation = _check_threshold_violation(
                    metric=metric,
                    config_key=key,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    threshold_type="ratio",
                    threshold_value=thresholds.norm_ratio,
                )

            if violation is not None:
                regressions.append(violation)

    return ComparisonResult(
        regressions=regressions,
        total_compared=total_compared,
        baseline_missing=False,
    )


def format_regression_report(result: ComparisonResult) -> str:
    if result.baseline_missing:
        return "Baseline summary unavailable; regression checks skipped."

    if not result.has_regressions:
        return f"No regressions detected across {result.total_compared} metric comparisons."

    lines = [
        f"Regression detected in {len(result.regressions)} metric(s) out of {result.total_compared} comparisons:",
    ]

    for record in result.regressions:
        detail_str = record.detail
        if record.p_value is not None:
            detail_str += f" [p={record.p_value:.4f}]"
        lines.append(
            "- {metric} @ {config}: baseline={baseline:.6f}, candidate={candidate:.6f} ({detail})".format(
                metric=record.metric,
                config=record.config_key,
                baseline=record.baseline_value,
                candidate=record.candidate_value,
                detail=detail_str,
            )
        )

    return "\n".join(lines)


def _load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FedProx regression metrics")
    parser.add_argument("--candidate", required=True, type=Path, help="Current summary JSON")
    parser.add_argument("--baseline", type=Path, help="Baseline summary JSON")
    parser.add_argument("--output", type=Path, help="Optional path to write regression report JSON")
    args = parser.parse_args()

    candidate_summary = _load_summary(args.candidate)
    baseline_summary = _load_summary(args.baseline) if args.baseline and args.baseline.exists() else None

    result = compare_summaries(baseline_summary, candidate_summary, MetricThresholds())
    report_text = format_regression_report(result)
    print(report_text)

    if args.output:
        payload = {
            "regressions": [record.__dict__ for record in result.regressions],
            "total_compared": result.total_compared,
            "baseline_missing": result.baseline_missing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
