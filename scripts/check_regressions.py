from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class ComparisonResult:
    regressions: list[RegressionRecord]
    total_compared: int
    baseline_missing: bool = False

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def _safe_ratio(baseline: float, candidate: float) -> float:
    if abs(baseline) < 1e-12:
        return float("inf") if candidate > baseline else 1.0
    return candidate / baseline


def _gather_convergence_rows(summary: dict) -> dict[str, dict[str, float | None]]:
    results: dict[str, dict[str, float | None]] = {}
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
    baseline_summary: dict | None,
    candidate_summary: dict,
    thresholds: MetricThresholds,
) -> ComparisonResult:
    if baseline_summary is None:
        return ComparisonResult(regressions=[], total_compared=0, baseline_missing=True)

    baseline_rows = _gather_convergence_rows(baseline_summary)
    candidate_rows = _gather_convergence_rows(candidate_summary)

    regressions: list[RegressionRecord] = []
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

            if metric == "final_l2_distance":
                ratio = _safe_ratio(baseline_value, candidate_value)
                if ratio > 1 + thresholds.l2_ratio:
                    regressions.append(
                        RegressionRecord(
                            metric=metric,
                            config_key=key,
                            baseline_value=baseline_value,
                            candidate_value=candidate_value,
                            threshold=thresholds.l2_ratio,
                            detail=f"ratio {ratio:.3f} exceeded allowed {(1 + thresholds.l2_ratio):.3f}",
                        )
                    )
            elif metric == "final_cosine_similarity":
                delta = baseline_value - candidate_value
                if delta > thresholds.cosine_delta:
                    regressions.append(
                        RegressionRecord(
                            metric=metric,
                            config_key=key,
                            baseline_value=baseline_value,
                            candidate_value=candidate_value,
                            threshold=thresholds.cosine_delta,
                            detail=f"drop {delta:.4f} exceeded allowed {thresholds.cosine_delta:.4f}",
                        )
                    )
            elif metric == "avg_aggregation_time":
                ratio = _safe_ratio(baseline_value, candidate_value)
                if ratio > 1 + thresholds.aggregation_ratio:
                    regressions.append(
                        RegressionRecord(
                            metric=metric,
                            config_key=key,
                            baseline_value=baseline_value,
                            candidate_value=candidate_value,
                            threshold=thresholds.aggregation_ratio,
                            detail=f"ratio {ratio:.3f} exceeded allowed {(1 + thresholds.aggregation_ratio):.3f}",
                        )
                    )
            elif metric == "update_norm_stability":
                ratio = _safe_ratio(baseline_value, candidate_value)
                if ratio > 1 + thresholds.norm_ratio:
                    regressions.append(
                        RegressionRecord(
                            metric=metric,
                            config_key=key,
                            baseline_value=baseline_value,
                            candidate_value=candidate_value,
                            threshold=thresholds.norm_ratio,
                            detail=f"ratio {ratio:.3f} exceeded allowed {(1 + thresholds.norm_ratio):.3f}",
                        )
                    )

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
        lines.append(
            f"- {record.metric} @ {record.config_key}: baseline={record.baseline_value:.6f}, candidate={record.candidate_value:.6f} ({record.detail})"
        )

    return "\n".join(lines)


def _load_summary(path: Path) -> dict:
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
