from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_regressions import (
    MetricThresholds,
    compare_summaries,
    format_regression_report,
)


def _summary_with_metrics(l2: float, cosine: float, agg: float, norm: float) -> dict:
    return {
        "raw_analysis_results": {
            "convergence_analysis": {
                "alpha_0.1_mu_0.0": {
                    "final_l2_distance": l2,
                    "final_cosine_similarity": cosine,
                    "avg_aggregation_time": agg,
                    "update_norm_stability": norm,
                }
            }
        }
    }


def test_compare_summaries_detects_no_regression(tmp_path: Path) -> None:
    baseline = _summary_with_metrics(0.5, 0.99, 1.0, 0.4)
    candidate = _summary_with_metrics(0.525, 0.9895, 1.05, 0.43)

    result = compare_summaries(baseline, candidate, MetricThresholds())

    assert result.regressions == []
    assert result.total_compared == 4


def test_compare_summaries_flags_l2_regression(tmp_path: Path) -> None:
    baseline = _summary_with_metrics(0.5, 0.99, 1.0, 0.4)
    candidate = _summary_with_metrics(0.56, 0.9895, 1.05, 0.43)

    result = compare_summaries(baseline, candidate, MetricThresholds())

    assert len(result.regressions) == 1
    regression = result.regressions[0]
    assert regression.metric == "final_l2_distance"
    assert regression.config_key == "alpha_0.1_mu_0.0"
    assert pytest.approx(regression.baseline_value) == 0.5
    assert pytest.approx(regression.candidate_value) == 0.56


def test_format_regression_report_outputs_readable_summary() -> None:
    baseline = _summary_with_metrics(0.5, 0.99, 1.0, 0.4)
    candidate = _summary_with_metrics(0.56, 0.98, 1.2, 0.55)

    result = compare_summaries(baseline, candidate, MetricThresholds())
    report = format_regression_report(result)

    assert "Regression detected" in report
    assert "final_l2_distance" in report
    assert "avg_aggregation_time" in report
    assert "final_cosine_similarity" in report
