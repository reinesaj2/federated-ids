from __future__ import annotations

import pandas as pd

from analyze_metric_issues import analyze_metrics_file


def test_analyze_metrics_file_flags_expected_issues(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "cos_to_benign_mean": [0.95, 0.91, 0.4],
            "l2_to_benign_mean": [0.0, 0.0, 0.3],
            "macro_f1": [0.999, 0.9995, 0.9999],
        }
    ).to_csv(csv_path, index=False)

    result = analyze_metrics_file(csv_path)

    assert result["file"] == str(csv_path)
    assert result["total_rows"] == 3
    assert any("Min cosine 0.400000" in issue for issue in result["cosine_issues"])
    assert any("L2 values = 0.0 exactly" in issue for issue in result["l2_issues"])
    assert any("Multiple L2=0 values" in issue for issue in result["l2_issues"])
    assert any("F1 values ≥ 0.999" in issue for issue in result["f1_issues"])

    all_ones_csv = tmp_path / "all_ones.csv"
    pd.DataFrame({"cos_to_benign_mean": [1.0, 1.0, 1.0]}).to_csv(all_ones_csv, index=False)
    all_ones_result = analyze_metrics_file(all_ones_csv)
    assert any("All cosine values" in issue for issue in all_ones_result["cosine_issues"])


def test_analyze_metrics_file_returns_empty_issue_lists_when_clean(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    pd.DataFrame({"some_metric": [0.1, 0.2]}).to_csv(csv_path, index=False)

    result = analyze_metrics_file(csv_path)

    assert result["cosine_issues"] == []
    assert result["l2_issues"] == []
    assert result["f1_issues"] == []
