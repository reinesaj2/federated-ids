from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.verify_hybrid_dataset import METADATA_COLS, audit_hybrid_dataset_csv


def test_audit_hybrid_dataset_csv_reports_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "hybrid.csv"
    df = pd.DataFrame(
        {
            "duration": [1.0, 2.0, 3.0, 4.0],
            "fwd_packets": [0.0, 1.0, 0.0, 2.0],
            "source_dataset": ["cic", "cic", "unsw", "iiot"],
            "attack_class": [0, 1, 6, 3],
            "attack_label_original": ["BENIGN", "DoS Hulk", "Generic", "SQL_injection"],
        }
    )
    df.to_csv(csv_path, index=False)

    report = audit_hybrid_dataset_csv(csv_path, chunksize=2, duplicate_sample_rows=10, seed=123)

    assert report["total_rows"] == 4
    assert report["n_columns"] == len(df.columns)
    assert report["n_features"] == len(df.columns) - len(METADATA_COLS)
    assert report["counts_by_source"] == {"cic": 2, "iiot": 1, "unsw": 1}
    assert report["counts_by_attack_class"] == {"0": 1, "1": 1, "3": 1, "6": 1}
    assert report["passes_minimum_checks"] is True
    assert report["issues"] == []


def test_audit_hybrid_dataset_csv_flags_nan_and_inf(tmp_path: Path) -> None:
    csv_path = tmp_path / "hybrid.csv"
    df = pd.DataFrame(
        {
            "duration": [1.0, float("nan"), float("inf")],
            "fwd_packets": [0.0, 1.0, 2.0],
            "source_dataset": ["cic", "unsw", "iiot"],
            "attack_class": [0, 1, 2],
            "attack_label_original": ["BENIGN", "DoS", "Port_Scanning"],
        }
    )
    df.to_csv(csv_path, index=False)

    report = audit_hybrid_dataset_csv(csv_path, chunksize=2, duplicate_sample_rows=10, seed=1)

    messages = [i["message"] for i in report["issues"]]
    assert any("NaNs" in m for m in messages)
    assert any("infinities" in m for m in messages)
    assert report["passes_minimum_checks"] is False


def test_audit_hybrid_dataset_csv_missing_metadata_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "hybrid.csv"
    df = pd.DataFrame({"duration": [1.0], "attack_class": [0], "attack_label_original": ["BENIGN"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        audit_hybrid_dataset_csv(csv_path, chunksize=1)


def test_audit_report_is_json_serializable(tmp_path: Path) -> None:
    csv_path = tmp_path / "hybrid.csv"
    df = pd.DataFrame(
        {
            "duration": [1.0, 2.0],
            "fwd_packets": [0.0, 1.0],
            "source_dataset": ["cic", "unsw"],
            "attack_class": [0, 6],
            "attack_label_original": ["BENIGN", "Generic"],
        }
    )
    df.to_csv(csv_path, index=False)

    report = audit_hybrid_dataset_csv(csv_path, chunksize=1, duplicate_sample_rows=2, seed=42)
    payload = json.dumps(report)
    assert isinstance(payload, str)

