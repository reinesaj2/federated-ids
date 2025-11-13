"""
Unit tests for Edge-IIoTset preprocessing functions.

Tests cover data loading, label normalization, and sample generation
to ensure compatibility with existing federated learning infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import load_edge_iiotset


@pytest.fixture
def mock_edge_iiotset_csv(tmp_path: Path) -> Path:
    """Create a minimal mock Edge-IIoTset CSV for testing."""
    data = {
        "frame.time": ["2022-01-01 00:00:00"] * 100,
        "ip.src_host": ["192.168.1.1"] * 100,
        "ip.dst_host": ["192.168.1.2"] * 100,
        "tcp.seq": np.random.randint(0, 1000, 100),
        "tcp.ack": np.random.randint(0, 1000, 100),
        "udp.port": np.random.randint(0, 65535, 100),
        "http.content_length": np.random.randint(0, 10000, 100),
        "Attack_label": [0] * 60 + [1] * 40,
        "Attack_type": ["Normal"] * 60 + ["DDoS_TCP"] * 30 + ["SQL_injection"] * 10,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "mock_edge_iiotset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_edge_iiotset_binary_classification(mock_edge_iiotset_csv: Path):
    """Test loading Edge-IIoTset with binary classification."""
    df, label_col, proto_col = load_edge_iiotset(mock_edge_iiotset_csv, use_multiclass=False)

    assert label_col == "Attack_label"
    assert proto_col is None
    assert len(df) == 100
    assert "Attack_label" in df.columns
    assert set(df["Attack_label"].unique()) == {0, 1}


def test_load_edge_iiotset_multiclass_classification(mock_edge_iiotset_csv: Path):
    """Test loading Edge-IIoTset with multi-class classification."""
    df, label_col, proto_col = load_edge_iiotset(mock_edge_iiotset_csv, use_multiclass=True)

    assert label_col == "Attack_type"
    assert proto_col is None
    assert len(df) == 100
    assert "Attack_type" in df.columns
    assert "BENIGN" in df["Attack_type"].values
    assert "Normal" not in df["Attack_type"].values
    assert "DDoS_TCP" in df["Attack_type"].values
    assert "SQL_injection" in df["Attack_type"].values


def test_load_edge_iiotset_normal_to_benign_normalization(mock_edge_iiotset_csv: Path):
    """Test that 'Normal' is normalized to 'BENIGN' for consistency."""
    df, label_col, _ = load_edge_iiotset(mock_edge_iiotset_csv, use_multiclass=True)

    assert "BENIGN" in df[label_col].values
    assert "Normal" not in df[label_col].values

    benign_count = (df[label_col] == "BENIGN").sum()
    assert benign_count == 60


def test_load_edge_iiotset_max_samples(mock_edge_iiotset_csv: Path):
    """Test loading with max_samples limit."""
    df, label_col, _ = load_edge_iiotset(mock_edge_iiotset_csv, use_multiclass=False, max_samples=50)

    assert len(df) <= 50
    assert label_col == "Attack_label"


def test_load_edge_iiotset_drops_duplicates(tmp_path: Path):
    """Test that duplicate rows are removed."""
    data = {
        "tcp.seq": [1, 1, 2, 2, 3],
        "tcp.ack": [10, 10, 20, 20, 30],
        "Attack_label": [0, 0, 1, 1, 0],
        "Attack_type": ["Normal", "Normal", "DDoS_TCP", "DDoS_TCP", "Normal"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "duplicates.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, _, _ = load_edge_iiotset(csv_path, use_multiclass=False)

    assert len(loaded_df) == 3


def test_load_edge_iiotset_handles_inf_values(tmp_path: Path):
    """Test that inf values are replaced with NaN and dropped."""
    data = {
        "tcp.seq": [1, 2, np.inf, 4, 5],
        "tcp.ack": [10, 20, 30, -np.inf, 50],
        "Attack_label": [0, 0, 1, 1, 0],
        "Attack_type": ["Normal", "Normal", "DDoS_TCP", "DDoS_TCP", "Normal"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "with_inf.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, _, _ = load_edge_iiotset(csv_path, use_multiclass=False)

    assert len(loaded_df) == 3
    assert not np.any(np.isinf(loaded_df.select_dtypes(include=[np.number]).values))


def test_load_edge_iiotset_missing_label_column_raises_error(tmp_path: Path):
    """Test that missing label column raises clear error."""
    data = {
        "tcp.seq": [1, 2, 3],
        "tcp.ack": [10, 20, 30],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "no_labels.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Expected label column"):
        load_edge_iiotset(csv_path, use_multiclass=True)


def test_load_edge_iiotset_whitespace_stripping(tmp_path: Path):
    """Test that column names and labels have whitespace stripped."""
    data = {
        " tcp.seq ": [1, 2, 3],
        "tcp.ack": [10, 20, 30],
        "Attack_label": [0, 1, 0],
        "Attack_type": [" Normal ", "DDoS_TCP", " SQL_injection "],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "whitespace.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, label_col, _ = load_edge_iiotset(csv_path, use_multiclass=True)

    assert "tcp.seq" in loaded_df.columns
    assert " tcp.seq " not in loaded_df.columns
    assert "BENIGN" in loaded_df[label_col].values
    assert " Normal " not in loaded_df[label_col].values
