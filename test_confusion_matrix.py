#!/usr/bin/env python3
"""Unit tests for confusion matrix generation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from scripts.generate_thesis_plots import (
    compute_confusion_matrix,
    render_confusion_matrix_heatmap,
    aggregate_confusion_matrices,
)


def test_compute_confusion_matrix_binary():
    """Test confusion matrix computation for binary classification."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)

    assert cm.shape == (2, 2)
    assert cm[0, 0] == 3
    assert cm[0, 1] == 0
    assert cm[1, 0] == 1
    assert cm[1, 1] == 2


def test_compute_confusion_matrix_multiclass():
    """Test confusion matrix for multi-class classification (CIC-IDS2017)."""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3])

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)

    assert cm.shape == (4, 4)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[2, 2] == 1
    assert cm[2, 3] == 1
    assert cm[3, 2] == 1


def test_compute_confusion_matrix_normalized():
    """Test normalized confusion matrix (percentages)."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2, normalize=True)

    assert cm.shape == (2, 2)
    assert cm[0, 0] == pytest.approx(2.0 / 3.0, abs=0.01)
    assert cm[0, 1] == pytest.approx(1.0 / 3.0, abs=0.01)
    assert cm[1, 1] == pytest.approx(1.0, abs=0.01)


def test_compute_confusion_matrix_empty():
    """Test confusion matrix with empty predictions."""
    y_true = np.array([])
    y_pred = np.array([])

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)

    assert cm.shape == (2, 2)
    assert np.all(cm == 0)


def test_compute_confusion_matrix_class_imbalance():
    """Test confusion matrix with severe class imbalance."""
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.array([0] * 98 + [1] * 2)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)

    assert cm.shape == (2, 2)
    assert cm[0, 0] >= 90
    assert cm.sum() == 100


def test_render_confusion_matrix_heatmap():
    """Test heatmap rendering creates PNG file."""
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "confusion_matrix.png"

        cm = np.array([[45, 5], [3, 47]])
        class_names = ["BENIGN", "ATTACK"]

        render_confusion_matrix_heatmap(cm, class_names=class_names, output_path=output_path, normalize=True)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_render_confusion_matrix_multiclass_cic():
    """Test heatmap rendering for CIC-IDS2017 10-class dataset."""
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "confusion_matrix_cic.png"

        cm = np.random.randint(0, 100, size=(10, 10))
        class_names = [
            "BENIGN",
            "DDoS",
            "DoS GoldenEye",
            "DoS Hulk",
            "FTP-Patator",
            "PortScan",
            "SSH-Patator",
            "Web Attack BF",
            "Web Attack SQLi",
            "Web Attack XSS",
        ]

        render_confusion_matrix_heatmap(cm, class_names=class_names, output_path=output_path, normalize=True)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_aggregate_confusion_matrices_multiple_clients():
    """Test aggregation of confusion matrices from multiple clients."""
    cm1 = np.array([[10, 2], [1, 12]])
    cm2 = np.array([[8, 3], [2, 10]])
    cm3 = np.array([[12, 1], [3, 14]])

    cms = [cm1, cm2, cm3]

    aggregated = aggregate_confusion_matrices(cms)

    assert aggregated.shape == (2, 2)
    assert aggregated[0, 0] == 30
    assert aggregated[0, 1] == 6
    assert aggregated[1, 0] == 6
    assert aggregated[1, 1] == 36


def test_aggregate_confusion_matrices_empty():
    """Test aggregation with no confusion matrices."""
    cms = []

    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        aggregate_confusion_matrices(cms)


def test_aggregate_confusion_matrices_mismatched_shapes():
    """Test aggregation rejects mismatched matrix shapes."""
    cm1 = np.array([[10, 2], [1, 12]])
    cm2 = np.array([[8, 3, 1], [2, 10, 0], [1, 0, 5]])

    cms = [cm1, cm2]

    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_confusion_matrices(cms)


def test_confusion_matrix_per_attack_accuracy():
    """Test computing per-attack detection rates from confusion matrix."""
    cm = np.array([[90, 5, 5], [10, 80, 10], [5, 5, 90]])

    diagonal = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_accuracy = diagonal / per_class_total

    assert per_class_accuracy[0] == pytest.approx(0.90, abs=0.01)
    assert per_class_accuracy[1] == pytest.approx(0.80, abs=0.01)
    assert per_class_accuracy[2] == pytest.approx(0.90, abs=0.01)


def test_confusion_matrix_overall_accuracy():
    """Test computing overall accuracy from confusion matrix."""
    cm = np.array([[85, 10, 5], [8, 82, 10], [7, 8, 85]])

    correct = np.diag(cm).sum()
    total = cm.sum()
    accuracy = correct / total

    assert accuracy == pytest.approx(0.84, abs=0.01)


def test_confusion_matrix_csv_export():
    """Test exporting confusion matrix stats to CSV."""
    with TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "confusion_matrix_stats.csv"

        cm = np.array([[90, 5, 5], [10, 80, 10], [5, 5, 90]])
        class_names = ["Class0", "Class1", "Class2"]

        diagonal = np.diag(cm)
        per_class_total = cm.sum(axis=1)
        per_class_accuracy = diagonal / per_class_total

        df = pd.DataFrame(
            {
                "class_name": class_names,
                "true_count": per_class_total,
                "correct_count": diagonal,
                "accuracy": per_class_accuracy,
            }
        )
        df.to_csv(csv_path, index=False)

        assert csv_path.exists()

        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 3
        assert "class_name" in loaded.columns
        assert "accuracy" in loaded.columns


def test_confusion_matrix_handles_unseen_classes():
    """Test confusion matrix handles classes not present in predictions."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1])

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

    assert cm.shape == (3, 3)
    assert cm[2, 2] == 0
    assert cm[2, 1] == 2


def test_confusion_matrix_json_serializable():
    """Test confusion matrix can be saved to JSON for summary.json."""
    cm = np.array([[45, 5], [3, 47]])

    cm_dict = {"confusion_matrix": cm.tolist(), "class_names": ["BENIGN", "ATTACK"]}

    with TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "summary.json"

        with open(json_path, "w") as f:
            json.dump(cm_dict, f)

        assert json_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)

        assert "confusion_matrix" in loaded
        assert len(loaded["confusion_matrix"]) == 2
