#!/usr/bin/env python3
"""
Unit tests for fairness metrics computation.

Tests the computation of fairness and disparity metrics across federated clients,
including worst/best performance, coefficient of variation, and low FPR fraction.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.summarize_metrics import compute_fairness_metrics


def test_compute_fairness_metrics_worst_best():
    """Compute worst and best client F1."""
    data = {
        "client_id": [0, 0, 1, 1, 2, 2],
        "round": [1, 5, 1, 5, 1, 5],
        "macro_f1_argmax": [0.80, 0.85, 0.65, 0.70, 0.88, 0.90],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert "worst_client_macro_f1_argmax" in result
    assert "best_client_macro_f1_argmax" in result
    assert result["worst_client_macro_f1_argmax"] == pytest.approx(0.70, abs=0.001)
    assert result["best_client_macro_f1_argmax"] == pytest.approx(0.90, abs=0.001)


def test_compute_fairness_metrics_cv():
    """Compute coefficient of variation."""
    data = {
        "client_id": [0, 1, 2, 3],
        "round": [5, 5, 5, 5],
        "macro_f1_argmax": [0.80, 0.85, 0.75, 0.78],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert "cv_macro_f1_argmax" in result
    f1_values = np.array([0.80, 0.85, 0.75, 0.78])
    expected_cv = np.std(f1_values, ddof=0) / np.mean(f1_values)
    assert result["cv_macro_f1_argmax"] == pytest.approx(expected_cv, abs=0.001)


def test_compute_fairness_metrics_low_fpr_fraction():
    """Compute fraction of clients with FPR <= 0.10."""
    data = {
        "client_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "round": [1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5],
        "benign_fpr_argmax": [0.03, 0.05, 0.12, 0.15, 0.08, 0.09, 0.18, 0.20, 0.04, 0.06, 0.25, 0.30],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert "fraction_clients_fpr_le_0_10" in result
    low_fpr_clients = 3
    total_clients = 6
    expected_fraction = low_fpr_clients / total_clients
    assert result["fraction_clients_fpr_le_0_10"] == pytest.approx(expected_fraction, abs=0.001)


def test_compute_fairness_metrics_no_rare_classes():
    """Handle case where no rare classes exist."""
    data = {
        "client_id": [0, 1, 2],
        "round": [5, 5, 5],
        "macro_f1_argmax": [0.85, 0.80, 0.88],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert "rare_class_f1_mean" not in result or result.get("rare_class_f1_mean") is None
    assert "rare_class_f1_min" not in result or result.get("rare_class_f1_min") is None


def test_compute_fairness_metrics_all_clients_equal():
    """Handle case where all clients have identical performance."""
    data = {
        "client_id": [0, 1, 2],
        "round": [5, 5, 5],
        "macro_f1_argmax": [0.85, 0.85, 0.85],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert result["worst_client_macro_f1_argmax"] == pytest.approx(0.85, abs=0.001)
    assert result["best_client_macro_f1_argmax"] == pytest.approx(0.85, abs=0.001)
    assert result["cv_macro_f1_argmax"] == pytest.approx(0.0, abs=0.001)


def test_compute_fairness_metrics_single_client():
    """Handle single client edge case."""
    data = {
        "client_id": [0, 0],
        "round": [1, 5],
        "macro_f1_argmax": [0.80, 0.85],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert result["worst_client_macro_f1_argmax"] == pytest.approx(0.85, abs=0.001)
    assert result["best_client_macro_f1_argmax"] == pytest.approx(0.85, abs=0.001)
    assert result["cv_macro_f1_argmax"] == pytest.approx(0.0, abs=0.001)


def test_compute_fairness_metrics_missing_fpr_column():
    """Handle missing FPR column gracefully."""
    data = {
        "client_id": [0, 1, 2],
        "round": [5, 5, 5],
        "macro_f1_argmax": [0.85, 0.80, 0.88],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert "fraction_clients_fpr_le_0_10" not in result


def test_compute_fairness_metrics_empty_dataframe():
    """Handle empty DataFrame gracefully."""
    df = pd.DataFrame()

    result = compute_fairness_metrics(df)

    assert result == {}


def test_compute_fairness_metrics_missing_macro_f1_column():
    """Handle missing macro_f1_argmax column gracefully."""
    data = {
        "client_id": [0, 1, 2],
        "round": [5, 5, 5],
        "benign_fpr_argmax": [0.05, 0.08, 0.15],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert "worst_client_macro_f1_argmax" not in result
    assert "best_client_macro_f1_argmax" not in result
    assert "cv_macro_f1_argmax" not in result
    assert "fraction_clients_fpr_le_0_10" in result


def test_compute_fairness_metrics_multiple_rounds():
    """Extract final round per client correctly."""
    data = {
        "client_id": [0, 0, 0, 1, 1, 1],
        "round": [1, 3, 5, 1, 3, 5],
        "macro_f1_argmax": [0.70, 0.75, 0.80, 0.65, 0.72, 0.78],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert result["worst_client_macro_f1_argmax"] == pytest.approx(0.78, abs=0.001)
    assert result["best_client_macro_f1_argmax"] == pytest.approx(0.80, abs=0.001)


def test_compute_fairness_metrics_zero_mean_f1():
    """Handle edge case where mean F1 is zero."""
    data = {
        "client_id": [0, 1, 2],
        "round": [5, 5, 5],
        "macro_f1_argmax": [0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)

    result = compute_fairness_metrics(df)

    assert result["cv_macro_f1_argmax"] == 0.0
