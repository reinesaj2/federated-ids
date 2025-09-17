from __future__ import annotations

import os
import csv
import tempfile
import time
from pathlib import Path
from typing import List
import numpy as np
import pytest


def test_client_metrics_csv_creation():
    """Test that client metrics CSV is created with correct headers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics.csv"

        from client_metrics import ClientMetricsLogger

        logger = ClientMetricsLogger(str(metrics_path), client_id=0)

        # Check file exists after creation
        assert metrics_path.exists()

        # Check headers
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected = [
                "client_id", "round", "dataset_size", "n_classes",
                "loss_before", "acc_before", "loss_after", "acc_after",
                "weight_norm_before", "weight_norm_after", "weight_update_norm",
                "t_fit_ms", "epochs_completed", "lr", "batch_size"
            ]
            assert headers == expected


def test_client_metrics_logging_complete_record():
    """Test logging a complete client metrics record."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics.csv"

        from client_metrics import ClientMetricsLogger

        logger = ClientMetricsLogger(str(metrics_path), client_id=5)

        # Log a complete record
        logger.log_round_metrics(
            round_num=3,
            dataset_size=1000,
            n_classes=2,
            loss_before=1.5,
            acc_before=0.6,
            loss_after=0.8,
            acc_after=0.85,
            weight_norm_before=10.5,
            weight_norm_after=12.3,
            weight_update_norm=2.1,
            t_fit_ms=2500.0,
            epochs_completed=5,
            lr=0.01,
            batch_size=32
        )

        # Read and verify
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)

            expected_row = [
                "5", "3", "1000", "2", "1.5", "0.6", "0.8", "0.85",
                "10.5", "12.3", "2.1", "2500.0", "5", "0.01", "32"
            ]
            assert row == expected_row


def test_client_metrics_multiple_rounds():
    """Test logging multiple rounds from same client."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics.csv"

        from client_metrics import ClientMetricsLogger

        logger = ClientMetricsLogger(str(metrics_path), client_id=2)

        # Log multiple rounds
        for round_num in range(1, 4):
            logger.log_round_metrics(
                round_num=round_num,
                dataset_size=500,
                n_classes=3,
                loss_before=2.0 - round_num * 0.2,
                acc_before=0.4 + round_num * 0.1,
                loss_after=1.5 - round_num * 0.2,
                acc_after=0.5 + round_num * 0.1,
                weight_norm_before=8.0,
                weight_norm_after=8.5,
                weight_update_norm=1.0,
                t_fit_ms=1000.0 + round_num * 100,
                epochs_completed=1,
                lr=0.02,
                batch_size=64
            )

        # Verify all rounds logged
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            rows = list(reader)
            assert len(rows) == 3

            # Check round numbers and client ID consistency
            assert [row[1] for row in rows] == ["1", "2", "3"]  # Round numbers
            assert all(row[0] == "2" for row in rows)  # All same client ID


def test_client_metrics_with_none_values():
    """Test handling of None values in client metrics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics.csv"

        from client_metrics import ClientMetricsLogger

        logger = ClientMetricsLogger(str(metrics_path), client_id=7)

        # Log with some None values
        logger.log_round_metrics(
            round_num=1,
            dataset_size=800,
            n_classes=4,
            loss_before=None,  # Not available
            acc_before=None,
            loss_after=0.9,
            acc_after=0.78,
            weight_norm_before=None,
            weight_norm_after=15.2,
            weight_update_norm=None,
            t_fit_ms=3000.0,
            epochs_completed=3,
            lr=0.005,
            batch_size=128
        )

        # Read and verify None values are handled
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            row = next(reader)

            # None values should be empty strings in CSV
            expected_row = [
                "7", "1", "800", "4", "", "", "0.9", "0.78",
                "", "15.2", "", "3000.0", "3", "0.005", "128"
            ]
            assert row == expected_row


def test_client_fit_timer():
    """Test client fit timing utilities."""
    from client_metrics import ClientFitTimer

    timer = ClientFitTimer()

    # Test timing context manager
    with timer.time_fit():
        time.sleep(0.01)  # Simulate 10ms training

    elapsed = timer.get_last_fit_time_ms()
    assert elapsed is not None
    assert 8.0 <= elapsed <= 50.0  # Allow some timing variance


def test_calculate_weight_norms():
    """Test weight norm calculation utilities."""
    from client_metrics import calculate_weight_norms

    # Create sample weight arrays (simulating neural network layers)
    weights = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2x2 weight matrix
        np.array([0.5, -0.5]),  # bias vector
        np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])  # 2x3 output layer
    ]

    norm = calculate_weight_norms(weights)

    # Verify it's a positive number
    assert norm >= 0.0
    assert isinstance(norm, float)

    # Test with zero weights
    zero_weights = [np.zeros((2, 2)), np.zeros(3)]
    zero_norm = calculate_weight_norms(zero_weights)
    assert zero_norm == 0.0


def test_calculate_weight_update_norm():
    """Test weight update norm calculation."""
    from client_metrics import calculate_weight_update_norm

    # Create before and after weights
    weights_before = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, -0.5])
    ]

    weights_after = [
        np.array([[1.1, 2.1], [3.1, 4.1]]),
        np.array([0.6, -0.4])
    ]

    update_norm = calculate_weight_update_norm(weights_before, weights_after)

    # Verify it's a positive number representing the change magnitude
    assert update_norm >= 0.0
    assert isinstance(update_norm, float)

    # Test with identical weights (no update)
    no_update_norm = calculate_weight_update_norm(weights_before, weights_before)
    assert no_update_norm == 0.0


def test_client_metrics_directory_creation():
    """Test that parent directories are created if they don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = Path(temp_dir) / "client_logs" / "experiment_1" / "client_metrics.csv"

        from client_metrics import ClientMetricsLogger

        # Should create parent directories
        logger = ClientMetricsLogger(str(nested_path), client_id=10)

        assert nested_path.exists()
        assert nested_path.parent.exists()


def test_client_data_distribution_analysis():
    """Test client data distribution analysis utilities."""
    from client_metrics import analyze_data_distribution

    # Create sample data
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 0, 1])  # 3 classes

    stats = analyze_data_distribution(labels)

    # Verify structure
    assert "dataset_size" in stats
    assert "n_classes" in stats

    # Verify values
    assert stats["dataset_size"] == 9
    assert stats["n_classes"] == 3

    # Test single class
    single_class_labels = np.array([1, 1, 1, 1])
    single_stats = analyze_data_distribution(single_class_labels)
    assert single_stats["n_classes"] == 1
    assert single_stats["dataset_size"] == 4