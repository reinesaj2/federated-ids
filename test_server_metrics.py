from __future__ import annotations

import os
import csv
import tempfile
import time
from pathlib import Path
from typing import List
import numpy as np
import pytest

from robust_aggregation import AggregationMethod


def test_server_metrics_csv_creation():
    """Test that server metrics CSV is created with correct headers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "metrics.csv"

        # Import here to avoid circular dependency issues during testing
        from server_metrics import ServerMetricsLogger

        logger = ServerMetricsLogger(str(metrics_path))

        # Check file exists after creation
        assert metrics_path.exists()

        # Check headers
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected = [
                "round", "agg_method", "n_clients", "byzantine_f",
                "l2_to_benign_mean", "cos_to_benign_mean", "coord_median_agree_pct",
                "update_norm_mean", "update_norm_std", "t_aggregate_ms", "t_round_ms",
                "pairwise_cosine_mean", "pairwise_cosine_std", "l2_dispersion_mean", "l2_dispersion_std"
            ]
            assert headers == expected


def test_server_metrics_logging_complete_record():
    """Test logging a complete metrics record."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "metrics.csv"

        from server_metrics import ServerMetricsLogger

        logger = ServerMetricsLogger(str(metrics_path))

        # Log a complete record
        logger.log_round_metrics(
            round_num=1,
            agg_method=AggregationMethod.KRUM,
            n_clients=5,
            byzantine_f=1,
            l2_to_benign_mean=0.1234,
            cos_to_benign_mean=0.9876,
            coord_median_agree_pct=85.5,
            update_norm_mean=1.5,
            update_norm_std=0.25,
            t_aggregate_ms=15.5,
            t_round_ms=1250.0
        )

        # Read and verify
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)

            expected_row = [
                "1", "krum", "5", "1", "0.1234", "0.9876", "85.5",
                "1.5", "0.25", "15.5", "1250.0", "", "", "", ""
            ]
            assert row == expected_row


def test_server_metrics_multiple_rounds():
    """Test logging multiple rounds."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "metrics.csv"

        from server_metrics import ServerMetricsLogger

        logger = ServerMetricsLogger(str(metrics_path))

        # Log multiple rounds
        for round_num in range(1, 4):
            logger.log_round_metrics(
                round_num=round_num,
                agg_method=AggregationMethod.FED_AVG,
                n_clients=3,
                byzantine_f=0,
                l2_to_benign_mean=0.1 * round_num,
                cos_to_benign_mean=0.9,
                coord_median_agree_pct=90.0,
                update_norm_mean=1.0,
                update_norm_std=0.1,
                t_aggregate_ms=10.0,
                t_round_ms=1000.0 + round_num * 100
            )

        # Verify all rounds logged
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            rows = list(reader)
            assert len(rows) == 3

            # Check round numbers
            assert [row[0] for row in rows] == ["1", "2", "3"]
            # Check increasing l2 distances (handle floating point precision)
            l2_values = [float(row[4]) for row in rows]
            expected_l2 = [0.1, 0.2, 0.3]
            assert len(l2_values) == len(expected_l2)
            for actual, expected in zip(l2_values, expected_l2):
                assert abs(actual - expected) < 1e-10


def test_server_metrics_with_none_values():
    """Test handling of None values in metrics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "metrics.csv"

        from server_metrics import ServerMetricsLogger

        logger = ServerMetricsLogger(str(metrics_path))

        # Log with some None values
        logger.log_round_metrics(
            round_num=1,
            agg_method=AggregationMethod.MEDIAN,
            n_clients=2,
            byzantine_f=None,  # Auto-determined
            l2_to_benign_mean=None,  # Not available
            cos_to_benign_mean=None,
            coord_median_agree_pct=None,
            update_norm_mean=2.0,
            update_norm_std=0.5,
            t_aggregate_ms=20.0,
            t_round_ms=2000.0
        )

        # Read and verify None values are handled
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            row = next(reader)

            # None values should be empty strings in CSV
            expected_row = [
                "1", "median", "2", "", "", "", "",
                "2.0", "0.5", "20.0", "2000.0", "", "", "", ""
            ]
            assert row == expected_row


def test_server_metrics_directory_creation():
    """Test that parent directories are created if they don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = Path(temp_dir) / "logs" / "experiment_1" / "metrics.csv"

        from server_metrics import ServerMetricsLogger

        # Should create parent directories
        logger = ServerMetricsLogger(str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()


def test_aggregation_timing_measurement():
    """Test timing measurement utilities."""
    from server_metrics import AggregationTimer

    timer = AggregationTimer()

    # Test timing context manager
    with timer.time_aggregation():
        time.sleep(0.01)  # Simulate 10ms work

    elapsed = timer.get_last_aggregation_time_ms()
    assert 8.0 <= elapsed <= 50.0  # Allow some timing variance


def test_robust_metrics_calculation():
    """Test calculation of robustness metrics."""
    from server_metrics import calculate_robustness_metrics

    # Create synthetic client updates and benign mean
    client_updates = [
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],
        [np.array([1.1, 2.1, 3.1]), np.array([4.1, 5.1])],  # Close to benign
        [np.array([10.0, 20.0, 30.0]), np.array([40.0, 50.0])],  # Outlier
    ]
    benign_mean = [np.array([1.05, 2.05, 3.05]), np.array([4.05, 5.05])]
    aggregated = [np.array([1.2, 2.2, 3.2]), np.array([4.2, 5.2])]

    metrics = calculate_robustness_metrics(client_updates, benign_mean, aggregated)

    # Verify metrics structure
    assert "l2_to_benign_mean" in metrics
    assert "cos_to_benign_mean" in metrics
    assert "coord_median_agree_pct" in metrics
    assert "update_norm_mean" in metrics
    assert "update_norm_std" in metrics

    # Verify reasonable values
    assert metrics["l2_to_benign_mean"] >= 0.0
    assert -1.0 <= metrics["cos_to_benign_mean"] <= 1.0
    assert 0.0 <= metrics["coord_median_agree_pct"] <= 100.0
    assert metrics["update_norm_mean"] >= 0.0
    assert metrics["update_norm_std"] >= 0.0