from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from robust_aggregation import AggregationMethod


def test_server_metrics_csv_creation():
    """Test that server metrics CSV is created with correct headers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "metrics.csv"

        # Import here to avoid circular dependency issues during testing
        from server_metrics import ServerMetricsLogger

        ServerMetricsLogger(str(metrics_path))

        # Check file exists after creation
        assert metrics_path.exists()

        # Check headers
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected = [
                "round",
                "agg_method",
                "n_clients",
                "byzantine_f",
                "l2_to_benign_mean",
                "cos_to_benign_mean",
                "coord_median_agree_pct",
                "update_norm_mean",
                "update_norm_std",
                "t_aggregate_ms",
                "t_round_ms",
                "pairwise_cosine_mean",
                "pairwise_cosine_std",
                "l2_dispersion_mean",
                "l2_dispersion_std",
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
            t_round_ms=1250.0,
        )

        # Read and verify
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            row = next(reader)

            expected_row = [
                "1",
                "krum",
                "5",
                "1",
                "0.1234",
                "0.9876",
                "85.5",
                "1.5",
                "0.25",
                "15.5",
                "1250.0",
                "",
                "",
                "",
                "",  # Dispersion metrics empty
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
                t_round_ms=1000.0 + round_num * 100,
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
            t_round_ms=2000.0,
        )

        # Read and verify None values are handled
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            row = next(reader)

            # None values should be empty strings in CSV
            expected_row = ["1", "median", "2", "", "", "", "", "2.0", "0.5", "20.0", "2000.0", "", "", "", ""]  # Dispersion metrics empty
            assert row == expected_row


def test_server_metrics_directory_creation():
    """Test that parent directories are created if they don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = Path(temp_dir) / "logs" / "experiment_1" / "metrics.csv"

        from server_metrics import ServerMetricsLogger

        # Should create parent directories
        ServerMetricsLogger(str(nested_path))

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


# Cosine similarity validation tests
def test_cosine_similarity_identical_vectors():
    """Test cosine similarity returns 1.0 for identical vectors."""
    from server_metrics import calculate_robustness_metrics

    identical_vec = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    client_updates = [identical_vec, identical_vec]
    benign_mean = identical_vec
    aggregated = identical_vec

    metrics = calculate_robustness_metrics(client_updates, benign_mean, aggregated)

    assert abs(metrics["cos_to_benign_mean"] - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity returns 0.0 for orthogonal vectors."""
    from server_metrics import calculate_robustness_metrics

    vec_a = [np.array([1.0, 0.0])]
    vec_b = [np.array([0.0, 1.0])]
    client_updates = [vec_a, vec_b]
    aggregated = vec_a

    metrics = calculate_robustness_metrics(client_updates, vec_b, aggregated)

    assert abs(metrics["cos_to_benign_mean"] - 0.0) < 1e-6


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity returns -1.0 for opposite direction vectors."""
    from server_metrics import calculate_robustness_metrics

    vec_a = [np.array([1.0, 2.0, 3.0])]
    vec_b = [np.array([-1.0, -2.0, -3.0])]
    client_updates = [vec_a, vec_b]
    aggregated = vec_a

    metrics = calculate_robustness_metrics(client_updates, vec_b, aggregated)

    assert abs(metrics["cos_to_benign_mean"] - (-1.0)) < 1e-6


def test_cosine_similarity_45_degree_angle():
    """Test cosine similarity returns approximately 0.7071 for 45 degree angle."""
    from server_metrics import calculate_robustness_metrics

    vec_a = [np.array([1.0, 0.0])]
    vec_b = [np.array([1.0, 1.0])]  # 45 degrees from vec_a
    client_updates = [vec_a, vec_b]
    aggregated = vec_a

    metrics = calculate_robustness_metrics(client_updates, vec_b, aggregated)

    expected_cosine = 1.0 / np.sqrt(2.0)  # cos(45°) = 0.7071...
    assert abs(metrics["cos_to_benign_mean"] - expected_cosine) < 0.01


def test_cosine_similarity_raises_on_zero_norm_aggregated():
    """Test cosine similarity raises ValueError when aggregated vector has zero norm."""
    from server_metrics import calculate_robustness_metrics

    vec_nonzero = [np.array([1.0, 2.0, 3.0])]
    vec_zero = [np.array([0.0, 0.0, 0.0])]
    client_updates = [vec_nonzero]
    benign_mean = vec_nonzero
    aggregated = vec_zero

    with pytest.raises(ValueError, match="zero norm"):
        calculate_robustness_metrics(client_updates, benign_mean, aggregated)


def test_cosine_similarity_raises_on_zero_norm_benign():
    """Test cosine similarity raises ValueError when benign mean has zero norm."""
    from server_metrics import calculate_robustness_metrics

    vec_nonzero = [np.array([1.0, 2.0, 3.0])]
    vec_zero = [np.array([0.0, 0.0, 0.0])]
    client_updates = [vec_nonzero]
    benign_mean = vec_zero
    aggregated = vec_nonzero

    with pytest.raises(ValueError, match="zero norm"):
        calculate_robustness_metrics(client_updates, benign_mean, aggregated)


def test_cosine_similarity_within_bounds():
    """Test cosine similarity is always within [-1, 1] for random vectors."""
    from server_metrics import calculate_robustness_metrics

    np.random.seed(42)
    for _ in range(10):
        vec_a = [np.random.randn(50)]
        vec_b = [np.random.randn(50)]
        client_updates = [vec_a, vec_b]
        aggregated = vec_a

        metrics = calculate_robustness_metrics(client_updates, vec_b, aggregated)

        assert -1.0 <= metrics["cos_to_benign_mean"] <= 1.0


def test_cosine_similarity_high_for_similar_fl_models():
    """Test cosine similarity is high (>0.9) for similar FL model updates."""
    from server_metrics import calculate_robustness_metrics

    base_vec = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    # Similar vector with small perturbation
    similar_vec = [np.array([1.01, 2.01, 3.01, 4.01, 5.01])]
    client_updates = [base_vec, similar_vec]
    aggregated = base_vec

    metrics = calculate_robustness_metrics(client_updates, similar_vec, aggregated)

    assert metrics["cos_to_benign_mean"] > 0.99


# Metric validation tests
def test_validate_metrics_detects_out_of_bounds_cosine():
    """Test validation detects cosine similarity outside [-1, 1]."""
    from server_metrics import validate_metrics

    invalid_cosine = 1.5  # Outside valid range [-1, 1]
    valid_l2 = 0.1

    metrics = {
        "cos_to_benign_mean": invalid_cosine,
        "l2_to_benign_mean": valid_l2,
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 1
    assert "outside valid range" in warnings[0]
    assert "1.5" in warnings[0]


def test_validate_metrics_detects_suspiciously_low_cosine():
    """Test validation warns on cosine <0.5 for FL models."""
    from server_metrics import validate_metrics

    suspiciously_low_cosine = 0.3  # Below FL threshold of 0.5
    normal_l2 = 0.5

    metrics = {
        "cos_to_benign_mean": suspiciously_low_cosine,
        "l2_to_benign_mean": normal_l2,
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 1
    assert "Suspiciously low cosine" in warnings[0]
    assert "0.3" in warnings[0]


def test_validate_metrics_detects_negative_l2():
    """Test validation detects negative L2 distance."""
    from server_metrics import validate_metrics

    typical_cosine = 0.99  # Normal for similar FL models
    negative_l2 = -0.1  # Physically impossible

    metrics = {
        "cos_to_benign_mean": typical_cosine,
        "l2_to_benign_mean": negative_l2,
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 1
    assert "Negative L2" in warnings[0]


def test_validate_metrics_warns_on_l2_zero():
    """Test validation warns when L2 distance is exactly 0.0."""
    from server_metrics import validate_metrics

    metrics = {
        "cos_to_benign_mean": 1.0,
        "l2_to_benign_mean": 0.0,
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 1
    assert "exactly 0.0" in warnings[0]


def test_validate_metrics_passes_valid_fl_metrics():
    """Test validation passes for typical valid FL metrics."""
    from server_metrics import validate_metrics

    typical_fl_cosine = 0.98  # High similarity for same-architecture models
    typical_l2_distance = 0.25  # Small drift from reference
    typical_norm_mean = 5.0  # Average update magnitude
    typical_norm_std = 0.5  # Low variance across clients

    metrics = {
        "cos_to_benign_mean": typical_fl_cosine,
        "l2_to_benign_mean": typical_l2_distance,
        "update_norm_mean": typical_norm_mean,
        "update_norm_std": typical_norm_std,
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 0


def test_validate_metrics_handles_none_values():
    """Test validation skips None values without error."""
    from server_metrics import validate_metrics

    metrics = {
        "cos_to_benign_mean": None,
        "l2_to_benign_mean": None,
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 0


def test_validate_metrics_handles_nan_values():
    """Test validation skips NaN values without error."""
    from server_metrics import validate_metrics

    metrics = {
        "cos_to_benign_mean": float("nan"),
        "l2_to_benign_mean": float("nan"),
    }

    warnings = validate_metrics(metrics, "test_dimension")

    assert len(warnings) == 0
