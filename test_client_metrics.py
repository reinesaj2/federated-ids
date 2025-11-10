from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def test_client_metrics_csv_creation():
    """Test that client metrics CSV is created with correct headers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics.csv"

        from client_metrics import ClientMetricsLogger

        ClientMetricsLogger(str(metrics_path), client_id=0)

        # Check file exists after creation
        assert metrics_path.exists()

        # Check headers - expect extended headers by default (Issue #77 fix)
        with open(metrics_path) as f:
            reader = csv.reader(f)
            headers = next(reader)

            # After Issue #77, extended metrics are enabled by default
            extended_mode_env = os.environ.get("D2_EXTENDED_METRICS", "1").lower() not in (
                "0",
                "false",
                "no",
                "",
            )

            if extended_mode_env or "D2_EXTENDED_METRICS" not in os.environ:
                expected = [
                    "client_id",
                    "round",
                    "dataset_size",
                    "n_classes",
                    "loss_before",
                    "acc_before",
                    "loss_after",
                    "acc_after",
                    "macro_f1_before",
                    "macro_f1_after",
                "macro_f1_argmax",
                "benign_fpr_argmax",
                "f1_per_class_after",
                "precision_per_class",
                "recall_per_class",
                "confusion_matrix_counts",
                "confusion_matrix_normalized",
                "confusion_matrix_class_names",
                "fpr_after",
                    "pr_auc_after",
                    "threshold_tau",
                    "f1_bin_tau",
                    "benign_fpr_bin_tau",
                    "tau_bin",
                    "seed",
                    "weight_norm_before",
                    "weight_norm_after",
                    "weight_update_norm",
                    "grad_norm_l2",
                    "t_fit_ms",
                    "epochs_completed",
                    "lr",
                    "batch_size",
                    "macro_f1_global",
                    "macro_f1_personalized",
                    "benign_fpr_global",
                    "benign_fpr_personalized",
                    "personalization_gain",
                    "dp_epsilon",
                    "dp_delta",
                    "dp_sigma",
                    "dp_clip_norm",
                    "dp_sample_rate",
                    "dp_total_steps",
                    "dp_enabled",
                    "secure_aggregation",
                    "secure_aggregation_seed",
                    "secure_aggregation_mask_checksum",
                ]
            else:
                expected = [
                    "client_id",
                    "round",
                    "dataset_size",
                    "n_classes",
                    "loss_before",
                    "acc_before",
                    "loss_after",
                    "acc_after",
                    "weight_norm_before",
                    "weight_norm_after",
                    "weight_update_norm",
                    "grad_norm_l2",
                    "t_fit_ms",
                    "epochs_completed",
                    "lr",
                    "batch_size",
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
            grad_norm_l2=1.8,
            t_fit_ms=2500.0,
            epochs_completed=5,
            lr=0.01,
            batch_size=32,
        )

        # Read and verify key values are logged correctly
        with open(metrics_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)

            # Verify basic structure - convert to dict for easier checking
            row_dict = dict(zip(headers, row, strict=False))
            assert row_dict["client_id"] == "5"
            assert row_dict["round"] == "3"
            assert row_dict["dataset_size"] == "1000"
            assert row_dict["n_classes"] == "2"
            assert row_dict["loss_before"] == "1.5"
            assert row_dict["acc_before"] == "0.6"
            assert row_dict["loss_after"] == "0.8"
            assert row_dict["acc_after"] == "0.85"
            assert row_dict["weight_norm_before"] == "10.5"
            assert row_dict["weight_norm_after"] == "12.3"
            assert row_dict["weight_update_norm"] == "2.1"
            assert row_dict["grad_norm_l2"] == "1.8"
            assert row_dict["t_fit_ms"] == "2500.0"
            assert row_dict["epochs_completed"] == "5"
            assert row_dict["lr"] == "0.01"
            assert row_dict["batch_size"] == "32"
            assert "dp_enabled" in row_dict
            assert row_dict.get("secure_aggregation") == "False"


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
                batch_size=64,
            )

        # Verify all rounds logged
        with open(metrics_path) as f:
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
            grad_norm_l2=None,
            t_fit_ms=3000.0,
            epochs_completed=3,
            lr=0.005,
            batch_size=128,
        )

        # Read and verify None values are handled
        with open(metrics_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)

            # None values should be empty strings in CSV - check by column name
            row_dict = dict(zip(headers, row, strict=False))
            assert row_dict["client_id"] == "7"
            assert row_dict["round"] == "1"
            assert row_dict["dataset_size"] == "800"
            assert row_dict["n_classes"] == "4"
            assert row_dict["loss_before"] == ""  # None -> empty string
            assert row_dict["acc_before"] == ""  # None -> empty string
            assert row_dict["loss_after"] == "0.9"
            assert row_dict["acc_after"] == "0.78"
            assert row_dict["weight_norm_before"] == ""  # None -> empty string
            assert row_dict["weight_norm_after"] == "15.2"
            assert row_dict["weight_update_norm"] == ""  # None -> empty string
            assert row_dict["grad_norm_l2"] == ""
            assert row_dict["t_fit_ms"] == "3000.0"
            assert row_dict["epochs_completed"] == "3"
            assert row_dict["lr"] == "0.005"
            assert row_dict["batch_size"] == "128"


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
        np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),  # 2x3 output layer
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
    weights_before = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, -0.5])]

    weights_after = [np.array([[1.1, 2.1], [3.1, 4.1]]), np.array([0.6, -0.4])]

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
        ClientMetricsLogger(str(nested_path), client_id=10)

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


def test_client_metrics_extended_enabled_by_default_issue77():
    """Test Issue #77 fix: Extended metrics are enabled by default (not just when env var set)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics_extended.csv"

        from client_metrics import ClientMetricsLogger

        old_env = os.environ.get("D2_EXTENDED_METRICS")
        try:
            if old_env is not None:
                del os.environ["D2_EXTENDED_METRICS"]

            ClientMetricsLogger(str(metrics_path), client_id=0)

            with open(metrics_path) as f:
                reader = csv.reader(f)
                headers = next(reader)

            expected_f1_columns = [
                "macro_f1_before",
                "macro_f1_after",
                "macro_f1_argmax",
                "f1_per_class_after",
            ]

            for col in expected_f1_columns:
                assert col in headers, f"Column {col} not found in headers. " "Extended metrics should be enabled by default."

        finally:
            if old_env is not None:
                os.environ["D2_EXTENDED_METRICS"] = old_env


def test_client_metrics_extended_explicit_false():
    """Test that extended metrics can still be disabled explicitly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_metrics_basic.csv"

        from client_metrics import ClientMetricsLogger

        ClientMetricsLogger(str(metrics_path), client_id=0, extended=False)

        with open(metrics_path) as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_basic_columns = [
            "client_id",
            "round",
            "loss_before",
            "acc_before",
            "loss_after",
            "acc_after",
        ]

        for col in expected_basic_columns:
            assert col in headers

        unexpected_f1_columns = ["macro_f1_after", "macro_f1_before"]
        for col in unexpected_f1_columns:
            assert col not in headers, f"Column {col} should not be in basic mode, " "but extended=False was set explicitly"


def test_client_metrics_logs_macro_f1_values_issue77():
    """Test Issue #77 fix: Verify macro_f1 values are correctly logged."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "client_f1_test.csv"

        from client_metrics import ClientMetricsLogger

        logger = ClientMetricsLogger(str(metrics_path), client_id=1)

        test_macro_f1_before = 0.75
        test_macro_f1_after = 0.92
        test_macro_f1_argmax = 0.91

        logger.log_round_metrics(
            round_num=1,
            dataset_size=1000,
            n_classes=2,
            loss_before=0.5,
            acc_before=0.7,
            loss_after=0.1,
            acc_after=0.95,
            macro_f1_before=test_macro_f1_before,
            macro_f1_after=test_macro_f1_after,
            macro_f1_argmax=test_macro_f1_argmax,
            weight_norm_before=1.0,
            weight_norm_after=2.0,
        )

        with open(metrics_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)

        f1_before_idx = headers.index("macro_f1_before")
        f1_after_idx = headers.index("macro_f1_after")
        f1_argmax_idx = headers.index("macro_f1_argmax")

        assert float(row[f1_before_idx]) == test_macro_f1_before
        assert float(row[f1_after_idx]) == test_macro_f1_after
        assert float(row[f1_argmax_idx]) == test_macro_f1_argmax
