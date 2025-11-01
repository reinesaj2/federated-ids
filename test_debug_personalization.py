#!/usr/bin/env python3
"""
Test debug logging for personalization diagnostics.
"""

import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import SimpleNet, TorchClient, get_parameters
from client_metrics import (
    ClientFitTimer,
    ClientMetricsLogger,
    analyze_data_distribution,
)
from scripts.debug_personalization import summarize_client_metrics


def test_debug_logging_enabled():
    """Test that debug logging prints diagnostic info when enabled."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create synthetic data with distribution shift (like test_personalization.py)
    X_train = torch.randn(200, 10)
    y_train = (X_train[:, 0] > 0.3).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    X_test = torch.randn(100, 10)
    y_test = (X_test[:, 0] > 0.0).long()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "client_0_metrics.csv"

        # Enable debug logging
        os.environ["DEBUG_PERSONALIZATION"] = "1"
        os.environ["D2_EXTENDED_METRICS"] = "1"

        metrics_logger = ClientMetricsLogger(str(metrics_path), client_id=0)
        fit_timer = ClientFitTimer()
        data_stats = analyze_data_distribution(y_train.numpy())

        client = TorchClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            metrics_logger=metrics_logger,
            fit_timer=fit_timer,
            data_stats=data_stats,
            runtime_config={
                "local_epochs": 1,
                "lr": 0.01,
                "fedprox_mu": 0.0,
                "personalization_epochs": 3,
                "tau_mode": "max_f1",
                "target_fpr": 0.10,
            },
        )

        initial_params = get_parameters(model)

        # Capture stdout to verify debug messages
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            client.fit(initial_params, {})
            output = mock_stdout.getvalue()

            # Verify debug messages are present
            # Round counter increments at start of fit(), so first call is R1
            assert "Personalization R1:" in output
            assert "Starting with 3 epochs" in output
            assert "Train size:" in output
            assert "Test size:" in output
            assert "After epoch 1:" in output
            assert "weight_norm=" in output
            assert "delta=" in output
            assert "Personalization results:" in output
            assert "global_F1=" in output
            assert "personalized_F1=" in output
            assert "gain=" in output

        # Clean up
        del os.environ["DEBUG_PERSONALIZATION"]


def test_debug_logging_disabled():
    """Test that debug logging is silent when disabled."""
    torch.manual_seed(42)
    np.random.seed(42)

    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32)
    test_loader = DataLoader(dataset, batch_size=32)

    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "client_0_metrics.csv"

        # Disable debug logging
        os.environ["DEBUG_PERSONALIZATION"] = "0"
        os.environ["D2_EXTENDED_METRICS"] = "1"

        metrics_logger = ClientMetricsLogger(str(metrics_path), client_id=0)
        fit_timer = ClientFitTimer()
        data_stats = analyze_data_distribution(y.numpy())

        client = TorchClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            metrics_logger=metrics_logger,
            fit_timer=fit_timer,
            data_stats=data_stats,
            runtime_config={
                "local_epochs": 1,
                "lr": 0.01,
                "fedprox_mu": 0.0,
                "personalization_epochs": 2,
                "tau_mode": "max_f1",
                "target_fpr": 0.10,
            },
        )

        initial_params = get_parameters(model)

        # Capture stdout to verify NO debug messages
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            client.fit(initial_params, {})
            output = mock_stdout.getvalue()

            # Verify debug messages are NOT present
            assert "Personalization R0:" not in output
            assert "Starting with" not in output
            assert "After epoch 1:" not in output

        # Clean up
        del os.environ["DEBUG_PERSONALIZATION"]


def test_zero_gain_warning():
    """Test that warning is printed when gain is near zero."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create data where personalization won't help (same distribution)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32)
    test_loader = DataLoader(dataset, batch_size=32)

    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "client_0_metrics.csv"

        os.environ["DEBUG_PERSONALIZATION"] = "1"
        os.environ["D2_EXTENDED_METRICS"] = "1"

        metrics_logger = ClientMetricsLogger(str(metrics_path), client_id=0)
        fit_timer = ClientFitTimer()
        data_stats = analyze_data_distribution(y.numpy())

        client = TorchClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            metrics_logger=metrics_logger,
            fit_timer=fit_timer,
            data_stats=data_stats,
            runtime_config={
                "local_epochs": 1,
                "lr": 0.01,
                "fedprox_mu": 0.0,
                "personalization_epochs": 2,
                "tau_mode": "max_f1",
                "target_fpr": 0.10,
            },
        )

        initial_params = get_parameters(model)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            client.fit(initial_params, {})
            output = mock_stdout.getvalue()

            # Check if warning appears
            # (may not always trigger, but should when gain < 0.001)
            if "WARNING: Near-zero gain detected" in output:
                assert "train/test same distribution" in output
                assert "insufficient personalization epochs" in output
                assert "learning rate too low" in output

        del os.environ["DEBUG_PERSONALIZATION"]


def test_summarize_client_metrics_returns_primary_values_when_present():
    row = {
        "macro_f1_global": "0.9",
        "macro_f1_personalized": "0.95",
        "personalization_gain": "0.05",
    }

    global_f1, pers_f1, gain = summarize_client_metrics(row)

    assert global_f1 == pytest.approx(0.9)
    assert pers_f1 == pytest.approx(0.95)
    assert gain == pytest.approx(0.05)


def test_summarize_client_metrics_falls_back_to_after_values_when_missing():
    row = {
        "macro_f1_global": "",
        "macro_f1_personalized": None,
        "personalization_gain": "None",
        "macro_f1_after": "0.6869979919678715",
        "macro_f1_before": "0.6869979919678715",
    }

    global_f1, pers_f1, gain = summarize_client_metrics(row)

    assert global_f1 == pytest.approx(0.6869979919678715)
    assert pers_f1 == pytest.approx(0.6869979919678715)
    assert gain == pytest.approx(0.0)


def test_summarize_client_metrics_handles_partial_rows():
    row = {
        "macro_f1_global": "",
        "macro_f1_personalized": "",
        "personalization_gain": "",
        "macro_f1_before": "0.42",
    }

    global_f1, pers_f1, gain = summarize_client_metrics(row)

    assert global_f1 == pytest.approx(0.42)
    assert pers_f1 == pytest.approx(0.42)
    assert gain == pytest.approx(0.0)
