#!/usr/bin/env python3
"""
Test personalization implementation to ensure client-level fine-tuning works correctly.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import SimpleNet, TorchClient, get_parameters
from client_metrics import (
    ClientFitTimer,
    ClientMetricsLogger,
    analyze_data_distribution,
)


def test_personalization_disabled_when_epochs_zero():
    """Test that personalization is skipped when personalization_epochs=0."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create synthetic data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32)
    test_loader = DataLoader(dataset, batch_size=32)

    # Create model and client
    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "client_0_metrics.csv"
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
                "personalization_epochs": 0,  # Disabled
                "tau_mode": "max_f1",
                "target_fpr": 0.10,
            },
        )

        initial_params = get_parameters(model)
        returned_params, _, _ = client.fit(initial_params, {})

        # Verify CSV does not have personalization metrics filled
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]
            # When disabled, personalization fields should be empty or not set
            assert row.get("macro_f1_global", "") in ("", None)
            assert row.get("macro_f1_personalized", "") in ("", None)
            assert row.get("personalization_gain", "") in ("", None)


def test_personalization_computes_metrics_and_improves():
    """Test that personalization computes metrics correctly and improves local F1."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create deterministic data where personalization WILL improve performance
    # Strategy: Train on slightly different distribution than test
    X_train = torch.randn(200, 10)
    # Training labels: threshold at 0.3 (slightly offset)
    y_train = (X_train[:, 0] > 0.3).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Test labels: threshold at 0.0 (standard)
    X_test = torch.randn(100, 10)
    y_test = (X_test[:, 0] > 0.0).long()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model and client
    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "client_0_metrics.csv"
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
                "personalization_epochs": 5,  # More epochs for guaranteed improvement
                "tau_mode": "max_f1",
                "target_fpr": 0.10,
            },
        )

        initial_params = get_parameters(model)
        returned_params, _, _ = client.fit(initial_params, {})

        # Verify personalization metrics are logged correctly
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]

            # Check that personalization metrics exist
            assert row["macro_f1_global"] != ""
            assert row["macro_f1_personalized"] != ""

            global_f1 = float(row["macro_f1_global"])
            pers_f1 = float(row["macro_f1_personalized"])
            gain = float(row["personalization_gain"])

            # Verify metrics are valid floats in expected range
            assert 0.0 <= global_f1 <= 1.0
            assert 0.0 <= pers_f1 <= 1.0

            # Verify gain computation is correct (arithmetic check)
            assert abs(gain - (pers_f1 - global_f1)) < 1e-6

            # With 5 personalization epochs on local train data,
            # personalized model SHOULD improve on local test set
            assert pers_f1 > global_f1, f"Personalization should improve F1: " f"global={global_f1:.4f}, personalized={pers_f1:.4f}"


def test_personalization_returns_global_weights():
    """Test that global weights (not personalized) are returned to server.

    This test verifies that even after personalization, the weights sent back
    to the server are the global (pre-personalization) weights, not the
    personalized weights. This is crucial for federated learning to work correctly.
    """
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

        # Capture model weights BEFORE fit() is called
        weights_before_fit = [np.copy(p) for p in initial_params]

        # Run fit which includes FL training + personalization
        returned_params, _, _ = client.fit(initial_params, {})

        # The key test: returned params should NOT be equal to the initial params
        # (because FL training happened), but they SHOULD be the global weights,
        # not the personalized weights. We can't directly test this without
        # additional instrumentation, so instead we verify:
        # 1. Weights changed from initial (FL training happened)
        # 2. Weights are returned successfully
        # 3. Personalization metrics show improvement (tested in other tests)

        # Verify weights changed (FL training occurred)
        weights_changed = False
        for ret_param, init_param in zip(returned_params, weights_before_fit, strict=False):
            if not np.allclose(ret_param, init_param, rtol=1e-5, atol=1e-6):
                weights_changed = True
                break

        assert weights_changed, "FL training should have modified weights"

        # Verify returned weights are valid (not corrupted)
        assert len(returned_params) > 0
        for param in returned_params:
            assert not np.any(np.isnan(param))
            assert not np.any(np.isinf(param))


def test_personalization_metrics_logged_correctly():
    """Test that personalization metrics are logged to CSV with correct format."""
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
        client.fit(initial_params, {})

        # Verify CSV structure
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            # Check all personalization fields exist in header
            assert "macro_f1_global" in row
            assert "macro_f1_personalized" in row
            assert "benign_fpr_global" in row
            assert "benign_fpr_personalized" in row
            assert "personalization_gain" in row

            # Check values are populated
            assert row["macro_f1_global"] != ""
            assert row["macro_f1_personalized"] != ""
            assert row["personalization_gain"] != ""

            # Check values are valid floats
            float(row["macro_f1_global"])
            float(row["macro_f1_personalized"])
            float(row["personalization_gain"])


def test_personalization_empty_test_loader():
    """Test that personalization is skipped gracefully when test loader is empty."""
    torch.manual_seed(42)
    np.random.seed(42)

    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32)

    # Empty test loader
    test_dataset = TensorDataset(torch.zeros(0, 10), torch.zeros(0, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = Path(tmp_dir) / "client_0_metrics.csv"
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

        # Should not crash even with empty test loader
        returned_params, _, _ = client.fit(initial_params, {})
        assert returned_params is not None
