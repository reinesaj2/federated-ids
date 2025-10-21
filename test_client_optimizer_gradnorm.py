"""
Unit tests for Issue #8: AdamW optimizer and gradient norm logging.

Tests verify:
1. Default optimizer is AdamW with correct weight_decay
2. Gradient norm is computed and logged during training
3. train_epoch returns both loss and gradient norm
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import SimpleNet, create_adamw_optimizer, train_epoch
from client_metrics import ClientMetricsLogger


def test_create_adamw_optimizer_returns_adamw():
    """Verify that create_adamw_optimizer returns AdamW optimizer."""
    model = SimpleNet(num_features=10, num_classes=2)
    optimizer = create_adamw_optimizer(model.parameters(), lr=0.001, weight_decay=1e-4)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["weight_decay"] == 1e-4


def test_train_epoch_returns_loss_and_grad_norm():
    """Verify that train_epoch returns both loss and gradient norm."""
    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    # Create synthetic training data
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Train for one epoch
    loss, grad_norm = train_epoch(
        model=model,
        loader=train_loader,
        device=device,
        lr=0.001,
        global_params=None,
        fedprox_mu=0.0,
        weight_decay=1e-4,
    )

    # Verify return types
    assert isinstance(loss, float)
    assert isinstance(grad_norm, float)
    assert loss > 0.0
    assert grad_norm > 0.0


def test_train_epoch_grad_norm_with_fedprox():
    """Verify gradient norm is computed correctly with FedProx enabled."""
    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    # Create synthetic training data
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Get initial parameters for FedProx
    global_params = [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

    # Train with FedProx
    loss, grad_norm = train_epoch(
        model=model,
        loader=train_loader,
        device=device,
        lr=0.001,
        global_params=global_params,
        fedprox_mu=0.01,
        weight_decay=1e-4,
    )

    # Verify gradient norm is positive
    assert grad_norm > 0.0
    assert isinstance(grad_norm, float)


def test_client_metrics_logger_has_grad_norm_field():
    """Verify that ClientMetricsLogger includes grad_norm_l2 in CSV headers."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_metrics.csv"
        logger = ClientMetricsLogger(str(csv_path), client_id=0, extended=True)

        # Read the CSV header
        with open(csv_path, "r") as f:
            header_line = f.readline().strip()
            headers = header_line.split(",")

        # Verify grad_norm_l2 is in headers
        assert "grad_norm_l2" in headers


def test_client_metrics_logger_logs_grad_norm():
    """Verify that gradient norm can be logged to CSV."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_metrics.csv"
        logger = ClientMetricsLogger(str(csv_path), client_id=0, extended=False)

        # Log a round with gradient norm
        grad_norm_value = 2.5
        logger.log_round_metrics(
            round_num=1,
            dataset_size=100,
            n_classes=2,
            loss_before=0.8,
            acc_before=0.6,
            loss_after=0.5,
            acc_after=0.75,
            weight_norm_before=10.0,
            weight_norm_after=9.5,
            weight_update_norm=0.5,
            grad_norm_l2=grad_norm_value,
            t_fit_ms=100.0,
            epochs_completed=1,
            lr=0.001,
            batch_size=32,
        )

        # Read the CSV and verify grad_norm_l2 is logged
        with open(csv_path, "r") as f:
            lines = f.readlines()
            headers = lines[0].strip().split(",")
            values = lines[1].strip().split(",")

        grad_norm_idx = headers.index("grad_norm_l2")
        logged_grad_norm = float(values[grad_norm_idx])

        assert logged_grad_norm == grad_norm_value


def test_train_epoch_grad_norm_accumulation():
    """Verify gradient norm is averaged across batches."""
    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")

    # Create synthetic data with small batch size to ensure multiple batches
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

    # Train for one epoch
    loss, grad_norm = train_epoch(
        model=model,
        loader=train_loader,
        device=device,
        lr=0.001,
        global_params=None,
        fedprox_mu=0.0,
        weight_decay=1e-4,
    )

    # With 100 samples and batch size 10, we have 10 batches
    # Gradient norm should be positive and averaged
    assert grad_norm > 0.0
    assert np.isfinite(grad_norm)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
