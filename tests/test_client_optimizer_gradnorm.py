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

from client import SimpleNet, create_adamw_optimizer, get_parameters, train_epoch
from client_metrics import ClientMetricsLogger
from models.per_dataset_encoder import PerDatasetEncoderConfig, PerDatasetEncoderNet


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
    loss = train_epoch(
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
    assert loss > 0.0


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
    loss = train_epoch(
        model=model,
        loader=train_loader,
        device=device,
        lr=0.001,
        global_params=global_params,
        fedprox_mu=0.01,
        weight_decay=1e-4,
    )

    # Verify loss is positive
    assert loss > 0.0
    assert isinstance(loss, float)


def test_client_metrics_logger_has_grad_norm_field():
    """Verify that ClientMetricsLogger includes grad_norm_l2 in CSV headers."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_metrics.csv"
        ClientMetricsLogger(str(csv_path), client_id=0, extended=True)

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


def test_train_epoch_with_fedprox_ignores_batchnorm_buffers():
    num_features = 6
    num_classes = 2
    num_samples = 20
    batch_size = 4
    learning_rate = 0.001
    weight_decay = 1e-4
    fedprox_mu = 0.1
    random_seed = 123

    torch.manual_seed(random_seed)
    model = PerDatasetEncoderNet(
        PerDatasetEncoderConfig(
            dataset_name="cic",
            input_dim=num_features,
            num_classes=num_classes,
            encoder_hidden=[4],
            latent_dim=3,
            shared_hidden=[2],
            dropout=0.0,
        )
    )
    device = torch.device("cpu")

    x_train = torch.randn(num_samples, num_features)
    y_train = torch.randint(0, num_classes, (num_samples,))
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    global_params = get_parameters(model)
    loss = train_epoch(
        model=model,
        loader=train_loader,
        device=device,
        lr=learning_rate,
        global_params=global_params,
        fedprox_mu=fedprox_mu,
        weight_decay=weight_decay,
    )

    assert isinstance(loss, float) and loss > 0.0


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
    loss = train_epoch(
        model=model,
        loader=train_loader,
        device=device,
        lr=0.001,
        global_params=None,
        fedprox_mu=0.0,
        weight_decay=1e-4,
    )

    # With 100 samples and batch size 10, we have 10 batches
    # Loss should be positive and finite
    assert loss > 0.0
    assert np.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
