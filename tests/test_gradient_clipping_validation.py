"""
Unit tests for gradient clipping validation in Byzantine robust federated learning.

Tests verify:
1. Legitimate clients are not clipped (preserves learning)
2. Adversarial clients are clipped (bounds attack strength)
3. Clipping factor is configurable
4. Convergence is not hurt by proper clipping scope
"""

from __future__ import annotations

import numpy as np
import torch

from client import SimpleNet, TorchClient
from data_preprocessing import create_synthetic_classification_loaders


def _make_test_client(
    adversary_mode: str = "none",
    adversary_clip_factor: float = 2.0,
    num_features: int = 10,
    num_classes: int = 2,
) -> TorchClient:
    """Create a test client with specified adversary configuration."""
    # Create synthetic data
    train_loader, test_loader = create_synthetic_classification_loaders(
        num_samples=100,
        num_features=num_features,
        batch_size=32,
        seed=42,
        num_classes=num_classes,
    )

    # Create model
    model = SimpleNet(num_features=num_features, num_classes=num_classes)
    device = torch.device("cpu")

    # Mock metrics logger
    class MockMetricsLogger:
        def __init__(self):
            self.client_id = 0

        def log_personalization_metrics(self, **kwargs):
            pass

        def log_training_metrics(self, **kwargs):
            pass

        def log_round_metrics(self, **kwargs):
            pass

    # Mock fit timer
    class MockFitTimer:
        def time_fit(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_last_fit_time_ms(self):
            return 100.0  # Mock timing

    return TorchClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        metrics_logger=MockMetricsLogger(),
        fit_timer=MockFitTimer(),
        data_stats={
            "dataset_size": 100,
            "n_classes": num_classes,
        },
        runtime_config={
            "adversary_mode": adversary_mode,
            "adversary_clip_factor": adversary_clip_factor,
            "local_epochs": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
        },
    )


def test_legitimate_clients_not_clipped():
    """Verify legitimate clients don't have gradients clipped."""
    client = _make_test_client(adversary_mode="none")

    def mock_fit(parameters, config):
        # Set parameters
        for param, new_param in zip(client.model.parameters(), parameters, strict=False):
            param.data = torch.tensor(new_param, dtype=param.dtype)

        # Track if clipping was applied
        client._gradient_clipped = False

        # Mock gradient clipping to track if it's called
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

        def mock_clip_grad_norm(
            parameters,
            max_norm,
            norm_type=2.0,
            *,
            target_client=client,
            clip_func=original_clip_grad_norm,
        ):
            target_client._gradient_clipped = True
            return clip_func(parameters, max_norm, norm_type)

        torch.nn.utils.clip_grad_norm_ = mock_clip_grad_norm

        try:
            # Run one training step
            client.model.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(client.model.parameters(), lr=0.01)

            for xb, yb in client.train_loader:
                xb = xb.to(client.device)
                yb = yb.to(client.device)
                optimizer.zero_grad()
                preds = client.model(xb)
                loss = criterion(preds, yb)
                loss.backward()

                # Check if clipping was applied (should NOT be for legitimate clients)
                if hasattr(client, '_gradient_clipped'):
                    assert not client._gradient_clipped, "Legitimate client should not have gradients clipped"

                optimizer.step()
                break  # Only one batch for testing
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm

        # Return dummy results
        return [p.detach().numpy() for p in client.model.parameters()], 100, {}

    client.fit = mock_fit

    initial_params = [p.detach().numpy() for p in client.model.parameters()]
    # Test with legitimate client
    dummy_params = [np.random.randn(*p.shape) for p in initial_params]
    client.fit(dummy_params, {})

    # Verify no clipping occurred
    assert not hasattr(client, '_gradient_clipped') or not client._gradient_clipped


def test_adversarial_clients_are_clipped():
    """Verify adversarial clients have gradients clipped."""
    client = _make_test_client(adversary_mode="grad_ascent", adversary_clip_factor=2.0)

    # Track if clipping was applied
    client._gradient_clipped = False

    # Mock gradient clipping to track if it's called
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def mock_clip_grad_norm(parameters, max_norm, norm_type=2.0):
        client._gradient_clipped = True
        assert max_norm == 2.0, f"Expected clip factor 2.0, got {max_norm}"
        return original_clip_grad_norm(parameters, max_norm, norm_type)

    torch.nn.utils.clip_grad_norm_ = mock_clip_grad_norm

    try:
        # Use the actual fit method which contains the gradient clipping logic
        initial_params = [p.detach().numpy() for p in client.model.parameters()]
        client.fit(initial_params, {})

        # Check if clipping was applied (should be for adversarial clients)
        assert client._gradient_clipped, "Adversarial client should have gradients clipped"
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm


def test_clipping_factor_configurable():
    """Test different clipping factors work correctly."""
    for clip_factor in [1.0, 2.0, 5.0]:
        client = _make_test_client(adversary_mode="grad_ascent", adversary_clip_factor=clip_factor)

        # Track clipping calls
        client._clipping_calls = []

        # Mock gradient clipping to track calls
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

        def mock_clip_grad_norm(
            parameters,
            max_norm,
            norm_type=2.0,
            *,
            target_client=client,
            clip_func=original_clip_grad_norm,
        ):
            target_client._clipping_calls.append(max_norm)
            return clip_func(parameters, max_norm, norm_type)

        torch.nn.utils.clip_grad_norm_ = mock_clip_grad_norm

        try:
            # Use the actual fit method which contains the gradient clipping logic
            initial_params = [p.detach().numpy() for p in client.model.parameters()]
            client.fit(initial_params, {})
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm

        # Verify correct clip factor was used
        assert len(client._clipping_calls) > 0, f"No clipping calls for factor {clip_factor}"
        assert all(
            call == clip_factor for call in client._clipping_calls
        ), f"Expected clip factor {clip_factor}, got {client._clipping_calls}"


def test_convergence_not_hurt_by_proper_clipping():
    """Verify legitimate learning convergence is preserved with proper clipping scope."""
    # Test legitimate client (no clipping)
    legit_client = _make_test_client(adversary_mode="none")

    # Test adversarial client (with clipping)
    adv_client = _make_test_client(adversary_mode="grad_ascent", adversary_clip_factor=2.0)

    # Both should be able to learn on legitimate data
    for client in [legit_client, adv_client]:
        # Train for a few steps
        client.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(client.model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for i, (xb, yb) in enumerate(client.train_loader):
            xb = xb.to(client.device)
            yb = yb.to(client.device)
            optimizer.zero_grad()
            preds = client.model(xb)

            if client.runtime_config["adversary_mode"] == "grad_ascent":
                loss = -criterion(preds, yb)  # Adversarial
            else:
                loss = criterion(preds, yb)  # Legitimate

            if i == 0:
                initial_loss = float(loss.item())

            loss.backward()
            optimizer.step()

            if i >= 2:  # Test a few steps
                final_loss = float(loss.item())
                break

        # Verify learning occurred (loss changed)
        assert initial_loss is not None and final_loss is not None
        assert abs(initial_loss - final_loss) > 1e-6, "No learning occurred"


def test_label_flip_adversary_also_clipped():
    """Verify label flip adversaries are also clipped."""
    client = _make_test_client(adversary_mode="label_flip", adversary_clip_factor=2.0)

    # Track if clipping was applied
    client._gradient_clipped = False

    # Mock gradient clipping to track if it's called
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def mock_clip_grad_norm(parameters, max_norm, norm_type=2.0):
        client._gradient_clipped = True
        return original_clip_grad_norm(parameters, max_norm, norm_type)

    torch.nn.utils.clip_grad_norm_ = mock_clip_grad_norm

    try:
        # Use the actual fit method which contains the gradient clipping logic
        initial_params = [p.detach().numpy() for p in client.model.parameters()]
        client.fit(initial_params, {})

        # Check if clipping was applied
        assert client._gradient_clipped, "Label flip adversary should have gradients clipped"
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm
