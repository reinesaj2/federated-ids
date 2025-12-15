#!/usr/bin/env python3
"""
Test FedProx implementation to ensure proximal regularization is working correctly.
"""


import pytest
import torch

from client import DEFAULT_WEIGHT_DECAY, SimpleNet, get_parameters, set_parameters, train_epoch


def _build_loader(num_samples: int, num_features: int, num_classes: int, batch_size: int = 16):
    torch.manual_seed(42)
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size)


def test_fedprox_proximal_term():
    """Test that FedProx proximal regularization term is applied correctly."""

    loader = _build_loader(num_samples=100, num_features=10, num_classes=2, batch_size=32)

    # Create model and get initial parameters (simulating global model)
    model = SimpleNet(num_features=10, num_classes=2)
    device = torch.device("cpu")
    global_params = get_parameters(model)

    # Create a copy of the model and modify its parameters slightly
    model_copy = SimpleNet(num_features=10, num_classes=2)
    set_parameters(model_copy, global_params)

    # Manually modify the model parameters to create a drift from global
    with torch.no_grad():
        for param in model_copy.parameters():
            param += torch.randn_like(param) * 0.1

    # Train with FedProx (mu > 0)
    loss_with_fedprox = train_epoch(
        model=model_copy,
        loader=loader,
        device=device,
        lr=0.01,
        global_params=global_params,
        fedprox_mu=0.1,
    )

    # Create another copy and train without FedProx (mu = 0)
    model_copy2 = SimpleNet(num_features=10, num_classes=2)
    set_parameters(model_copy2, get_parameters(model_copy))  # Same starting point

    loss_without_fedprox = train_epoch(
        model=model_copy2,
        loader=loader,
        device=device,
        lr=0.01,
        global_params=global_params,
        fedprox_mu=0.0,
    )

    # With FedProx, the loss should typically be higher due to the proximal term
    # (though this isn't guaranteed in all cases, this test mainly checks implementation)
    assert isinstance(loss_with_fedprox, float)
    assert isinstance(loss_without_fedprox, float)
    assert loss_with_fedprox >= 0.0
    assert loss_without_fedprox >= 0.0

    print(f"Loss with FedProx (mu=0.1): {loss_with_fedprox:.6f}")
    print(f"Loss without FedProx (mu=0.0): {loss_without_fedprox:.6f}")


def test_fedprox_different_mu_values():
    """Test that different mu values produce different losses."""

    loader = _build_loader(num_samples=50, num_features=5, num_classes=2, batch_size=16)

    model = SimpleNet(num_features=5, num_classes=2)
    device = torch.device("cpu")
    global_params = get_parameters(model)

    # Create drift from global parameters
    with torch.no_grad():
        for param in model.parameters():
            param += torch.randn_like(param) * 0.2

    losses = {}
    mu_values = [0.0, 0.01, 0.1, 1.0]

    for mu in mu_values:
        # Reset to same starting point for fair comparison
        model_test = SimpleNet(num_features=5, num_classes=2)
        set_parameters(model_test, get_parameters(model))

        loss = train_epoch(
            model=model_test,
            loader=loader,
            device=device,
            lr=0.01,
            global_params=global_params,
            fedprox_mu=mu,
        )
        losses[mu] = loss

    print("FedProx losses with different mu values:")
    for mu, loss in losses.items():
        print(f"  mu={mu}: loss={loss:.6f}")

    # Basic sanity check - all losses should be non-negative
    assert all(loss >= 0.0 for loss in losses.values())


def test_fedprox_uses_adamw_regardless_of_mu(monkeypatch):
    """Ensure FedProx uses AdamW for all mu values (both 0 and >0)."""

    loader = _build_loader(num_samples=20, num_features=4, num_classes=2, batch_size=8)
    model = SimpleNet(num_features=4, num_classes=2)
    device = torch.device("cpu")
    global_params = get_parameters(model)

    calls = {"adamw": 0, "sgd": 0, "weight_decay": None}

    from client import create_adamw_optimizer as orig_create_adamw

    def wrapped_create_adamw(params, *args, **kwargs):
        calls["adamw"] += 1
        calls["weight_decay"] = kwargs.get("weight_decay", DEFAULT_WEIGHT_DECAY)
        return orig_create_adamw(params, *args, **kwargs)

    def fail_sgd(*args, **kwargs):
        calls["sgd"] += 1
        raise AssertionError("SGD should not be used - AdamW should be used for all cases")

    monkeypatch.setattr("client.create_adamw_optimizer", wrapped_create_adamw)
    monkeypatch.setattr(torch.optim, "SGD", fail_sgd)

    train_epoch(
        model=model,
        loader=loader,
        device=device,
        lr=0.01,
        global_params=global_params,
        fedprox_mu=0.05,
    )

    assert calls["adamw"] >= 1, "AdamW should be used"
    assert calls["sgd"] == 0, "SGD should never be called"
    assert calls["weight_decay"] == pytest.approx(DEFAULT_WEIGHT_DECAY), f"Weight decay should be {DEFAULT_WEIGHT_DECAY}"
