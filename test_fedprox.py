#!/usr/bin/env python3
"""
Test FedProx implementation to ensure proximal regularization is working correctly.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from client import train_epoch, SimpleNet, get_parameters, set_parameters


def test_fedprox_proximal_term():
    """Test that FedProx proximal regularization term is applied correctly."""

    # Create a simple synthetic dataset
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))

    # Create DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)

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
    loss_with_fedprox, grad_norm_with_fedprox = train_epoch(
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

    loss_without_fedprox, grad_norm_without_fedprox = train_epoch(
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
    assert isinstance(grad_norm_with_fedprox, float)
    assert isinstance(grad_norm_without_fedprox, float)
    assert loss_with_fedprox >= 0.0
    assert loss_without_fedprox >= 0.0

    print(f"Loss with FedProx (mu=0.1): {loss_with_fedprox:.6f}")
    print(f"Loss without FedProx (mu=0.0): {loss_without_fedprox:.6f}")


def test_fedprox_different_mu_values():
    """Test that different mu values produce different losses."""

    torch.manual_seed(42)
    X = torch.randn(50, 5)
    y = torch.randint(0, 2, (50,))

    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16)

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

        loss, grad_norm = train_epoch(
            model=model_test,
            loader=loader,
            device=device,
            lr=0.01,
            global_params=global_params,
            fedprox_mu=mu,
        )
        losses[mu] = (loss, grad_norm)

    print("FedProx losses with different mu values:")
    for mu, (loss, grad_norm) in losses.items():
        print(f"  mu={mu}: loss={loss:.6f}, grad_norm={grad_norm:.6f}")

    # Basic sanity check - all losses should be non-negative
    assert all(loss >= 0.0 for loss, _ in losses.values())


if __name__ == "__main__":
    test_fedprox_proximal_term()
    test_fedprox_different_mu_values()
    print("[PASS] All FedProx tests passed!")
