from __future__ import annotations

import numpy as np

from robust_aggregation import (
    AggregationMethod,
    aggregate_weights,
)


def _make_client_updates(n_clients: int = 5, n_layers: int = 3, shape=(4,)):
    rng = np.random.default_rng(42)
    clients = []
    for i in range(n_clients):
        layers = []
        for _ in range(n_layers):
            layers.append(rng.normal(0, 1, size=shape))
        clients.append(layers)
    return clients


def test_median_robust_to_outliers_coordinatewise():
    clients = _make_client_updates(n_clients=7)
    # Inject a strong outlier on client 0
    for l in range(len(clients[0])):
        clients[0][l] = clients[0][l] + 1000.0
    agg = aggregate_weights(clients, AggregationMethod.MEDIAN)
    # Median should not be ~1000 shifted; check it's finite and near typical scale
    vals = np.concatenate([a.reshape(-1) for a in agg])
    assert float(np.abs(vals).mean()) < 10.0


def test_krum_selects_single_reasonable_candidate():
    clients = _make_client_updates(n_clients=6)
    # Make one malicious far-away
    for l in range(len(clients[1])):
        clients[1][l] = clients[1][l] + 50.0
    agg = aggregate_weights(clients, AggregationMethod.KRUM)
    # Krum returns a single candidate's update; verify shapes match and values are finite
    assert len(agg) == len(clients[0])
    assert all(a.shape == c.shape for a, c in zip(agg, clients[0]))
    vals = np.concatenate([a.reshape(-1) for a in agg])
    assert np.isfinite(vals).all()


def test_bulyan_behaves_reasonably_with_outliers():
    clients = _make_client_updates(n_clients=11)  # Bulyan requires n >= 4f+3 (for f=2, need 11)
    # Two outliers
    for l in range(len(clients[2])):
        clients[2][l] = clients[2][l] - 50.0
    for l in range(len(clients[5])):
        clients[5][l] = clients[5][l] + 50.0
    agg = aggregate_weights(clients, AggregationMethod.BULYAN)
    vals = np.concatenate([a.reshape(-1) for a in agg])
    assert np.isfinite(vals).all()
    # Not excessively dominated by outliers
    assert float(np.abs(vals).mean()) < 10.0


def test_krum_and_bulyan_accept_explicit_byzantine_f():
    clients = _make_client_updates(n_clients=7)
    for l in range(len(clients[0])):
        clients[0][l] = clients[0][l] + 80.0
    agg1 = aggregate_weights(clients, AggregationMethod.KRUM, byzantine_f=1)
    agg2 = aggregate_weights(clients, AggregationMethod.BULYAN, byzantine_f=1)
    v1 = np.concatenate([a.reshape(-1) for a in agg1])
    v2 = np.concatenate([a.reshape(-1) for a in agg2])
    assert np.isfinite(v1).all() and np.isfinite(v2).all()


def test_krum_selects_honest_with_bounded_adversaries():
    """Test Krum with 9 honest clients + 2 bounded adversaries (15% Byzantine)."""
    clients = _make_client_updates(n_clients=11, shape=(10,))

    # First 9 are honest (small, coordinated gradients)
    for i in range(9):
        for layer_idx in range(len(clients[i])):
            clients[i][layer_idx] = np.ones(10) * 0.1

    # Clients 9-10 are bounded adversaries (large but not extreme)
    for i in range(9, 11):
        for layer_idx in range(len(clients[i])):
            clients[i][layer_idx] = np.ones(10) * 5.0  # 50x honest magnitude

    agg = aggregate_weights(clients, AggregationMethod.KRUM, byzantine_f=2)

    # Verify result is closer to honest (0.1) than adversary (5.0)
    vals = np.concatenate([a.reshape(-1) for a in agg])
    mean_val = float(np.abs(vals).mean())

    # Should be close to honest updates, not adversary updates
    assert 0.05 < mean_val < 1.0, f"Krum result {mean_val} not in honest range"


def test_median_ignores_bounded_outliers_coordinatewise():
    """Test Median coordinate-wise with 7 honest + 2 bounded adversaries."""
    clients = _make_client_updates(n_clients=9, shape=(8,))

    # First 7 are honest
    for i in range(7):
        for layer_idx in range(len(clients[i])):
            clients[i][layer_idx] = np.ones(8) * 0.2

    # Last 2 are adversaries with clipped gradients
    for i in range(7, 9):
        for layer_idx in range(len(clients[i])):
            clients[i][layer_idx] = np.ones(8) * 3.0  # 15x honest magnitude

    agg = aggregate_weights(clients, AggregationMethod.MEDIAN)

    # Median should select majority (honest) value at each coordinate
    vals = np.concatenate([a.reshape(-1) for a in agg])
    mean_val = float(np.median(np.abs(vals)))

    # Should be close to honest value, not adversary value
    assert 0.1 < mean_val < 1.0, f"Median result {mean_val} not in honest range"


def test_bulyan_byzantine_resilience_with_bounded_attacks():
    """Test Bulyan with n=11, f=2 Byzantine tolerance and realistic bounded attacks."""
    clients = _make_client_updates(n_clients=11, shape=(6,))

    # First 9 honest clients
    for i in range(9):
        for layer_idx in range(len(clients[i])):
            clients[i][layer_idx] = np.ones(6) * 0.15

    # 2 Byzantine clients with bounded gradients (not extreme)
    for i in range(9, 11):
        for layer_idx in range(len(clients[i])):
            clients[i][layer_idx] = np.ones(6) * 2.5  # 16x honest magnitude

    agg = aggregate_weights(clients, AggregationMethod.BULYAN, byzantine_f=2)

    # Verify Bulyan produces reasonable output
    vals = np.concatenate([a.reshape(-1) for a in agg])
    mean_val = float(np.mean(np.abs(vals)))

    # Should be between honest and adversary but closer to honest
    assert np.isfinite(vals).all(), "Bulyan output contains NaN or Inf"
    assert 0.05 < mean_val < 1.5, f"Bulyan result {mean_val} out of expected range"
