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
    clients = _make_client_updates(n_clients=8)
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


