from __future__ import annotations

import time
import numpy as np
import hypothesis as hy
import hypothesis.strategies as st

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


# Property-based tests with Hypothesis
@st.composite
def client_updates_with_benign_mean(draw, n_clients=st.integers(min_value=5, max_value=20),
                                   n_layers=st.integers(min_value=1, max_value=3),
                                   layer_dim=st.integers(min_value=4, max_value=32)):
    n = draw(n_clients)
    layers = draw(n_layers)
    dim = draw(layer_dim)

    # Generate benign mean for each layer
    benign_means = []
    for _ in range(layers):
        mean_vals = draw(st.lists(st.floats(-1, 1, allow_nan=False, allow_infinity=False),
                                min_size=dim, max_size=dim))
        benign_means.append(np.array(mean_vals, dtype=np.float32))

    # Generate benign clients (n-1) with small noise around means
    benign_clients = []
    for _ in range(n - 1):
        client_layers = []
        for layer_idx in range(layers):
            noise = 0.01 * np.random.randn(*benign_means[layer_idx].shape).astype(np.float32)
            client_layers.append(benign_means[layer_idx] + noise)
        benign_clients.append(client_layers)

    # Generate one adversarial client
    adv_scale = draw(st.one_of(st.floats(5, 50), st.floats(-50, -5)))
    adversary_layers = []
    for layer_idx in range(layers):
        adv_noise = adv_scale * np.random.randn(*benign_means[layer_idx].shape).astype(np.float32)
        adversary_layers.append(benign_means[layer_idx] + adv_noise)

    all_clients = benign_clients + [adversary_layers]
    return all_clients, benign_means


def _l2_distance(a, b):
    """Compute L2 distance between two lists of arrays."""
    total_dist = 0.0
    for arr_a, arr_b in zip(a, b):
        total_dist += float(np.linalg.norm(arr_a - arr_b) ** 2)
    return np.sqrt(total_dist)


@hy.given(client_updates_with_benign_mean())
def test_benign_invariance_property(data):
    """When all clients have identical updates, aggregation should equal that update."""
    clients, benign_means = data

    # Create scenario where all clients have identical updates
    identical_clients = [benign_means for _ in range(len(clients))]

    for method in [AggregationMethod.FED_AVG, AggregationMethod.MEDIAN,
                   AggregationMethod.KRUM, AggregationMethod.BULYAN]:
        agg = aggregate_weights(identical_clients, method)

        # Aggregation should equal the identical update
        for expected, actual in zip(benign_means, agg):
            assert np.allclose(expected, actual, rtol=1e-5, atol=1e-6), \
                f"Benign invariance failed for {method.value}"


@hy.given(client_updates_with_benign_mean())
def test_outlier_resistance_property(data):
    """Robust methods should be closer to benign mean than FedAvg when adversary present."""
    clients, benign_means = data

    # Only test if we have enough clients for meaningful comparison
    if len(clients) < 5:
        return

    fedavg_agg = aggregate_weights(clients, AggregationMethod.FED_AVG)
    median_agg = aggregate_weights(clients, AggregationMethod.MEDIAN)
    krum_agg = aggregate_weights(clients, AggregationMethod.KRUM)
    bulyan_agg = aggregate_weights(clients, AggregationMethod.BULYAN)

    fedavg_dist = _l2_distance(fedavg_agg, benign_means)
    median_dist = _l2_distance(median_agg, benign_means)
    krum_dist = _l2_distance(krum_agg, benign_means)
    bulyan_dist = _l2_distance(bulyan_agg, benign_means)

    # At least one robust method should be closer to benign mean than FedAvg
    robust_better = (median_dist <= fedavg_dist or
                    krum_dist <= fedavg_dist or
                    bulyan_dist <= fedavg_dist)

    assert robust_better, (
        f"No robust method closer to benign mean than FedAvg: "
        f"FedAvg={fedavg_dist:.4f}, Median={median_dist:.4f}, "
        f"Krum={krum_dist:.4f}, Bulyan={bulyan_dist:.4f}"
    )


def test_empty_input_returns_empty_list():
    """Empty client list should return empty list."""
    for method in [AggregationMethod.FED_AVG, AggregationMethod.MEDIAN,
                   AggregationMethod.KRUM, AggregationMethod.BULYAN]:
        result = aggregate_weights([], method)
        assert result == [], f"Empty input should return empty list for {method.value}"


def test_single_client_returns_that_update():
    """Single client should return that client's update unchanged."""
    client_update = [np.array([1.0, 2.0, 3.0]), np.array([[4.0, 5.0], [6.0, 7.0]])]
    clients = [client_update]

    for method in [AggregationMethod.FED_AVG, AggregationMethod.MEDIAN,
                   AggregationMethod.KRUM, AggregationMethod.BULYAN]:
        agg = aggregate_weights(clients, method)

        for expected, actual in zip(client_update, agg):
            assert np.allclose(expected, actual), f"Single client test failed for {method.value}"


def test_mismatched_shapes_handled():
    """Clients with mismatched layer shapes should be handled gracefully."""
    client1 = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    client2 = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]  # Different first layer shape
    clients = [client1, client2]

    # Should either work (if implementation handles it) or raise clear error
    for method in [AggregationMethod.FED_AVG, AggregationMethod.MEDIAN]:
        try:
            agg = aggregate_weights(clients, method)
            # If it works, verify output is reasonable
            assert len(agg) == 2
            assert all(np.isfinite(arr).all() for arr in agg)
        except (ValueError, IndexError):
            pass  # Acceptable to raise error for mismatched shapes


def test_aggregation_timing_performance():
    """Aggregation should complete within reasonable time bounds."""
    n_clients = 50
    n_params = 1000
    clients = _make_client_updates(n_clients=n_clients, n_layers=2, shape=(n_params,))

    for method in [AggregationMethod.FED_AVG, AggregationMethod.MEDIAN,
                   AggregationMethod.KRUM, AggregationMethod.BULYAN]:
        start_time = time.perf_counter()
        agg = aggregate_weights(clients, method)
        elapsed = time.perf_counter() - start_time

        # Should complete within 500ms for this size
        assert elapsed < 0.5, f"{method.value} took {elapsed:.3f}s, expected <0.5s"
        assert all(np.isfinite(arr).all() for arr in agg), f"{method.value} produced non-finite values"


def test_large_magnitude_stability():
    """Aggregation should handle large magnitude values without overflow/NaN."""
    large_val = 1e6
    clients = []
    for i in range(5):
        client_layers = [
            np.array([large_val + i, -large_val + i]),
            np.array([[large_val * 2, -large_val * 2]])
        ]
        clients.append(client_layers)

    for method in [AggregationMethod.FED_AVG, AggregationMethod.MEDIAN,
                   AggregationMethod.KRUM, AggregationMethod.BULYAN]:
        agg = aggregate_weights(clients, method)

        # Results should be finite
        for arr in agg:
            assert np.isfinite(arr).all(), f"{method.value} produced non-finite values with large inputs"
            assert not np.isnan(arr).any(), f"{method.value} produced NaN values"
