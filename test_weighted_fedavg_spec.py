from __future__ import annotations

import numpy as np

from robust_aggregation import aggregate_weighted_mean


def test_weighted_mean_matches_manual_per_layer_average():
    rng = np.random.default_rng(123)
    n_clients = 4
    sample_counts = np.array([10, 20, 30, 40], dtype=float)
    layers_per_client = []
    for _ in range(n_clients):
        layers_per_client.append(
            [
                rng.normal(0, 1, size=(3,)),
                rng.normal(0, 1, size=(2, 2)),
            ]
        )
    agg = aggregate_weighted_mean(layers_per_client, sample_counts)

    totals = np.sum(sample_counts)
    # Check each layer
    for layer_idx in range(len(agg)):
        manual = sum(
            layers_per_client[i][layer_idx] * (sample_counts[i] / totals)
            for i in range(n_clients)
        )
        assert np.allclose(agg[layer_idx], manual)
