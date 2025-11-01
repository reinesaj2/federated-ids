from __future__ import annotations

import os
from collections.abc import Sequence
from enum import Enum

import numpy as np

DEBUG_AGGREGATION = os.environ.get("DEBUG_AGGREGATION", "0").lower() in ("1", "true", "yes")


class AggregationMethod(Enum):
    FED_AVG = "fedavg"
    MEDIAN = "median"
    KRUM = "krum"
    BULYAN = "bulyan"

    @staticmethod
    def from_string(value: str) -> AggregationMethod:
        try:
            return AggregationMethod(value.lower())
        except Exception:
            return AggregationMethod.FED_AVG


def _stack_layers(weights_per_client: list[list[np.ndarray]], layer_idx: int) -> np.ndarray:
    return np.stack([client[layer_idx] for client in weights_per_client], axis=0)


def _flatten_client_update(
    client_layers: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[tuple[tuple[int, ...], int]]]:
    shapes_and_sizes: list[tuple[tuple[int, ...], int]] = []
    flat_parts: list[np.ndarray] = []
    for arr in client_layers:
        shapes_and_sizes.append((arr.shape, arr.size))
        flat_parts.append(arr.reshape(-1))
    flat = np.concatenate(flat_parts, axis=0)
    return flat, shapes_and_sizes


def _pairwise_sq_dists(vectors: np.ndarray) -> np.ndarray:
    # vectors: (n_clients, dim)
    # Efficient pairwise squared distances: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    norms = np.sum(vectors * vectors, axis=1, keepdims=True)
    dists = norms + norms.T - 2.0 * vectors @ vectors.T
    np.fill_diagonal(dists, 0.0)
    return dists


def _guess_f_byzantine(n: int) -> int:
    """
    Estimate maximum Byzantine clients to tolerate.

    Uses Bulyan's constraint: n >= 4f + 3, which means f <= (n - 3) / 4.
    This is more conservative than Krum's f < n/2 but ensures compatibility
    across all aggregation methods.

    BREAKING CHANGE (Issue #70): Changed from (n-2)//2 - 1 to (n-3)//4 to
    satisfy Bulyan's Byzantine resilience requirement. This affects default
    behavior when byzantine_f is not explicitly specified:
    - n=6: f=1 -> f=0
    - n=7: f=1 -> f=1
    - n=11: f=3 -> f=2

    For reproducibility of prior experiments, always specify byzantine_f explicitly.

    Args:
        n: Total number of clients

    Returns:
        Maximum f value that satisfies Bulyan constraint (f <= (n-3)//4)
    """
    if n <= 4:
        return 0
    return max(0, (n - 3) // 4)


def _krum_candidate_indices(vectors: np.ndarray, f: int, multi: bool) -> list[int]:
    n = vectors.shape[0]

    # Handle edge case: single client
    if n <= 1:
        return [0] if n == 1 else []

    dists = _pairwise_sq_dists(vectors)
    # For each client, sum the distances to its closest n - f - 2 others
    m = max(1, min(n - f - 2, n - 1))  # Ensure m doesn't exceed available neighbors
    scores: list[tuple[float, int]] = []
    for i in range(n):
        # Handle case where we have fewer neighbors than needed
        available_neighbors = min(m, n - 1)  # Exclude self
        if available_neighbors > 0:
            neighbors = np.partition(dists[i], available_neighbors)[:available_neighbors]
            scores.append((float(np.sum(neighbors)), i))
        else:
            scores.append((0.0, i))
    scores.sort(key=lambda x: x[0])
    if multi:
        # Select top k candidates (Multi-Krum), average their updates
        k = max(1, min(n - f - 2, n))
        selected = [idx for _, idx in scores[:k]]
    else:
        # Krum: select single best
        selected = [scores[0][1]]

    if DEBUG_AGGREGATION:
        grad_norms = np.linalg.norm(vectors.reshape(n, -1), axis=1)
        print(
            f"[AGGREGATION] Krum: selected={selected}, "
            f"grad_norms (min/median/max)={grad_norms.min():.3f}/{np.median(grad_norms):.3f}/{grad_norms.max():.3f}, "
            f"f={f}, multi={multi}"
        )

    return selected


def _average_selected(weights_per_client: list[list[np.ndarray]], selected: Sequence[int]) -> list[np.ndarray]:
    num_layers = len(weights_per_client[0])
    aggregated: list[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = np.stack([weights_per_client[i][layer_idx] for i in selected], axis=0)
        aggregated.append(np.mean(stacked, axis=0))
    return aggregated


def _median_aggregate(weights_per_client: list[list[np.ndarray]]) -> list[np.ndarray]:
    num_layers = len(weights_per_client[0])
    aggregated: list[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = _stack_layers(weights_per_client, layer_idx)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated


def _coordinate_wise_trimmed_mean(stacked: np.ndarray, beta: int) -> np.ndarray:
    """
    Compute coordinate-wise trimmed mean per El Mhamdi et al. 2018.

    For each coordinate (feature), sort values across candidates, remove
    beta/2 smallest and beta/2 largest values, then average the remaining.

    Args:
        stacked: Array of shape (n_candidates, ...) where axis 0 is candidates
        beta: Number of values to trim (must be even, beta/2 from each end)

    Returns:
        Trimmed mean with shape matching stacked[0]

    Raises:
        TypeError: If stacked is not a numpy array
        ValueError: If beta is invalid or stacked has insufficient dimensions
    """
    # Input type validation
    if not isinstance(stacked, np.ndarray):
        raise TypeError(f"stacked must be numpy.ndarray, got {type(stacked).__name__}")
    if stacked.ndim < 1:
        raise ValueError(f"stacked must have at least 1 dimension, got shape {stacked.shape}")

    n_candidates = stacked.shape[0]

    # Parameter validation
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if beta % 2 != 0:
        raise ValueError(f"beta must be even for symmetric trimming, got {beta}")
    if beta >= n_candidates:
        raise ValueError(f"beta ({beta}) must be less than number of candidates ({n_candidates})")

    # Handle no-trimming case efficiently
    if beta == 0:
        return np.mean(stacked, axis=0)

    # Sort along candidate axis (axis=0)
    sorted_values = np.sort(stacked, axis=0)

    # Trim beta/2 from each end
    trim_count = beta // 2
    trimmed = sorted_values[trim_count : n_candidates - trim_count]

    # Return mean of remaining values
    return np.mean(trimmed, axis=0)


def _bulyan_aggregate(weights_per_client: list[list[np.ndarray]], f: int) -> list[np.ndarray]:
    """
    True Bulyan aggregation per El Mhamdi et al. 2018.

    Algorithm (Theorem 2 from paper):
    1. Use Multi-Krum to select θ = n - 2f candidates
       - Rationale: Removes up to f Byzantine-influenced selections
    2. Apply coordinate-wise trimmed mean (trim β = 2f extreme values)
       - Rationale: From θ candidates, at most f are Byzantine, so trimming
         2f values (f from each tail) guarantees ≥1 honest value remains

    Requires: n ≥ 4f + 3 for Byzantine resilience guarantees
    - Ensures θ = n - 2f ≥ 2f + 3 (enough candidates for trimming)
    - Guarantees θ - β = (n - 2f) - 2f = n - 4f > f (more honest than Byzantine)

    Args:
        weights_per_client: List of client weight updates (each is list of layers)
        f: Maximum number of Byzantine (malicious) clients to tolerate

    Returns:
        Aggregated weight update (list of layer arrays)

    Raises:
        ValueError: If n < 4f + 3 (insufficient clients for Byzantine resilience)

    Reference:
        El Mhamdi et al. "The Hidden Vulnerability of Distributed Learning in
        Byzantium." ICML 2018. https://arxiv.org/abs/1802.07927
    """
    if not weights_per_client:
        return []

    n = len(weights_per_client)

    # Validate minimum client requirement per El Mhamdi et al. 2018
    if n < 4 * f + 3:
        raise ValueError(f"Bulyan requires n >= 4f + 3 for Byzantine resilience. " f"Got n={n}, f={f}, but need n >= {4 * f + 3}")

    # Flatten client updates for distance-based selection
    flats: list[np.ndarray] = []
    for client_layers in weights_per_client:
        flat, _ = _flatten_client_update(client_layers)
        flats.append(flat)
    vectors = np.stack(flats, axis=0)

    # Step 1: Multi-Krum selection of θ = n - 2f candidates
    theta = n - 2 * f
    krum_candidates = _krum_candidate_indices(vectors, f=f, multi=True)
    selected = krum_candidates[:theta]

    # Step 2: Coordinate-wise trimmed mean with β = 2f trimming
    beta = 2 * f
    num_layers = len(weights_per_client[0])
    aggregated: list[np.ndarray] = []

    for layer_idx in range(num_layers):
        stacked = np.stack([weights_per_client[i][layer_idx] for i in selected], axis=0)
        aggregated.append(_coordinate_wise_trimmed_mean(stacked, beta))

    return aggregated


def aggregate_weights(
    weights_per_client: list[list[np.ndarray]],
    method: AggregationMethod,
    byzantine_f: int | None = None,
) -> list[np.ndarray]:
    """
    Aggregate model weights across clients. Each client contributes a list of ndarrays (one per layer).
    - fedavg: simple mean
    - median: coordinate-wise median
    - krum: Krum selection (single candidate)
    - bulyan: true Bulyan per El Mhamdi et al. 2018 (Multi-Krum + trimmed mean)
    """
    if not weights_per_client:
        return []

    if DEBUG_AGGREGATION:
        print(f"[AGGREGATION] Starting aggregation: method={method.value}, num_clients={len(weights_per_client)}")

    if method == AggregationMethod.MEDIAN:
        return _median_aggregate(weights_per_client)

    if method in (AggregationMethod.KRUM, AggregationMethod.BULYAN):
        # Prepare flattened vectors for selection
        flats: list[np.ndarray] = []
        for client_layers in weights_per_client:
            flat, _ = _flatten_client_update(client_layers)
            flats.append(flat)
        vectors = np.stack(flats, axis=0)
        f = byzantine_f if byzantine_f is not None else _guess_f_byzantine(vectors.shape[0])
        if method == AggregationMethod.KRUM:
            selected = _krum_candidate_indices(vectors, f=f, multi=False)
            # Return the single selected client's update as the aggregate
            return [arr.copy() for arr in weights_per_client[selected[0]]]
        # Bulyan: simplified multi-krum + median
        return _bulyan_aggregate(weights_per_client, f=f)

    # Default: simple mean (FedAvg-like without sample weighting)
    num_layers = len(weights_per_client[0])
    aggregated_mean: list[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = _stack_layers(weights_per_client, layer_idx)
        aggregated_mean.append(np.mean(stacked, axis=0))
    return aggregated_mean


def aggregate_weighted_mean(weights_per_client: list[list[np.ndarray]], sample_counts: Sequence[float]) -> list[np.ndarray]:
    """
    Compute sample-size weighted average of client weights per layer.
    sample_counts can be ints or floats; must be non-negative and length match clients.
    """
    if not weights_per_client:
        return []
    if len(weights_per_client) != len(sample_counts):
        raise ValueError("sample_counts length must match number of clients")
    totals = float(np.sum(sample_counts))
    if totals <= 0:
        raise ValueError("Total sample count must be positive")
    num_layers = len(weights_per_client[0])
    aggregated: list[np.ndarray] = []
    for layer_idx in range(num_layers):
        acc: np.ndarray | None = None
        for client_idx, client_layers in enumerate(weights_per_client):
            weight = float(sample_counts[client_idx]) / totals
            contrib = client_layers[layer_idx] * weight
            if acc is None:
                acc = contrib
            else:
                acc = acc + contrib
        assert acc is not None
        aggregated.append(acc)
    return aggregated
