from __future__ import annotations

from enum import Enum
from typing import List, Sequence, Tuple, Optional

import numpy as np


class AggregationMethod(Enum):
    FED_AVG = "fedavg"
    MEDIAN = "median"
    KRUM = "krum"
    BULYAN = "bulyan"

    @staticmethod
    def from_string(value: str) -> "AggregationMethod":
        try:
            return AggregationMethod(value.lower())
        except Exception:
            return AggregationMethod.FED_AVG


def _stack_layers(weights_per_client: List[List[np.ndarray]], layer_idx: int) -> np.ndarray:
    return np.stack([client[layer_idx] for client in weights_per_client], axis=0)


def _flatten_client_update(
    client_layers: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, ...], int]]]:
    shapes_and_sizes: List[Tuple[Tuple[int, ...], int]] = []
    flat_parts = []
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
    This is more conservative than Krum's f < n/2 but ensures compatibility.

    Args:
        n: Total number of clients

    Returns:
        Maximum f value that satisfies Bulyan constraint
    """
    if n <= 4:
        return 0
    return max(0, (n - 3) // 4)


def _krum_candidate_indices(vectors: np.ndarray, f: int, multi: bool) -> List[int]:
    n = vectors.shape[0]

    # Handle edge case: single client
    if n <= 1:
        return [0] if n == 1 else []

    dists = _pairwise_sq_dists(vectors)
    # For each client, sum the distances to its closest n - f - 2 others
    m = max(1, min(n - f - 2, n - 1))  # Ensure m doesn't exceed available neighbors
    scores = []
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
        return [idx for _, idx in scores[:k]]
    # Krum: select single best
    return [scores[0][1]]


def _average_selected(weights_per_client: List[List[np.ndarray]], selected: Sequence[int]) -> List[np.ndarray]:
    num_layers = len(weights_per_client[0])
    aggregated: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = np.stack([weights_per_client[i][layer_idx] for i in selected], axis=0)
        aggregated.append(np.mean(stacked, axis=0))
    return aggregated


def _median_aggregate(weights_per_client: List[List[np.ndarray]]) -> List[np.ndarray]:
    num_layers = len(weights_per_client[0])
    aggregated: List[np.ndarray] = []
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
        ValueError: If beta is invalid (not even, too large, or negative)
    """
    n_candidates = stacked.shape[0]

    # Validation
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


def _bulyan_aggregate(weights_per_client: List[List[np.ndarray]], f: int) -> List[np.ndarray]:
    """
    True Bulyan aggregation per El Mhamdi et al. 2018.

    Algorithm:
    1. Use Multi-Krum to select θ = n - 2f candidates
    2. Apply coordinate-wise trimmed mean (trim β = 2f extreme values)

    Requires: n ≥ 4f + 3 for Byzantine resilience guarantees

    Args:
        weights_per_client: List of client weight updates (each is list of layers)
        f: Maximum number of Byzantine (malicious) clients to tolerate

    Returns:
        Aggregated weight update (list of layer arrays)

    Raises:
        ValueError: If n < 4f + 3 (insufficient clients for Byzantine resilience)
    """
    if not weights_per_client:
        return []

    n = len(weights_per_client)

    # Validate minimum client requirement per El Mhamdi et al. 2018
    if n < 4 * f + 3:
        raise ValueError(f"Bulyan requires n >= 4f + 3 for Byzantine resilience. " f"Got n={n}, f={f}, but need n >= {4 * f + 3}")

    # Flatten client updates for distance-based selection
    flats = []
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
    aggregated: List[np.ndarray] = []

    for layer_idx in range(num_layers):
        stacked = np.stack([weights_per_client[i][layer_idx] for i in selected], axis=0)
        aggregated.append(_coordinate_wise_trimmed_mean(stacked, beta))

    return aggregated


def aggregate_weights(
    weights_per_client: List[List[np.ndarray]],
    method: AggregationMethod,
    byzantine_f: int | None = None,
) -> List[np.ndarray]:
    """
    Aggregate model weights across clients. Each client contributes a list of ndarrays (one per layer).
    - fedavg: simple mean
    - median: coordinate-wise median
    - krum: Krum selection (single candidate)
    - bulyan: true Bulyan per El Mhamdi et al. 2018 (Multi-Krum + trimmed mean)
    """
    if not weights_per_client:
        return []

    if method == AggregationMethod.MEDIAN:
        return _median_aggregate(weights_per_client)

    if method in (AggregationMethod.KRUM, AggregationMethod.BULYAN):
        # Prepare flattened vectors for selection
        flats = []
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
    aggregated_mean: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = _stack_layers(weights_per_client, layer_idx)
        aggregated_mean.append(np.mean(stacked, axis=0))
    return aggregated_mean


def aggregate_weighted_mean(weights_per_client: List[List[np.ndarray]], sample_counts: Sequence[float]) -> List[np.ndarray]:
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
    aggregated: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        acc: Optional[np.ndarray] = None
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
