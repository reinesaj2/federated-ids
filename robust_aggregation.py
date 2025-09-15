from __future__ import annotations

from enum import Enum
from typing import List

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


def aggregate_weights(weights_per_client: List[List[np.ndarray]], method: AggregationMethod) -> List[np.ndarray]:
    """
    Aggregate model weights across clients. Each client contributes a list of ndarrays (one per layer).
    Currently: MEDIAN is implemented; KRUM and BULYAN fall back to simple mean (FedAvg-like) for demo.
    """
    if not weights_per_client:
        return []

    num_layers = len(weights_per_client[0])

    if method == AggregationMethod.MEDIAN:
        aggregated: List[np.ndarray] = []
        for layer_idx in range(num_layers):
            stacked = np.stack([client[layer_idx] for client in weights_per_client], axis=0)
            aggregated.append(np.median(stacked, axis=0))
        return aggregated

    if method in (AggregationMethod.KRUM, AggregationMethod.BULYAN):
        # Placeholders for future robust implementations; use mean for now
        pass

    # Default: simple mean (FedAvg-like without sample weighting)
    aggregated_mean: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = np.stack([client[layer_idx] for client in weights_per_client], axis=0)
        aggregated_mean.append(np.mean(stacked, axis=0))
    return aggregated_mean
