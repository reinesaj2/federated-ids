from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def generate_secret_shares(shape: Tuple[int, ...], seed: int = -1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random secret share for masking model updates."""
    if rng is None:
        rng = np.random.default_rng(seed if seed >= 0 else None)
    return rng.uniform(-0.5, 0.5, size=shape)


def generate_mask_sequence(seed: int, shapes: Iterable[Tuple[int, ...]]) -> List[np.ndarray]:
    """Produce a deterministic sequence of masks for the supplied shapes."""
    rng = np.random.default_rng(seed)
    return [generate_secret_shares(shape, rng=rng) for shape in shapes]


def mask_updates(updates: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Apply additive masking to model updates."""
    if updates.shape != share.shape:
        raise ValueError(f"Shape mismatch: updates {updates.shape} vs share {share.shape}")
    return updates + share


def sum_updates(updates_list: Sequence[np.ndarray]) -> np.ndarray:
    """Sum multiple update arrays element-wise."""
    if not updates_list:
        raise ValueError("Cannot sum empty list of updates")
    return np.sum(updates_list, axis=0)


def unmask_aggregate(aggregate: np.ndarray, shares_list: Sequence[np.ndarray]) -> np.ndarray:
    """Remove secret shares from aggregated result."""
    if not shares_list:
        return aggregate.copy()
    shares_sum = sum_updates(shares_list)
    if aggregate.shape != shares_sum.shape:
        raise ValueError(f"Shape mismatch: aggregate {aggregate.shape} vs shares_sum {shares_sum.shape}")
    return aggregate - shares_sum
