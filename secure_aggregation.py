from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def generate_secret_shares(shape: Tuple[int, ...], seed: int = -1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random secret share for masking model updates."""
    if rng is None:
        rng = np.random.default_rng(seed if seed >= 0 else None)
    return rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)


def generate_mask_sequence(seed: int, shapes: Iterable[Tuple[int, ...]]) -> List[np.ndarray]:
    """Produce a deterministic sequence of masks for the supplied shapes."""
    rng = np.random.default_rng(seed)
    return [generate_secret_shares(shape, rng=rng) for shape in shapes]


def generate_client_mask_sequence(
    client_id: str,
    shapes: Iterable[Tuple[int, ...]],
    personal_seed: int | None,
    pairwise_seeds: Mapping[str, int] | None = None,
) -> List[np.ndarray]:
    """Generate the full masking sequence for a client.

    Combines the client's personal mask with deterministic pairwise masks so that
    peers receive opposite-signed contributions. This mirrors the server's
    reconstruction logic, enabling the server to remove all applied masks even
    if a peer drops out before sending an update.
    """
    shape_list = [tuple(shape) for shape in shapes]
    masks: List[np.ndarray] = [np.zeros(shape, dtype=np.float32) for shape in shape_list]

    if personal_seed is not None:
        for idx, personal_mask in enumerate(generate_mask_sequence(personal_seed, shape_list)):
            masks[idx] = masks[idx] + personal_mask

    if pairwise_seeds:
        client_key = str(client_id)
        for peer_id, seed in pairwise_seeds.items():
            peer_key = str(peer_id)
            pair_masks = generate_mask_sequence(seed, shape_list)
            sign = 1.0 if client_key < peer_key else -1.0
            for idx, pair_mask in enumerate(pair_masks):
                masks[idx] = masks[idx] + sign * pair_mask

    return masks


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
