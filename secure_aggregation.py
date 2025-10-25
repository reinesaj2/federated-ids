import numpy as np


def generate_secret_shares(shape: tuple, seed: int = -1) -> np.ndarray:
    """Generate random secret shares for masking model updates.

    Uses additive secret sharing: each client generates a random vector
    to mask its update. Sum of all masks subtracted from the aggregate
    recovers the true sum.

    Args:
        shape: Shape of the secret share array
        seed: Random seed for reproducibility

    Returns:
        Array of shape 'shape' with random values in (-1, 1)
    """
    if seed >= 0:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return rng.uniform(-0.5, 0.5, size=shape)


def mask_updates(updates: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Apply additive masking to model updates.

    Performs: masked = updates + share

    Args:
        updates: Model update array
        share: Secret share array (same shape as updates)

    Returns:
        Masked update array

    Raises:
        ValueError: If shapes don't match
    """
    if updates.shape != share.shape:
        raise ValueError(f"Shape mismatch: updates {updates.shape} vs share {share.shape}")
    return updates + share


def sum_updates(updates_list: list) -> np.ndarray:
    """Sum multiple update arrays element-wise.

    Args:
        updates_list: List of numpy arrays to sum

    Returns:
        Sum of all arrays

    Raises:
        ValueError: If list is empty
    """
    if not updates_list:
        raise ValueError("Cannot sum empty list of updates")
    return np.sum(updates_list, axis=0)


def unmask_aggregate(aggregate: np.ndarray, shares_list: list) -> np.ndarray:
    """Remove secret shares from aggregated result.

    Performs: clean = aggregate - sum(shares)

    Args:
        aggregate: Summed masked updates from server
        shares_list: List of secret share arrays

    Returns:
        Unmasked aggregate (true sum)

    Raises:
        ValueError: If shares shape doesn't match aggregate
    """
    if not shares_list:
        return aggregate.copy()

    shares_sum = sum_updates(shares_list)

    if aggregate.shape != shares_sum.shape:
        raise ValueError(f"Shape mismatch: aggregate {aggregate.shape} vs shares_sum {shares_sum.shape}")

    return aggregate - shares_sum
