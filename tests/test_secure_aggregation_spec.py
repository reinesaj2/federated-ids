import numpy as np
import pytest

from secure_aggregation import (
    generate_mask_sequence,
    generate_secret_shares,
    mask_updates,
    sum_updates,
    unmask_aggregate,
)


def test_generate_mask_sequence_is_deterministic():
    shapes = [(4,), (2, 3)]
    first = generate_mask_sequence(1234, shapes)
    second = generate_mask_sequence(1234, shapes)
    for mask_a, mask_b in zip(first, second):
        np.testing.assert_array_equal(mask_a, mask_b)


def test_generate_secret_shares_rejects_shape_mismatch():
    share = generate_secret_shares((2, 2), seed=1)
    update = np.ones((2, 3))
    with pytest.raises(ValueError):
        mask_updates(update, share)


def test_mask_round_trip_with_deterministic_seed():
    updates = [np.random.randn(4, 4), np.random.randn(2, 3)]
    seed = 4321
    shapes = [u.shape for u in updates]
    masks = generate_mask_sequence(seed, shapes)
    masked = [mask_updates(u, m) for u, m in zip(updates, masks)]
    server_masks = generate_mask_sequence(seed, shapes)
    recovered = [masked_layer - mask for masked_layer, mask in zip(masked, server_masks)]
    for rec, original in zip(recovered, updates):
        np.testing.assert_allclose(rec, original, atol=1e-6)


def test_unmask_aggregate_restores_single_layer_sum():
    update_a = np.array([[1.0, -2.0], [0.5, 3.5]])
    update_b = np.array([[0.1, 0.2], [-0.3, 0.4]])
    mask_a = np.array([[0.5, 0.5], [0.5, 0.5]])
    mask_b = np.array([[-0.5, -0.5], [-0.5, -0.5]])
    masked_a = mask_updates(update_a, mask_a)
    masked_b = mask_updates(update_b, mask_b)
    aggregate = sum_updates([masked_a, masked_b])
    unmasked = unmask_aggregate(aggregate, [mask_a + mask_b])
    np.testing.assert_allclose(unmasked, sum_updates([update_a, update_b]), atol=1e-6)
