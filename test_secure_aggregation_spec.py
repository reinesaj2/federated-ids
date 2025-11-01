import numpy as np
import pytest

from secure_aggregation import (
    generate_client_mask_sequence,
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


def test_pairwise_masks_cancel_between_clients():
    shapes = [(5,)]
    pair_seed = 7788
    mask_a = generate_client_mask_sequence("a", shapes, 1357, {"b": pair_seed})
    mask_b = generate_client_mask_sequence("b", shapes, 2468, {"a": pair_seed})

    personal_a = generate_mask_sequence(1357, shapes)
    personal_b = generate_mask_sequence(2468, shapes)

    for combined_a, combined_b, personal_component_a, personal_component_b in zip(
        mask_a, mask_b, personal_a, personal_b
    ):
        pair_component_a = combined_a - personal_component_a
        pair_component_b = combined_b - personal_component_b
        np.testing.assert_allclose(pair_component_a, -pair_component_b, atol=1e-6)


def test_client_server_masks_align_with_pairwise_peers():
    shapes = [(3, 2)]
    client_id = "alice"
    peers = {"bob": 9991, "carol": 9992}
    personal_seed = 4242

    client_masks = generate_client_mask_sequence(client_id, shapes, personal_seed, peers)
    update = [np.random.randn(*shape) for shape in shapes]
    masked = [mask_updates(layer, mask) for layer, mask in zip(update, client_masks)]

    server_masks = generate_client_mask_sequence(client_id, shapes, personal_seed, peers)
    recovered = [layer - mask for layer, mask in zip(masked, server_masks)]

    for original, restored in zip(update, recovered):
        np.testing.assert_allclose(restored, original, atol=1e-6)
