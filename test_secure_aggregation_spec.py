import numpy as np
import pytest
from secure_aggregation import (
    generate_secret_shares,
    mask_updates,
    unmask_aggregate,
    sum_updates,
)


class TestGenerateSecretShares:
    def test_generates_shares_with_correct_shape(self):
        shapes = [(10,), (5, 3), (2, 4, 6)]
        for shape in shapes:
            shares = generate_secret_shares(shape, seed=42)
            assert shares.shape == shape

    def test_deterministic_with_seed(self):
        shape = (10, 5)
        shares1 = generate_secret_shares(shape, seed=42)
        shares2 = generate_secret_shares(shape, seed=42)
        np.testing.assert_array_equal(shares1, shares2)

    def test_different_seeds_produce_different_shares(self):
        shape = (10, 5)
        shares1 = generate_secret_shares(shape, seed=42)
        shares2 = generate_secret_shares(shape, seed=43)
        assert not np.allclose(shares1, shares2)

    def test_shares_are_floats(self):
        shares = generate_secret_shares((5, 5), seed=42)
        assert shares.dtype in (np.float32, np.float64)

    def test_shares_are_scaled_reasonably(self):
        shares = generate_secret_shares((1000,), seed=42)
        mean_abs = np.mean(np.abs(shares))
        assert mean_abs < 1.0


class TestMaskUpdates:
    def test_mask_single_update(self):
        update = np.array([[1.0, 2.0], [3.0, 4.0]])
        share = generate_secret_shares(update.shape, seed=42)
        masked = mask_updates(update, share)
        assert masked.shape == update.shape
        assert masked.dtype == update.dtype

    def test_mask_multiple_updates_independently(self):
        updates = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
        ]
        shares = [
            generate_secret_shares(updates[0].shape, seed=42),
            generate_secret_shares(updates[1].shape, seed=43),
        ]
        masked = [mask_updates(u, s) for u, s in zip(updates, shares)]
        assert len(masked) == 2
        assert masked[0].shape == updates[0].shape
        assert masked[1].shape == updates[1].shape

    def test_masked_differs_from_original(self):
        update = np.random.randn(10, 5)
        share = generate_secret_shares(update.shape, seed=42)
        masked = mask_updates(update, share)
        assert not np.allclose(masked, update)


class TestSumUpdates:
    def test_sum_single_update(self):
        updates = [np.array([[1.0, 2.0]])]
        result = sum_updates(updates)
        np.testing.assert_array_equal(result, updates[0])

    def test_sum_multiple_updates(self):
        updates = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
        ]
        result = sum_updates(updates)
        expected = np.array([[4.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_sum_empty_raises(self):
        with pytest.raises((ValueError, IndexError)):
            sum_updates([])

    def test_sum_handles_negative_values(self):
        updates = [
            np.array([[1.0, -2.0]]),
            np.array([[-3.0, 4.0]]),
        ]
        result = sum_updates(updates)
        expected = np.array([[-2.0, 2.0]])
        np.testing.assert_array_equal(result, expected)


class TestUnmaskAggregate:
    def test_unmask_with_single_share(self):
        aggregate = np.array([[10.0, 20.0]])
        shares = [np.array([[1.0, 2.0]])]
        result = unmask_aggregate(aggregate, shares)
        expected = np.array([[9.0, 18.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_unmask_with_multiple_shares(self):
        aggregate = np.array([[10.0, 20.0]])
        shares = [
            np.array([[1.0, 2.0]]),
            np.array([[2.0, 3.0]]),
        ]
        result = unmask_aggregate(aggregate, shares)
        expected = np.array([[7.0, 15.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_unmask_zero_shares_returns_aggregate(self):
        aggregate = np.array([[10.0, 20.0]])
        shares = []
        result = unmask_aggregate(aggregate, shares)
        np.testing.assert_array_equal(result, aggregate)


class TestSecureAggregationRoundTrip:
    def test_mask_sum_unmask_roundtrip(self):
        updates = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
        ]
        shares = [
            generate_secret_shares(updates[0].shape, seed=42),
            generate_secret_shares(updates[1].shape, seed=43),
        ]
        masked = [mask_updates(u, s) for u, s in zip(updates, shares)]
        aggregate = sum_updates(masked)
        unmasked = unmask_aggregate(aggregate, shares)
        expected = sum_updates(updates)
        np.testing.assert_array_almost_equal(unmasked, expected, decimal=5)

    def test_roundtrip_with_large_weights(self):
        updates = [
            np.random.randn(100, 50) * 1000.0,
            np.random.randn(100, 50) * 1000.0,
        ]
        shares = [generate_secret_shares(u.shape, seed=42 + i) for i, u in enumerate(updates)]
        masked = [mask_updates(u, s) for u, s in zip(updates, shares)]
        aggregate = sum_updates(masked)
        unmasked = unmask_aggregate(aggregate, shares)
        expected = sum_updates(updates)
        np.testing.assert_array_almost_equal(unmasked, expected, decimal=3)

    def test_roundtrip_preserves_dimension_mismatch_error(self):
        updates = [np.array([[1.0, 2.0]])]
        shares = [np.array([[1.0, 2.0, 3.0]])]
        with pytest.raises(ValueError):
            mask_updates(updates[0], shares[0])


class TestSecureAggregationWithWeights:
    def test_roundtrip_multilayer_neural_network_weights(self):
        layer1_w = np.random.randn(10, 5)
        layer2_w = np.random.randn(5, 2)
        layer1_b = np.random.randn(5)
        layer2_b = np.random.randn(2)

        updates = [
            [layer1_w, layer2_w, layer1_b, layer2_b],
            [layer1_w * 0.5, layer2_w * 0.5, layer1_b * 0.5, layer2_b * 0.5],
        ]

        shares = []
        masked_updates = []
        for update in updates:
            client_shares = [generate_secret_shares(layer.shape, seed=42 + i) for i, layer in enumerate(update)]
            shares.append(client_shares)
            masked = [mask_updates(layer, share) for layer, share in zip(update, client_shares)]
            masked_updates.append(masked)

        aggregated = [sum_updates([m[i] for m in masked_updates]) for i in range(len(updates[0]))]
        unmasked = [unmask_aggregate(agg, [s[i] for s in shares]) for i, agg in enumerate(aggregated)]

        expected = [sum_updates([u[i] for u in updates]) for i in range(len(updates[0]))]
        for un, exp in zip(unmasked, expected):
            np.testing.assert_array_almost_equal(un, exp, decimal=3)
