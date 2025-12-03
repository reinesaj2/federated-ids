"""Property-based tests for data partitioning using hypothesis.

These tests verify universal properties that should hold for any valid partition
across a wide range of random inputs, catching edge cases that unit tests miss.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from data_preprocessing import MIN_SAMPLES_PER_CLASS, dirichlet_partition, iid_partition


@given(
    num_samples=st.integers(min_value=200, max_value=1000),
    num_clients=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=50, deadline=None)
def test_iid_partition_always_covers_all_indices(num_samples, num_clients, seed):
    """Property: IID partition uses every index exactly once."""
    shards = iid_partition(num_samples, num_clients, seed)

    all_indices = sorted(idx for shard in shards for idx in shard)
    assert all_indices == list(range(num_samples)), "Index coverage violation"


@given(
    num_samples_per_class=st.integers(min_value=200, max_value=400),
    num_clients=st.integers(min_value=2, max_value=6),
    alpha=st.floats(min_value=0.5, max_value=5.0),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=None)
def test_dirichlet_partition_always_covers_all_indices(num_samples_per_class, num_clients, alpha, seed):
    """Property: Dirichlet partition uses every index exactly once."""
    # Create balanced binary labels
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

    shards = dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_samples_per_class=MIN_SAMPLES_PER_CLASS,
    )

    all_indices = sorted(idx for shard in shards for idx in shard)
    assert all_indices == list(range(len(labels))), "Index coverage violation"


@given(
    num_samples_per_class=st.integers(min_value=200, max_value=400),
    num_clients=st.integers(min_value=2, max_value=6),
    alpha=st.floats(min_value=0.5, max_value=5.0),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=None)
def test_dirichlet_partition_meets_min_samples_constraint(num_samples_per_class, num_clients, alpha, seed):
    """Property: Every client has >= min_samples_per_class for every class."""
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

    shards = dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_samples_per_class=MIN_SAMPLES_PER_CLASS,
    )

    # Verify constraint for all clients
    for client_id, shard in enumerate(shards):
        shard_labels = labels[shard]
        for class_idx in [0, 1]:
            count = np.sum(shard_labels == class_idx)
            assert count >= MIN_SAMPLES_PER_CLASS, (
                f"Property violation: Client {client_id}, class {class_idx} " f"has {count} < {MIN_SAMPLES_PER_CLASS}"
            )


@given(
    num_samples=st.integers(min_value=200, max_value=1000),
    num_clients=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=50, deadline=None)
def test_iid_partition_produces_equal_size_shards(num_samples, num_clients, seed):
    """Property: IID partition creates roughly equal-sized shards."""
    shards = iid_partition(num_samples, num_clients, seed)

    sizes = [len(shard) for shard in shards]
    min_size = min(sizes)
    max_size = max(sizes)

    # Size difference should be at most 1 (due to integer division)
    assert max_size - min_size <= 1, f"IID shards not balanced: sizes={sizes}"


@given(
    num_samples_per_class=st.integers(min_value=150, max_value=400),
    num_clients=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=None)
def test_alpha_infinity_produces_balanced_distribution(num_samples_per_class, num_clients, seed):
    """Property: Alpha=infinity produces nearly balanced class distributions."""
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class + [2] * num_samples_per_class)

    shards = dirichlet_partition(labels=labels, num_clients=num_clients, alpha=float("inf"), seed=seed)

    # For each client, class counts should be roughly equal
    for client_id, shard in enumerate(shards):
        shard_labels = labels[shard]
        counts = [np.sum(shard_labels == c) for c in [0, 1, 2]]

        # Calculate coefficient of variation (should be low for IID)
        mean_count = np.mean(counts)
        if mean_count > 0:
            std_count = np.std(counts)
            cv = std_count / mean_count

            assert cv < 0.5, f"Alpha=infinity not balanced for client {client_id}: " f"counts={counts}, CV={cv:.3f}"


@given(
    num_classes=st.integers(min_value=2, max_value=3),
    samples_per_class=st.integers(min_value=300, max_value=400),
    num_clients=st.integers(min_value=3, max_value=4),
    alpha=st.floats(min_value=1.0, max_value=3.0),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=20, deadline=None)
def test_multiclass_partition_coverage(num_classes, samples_per_class, num_clients, alpha, seed):
    """Property: Multiclass partitioning covers all classes for all clients."""
    labels = np.concatenate([np.full(samples_per_class, c) for c in range(num_classes)])

    shards = dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_samples_per_class=MIN_SAMPLES_PER_CLASS,
    )

    # Every client must have samples from every class
    for client_id, shard in enumerate(shards):
        shard_labels = labels[shard]
        unique_classes = set(shard_labels.tolist())

        assert len(unique_classes) == num_classes, (
            f"Client {client_id} missing classes. " f"Has {unique_classes}, expected {set(range(num_classes))}"
        )
