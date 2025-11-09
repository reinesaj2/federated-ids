from __future__ import annotations

import numpy as np
import pandas as pd

from data_preprocessing import (
    MIN_SAMPLES_PER_CLASS,
    dirichlet_partition,
    fit_preprocessor_global,
    prepare_partitions_from_dataframe,
    protocol_partition,
    transform_with_preprocessor,
)


def _make_dummy_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Numeric features
    num1 = rng.normal(0, 1, size=n)
    num2 = rng.normal(5, 2, size=n)
    # Categorical features
    cat1 = rng.choice(["HTTP", "DNS", "FTP"], size=n)
    cat2 = rng.choice(["A", "B"], size=n)
    # Labels (binary)
    label = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "proto": cat1,
            "flag": cat2,
            "label": label,
        }
    )


def test_preprocessor_feature_dimension_consistency():
    df = _make_dummy_df(n=300)
    pre, X_all, y_all = fit_preprocessor_global(df, label_col="label")
    # Ensure transform yields same shape when applied again
    X_again, y_again = transform_with_preprocessor(df, label_col="label", pre=pre)
    assert X_all.shape == X_again.shape
    assert np.array_equal(y_all, y_again)


def test_partition_outputs_have_same_feature_dimension_across_clients():
    df = _make_dummy_df(n=300)
    _, X_parts, y_parts, _ = prepare_partitions_from_dataframe(
        df=df,
        label_col="label",
        partition_strategy="dirichlet",
        num_clients=3,
        seed=123,
        alpha=0.1,
        protocol_col="proto",
    )
    dims = [Xp.shape[1] for Xp in X_parts]
    assert len(set(dims)) == 1
    # Sanity: non-empty splits sum to total samples
    assert sum(len(y) for y in y_parts) == len(df)


def test_dirichlet_partition_index_coverage_and_disjointness():
    labels = np.array([0] * 50 + [1] * 50)
    shards = dirichlet_partition(labels=labels, num_clients=3, alpha=0.1, seed=77, min_samples_per_class=1)
    all_indices = sorted(idx for shard in shards for idx in shard)
    assert all_indices == list(range(len(labels)))
    # Mostly disjoint (allow minimal collisions from algorithmic integer splits)
    sets = [set(s) for s in shards]
    intersection = set.intersection(*sets)
    assert len(intersection) == 0


def test_dirichlet_partition_enforces_minimum_samples_per_class():
    """
    Test that Dirichlet partitioning enforces minimum sample threshold per class.

    Regression test for alpha=0.5 pathological imbalance where clients received
    only 3-5 attack samples, causing F1=0.15 training failures.
    """
    # Create dataset large enough to support minimum threshold for all clients
    # With 11 clients and min 50/class, need at least 11 * 50 = 550 per class
    labels = np.array([0] * 600 + [1] * 600)
    num_clients = 11
    alpha = 0.5  # Known to create extreme imbalance

    # Use seed that previously caused failures
    shards = dirichlet_partition(labels=labels, num_clients=num_clients, alpha=alpha, seed=46, min_samples_per_class=MIN_SAMPLES_PER_CLASS)

    # Verify all clients got data
    assert len(shards) == num_clients
    assert all(len(shard) > 0 for shard in shards)

    # Verify minimum samples per class for each client
    for client_id, shard in enumerate(shards):
        shard_labels = labels[shard]
        class_0_count = np.sum(shard_labels == 0)
        class_1_count = np.sum(shard_labels == 1)

        assert class_0_count >= MIN_SAMPLES_PER_CLASS, (
            f"Client {client_id} has only {class_0_count} class 0 samples " f"(minimum: {MIN_SAMPLES_PER_CLASS})"
        )
        assert class_1_count >= MIN_SAMPLES_PER_CLASS, (
            f"Client {client_id} has only {class_1_count} class 1 samples " f"(minimum: {MIN_SAMPLES_PER_CLASS})"
        )

    # Verify all indices are covered
    all_indices = sorted(idx for shard in shards for idx in shard)
    assert all_indices == list(range(len(labels)))


def test_protocol_partition_assigns_each_protocol_to_single_client():
    df = _make_dummy_df(n=240)
    shards = protocol_partition(protocols=df["proto"].tolist(), num_clients=3, seed=42)
    proto_values = df["proto"].astype(str).to_numpy()
    # For each protocol, ensure it appears in exactly one client shard
    proto_to_clients = {}
    for client_id, shard in enumerate(shards):
        for idx in shard:
            proto = proto_values[idx]
            proto_to_clients.setdefault(proto, set()).add(client_id)
    assert all(len(clients) == 1 for clients in proto_to_clients.values())


def test_preprocessor_yields_same_feature_dim_when_fitted_on_shuffled_rows():
    df = _make_dummy_df(n=400, seed=101)
    df_shuffled = df.sample(frac=1.0, random_state=202).reset_index(drop=True)
    _, X1, _ = fit_preprocessor_global(df, label_col="label")
    _, X2, _ = fit_preprocessor_global(df_shuffled, label_col="label")
    assert X1.shape[1] == X2.shape[1]
