from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from data_preprocessing import (
    dirichlet_partition,
    fit_preprocessor_global,
    load_cic_ids2017,
    numpy_to_loaders,
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
    _, X_parts, y_parts, num_classes_global = prepare_partitions_from_dataframe(
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
    # Global classes equals number of unique labels
    import numpy as _np

    assert num_classes_global == len(_np.unique(df["label"]))


def test_dirichlet_partition_index_coverage_and_disjointness():
    labels = np.array([0] * 50 + [1] * 50)
    shards = dirichlet_partition(labels=labels, num_clients=3, alpha=0.1, seed=77)
    all_indices = sorted(idx for shard in shards for idx in shard)
    assert all_indices == list(range(len(labels)))
    # Mostly disjoint (allow minimal collisions from algorithmic integer splits)
    sets = [set(s) for s in shards]
    intersection = set.intersection(*sets)
    assert len(intersection) == 0


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


def test_protocol_partition_supports_explicit_mapping():
    df = _make_dummy_df(n=120)
    mapping = {"HTTP": 0, "FTP": 2}
    shards = protocol_partition(protocols=df["proto"].tolist(), num_clients=4, seed=7, protocol_mapping=mapping)
    proto_values = df["proto"].astype(str).str.strip().str.upper().to_numpy()
    for client_id, shard in enumerate(shards):
        for idx in shard:
            proto = proto_values[idx]
            if proto in mapping:
                assert client_id == mapping[proto]


def test_preprocessor_yields_same_feature_dim_when_fitted_on_shuffled_rows():
    df = _make_dummy_df(n=400, seed=101)
    df_shuffled = df.sample(frac=1.0, random_state=202).reset_index(drop=True)
    _, X1, _ = fit_preprocessor_global(df, label_col="label")
    _, X2, _ = fit_preprocessor_global(df_shuffled, label_col="label")
    assert X1.shape[1] == X2.shape[1]


def test_load_cic_ids2017_trims_whitespace_headers(tmp_path):
    df = pd.DataFrame(
        {
            " FlowDuration": [1.0, 2.0],
            " Label": ["benign", "Attack"],
            " Protocol": ["tcp", "udp"],
        }
    )
    csv_path = tmp_path / "cic_whitespace.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, label_col, proto_col = load_cic_ids2017(str(csv_path))

    assert label_col == "Label"
    assert proto_col == "Protocol"
    assert {"BENIGN", "ATTACK"} == set(loaded_df[label_col].unique())


def test_load_cic_ids2017_drops_infinite_rows(tmp_path):
    df = pd.DataFrame(
        {
            "Dst Port": [80, 443],
            "Label": ["BENIGN", "Attack"],
            "Flow Bytes/s": [np.inf, 123.0],
        }
    )
    csv_path = tmp_path / "cic_inf.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, label_col, _ = load_cic_ids2017(str(csv_path))

    assert len(loaded_df) == 1
    assert np.isfinite(loaded_df["Flow Bytes/s"].values).all()
    assert label_col == "Label"


def test_numpy_to_loaders_supports_sparse_feature_matrix(monkeypatch):
    monkeypatch.setenv("OHE_SPARSE", "1")

    rng = np.random.default_rng(123)
    n = 200
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(5, 2, size=n),
            "cat_high": [f"C{int(i)}" for i in rng.integers(0, 100, size=n)],
            "label": rng.integers(0, 2, size=n),
        }
    )

    _pre, X_all, y_all = fit_preprocessor_global(df, label_col="label")
    assert sp.issparse(X_all)

    train_loader, _test_loader = numpy_to_loaders(X_all, y_all, batch_size=32, seed=42)
    xb, yb = next(iter(train_loader))

    assert xb.shape == (32, X_all.shape[1])
    assert xb.dtype == torch.float32
    assert yb.dtype == torch.long


def test_sparse_and_dense_paths_produce_identical_splits():
    rng = np.random.default_rng(777)
    n = 500
    n_features = 20
    X_dense = rng.normal(size=(n, n_features)).astype(np.float32)
    y = rng.integers(0, 3, size=n).astype(np.int64)

    X_sparse = sp.csr_matrix(X_dense)

    seed = 42
    batch_size = 64

    train_dense, test_dense = numpy_to_loaders(X_dense, y, batch_size=batch_size, seed=seed)
    train_sparse, test_sparse = numpy_to_loaders(X_sparse, y, batch_size=batch_size, seed=seed)

    assert len(train_dense.dataset) == len(train_sparse.dataset)
    assert len(test_dense.dataset) == len(test_sparse.dataset)

    def collect_all_batches(loader):
        all_x, all_y = [], []
        for xb, yb in loader:
            all_x.append(xb)
            all_y.append(yb)
        return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)

    train_x_dense, train_y_dense = collect_all_batches(train_dense)
    train_x_sparse, train_y_sparse = collect_all_batches(train_sparse)

    train_dense_sorted = torch.argsort(train_y_dense * 1000000 + torch.arange(len(train_y_dense)))
    train_sparse_sorted = torch.argsort(train_y_sparse * 1000000 + torch.arange(len(train_y_sparse)))

    assert torch.equal(train_y_dense[train_dense_sorted], train_y_sparse[train_sparse_sorted])

    test_x_dense, test_y_dense = collect_all_batches(test_dense)
    test_x_sparse, test_y_sparse = collect_all_batches(test_sparse)

    dense_train_set = set(tuple(row.tolist()) for row in train_x_dense)
    sparse_train_set = set(tuple(row.tolist()) for row in train_x_sparse)
    assert dense_train_set == sparse_train_set

    dense_test_set = set(tuple(row.tolist()) for row in test_x_dense)
    sparse_test_set = set(tuple(row.tolist()) for row in test_x_sparse)
    assert dense_test_set == sparse_test_set
