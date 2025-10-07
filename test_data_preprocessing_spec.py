from __future__ import annotations

import numpy as np
import pandas as pd

from data_preprocessing import (
    fit_preprocessor_global,
    transform_with_preprocessor,
    prepare_partitions_from_dataframe,
    dirichlet_partition,
    protocol_partition,
    load_cic_ids2017,
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
    # For each protocol, ensure it appears in exactly one client shard
    proto_to_clients = {}
    for client_id, shard in enumerate(shards):
        for idx in shard:
            proto = str(df.iloc[idx]["proto"])  # type: ignore[index]
            proto_to_clients.setdefault(proto, set()).add(client_id)
    assert all(len(clients) == 1 for clients in proto_to_clients.values())


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
