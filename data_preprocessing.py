from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Drop silently if missing; helps avoid leakage/time proxies in IDS datasets
DEFAULT_DROP_COLS: List[str] = [
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
]

@dataclass
class DatasetStats:
    num_samples: int
    class_counts: Dict[int, int]


def _compute_class_counts(labels: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def create_synthetic_classification_loaders(
    num_samples: int,
    num_features: int,
    batch_size: int,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    rng = np.random.default_rng(seed)

    # Two-class synthetic data with controlled separation
    means = np.zeros(num_features)
    cov = np.eye(num_features)
    x0 = rng.multivariate_normal(mean=means - 0.5, cov=cov, size=num_samples // 2)
    x1 = rng.multivariate_normal(
        mean=means + 0.5, cov=cov, size=num_samples - x0.shape[0]
    )
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.array([0] * x0.shape[0] + [1] * x1.shape[0], dtype=np.int64)

    # Shuffle consistently
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_stats = _compute_class_counts(y_train)
    test_stats = _compute_class_counts(y_test)
    print(
        f"[Data] Train samples={len(train_ds)}, class_counts={train_stats}; Test samples={len(test_ds)}, class_counts={test_stats}"
    )

    return train_loader, test_loader


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition indices into num_clients shards using a Dirichlet distribution over label proportions.
    Returns a list of index lists per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    # For each class, sample proportions for clients
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for idxs in class_indices:
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        shards = np.split(idxs, splits)
        for i, shard in enumerate(shards):
            client_indices[i].extend(shard.astype(np.int64).tolist())

    for shard in client_indices:
        rng.shuffle(shard)

    return client_indices


def iid_partition(
    num_samples: int, num_clients: int, seed: int = 42
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [arr.tolist() for arr in np.array_split(indices, num_clients)]


def protocol_partition(
    protocols: Sequence[object],
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition deterministically by protocol: samples with the same protocol label go to the same client
    (round-robin assign unique protocol values to clients for balance).
    """
    rng = np.random.default_rng(seed)
    protocols = np.asarray(protocols)
    unique_protocols = sorted({str(p) for p in protocols})
    rng.shuffle(unique_protocols)
    proto_to_client: Dict[str, int] = {}
    for i, proto in enumerate(unique_protocols):
        proto_to_client[proto] = i % num_clients

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for idx, p in enumerate(protocols):
        client_indices[proto_to_client[str(p)]].append(idx)
    return client_indices


def infer_feature_columns(
    df: pd.DataFrame, label_col: str, drop_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    drop_cols = drop_cols or []
    feature_df = df.drop(columns=[label_col] + drop_cols, errors="ignore")
    categorical_cols = [
        c
        for c in feature_df.columns
        if feature_df[c].dtype == "object"
        or str(feature_df[c].dtype).startswith("category")
    ]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    # Toggle sparsity via env var for high-cardinality safety; default to dense for simplicity
    import os as _os

    _sparse_flag = _os.environ.get("OHE_SPARSE", "0").lower() not in (
        "0",
        "false",
        "no",
        "",
    )
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=_sparse_flag
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return pre


def _encode_labels_to_ints(labels: pd.Series) -> np.ndarray:
    # If labels are already numeric, cast to int64; otherwise map strings to ints
    if pd.api.types.is_integer_dtype(labels) or pd.api.types.is_bool_dtype(labels):
        return labels.astype(np.int64).to_numpy()
    if pd.api.types.is_float_dtype(labels):
        # Reject non-integer floats to avoid silent truncation
        vals = labels.to_numpy()
        if not np.all(np.equal(vals, np.floor(vals))):
            raise ValueError(
                "Label column contains non-integer floats; please map labels explicitly"
            )
        return labels.astype(np.int64).to_numpy()
    # String-like: normalize and enforce BENIGN maps to index 0 when present
    str_labels = labels.astype(str).str.strip().str.upper()
    uniques = list(pd.unique(str_labels))
    if "BENIGN" in uniques:
        # Put BENIGN first, keep order of the rest
        ordered = ["BENIGN"] + [u for u in uniques if u != "BENIGN"]
        cat = pd.Categorical(str_labels, categories=ordered)
        codes = cat.codes
    else:
        codes, _ = pd.factorize(str_labels)
    return codes.astype(np.int64)


def fit_preprocessor_global(
    df: pd.DataFrame, label_col: str, drop_cols: Optional[List[str]] = None
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    numeric_cols, categorical_cols = infer_feature_columns(df, label_col, drop_cols)
    pre = build_preprocessor(numeric_cols, categorical_cols)
    X = pre.fit_transform(df)
    y = _encode_labels_to_ints(df[label_col])
    X = X.astype(np.float32)
    return pre, X, y


def transform_with_preprocessor(
    df: pd.DataFrame, label_col: str, pre: ColumnTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    X = pre.transform(df)
    y = _encode_labels_to_ints(df[label_col])
    X = X.astype(np.float32)
    return X, y


def numpy_to_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int = 42,
    test_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def numpy_to_train_val_test_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int = 42,
    splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create stratified train/val/test loaders from numpy arrays.

    Default splits are 70/15/15. The splits tuple must sum to 1.0.
    """
    train_frac, val_frac, test_frac = splits
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("splits must sum to 1.0")

    # First split off test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    # Compute validation proportion relative to remaining train_val
    denom = max(train_frac + val_frac, 1e-12)
    val_relative = val_frac / denom
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=seed, stratify=y_train_val
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def load_csv_dataset(
    csv_path: str,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic cleanup: drop fully null columns
    df = df.dropna(axis=1, how="all")
    # Drop rows with missing label
    df = df[~df[label_col].isna()].reset_index(drop=True)
    return df


def load_unsw_nb15(csv_path: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Load UNSW-NB15 CSV and return (dataframe, label_col, protocol_col).
    Tries common column names across variants.
    """
    df = pd.read_csv(csv_path)
    # Standardize: some versions use 'Label' with 'Normal'/'Attack' or binary 0/1
    candidate_labels = [
        "label",
        "Label",
        "class",
        "Class",
    ]
    label_col = next((c for c in candidate_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            "Could not find label column in UNSW-NB15. Tried: 'label', 'Label', 'class', 'Class'"
        )
    # Normalize negative/benign label naming to BENIGN for consistency
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    df[label_col] = df[label_col].replace({"NORMAL": "BENIGN"})

    # Protocol column often 'proto'
    proto_col = "proto" if "proto" in df.columns else None
    # Basic cleanup
    df = df.dropna(axis=1, how="all")
    df = df[~df[label_col].isna()].reset_index(drop=True)
    return df, label_col, proto_col


def load_cic_ids2017(csv_path: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Load CIC-IDS2017 CSV and return (dataframe, label_col, protocol_col).
    Tries common column names across merged/day CSVs.
    """
    df = pd.read_csv(csv_path)
    candidate_labels = [
        "Label",
        "label",
        "Attack",
        "attack",
    ]
    label_col = next((c for c in candidate_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            "Could not find label column in CIC-IDS2017. Tried: 'Label', 'label', 'Attack', 'attack'"
        )
    # Normalize negative/benign label naming to BENIGN for consistency
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    df[label_col] = df[label_col].replace({"NORMAL": "BENIGN"})

    # Protocol column may be 'Protocol', 'ProtocolName', or 'proto'
    proto_col_candidates = ["Protocol", "ProtocolName", "proto"]
    proto_col = next((c for c in proto_col_candidates if c in df.columns), None)
    df = df.dropna(axis=1, how="all")
    df = df[~df[label_col].isna()].reset_index(drop=True)
    return df, label_col, proto_col


def fit_preprocessor_train_only_and_transform_all(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """Split first; fit preprocessor on train only; transform all rows.

    Returns (pre, X_all, y_all) where pre is fitted on the train subset.
    """
    drop_cols = drop_cols or DEFAULT_DROP_COLS
    df_ = df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_all = _encode_labels_to_ints(df_[label_col])
    # Stratified split to get train indices only
    idx = np.arange(len(df_))
    _, idx_train = train_test_split(
        idx, test_size=0.3, random_state=seed, stratify=y_all
    )
    # Infer columns after drops
    numeric_cols, categorical_cols = infer_feature_columns(df_, label_col, drop_cols=[])
    pre = build_preprocessor(numeric_cols, categorical_cols)
    # Fit on train subset
    pre.fit(df_.iloc[idx_train])
    # Transform all
    X_all = pre.transform(df_).astype(np.float32)
    return pre, X_all, y_all


def prepare_partitions_from_dataframe(
    df: pd.DataFrame,
    label_col: str,
    partition_strategy: str,
    num_clients: int,
    seed: int = 42,
    alpha: float = 0.1,
    protocol_col: Optional[str] = None,
    leakage_safe: bool = False,
) -> Tuple[ColumnTransformer, List[np.ndarray], List[np.ndarray], int]:
    # Drop default identifiers/time proxies if leakage_safe
    drop_cols = DEFAULT_DROP_COLS if leakage_safe else None
    if leakage_safe:
        pre, X_all, y_all = fit_preprocessor_train_only_and_transform_all(
            df, label_col, drop_cols=drop_cols, seed=seed
        )
    else:
        pre, X_all, y_all = fit_preprocessor_global(df.drop(columns=(drop_cols or []), errors="ignore"), label_col)
    num_classes_global = int(len(np.unique(y_all)))
    if partition_strategy == "iid":
        shards = iid_partition(num_samples=len(df), num_clients=num_clients, seed=seed)
    elif partition_strategy == "dirichlet":
        shards = dirichlet_partition(
            labels=y_all, num_clients=num_clients, alpha=alpha, seed=seed
        )
    elif partition_strategy == "protocol":
        if not protocol_col or protocol_col not in df.columns:
            raise ValueError(
                "protocol partition requires a valid protocol_col present in dataframe"
            )
        shards = protocol_partition(
            df[protocol_col].tolist(), num_clients=num_clients, seed=seed
        )
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")

    X_parts = [X_all[np.array(idx_list, dtype=np.int64)] for idx_list in shards]
    y_parts = [y_all[np.array(idx_list, dtype=np.int64)] for idx_list in shards]
    return pre, X_parts, y_parts, num_classes_global
