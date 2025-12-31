from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None

# Drop silently if missing; helps avoid leakage/time proxies in IDS datasets
DEFAULT_DROP_COLS: list[str] = [
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
]

# Minimum samples per class required for meaningful model training
# Dirichlet partitioning will resample until all clients meet this threshold
# NOTE: Set to 5 (not 50) to allow true heterogeneity at low alpha values.
# Higher values cause repeated resampling failures, preventing extreme heterogeneity.
MIN_SAMPLES_PER_CLASS: int = 5

# Maximum resampling attempts for Dirichlet partitioning before falling back to even distribution
# Empirically chosen to balance distribution quality with computational cost
MAX_PARTITION_ATTEMPTS: int = 100

# Temporal validation protocol constants
TEMPORAL_TRAIN_FRAC: float = 0.70
TEMPORAL_VAL_FRAC: float = 0.15
TEMPORAL_TEST_FRAC: float = 0.15
FRAME_TIME_FORMAT: str = "%Y %H:%M:%S.%f"


def parse_frame_time(series: pd.Series) -> pd.Series:
    """Parse Edge-IIoTset frame.time column as datetime.

    Format: "2021 19:46:24.393481000" (year + time, no month/day)

    Args:
        series: Pandas Series containing frame.time strings

    Returns:
        Series of datetime64 values; unparsable values become NaT
    """
    return pd.to_datetime(series.str.strip(), format=FRAME_TIME_FORMAT, errors="coerce")


def temporal_sort_indices(
    df: pd.DataFrame,
    time_col: str = "frame.time",
) -> tuple[np.ndarray, int]:
    """Compute indices that sort dataframe by temporal order.

    Rows with unparsable timestamps retain their original relative position
    among other unparsable rows, placed after all parsable rows.

    Args:
        df: DataFrame containing the time column
        time_col: Name of the timestamp column

    Returns:
        Tuple of (sort_indices, n_unparsable) where:
        - sort_indices: Array of row indices in temporal order
        - n_unparsable: Count of rows that fell back to original order
    """
    if time_col not in df.columns:
        return np.arange(len(df), dtype=np.int64), len(df)

    timestamps = parse_frame_time(df[time_col])
    valid_mask = ~timestamps.isna()
    n_unparsable = int((~valid_mask).sum())

    valid_indices = np.where(valid_mask)[0]
    invalid_indices = np.where(~valid_mask)[0]

    if len(valid_indices) > 0:
        valid_timestamps = timestamps.iloc[valid_indices]
        sorted_valid_order = valid_timestamps.argsort()
        sorted_valid_indices = valid_indices[sorted_valid_order]
    else:
        sorted_valid_indices = np.array([], dtype=np.int64)

    sort_indices = np.concatenate([sorted_valid_indices, invalid_indices]).astype(np.int64)
    return sort_indices, n_unparsable


def temporal_train_val_test_split_indices(
    n_samples: int,
    train_frac: float = TEMPORAL_TRAIN_FRAC,
    val_frac: float = TEMPORAL_VAL_FRAC,
    test_frac: float = TEMPORAL_TEST_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split pre-sorted indices into train/val/test by position (not random).

    Args:
        n_samples: Total number of samples (assumed already temporally sorted)
        train_frac: Fraction for training (earliest samples)
        val_frac: Fraction for validation (middle samples)
        test_frac: Fraction for test (latest samples)

    Returns:
        Tuple of (train_indices, val_indices, test_indices)

    Raises:
        ValueError: If fractions don't sum to 1.0
    """
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    train_end = int(n_samples * train_frac)
    val_end = train_end + int(n_samples * val_frac)

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, n_samples, dtype=np.int64)

    return train_idx, val_idx, test_idx


@dataclass
class DatasetStats:
    num_samples: int
    class_counts: dict[int, int]


def _compute_class_counts(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts, strict=False)}


def create_synthetic_classification_loaders(
    num_samples: int,
    num_features: int,
    batch_size: int,
    seed: int = 42,
    num_classes: int = 2,
) -> tuple[DataLoader, DataLoader]:
    rng = np.random.default_rng(seed)

    # Multi-class synthetic data with controlled separation
    # Generate K Gaussian clusters arranged in a circle pattern
    X_list = []
    y_list = []
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    cov = np.eye(num_features)

    for class_idx in range(num_classes):
        # Distribute classes in a circle for max separation (first 2 dimensions)
        angle = 2 * np.pi * class_idx / num_classes
        mean = np.zeros(num_features)
        mean[0] = 2.0 * np.cos(angle)
        if num_features > 1:
            mean[1] = 2.0 * np.sin(angle)

        # Distribute remainder samples across first few classes
        n_samples_for_class = samples_per_class + (1 if class_idx < remainder else 0)

        x_class = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples_for_class)
        X_list.append(x_class)
        y_list.extend([class_idx] * n_samples_for_class)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle consistently
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_stats = _compute_class_counts(y_train)
    test_stats = _compute_class_counts(y_test)
    print(f"[Data] Train samples={len(train_ds)}, class_counts={train_stats}; Test samples={len(test_ds)}, class_counts={test_stats}")

    return train_loader, test_loader


def _validate_partition_constraints(
    client_indices: list[list[int]],
    labels: np.ndarray,
    min_samples_per_class: int,
) -> tuple[bool, str]:
    """Check if partition meets min_samples_per_class constraint for all clients.

    Args:
        client_indices: List of index lists per client
        labels: Array of labels for all samples
        min_samples_per_class: Minimum required samples per class per client

    Returns:
        (is_valid, diagnostic_message)
    """
    num_classes = int(labels.max()) + 1
    violations = []

    for client_id, shard in enumerate(client_indices):
        if len(shard) == 0:
            violations.append(f"Client {client_id}: empty shard")
            continue

        shard_labels = labels[shard]
        for class_idx in range(num_classes):
            count = np.sum(shard_labels == class_idx)
            if count < min_samples_per_class:
                violations.append(f"Client {client_id}: class {class_idx} has {count} samples (need {min_samples_per_class})")

    if violations:
        return False, "; ".join(violations[:5])  # Limit to first 5 violations
    return True, "OK"


def _attempt_dirichlet_partition_single(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Single attempt at Dirichlet partitioning (pure function).

    Args:
        labels: Array of labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        rng: Random number generator

    Returns:
        List of index lists per client
    """
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for idxs in class_indices:
        if len(idxs) == 0:
            continue
        rng.shuffle(idxs)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        shards = np.split(idxs, splits)

        for i, shard in enumerate(shards):
            client_indices[i].extend(shard.astype(np.int64).tolist())

    # Shuffle each client's indices
    for shard in client_indices:
        rng.shuffle(shard)

    return client_indices


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
) -> list[list[int]]:
    """Partition indices using Dirichlet distribution with min samples per class constraint.

    Deterministic constraint handling:
    - Seed each client with `min_samples_per_class` examples from every class (round-robin).
    - Distribute remaining examples per class using Dirichlet proportions.
    This guarantees the minimum per-class coverage while preserving heterogeneity on the
    remaining samples.

    Args:
        labels: Sequence of integer labels
        num_clients: Number of clients to partition data across
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
               Special case: alpha=inf delegates to iid_partition()
        seed: Random seed for reproducibility
        min_samples_per_class: Minimum samples per class per client (default: MIN_SAMPLES_PER_CLASS)

    Returns:
        List of index lists, one per client

    Raises:
        ValueError: If unable to satisfy min_samples_per_class constraint after MAX_PARTITION_ATTEMPTS

    References:
        - Dirichlet distribution for non-IID partitioning: Hsu et al. (2019)
        - MIN_SAMPLES_PER_CLASS rationale: See data_preprocessing.py:26-29
    """
    # Special case: alpha=infinity means IID (uniform distribution)
    if np.isinf(alpha):
        return iid_partition(len(labels), num_clients, seed)

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    if min_samples_per_class == 0:
        for _ in range(MAX_PARTITION_ATTEMPTS):
            candidate = _attempt_dirichlet_partition_single(labels, num_clients, alpha, rng)
            if all(len(shard) > 0 for shard in candidate):
                return candidate
        return iid_partition(len(labels), num_clients, seed)

    # Feasibility check: each class must have enough samples to meet the floor.
    class_counts = np.bincount(labels, minlength=num_classes)
    required_per_class = num_clients * min_samples_per_class
    infeasible = [i for i, count in enumerate(class_counts) if count < required_per_class]
    if infeasible:
        raise ValueError(
            f"Dirichlet partition infeasible: classes {infeasible} have fewer than "
            f"{required_per_class} samples required to allocate {min_samples_per_class} "
            f"per client (counts={class_counts.tolist()})"
        )

    # Prepare per-class indices
    class_indices = [np.where(labels == c)[0].tolist() for c in range(num_classes)]

    # Initialize client shards
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    # Step 1: seed each client with the minimum per class (round-robin)
    for class_id, idxs in enumerate(class_indices):
        rng.shuffle(idxs)
        needed = required_per_class
        seed_chunk, remainder = idxs[:needed], idxs[needed:]

        for client_id in range(num_clients):
            start = client_id * min_samples_per_class
            end = start + min_samples_per_class
            client_indices[client_id].extend(seed_chunk[start:end])

        # Store remaining indices for Dirichlet-based distribution
        class_indices[class_id] = remainder

    # Step 2: distribute remaining indices per class using Dirichlet proportions
    for idxs in class_indices:
        if not idxs:
            continue
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        shards = np.split(np.array(idxs, dtype=np.int64), splits)
        for client_id, shard in enumerate(shards):
            if shard.size:
                client_indices[client_id].extend(shard.tolist())

    # Final shuffle per client for randomness
    for shard in client_indices:
        rng.shuffle(shard)

    # Validate constraints post-construction
    is_valid, diagnostic = _validate_partition_constraints(client_indices, labels, min_samples_per_class)
    if not is_valid:
        raise ValueError(
            f"Dirichlet partition failed post-validation: {diagnostic}. "
            f"Constraints: {num_clients} clients, {len(labels)} samples, {num_classes} classes, "
            f"alpha={alpha:.4f}, min_samples_per_class={min_samples_per_class}."
        )

    return client_indices


def iid_partition(num_samples: int, num_clients: int, seed: int = 42) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [arr.tolist() for arr in np.array_split(indices, num_clients)]


def protocol_partition(
    protocols: Sequence[object],
    num_clients: int,
    seed: int = 42,
    protocol_mapping: dict[str, int] | None = None,
) -> list[list[int]]:
    """
    Partition deterministically by protocol: samples with the same protocol label go to the same client
    (round-robin assign unique protocol values to clients for balance). Accepts optional mapping to
    pin specific protocols to specific clients.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive for protocol partitioning")

    rng = np.random.default_rng(seed)
    normalized_protocols = np.array([str(p).strip().upper() for p in protocols], dtype=object)
    unique_protocols = sorted(set(normalized_protocols.tolist()))

    proto_to_client: dict[str, int] = {}
    if protocol_mapping:
        for proto_name, client_id in protocol_mapping.items():
            try:
                normalized_proto = str(proto_name).strip().upper()
                client_idx = int(client_id)
            except (TypeError, ValueError):
                continue
            proto_to_client[normalized_proto] = client_idx % num_clients

    unassigned = [proto for proto in unique_protocols if proto not in proto_to_client]
    rng.shuffle(unassigned)
    for i, proto in enumerate(unassigned):
        proto_to_client[proto] = i % num_clients

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for idx, proto in enumerate(normalized_protocols):
        client_idx = proto_to_client.get(proto)
        if client_idx is None:
            client_idx = int(rng.integers(0, num_clients))
            proto_to_client[proto] = client_idx
        client_indices[client_idx].append(idx)
    return client_indices


def infer_feature_columns(df: pd.DataFrame, label_col: str, drop_cols: list[str] | None = None) -> tuple[list[str], list[str]]:
    drop_cols = drop_cols or []
    feature_df = df.drop(columns=[label_col] + drop_cols, errors="ignore")
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object" or str(feature_df[c].dtype).startswith("category")]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    # Toggle sparsity via env var for high-cardinality safety; default to dense for simplicity
    import os as _os

    _sparse_flag = _os.environ.get("OHE_SPARSE", "0").lower() not in (
        "0",
        "false",
        "no",
        "",
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=_sparse_flag)
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
            raise ValueError("Label column contains non-integer floats; please map labels explicitly")
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


def infer_class_names_from_series(labels: pd.Series) -> list[str]:
    """
    Infer stable class name ordering consistent with _encode_labels_to_ints.
    """
    if pd.api.types.is_integer_dtype(labels) or pd.api.types.is_bool_dtype(labels):
        uniques = list(pd.unique(labels))
        return [str(int(u)) for u in uniques]
    if pd.api.types.is_float_dtype(labels):
        vals = labels.to_numpy()
        if not np.all(np.equal(vals, np.floor(vals))):
            raise ValueError("Label column contains non-integer floats; please map labels explicitly")
        uniques = list(pd.unique(labels.astype(np.int64)))
        return [str(int(u)) for u in uniques]

    str_labels = labels.astype(str).str.strip().str.upper()
    uniques = list(pd.unique(str_labels))
    if "BENIGN" in uniques:
        return ["BENIGN"] + [u for u in uniques if u != "BENIGN"]
    return uniques


def fit_preprocessor_global(
    df: pd.DataFrame, label_col: str, drop_cols: list[str] | None = None
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    numeric_cols, categorical_cols = infer_feature_columns(df, label_col, drop_cols)
    pre = build_preprocessor(numeric_cols, categorical_cols)
    X = pre.fit_transform(df)
    y = _encode_labels_to_ints(df[label_col])
    X = X.astype(np.float32)
    return pre, X, y


def transform_with_preprocessor(df: pd.DataFrame, label_col: str, pre: ColumnTransformer) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[DataLoader, DataLoader]:
    # Defensive guard: handle empty shards gracefully
    if X.shape[0] == 0 or y.size == 0:
        # Return dummy loaders with minimal tensors
        dummy_X = torch.zeros((1, X.shape[1] if X.size > 0 else 1), dtype=torch.float32)
        dummy_y = torch.zeros((1,), dtype=torch.long)
        dummy_ds = TensorDataset(dummy_X, dummy_y)
        dummy_loader = DataLoader(dummy_ds, batch_size=batch_size, shuffle=False)
        return dummy_loader, dummy_loader

    if sp is not None and sp.issparse(X):
        row_indices = np.arange(X.shape[0], dtype=np.int64)

        try:
            train_rows, test_rows = train_test_split(row_indices, test_size=test_size, random_state=seed, stratify=y)
        except ValueError as e:
            if "least populated class" in str(e) or "minimum number of groups" in str(e):
                train_rows, test_rows = train_test_split(row_indices, test_size=test_size, random_state=seed, stratify=None)
            else:
                raise e

        class _RowIndexDataset(Dataset):
            def __init__(self, rows: np.ndarray) -> None:
                self.rows = rows.astype(np.int64, copy=False)

            def __len__(self) -> int:
                return int(self.rows.shape[0])

            def __getitem__(self, idx: int) -> int:
                return int(self.rows[idx])

        def _collate_sparse_rows(batch: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
            rows = np.fromiter(batch, dtype=np.int64, count=len(batch))
            xb = X[rows].toarray().astype(np.float32, copy=False)
            yb = y[rows]
            return torch.from_numpy(xb), torch.from_numpy(yb)

        train_loader = DataLoader(
            _RowIndexDataset(train_rows),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_sparse_rows,
        )
        test_loader = DataLoader(
            _RowIndexDataset(test_rows),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_sparse_rows,
        )
        return train_loader, test_loader

    # Try stratified split first, fall back to simple split if stratification fails
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError as e:
        # Handle case where stratification is impossible (e.g., only 1 sample per class)
        if "least populated class" in str(e) or "minimum number of groups" in str(e):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)
        else:
            raise e

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
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create stratified train/val/test loaders from numpy arrays.

    Default splits are 70/15/15. The splits tuple must sum to 1.0.
    """
    train_frac, val_frac, test_frac = splits
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("splits must sum to 1.0")

    # First split off test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_frac, random_state=seed, stratify=y)
    # Compute validation proportion relative to remaining train_val
    denom = max(train_frac + val_frac, 1e-12)
    val_relative = val_frac / denom
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=seed,
        stratify=y_train_val,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def numpy_to_temporal_train_val_test_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    sort_indices: np.ndarray | None = None,
    splits: tuple[float, float, float] = (
        TEMPORAL_TRAIN_FRAC,
        TEMPORAL_VAL_FRAC,
        TEMPORAL_TEST_FRAC,
    ),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders using temporal (positional) split.

    Unlike numpy_to_train_val_test_loaders which uses random stratified splitting,
    this function splits by position in the (pre-sorted) data:
    - Train: first 70% (earliest samples)
    - Val: next 15% (middle samples)
    - Test: final 15% (latest samples)

    Supports both dense numpy arrays and sparse scipy matrices.

    Args:
        X: Feature matrix (n_samples, n_features), dense or sparse
        y: Label vector (n_samples,)
        batch_size: Batch size for DataLoaders
        sort_indices: Optional pre-computed temporal sort indices. If None,
                      assumes X and y are already in temporal order.
        splits: Tuple of (train_frac, val_frac, test_frac), must sum to 1.0

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_frac, val_frac, test_frac = splits
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("splits must sum to 1.0")

    if sort_indices is not None:
        X = X[sort_indices]
        y = y[sort_indices]

    n_samples = X.shape[0]
    train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples, train_frac, val_frac, test_frac)

    if sp is not None and sp.issparse(X):

        class _RowIndexDataset(Dataset):
            def __init__(self, rows: np.ndarray) -> None:
                self.rows = rows.astype(np.int64, copy=False)

            def __len__(self) -> int:
                return int(self.rows.shape[0])

            def __getitem__(self, idx: int) -> int:
                return int(self.rows[idx])

        def _make_collate(X_ref, y_ref):
            def _collate_sparse_rows(batch: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
                rows = np.fromiter(batch, dtype=np.int64, count=len(batch))
                xb = X_ref[rows].toarray().astype(np.float32, copy=False)
                yb = y_ref[rows]
                return torch.from_numpy(xb), torch.from_numpy(yb.astype(np.int64, copy=False))

            return _collate_sparse_rows

        train_ds = _RowIndexDataset(train_idx)
        val_ds = _RowIndexDataset(val_idx)
        test_ds = _RowIndexDataset(test_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_make_collate(X, y))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_make_collate(X, y))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_make_collate(X, y))

        return train_loader, val_loader, test_loader

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32, copy=False)),
        torch.from_numpy(y_train.astype(np.int64, copy=False)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32, copy=False)),
        torch.from_numpy(y_val.astype(np.int64, copy=False)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32, copy=False)),
        torch.from_numpy(y_test.astype(np.int64, copy=False)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_csv_dataset(
    csv_path: str,
    label_col: str,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    # Basic cleanup: drop fully null columns
    df = df.dropna(axis=1, how="all")
    # Drop rows with missing label
    df = df[~df[label_col].isna()].reset_index(drop=True)
    return df


def _collapse_labels_to_binary(series: pd.Series) -> pd.Series:
    benign_tokens = {"BENIGN", "0", "0.0"}
    return series.apply(lambda value: "BENIGN" if value in benign_tokens else "ATTACK")


def load_unsw_nb15(csv_path: str, use_multiclass: bool = True) -> tuple[pd.DataFrame, str, str | None]:
    """
    Load UNSW-NB15 CSV and return (dataframe, label_col, protocol_col).
    Tries common column names across variants.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    # Standardize: some versions use 'Label' with 'Normal'/'Attack' or binary 0/1
    candidate_labels = [
        "label",
        "Label",
        "class",
        "Class",
    ]
    label_col = next((c for c in candidate_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Could not find label column in UNSW-NB15. Tried: 'label', 'Label', 'class', 'Class'")
    # Normalize negative/benign label naming to BENIGN for consistency
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    df[label_col] = df[label_col].replace({"NORMAL": "BENIGN"})

    if not use_multiclass:
        df[label_col] = _collapse_labels_to_binary(df[label_col])

    # Protocol column often 'proto'
    proto_col = "proto" if "proto" in df.columns else None
    # Basic cleanup
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0).reset_index(drop=True)
    return df, label_col, proto_col


def load_cic_ids2017(csv_path: str, use_multiclass: bool = True) -> tuple[pd.DataFrame, str, str | None]:
    """
    Load CIC-IDS2017 CSV and return (dataframe, label_col, protocol_col).
    Tries common column names across merged/day CSVs.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    candidate_labels = [
        "Label",
        "label",
        "Attack",
        "attack",
    ]
    label_col = next((c for c in candidate_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Could not find label column in CIC-IDS2017. Tried: 'Label', 'label', 'Attack', 'attack'")
    # Normalize negative/benign label naming to BENIGN for consistency
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    df[label_col] = df[label_col].replace({"NORMAL": "BENIGN"})
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

    if not use_multiclass:
        df[label_col] = _collapse_labels_to_binary(df[label_col])

    # Protocol column may be 'Protocol', 'ProtocolName', or 'proto'
    proto_col_candidates = ["Protocol", "ProtocolName", "proto"]
    proto_col = next((c for c in proto_col_candidates if c in df.columns), None)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0).reset_index(drop=True)
    return df, label_col, proto_col


def load_edge_iiotset(
    csv_path: str | Path,
    use_multiclass: bool = True,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Load Edge-IIoTset CSV and return (dataframe, label_col, protocol_col).

    Args:
        csv_path: Path to Edge-IIoTset CSV (DNN-EdgeIIoT-dataset.csv or ML-EdgeIIoT-dataset.csv)
        use_multiclass: If True, use Attack_type (15 classes); if False, use Attack_label (binary)
        max_samples: Optional limit on number of samples to load (for tiered testing)

    Returns:
        Tuple of (dataframe, label_column_name, protocol_column_name)

    Note:
        Dataset citation: Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic
        Cyber Security Dataset of IoT and IIoT Applications for Centralized and
        Federated Learning", IEEE Access, 2022.
    """
    df = pd.read_csv(str(csv_path), low_memory=False, nrows=max_samples)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)

    # Choose label column based on classification type
    label_col = "Attack_type" if use_multiclass else "Attack_label"
    if label_col not in df.columns:
        raise ValueError(
            f"Expected label column '{label_col}' not found in Edge-IIoTset dataset. " f"Available columns: {list(df.columns)}"
        )

    # Normalize "Normal" to "BENIGN" for consistency with other datasets
    if use_multiclass and "Attack_type" in df.columns:
        df[label_col] = df[label_col].astype(str).str.strip()
        df[label_col] = df[label_col].replace({"Normal": "BENIGN"})

    # Edge-IIoTset doesn't have a traditional protocol column like UNSW/CIC
    # Protocol info is embedded in features (tcp.*, udp.*, mqtt.*, etc.)
    proto_col = None

    # Drop high-cardinality columns that cause memory explosion during one-hot encoding
    # These are identifiers/metadata that don't contribute to generalizable attack patterns
    drop_cols = [
        "frame.time",  # Timestamps - not predictive features
        "ip.src_host",  # Source IPs - would memorize specific IPs instead of learning patterns
        "ip.dst_host",  # Destination IPs - same issue
        "tcp.payload",  # Raw packet data - too specific, causes overfitting
        "tcp.options",  # TCP flags - too granular
        "tcp.srcport",  # Port as string - duplicate of numeric tcp.dstport
        "http.request.full_uri",  # Full URLs - application-specific
        "http.file_data",  # File content - not statistical features
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Basic cleanup
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    return df, label_col, proto_col


def load_edge_iiotset_with_temporal_order(
    csv_path: str | Path,
    use_multiclass: bool = True,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, str, str | None, np.ndarray, int]:
    """Load Edge-IIoTset CSV with temporal ordering for validation protocol.

    Similar to load_edge_iiotset() but preserves frame.time for temporal ordering
    before dropping it from features.

    Args:
        csv_path: Path to Edge-IIoTset CSV
        use_multiclass: If True, use Attack_type (15 classes); if False, use Attack_label (binary)
        max_samples: Optional limit on number of samples to load

    Returns:
        Tuple of (dataframe, label_col, protocol_col, temporal_sort_indices, n_unparsable)
        where:
        - dataframe: Cleaned DataFrame with frame.time dropped
        - label_col: Name of label column
        - protocol_col: Protocol column name (None for Edge-IIoTset)
        - temporal_sort_indices: Indices that sort data by frame.time
        - n_unparsable: Count of rows with unparsable timestamps
    """
    df = pd.read_csv(str(csv_path), low_memory=False, nrows=max_samples)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)

    label_col = "Attack_type" if use_multiclass else "Attack_label"
    if label_col not in df.columns:
        raise ValueError(
            f"Expected label column '{label_col}' not found in Edge-IIoTset dataset. " f"Available columns: {list(df.columns)}"
        )

    if use_multiclass and "Attack_type" in df.columns:
        df[label_col] = df[label_col].astype(str).str.strip()
        df[label_col] = df[label_col].replace({"Normal": "BENIGN"})

    proto_col = None

    sort_indices, n_unparsable = temporal_sort_indices(df, time_col="frame.time")

    drop_cols = [
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "tcp.payload",
        "tcp.options",
        "tcp.srcport",
        "http.request.full_uri",
        "http.file_data",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.dropna(axis=1, how="all")

    valid_rows_mask = ~df.isna().any(axis=1)
    valid_rows_mask &= ~df.duplicated()

    original_to_new = np.full(len(valid_rows_mask), -1, dtype=np.int64)
    new_idx = 0
    for old_idx in range(len(valid_rows_mask)):
        if valid_rows_mask.iloc[old_idx]:
            original_to_new[old_idx] = new_idx
            new_idx += 1

    df = df[valid_rows_mask].reset_index(drop=True)

    new_sort_indices = []
    for old_idx in sort_indices:
        new_idx = original_to_new[old_idx]
        if new_idx >= 0:
            new_sort_indices.append(new_idx)
    sort_indices = np.array(new_sort_indices, dtype=np.int64)

    return df, label_col, proto_col, sort_indices, n_unparsable


def load_hybrid_dataset(
    csv_path: str | Path,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, str, str | None, str]:
    """Load hybrid IDS dataset with source provenance preserved.

    The hybrid dataset combines CIC-IDS2017, UNSW-NB15, and Edge-IIoTset
    with unified 7-class attack taxonomy and source_dataset column.

    Args:
        csv_path: Path to hybrid dataset CSV (may be gzipped)
        max_samples: Optional limit on number of samples to load

    Returns:
        Tuple of (dataframe, label_col, protocol_col, source_col) where:
        - dataframe: Cleaned DataFrame with features and metadata
        - label_col: "attack_class" (unified 7-class taxonomy)
        - protocol_col: None (not applicable to hybrid)
        - source_col: "source_dataset" (values: cic, unsw, iiot)
    """
    path = Path(csv_path)
    compression = "gzip" if path.suffix == ".gz" else None
    header_df = pd.read_csv(str(path), nrows=0, compression=compression)
    raw_columns = list(header_df.columns)
    normalized_columns = [col.strip() if isinstance(col, str) else col for col in raw_columns]

    label_col = "attack_class"
    source_col = "source_dataset"

    string_cols = {source_col, "attack_label_original"}
    dtype_map = {}
    for raw_col, normalized in zip(raw_columns, normalized_columns):
        if normalized in string_cols:
            dtype_map[raw_col] = "string"
        else:
            dtype_map[raw_col] = np.float32

    file_size = path.stat().st_size if path.exists() else 0
    use_chunks = max_samples is None and file_size >= 100 * 1024 * 1024
    if use_chunks:
        reader = pd.read_csv(
            str(path),
            compression=compression,
            dtype=dtype_map,
            chunksize=200_000,
            engine="python",
        )
        df = pd.concat(reader, ignore_index=True)
    else:
        df = pd.read_csv(
            str(path),
            low_memory=False,
            nrows=max_samples,
            compression=compression,
            dtype=dtype_map,
        )
    df.columns = normalized_columns
    df = df.replace([np.inf, -np.inf], np.nan)

    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}' not found. Available: {list(df.columns)}")
    if source_col not in df.columns:
        raise ValueError(f"Expected source column '{source_col}' not found. Available: {list(df.columns)}")

    metadata_cols = [source_col, label_col, "attack_label_original"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    df[feature_cols] = df[feature_cols].fillna(0).astype(np.float32)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    return df, label_col, None, source_col


def source_aware_partition(
    source_labels: Sequence[str],
    attack_labels: Sequence[int],
    clients_per_source: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
) -> list[list[int]]:
    """Partition data by source dataset with optional within-source Dirichlet.

    Each source dataset (cic, unsw, iiot) is assigned `clients_per_source` clients.
    Within each source, samples are distributed using Dirichlet partitioning
    on the attack labels to create heterogeneity.

    Args:
        source_labels: Sequence of source dataset identifiers (cic, unsw, iiot)
        attack_labels: Sequence of attack class labels (0-6)
        clients_per_source: Number of clients per source dataset
        alpha: Dirichlet concentration for within-source partitioning
        seed: Random seed for reproducibility
        min_samples_per_class: Minimum samples per class per client

    Returns:
        List of index lists, one per client (total = num_sources * clients_per_source)
        Clients are ordered: [cic_0, cic_1, ..., unsw_0, unsw_1, ..., iiot_0, ...]
    """
    source_labels = np.asarray(source_labels)
    attack_labels = np.asarray(attack_labels)
    unique_sources = sorted(set(source_labels))

    all_client_indices: list[list[int]] = []

    for source_idx, source in enumerate(unique_sources):
        source_mask = source_labels == source
        source_indices = np.where(source_mask)[0]
        source_attack_labels = attack_labels[source_indices]

        source_seed = seed + source_idx * 1000

        if np.isinf(alpha):
            within_source_shards = iid_partition(len(source_indices), clients_per_source, source_seed)
        else:
            try:
                within_source_shards = dirichlet_partition(
                    labels=source_attack_labels,
                    num_clients=clients_per_source,
                    alpha=alpha,
                    seed=source_seed,
                    min_samples_per_class=min_samples_per_class,
                )
            except ValueError as e:
                warnings.warn(
                    f"Dirichlet partition infeasible for source '{source}' "
                    f"(alpha={alpha}, clients={clients_per_source}): {e}. "
                    f"Falling back to IID partition.",
                    UserWarning,
                    stacklevel=2,
                )
                within_source_shards = iid_partition(len(source_indices), clients_per_source, source_seed)

        for shard in within_source_shards:
            global_indices = source_indices[shard].tolist()
            all_client_indices.append(global_indices)

    return all_client_indices


def fit_preprocessor_train_only_and_transform_all(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: list[str] | None = None,
    seed: int = 42,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """Split first; fit preprocessor on train only; transform all rows.

    Returns (pre, X_all, y_all) where pre is fitted on the train subset.
    """
    drop_cols = drop_cols or DEFAULT_DROP_COLS
    df_ = df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_all = _encode_labels_to_ints(df_[label_col])
    # Stratified split to get train indices only
    idx = np.arange(len(df_))
    _, idx_train = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y_all)
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
    protocol_col: str | None = None,
    leakage_safe: bool = False,
    protocol_mapping: dict[str, int] | None = None,
    source_col: str | None = None,
) -> tuple[ColumnTransformer, list[np.ndarray], list[np.ndarray], int]:
    # Drop default identifiers/time proxies if leakage_safe
    drop_cols = DEFAULT_DROP_COLS if leakage_safe else None
    if leakage_safe:
        pre, X_all, y_all = fit_preprocessor_train_only_and_transform_all(df, label_col, drop_cols=drop_cols, seed=seed)
    else:
        pre, X_all, y_all = fit_preprocessor_global(df.drop(columns=(drop_cols or []), errors="ignore"), label_col)
    num_classes_global = int(len(np.unique(y_all)))
    if partition_strategy == "iid":
        shards = iid_partition(num_samples=len(df), num_clients=num_clients, seed=seed)
    elif partition_strategy == "dirichlet":
        class_counts = np.bincount(y_all, minlength=num_classes_global)
        min_class_count = int(class_counts.min()) if class_counts.size else 0
        per_client_floor = min_class_count // num_clients if num_clients > 0 else 0
        effective_min_samples = min(MIN_SAMPLES_PER_CLASS, per_client_floor)
        shards = dirichlet_partition(
            labels=y_all,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_class=effective_min_samples,
        )
    elif partition_strategy == "protocol":
        if not protocol_col or protocol_col not in df.columns:
            raise ValueError("protocol partition requires a valid protocol_col present in dataframe")
        shards = protocol_partition(
            df[protocol_col].tolist(),
            num_clients=num_clients,
            seed=seed,
            protocol_mapping=protocol_mapping,
        )
    elif partition_strategy == "source":
        if not source_col or source_col not in df.columns:
            raise ValueError("source partition requires a valid source_col present in dataframe")
        source_labels = df[source_col].astype(str).tolist()
        unique_sources = sorted(set(source_labels))
        if not unique_sources:
            raise ValueError("source partition requires at least one source label")
        if num_clients % len(unique_sources) != 0:
            raise ValueError("source partition requires num_clients divisible by number of sources")
        clients_per_source = num_clients // len(unique_sources)
        shards = source_aware_partition(
            source_labels=source_labels,
            attack_labels=y_all,
            clients_per_source=clients_per_source,
            alpha=alpha,
            seed=seed,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS,
        )
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")

    X_parts = [X_all[np.array(idx_list, dtype=np.int64)] for idx_list in shards]
    y_parts = [y_all[np.array(idx_list, dtype=np.int64)] for idx_list in shards]
    return pre, X_parts, y_parts, num_classes_global
