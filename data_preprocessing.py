from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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

    Resamples up to MAX_PARTITION_ATTEMPTS times to ensure each client receives at least
    min_samples_per_class samples from EVERY class. This prevents pathological imbalance
    where clients miss entire classes, which breaks stratified splitting and metrics.

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

    # Attempt partitioning with resampling
    for attempt in range(MAX_PARTITION_ATTEMPTS):
        client_indices = _attempt_dirichlet_partition_single(labels, num_clients, alpha, rng)

        # Validate constraints
        is_valid, diagnostic = _validate_partition_constraints(client_indices, labels, min_samples_per_class)

        if is_valid:
            return client_indices

        # Log warning on later attempts
        if attempt > MAX_PARTITION_ATTEMPTS // 2:
            import logging

            logging.warning(f"Dirichlet partition attempt {attempt + 1}/{MAX_PARTITION_ATTEMPTS} " f"failed constraint: {diagnostic}")

    # Failed after all attempts
    num_classes = int(labels.max()) + 1
    raise ValueError(
        f"Failed to create valid Dirichlet partition after {MAX_PARTITION_ATTEMPTS} attempts. "
        f"Constraints: {num_clients} clients, {len(labels)} samples, {num_classes} classes, "
        f"alpha={alpha:.4f}, min_samples_per_class={min_samples_per_class}. "
        f"Last attempt violation: {diagnostic}. "
        f"Suggestions: (1) Increase dataset size, (2) Decrease num_clients, "
        f"(3) Increase alpha (less heterogeneous), (4) Decrease min_samples_per_class"
    )


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
    if len(X) == 0 or len(y) == 0:
        # Return dummy loaders with minimal tensors
        dummy_X = torch.zeros((1, X.shape[1] if X.size > 0 else 1), dtype=torch.float32)
        dummy_y = torch.zeros((1,), dtype=torch.long)
        dummy_ds = TensorDataset(dummy_X, dummy_y)
        dummy_loader = DataLoader(dummy_ds, batch_size=batch_size, shuffle=False)
        return dummy_loader, dummy_loader

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


def load_unsw_nb15(csv_path: str) -> tuple[pd.DataFrame, str, str | None]:
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

    # Protocol column often 'proto'
    proto_col = "proto" if "proto" in df.columns else None
    # Basic cleanup
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0).reset_index(drop=True)
    return df, label_col, proto_col


def load_cic_ids2017(csv_path: str) -> tuple[pd.DataFrame, str, str | None]:
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
        shards = dirichlet_partition(labels=y_all, num_clients=num_clients, alpha=alpha, seed=seed)
    elif partition_strategy == "protocol":
        if not protocol_col or protocol_col not in df.columns:
            raise ValueError("protocol partition requires a valid protocol_col present in dataframe")
        shards = protocol_partition(
            df[protocol_col].tolist(),
            num_clients=num_clients,
            seed=seed,
            protocol_mapping=protocol_mapping,
        )
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")

    X_parts = [X_all[np.array(idx_list, dtype=np.int64)] for idx_list in shards]
    y_parts = [y_all[np.array(idx_list, dtype=np.int64)] for idx_list in shards]
    return pre, X_parts, y_parts, num_classes_global
