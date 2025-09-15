from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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
    x1 = rng.multivariate_normal(mean=means + 0.5, cov=cov, size=num_samples - x0.shape[0])
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.array([0] * x0.shape[0] + [1] * x1.shape[0], dtype=np.int64)

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


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition indices into num_clients shards using a Dirichlet distribution over label proportions.
    Returns a list of index lists per client. Intended for future use with real datasets.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    # For each class, sample proportions for clients
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c, idxs in enumerate(class_indices):
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Split idxs according to proportions
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        shards = np.split(idxs, splits)
        for i, shard in enumerate(shards):
            client_indices[i].extend(shard.tolist())

    # Shuffle within each client shard
    for shard in client_indices:
        rng.shuffle(shard)

    return client_indices
