from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np


class ClientMetricsLogger:
    """Handles CSV logging of client-side federated learning metrics."""

    def __init__(self, csv_path: str, client_id: int) -> None:
        """Initialize the client metrics logger with a CSV file path and client ID."""
        self.csv_path = Path(csv_path)
        self.client_id = client_id
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        # Create parent directories if needed
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with headers if it doesn't exist
        if not self.csv_path.exists():
            headers = [
                "client_id", "round", "dataset_size", "n_classes",
                "loss_before", "acc_before", "loss_after", "acc_after",
                "weight_norm_before", "weight_norm_after", "weight_update_norm",
                "t_fit_ms", "epochs_completed", "lr", "batch_size"
            ]
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_round_metrics(
        self,
        round_num: int,
        dataset_size: int,
        n_classes: int,
        loss_before: Optional[float],
        acc_before: Optional[float],
        loss_after: Optional[float],
        acc_after: Optional[float],
        weight_norm_before: Optional[float],
        weight_norm_after: Optional[float],
        weight_update_norm: Optional[float],
        t_fit_ms: Optional[float],
        epochs_completed: int,
        lr: float,
        batch_size: int,
    ) -> None:
        """Log metrics for a single client training round."""
        row = [
            str(self.client_id),
            str(round_num),
            str(dataset_size),
            str(n_classes),
            str(loss_before) if loss_before is not None else "",
            str(acc_before) if acc_before is not None else "",
            str(loss_after) if loss_after is not None else "",
            str(acc_after) if acc_after is not None else "",
            str(weight_norm_before) if weight_norm_before is not None else "",
            str(weight_norm_after) if weight_norm_after is not None else "",
            str(weight_update_norm) if weight_update_norm is not None else "",
            str(t_fit_ms) if t_fit_ms is not None else "",
            str(epochs_completed),
            str(lr),
            str(batch_size),
        ]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class ClientFitTimer:
    """Utility for measuring client fit timing."""

    def __init__(self) -> None:
        self._last_fit_time_ms: Optional[float] = None

    @contextmanager
    def time_fit(self):
        """Context manager for timing client fit operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self._last_fit_time_ms = (end_time - start_time) * 1000.0

    def get_last_fit_time_ms(self) -> Optional[float]:
        """Get the time in milliseconds for the last fit operation."""
        return self._last_fit_time_ms


def calculate_weight_norms(weights: List[np.ndarray]) -> float:
    """Calculate the L2 norm of a list of weight arrays."""
    total_norm_sq = 0.0
    for arr in weights:
        total_norm_sq += float(np.sum(arr * arr))
    return np.sqrt(total_norm_sq)


def calculate_weight_update_norm(
    weights_before: List[np.ndarray], weights_after: List[np.ndarray]
) -> float:
    """Calculate the L2 norm of the weight update (difference)."""
    total_norm_sq = 0.0
    for arr_before, arr_after in zip(weights_before, weights_after):
        diff = arr_after - arr_before
        total_norm_sq += float(np.sum(diff * diff))
    return np.sqrt(total_norm_sq)


def analyze_data_distribution(labels: np.ndarray) -> dict[str, int]:
    """Analyze the data distribution for a client."""
    unique_labels = np.unique(labels)
    return {
        "dataset_size": len(labels),
        "n_classes": len(unique_labels),
    }


def create_label_histogram_json(labels: np.ndarray) -> str:
    """Create a JSON representation of label distribution histogram."""
    import json
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_hist = {str(label): int(count) for label, count in zip(unique_labels, counts)}
    return json.dumps(label_hist, sort_keys=True)