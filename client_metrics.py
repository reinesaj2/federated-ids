from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np


class ClientMetricsLogger:
    """Handles CSV logging of client-side federated learning metrics."""

    def __init__(
        self, csv_path: str, client_id: int, extended: Optional[bool] = None
    ) -> None:
        """Initialize the client metrics logger with a CSV file path and client ID."""
        self.csv_path = Path(csv_path)
        self.client_id = client_id
        if extended is None:
            import os as _os

            self.extended = _os.environ.get("D2_EXTENDED_METRICS", "0").lower() not in (
                "0",
                "false",
                "no",
                "",
            )
        else:
            self.extended = bool(extended)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        # Create parent directories if needed
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with headers if it doesn't exist
        if not self.csv_path.exists():
            if self.extended:
                headers = [
                    "client_id",
                    "round",
                    "dataset_size",
                    "n_classes",
                    "loss_before",
                    "acc_before",
                    "loss_after",
                    "acc_after",
                    "macro_f1_before",
                    "macro_f1_after",
                    "macro_f1_argmax",
                    "benign_fpr_argmax",
                    "f1_per_class_after",
                    "fpr_after",
                    "pr_auc_after",
                    "threshold_tau",
                    "f1_bin_tau",
                    "benign_fpr_bin_tau",
                    "tau_bin",
                    "seed",
                    "weight_norm_before",
                    "weight_norm_after",
                    "weight_update_norm",
                    "grad_norm_l2",
                    "t_fit_ms",
                    "epochs_completed",
                    "lr",
                    "batch_size",
                ]
            else:
                headers = [
                    "client_id",
                    "round",
                    "dataset_size",
                    "n_classes",
                    "loss_before",
                    "acc_before",
                    "loss_after",
                    "acc_after",
                    "weight_norm_before",
                    "weight_norm_after",
                    "weight_update_norm",
                    "t_fit_ms",
                    "epochs_completed",
                    "lr",
                    "batch_size",
                ]
            with open(self.csv_path, "w", newline="") as f:
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
        macro_f1_before: Optional[float] = None,
        macro_f1_after: Optional[float] = None,
        macro_f1_argmax: Optional[float] = None,
        benign_fpr_argmax: Optional[float] = None,
        f1_per_class_after_json: Optional[str] = None,
        fpr_after: Optional[float] = None,
        pr_auc_after: Optional[float] = None,
        threshold_tau: Optional[float] = None,
        f1_bin_tau: Optional[float] = None,
        benign_fpr_bin_tau: Optional[float] = None,
        tau_bin: Optional[float] = None,
        seed: Optional[int] = None,
        weight_norm_before: Optional[float] = None,
        weight_norm_after: Optional[float] = None,
        weight_update_norm: Optional[float] = None,
        grad_norm_l2: Optional[float] = None,
        t_fit_ms: Optional[float] = None,
        epochs_completed: int = 0,
        lr: float = 0.0,
        batch_size: int = 0,
    ) -> None:
        """Log metrics for a single client training round."""
        if self.extended:
            row = [
                str(self.client_id),
                str(round_num),
                str(dataset_size),
                str(n_classes),
                str(loss_before) if loss_before is not None else "",
                str(acc_before) if acc_before is not None else "",
                str(loss_after) if loss_after is not None else "",
                str(acc_after) if acc_after is not None else "",
                str(macro_f1_before) if macro_f1_before is not None else "",
                str(macro_f1_after) if macro_f1_after is not None else "",
                str(macro_f1_argmax) if macro_f1_argmax is not None else "",
                str(benign_fpr_argmax) if benign_fpr_argmax is not None else "",
                f1_per_class_after_json or "",
                str(fpr_after) if fpr_after is not None else "",
                str(pr_auc_after) if pr_auc_after is not None else "",
                str(threshold_tau) if threshold_tau is not None else "",
                str(f1_bin_tau) if f1_bin_tau is not None else "",
                str(benign_fpr_bin_tau) if benign_fpr_bin_tau is not None else "",
                str(tau_bin) if tau_bin is not None else "",
                str(seed) if seed is not None else "",
                str(weight_norm_before) if weight_norm_before is not None else "",
                str(weight_norm_after) if weight_norm_after is not None else "",
                str(weight_update_norm) if weight_update_norm is not None else "",
                    str(grad_norm_l2) if grad_norm_l2 is not None else "",
                str(t_fit_ms) if t_fit_ms is not None else "",
                str(epochs_completed),
                str(lr),
                str(batch_size),
            ]
        else:
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

        with open(self.csv_path, "a", newline="") as f:
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
