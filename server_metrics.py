from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np

from robust_aggregation import AggregationMethod
from logging_utils import get_logger

logger = get_logger(__name__)


class ServerMetricsLogger:
    """Handles CSV logging of server-side federated learning metrics."""

    def __init__(self, csv_path: str) -> None:
        """Initialize the metrics logger with a CSV file path."""
        self.csv_path = Path(csv_path)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        # Create parent directories if needed
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with headers if it doesn't exist
        if not self.csv_path.exists():
            headers = [
                "round",
                "agg_method",
                "n_clients",
                "byzantine_f",
                "l2_to_benign_mean",
                "cos_to_benign_mean",
                "coord_median_agree_pct",
                "update_norm_mean",
                "update_norm_std",
                "t_aggregate_ms",
                "t_round_ms",
                "pairwise_cosine_mean",
                "pairwise_cosine_std",
                "l2_dispersion_mean",
                "l2_dispersion_std",
            ]
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_round_metrics(
        self,
        round_num: int,
        agg_method: AggregationMethod,
        n_clients: int,
        byzantine_f: Optional[int],
        l2_to_benign_mean: Optional[float],
        cos_to_benign_mean: Optional[float],
        coord_median_agree_pct: Optional[float],
        update_norm_mean: Optional[float],
        update_norm_std: Optional[float],
        t_aggregate_ms: Optional[float],
        t_round_ms: Optional[float],
        pairwise_cosine_mean: Optional[float] = None,
        pairwise_cosine_std: Optional[float] = None,
        l2_dispersion_mean: Optional[float] = None,
        l2_dispersion_std: Optional[float] = None,
    ) -> None:
        """Log metrics for a single federated learning round."""
        row = [
            str(round_num),
            agg_method.value,
            str(n_clients),
            str(byzantine_f) if byzantine_f is not None else "",
            str(l2_to_benign_mean) if l2_to_benign_mean is not None else "",
            str(cos_to_benign_mean) if cos_to_benign_mean is not None else "",
            str(coord_median_agree_pct) if coord_median_agree_pct is not None else "",
            str(update_norm_mean) if update_norm_mean is not None else "",
            str(update_norm_std) if update_norm_std is not None else "",
            str(t_aggregate_ms) if t_aggregate_ms is not None else "",
            str(t_round_ms) if t_round_ms is not None else "",
            str(pairwise_cosine_mean) if pairwise_cosine_mean is not None else "",
            str(pairwise_cosine_std) if pairwise_cosine_std is not None else "",
            str(l2_dispersion_mean) if l2_dispersion_mean is not None else "",
            str(l2_dispersion_std) if l2_dispersion_std is not None else "",
        ]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class AggregationTimer:
    """Utility for measuring aggregation timing."""

    def __init__(self) -> None:
        self._last_aggregation_time_ms: Optional[float] = None

    @contextmanager
    def time_aggregation(self):
        """Context manager for timing aggregation operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self._last_aggregation_time_ms = (end_time - start_time) * 1000.0

    def get_last_aggregation_time_ms(self) -> Optional[float]:
        """Get the time in milliseconds for the last aggregation operation."""
        return self._last_aggregation_time_ms


def validate_metrics(metrics: dict[str, float], dimension: str, min_expected_cosine: float = 0.5) -> list[str]:
    """Validate robustness metrics for data quality issues.

    Args:
        metrics: Dictionary of computed metrics
        dimension: Description of what dimension/experiment these metrics are from
        min_expected_cosine: Minimum expected cosine similarity for same-architecture
            FL models (default 0.5). Lower values trigger warning but not error.

    Returns:
        List of warning messages (empty if no issues detected)
    """
    warnings = []

    # Validate cosine similarity
    if "cos_to_benign_mean" in metrics:
        cosine = metrics["cos_to_benign_mean"]
        if cosine is not None and not np.isnan(cosine):
            if not (-1.0 <= cosine <= 1.0):
                warnings.append(f"{dimension}: Cosine {cosine:.6f} outside valid range [-1, 1]")
            elif cosine < min_expected_cosine:
                warnings.append(
                    f"{dimension}: Suspiciously low cosine {cosine:.6f} for FL models "
                    f"(expected >{min_expected_cosine} for same-architecture models)"
                )

    # Validate L2 distance
    if "l2_to_benign_mean" in metrics:
        l2 = metrics["l2_to_benign_mean"]
        if l2 is not None and not np.isnan(l2):
            if l2 < 0.0:
                warnings.append(f"{dimension}: Negative L2 distance {l2:.6f}")
            elif l2 == 0.0:
                warnings.append(f"{dimension}: L2 distance is exactly 0.0 " "(aggregated model identical to reference)")

    # Validate update norms
    if "update_norm_mean" in metrics:
        norm_mean = metrics["update_norm_mean"]
        if norm_mean is not None and norm_mean < 0.0:
            warnings.append(f"{dimension}: Negative norm mean {norm_mean:.6f}")

    if "update_norm_std" in metrics:
        norm_std = metrics["update_norm_std"]
        if norm_std is not None and norm_std < 0.0:
            warnings.append(f"{dimension}: Negative norm std {norm_std:.6f}")

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    return warnings


def calculate_robustness_metrics(
    client_updates: List[List[np.ndarray]],
    benign_mean: List[np.ndarray],
    aggregated: List[np.ndarray],
) -> dict[str, float]:
    """Calculate robustness metrics for federated learning aggregation."""

    def _flatten_update(update: List[np.ndarray]) -> np.ndarray:
        """Flatten a multi-layer update into a single vector."""
        return np.concatenate([arr.reshape(-1) for arr in update])

    def _l2_distance(a: List[np.ndarray], b: List[np.ndarray]) -> float:
        """Calculate L2 distance between two multi-layer updates."""
        total_dist_sq = 0.0
        for arr_a, arr_b in zip(a, b):
            total_dist_sq += float(np.sum((arr_a - arr_b) ** 2))
        return np.sqrt(total_dist_sq)

    def _cosine_similarity(a: List[np.ndarray], b: List[np.ndarray]) -> float:
        """Calculate cosine similarity between two multi-layer updates.

        Args:
            a: First multi-layer update
            b: Second multi-layer update

        Returns:
            Cosine similarity in range [-1.0, 1.0]

        Raises:
            ValueError: If either vector has zero norm (cosine undefined)
        """
        flat_a = _flatten_update(a)
        flat_b = _flatten_update(b)

        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)

        if norm_a == 0.0:
            raise ValueError(f"First vector has zero norm (length={len(flat_a)}). " "Cosine similarity is undefined for zero vectors.")
        if norm_b == 0.0:
            raise ValueError(f"Second vector has zero norm (length={len(flat_b)}). " "Cosine similarity is undefined for zero vectors.")

        cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

        # Clamp to [-1, 1] range to handle floating point errors
        # Cauchy-Schwarz guarantees cosine is in [-1, 1] mathematically
        cosine = np.clip(cosine, -1.0, 1.0)

        # Handle floating point inaccuracies with strict tolerance
        # Use rtol=1e-10 to avoid rounding DP-induced variations (0.99999x) to 1.0
        if np.isclose(cosine, 1.0, rtol=1e-10, atol=1e-12):
            cosine = 1.0
        elif np.isclose(cosine, -1.0, rtol=1e-10, atol=1e-12):
            cosine = -1.0

        return cosine

    def _coordinate_median_agreement(client_updates: List[List[np.ndarray]], aggregated: List[np.ndarray]) -> float:
        """Calculate percentage of coordinates where aggregated equals coordinate-wise median."""
        if not client_updates:
            return 0.0

        # Calculate coordinate-wise median
        num_layers = len(client_updates[0])
        median_update: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            layer_stack = np.stack([client[layer_idx] for client in client_updates], axis=0)
            median_layer = np.median(layer_stack, axis=0)
            median_update.append(median_layer)

        # Compare aggregated to median coordinate-wise
        total_coords = 0
        matching_coords = 0

        for agg_layer, median_layer in zip(aggregated, median_update):
            total_coords += int(agg_layer.size)
            # Use small tolerance for floating point comparison
            matches = np.isclose(agg_layer, median_layer, rtol=1e-10, atol=1e-10)
            matching_coords += int(np.sum(matches))

        return float(matching_coords / total_coords * 100.0) if total_coords > 0 else 0.0

    def _update_norm_stats(client_updates: List[List[np.ndarray]]) -> tuple[float, float]:
        """Calculate mean and std of client update norms."""
        norms = []
        for update in client_updates:
            flat_update = _flatten_update(update)
            norm = float(np.linalg.norm(flat_update))
            norms.append(norm)

        if not norms:
            return 0.0, 0.0

        return float(np.mean(norms)), float(np.std(norms))

    # Calculate all metrics
    l2_to_benign = _l2_distance(aggregated, benign_mean)
    cos_to_benign = _cosine_similarity(aggregated, benign_mean)
    coord_median_agree = _coordinate_median_agreement(client_updates, aggregated)
    norm_mean, norm_std = _update_norm_stats(client_updates)

    return {
        "l2_to_benign_mean": l2_to_benign,
        "cos_to_benign_mean": cos_to_benign,
        "coord_median_agree_pct": coord_median_agree,
        "update_norm_mean": norm_mean,
        "update_norm_std": norm_std,
    }
