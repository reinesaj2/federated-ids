"""Data loading and transformation utilities for plotting."""

from plots.data.loader import DataLoader, load_chapter4_data
from plots.data.transforms import (
    aggregate_by_seed,
    compute_confidence_interval,
    pivot_for_heatmap,
)
from plots.data.validators import MetricDetector

__all__ = [
    "DataLoader",
    "MetricDetector",
    "aggregate_by_seed",
    "compute_confidence_interval",
    "load_chapter4_data",
    "pivot_for_heatmap",
]
