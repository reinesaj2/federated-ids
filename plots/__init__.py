"""
Unified Plotting Package for Federated Learning Thesis

This package consolidates all plotting functionality into a single,
consistent, and maintainable codebase.

Usage:
    python -m plotting chapter4 --data results/summary.csv --output results/plots
    python -m plotting thesis --data results/experiments/ --output results/plots
"""

from plots.config.constants import (
    ADVERSARIAL_LEVELS,
    AGGREGATOR_COLORS,
    AGGREGATOR_LABELS,
    AGGREGATOR_ORDER,
    ALPHA_VALUES,
    DATASET_CONFIG,
    DATASETS,
    MU_VALUES,
)
from plots.config.style import PALETTES, PlotStyle, ThesisStyle

__all__ = [
    "ADVERSARIAL_LEVELS",
    "AGGREGATOR_COLORS",
    "AGGREGATOR_LABELS",
    "AGGREGATOR_ORDER",
    "ALPHA_VALUES",
    "DATASET_CONFIG",
    "DATASETS",
    "MU_VALUES",
    "PALETTES",
    "PlotStyle",
    "ThesisStyle",
]

__version__ = "1.0.0"
