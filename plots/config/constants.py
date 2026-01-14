"""
Plotting Constants

Single source of truth for all plotting-related constants across the thesis.
Consolidates definitions from multiple legacy scripts to ensure consistency.
"""

from typing import TypedDict

from plots.config.style import PALETTES


class DatasetConfigEntry(TypedDict):
    """Type definition for dataset configuration entries."""

    label: str
    color: str
    marker: str


AGGREGATOR_ORDER: list[str] = ["fedavg", "krum", "bulyan", "median"]

AGGREGATOR_LABELS: dict[str, str] = {
    "fedavg": "FedAvg",
    "krum": "Krum",
    "bulyan": "Bulyan",
    "median": "Median",
    "fedprox": "FedProx",
}

_colorblind = PALETTES["colorblind"]

AGGREGATOR_COLORS: dict[str, str] = {
    "fedavg": _colorblind[0],
    "krum": _colorblind[1],
    "bulyan": _colorblind[2],
    "median": _colorblind[3],
}

DATASETS: list[str] = ["iiot", "cic", "unsw"]

DATASET_CONFIG: dict[str, DatasetConfigEntry] = {
    "iiot": {
        "label": "Edge-IIoTset",
        "color": _colorblind[0],
        "marker": "o",
    },
    "cic": {
        "label": "CIC-IDS2017",
        "color": _colorblind[1],
        "marker": "s",
    },
    "unsw": {
        "label": "UNSW-NB15",
        "color": _colorblind[2],
        "marker": "^",
    },
}

ADVERSARIAL_LEVELS: list[int] = [0, 10, 20, 30]

ALPHA_VALUES: list[float] = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

MU_VALUES: list[float] = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2]
