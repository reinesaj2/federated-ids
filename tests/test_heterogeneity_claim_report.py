"""Unit tests for heterogeneity claim reporting."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.heterogeneity_claim_report import (
    build_regime_summary,
    classify_heterogeneity_regime,
    tightened_claim,
)


def test_classify_heterogeneity_regime_detects_iid_favored() -> None:
    alpha_to_metric = {
        0.02: 0.39,
        0.05: 0.49,
        0.1: 0.59,
        0.2: 0.64,
        0.5: 0.67,
        1.0: 0.68,
        float("inf"): 0.71,
    }

    assert classify_heterogeneity_regime(alpha_to_metric) == "iid_favored"


def test_classify_heterogeneity_regime_detects_rare_class_concentration() -> None:
    alpha_to_metric = {
        0.02: 0.74,
        0.05: 0.53,
        0.1: 0.52,
        0.2: 0.44,
        0.5: 0.39,
        1.0: 0.43,
        float("inf"): 0.42,
    }

    assert classify_heterogeneity_regime(alpha_to_metric) == "rare_class_concentration_favored"


def test_build_regime_summary_and_claim_text() -> None:
    heterogeneity = pd.DataFrame(
        [
            {"dataset": "cic", "alpha": 0.02, "metric_mean": 0.33},
            {"dataset": "cic", "alpha": float("inf"), "metric_mean": 0.52},
            {"dataset": "edge-iiotset-full", "alpha": 0.02, "metric_mean": 0.39},
            {"dataset": "edge-iiotset-full", "alpha": float("inf"), "metric_mean": 0.71},
            {"dataset": "unsw", "alpha": 0.02, "metric_mean": 0.74},
            {"dataset": "unsw", "alpha": float("inf"), "metric_mean": 0.42},
        ]
    )

    summary = build_regime_summary(heterogeneity)

    assert set(summary["dataset"]) == {"cic", "edge-iiotset-full", "unsw"}
    assert tightened_claim(summary) == (
        "FedAvg heterogeneity response is dataset-dependent: CIC-IDS2017, Edge-IIoTset improve as partitions "
        "approach IID, while UNSW-NB15 benefits from extreme heterogeneity because minority attacks become "
        "concentrated within fewer clients."
    )
