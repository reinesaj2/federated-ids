#!/usr/bin/env python3
"""Emit CIC-IDS2017 experiment manifest summaries."""

import json
import sys
from pathlib import Path
from typing import Dict, List

import math

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.comparative_analysis import ComparisonMatrix


def _serialize(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def summarize_dimension(matrix: ComparisonMatrix, dimension: str) -> Dict:
    configs = matrix.generate_configs(filter_dimension=dimension)
    presets = [cfg.to_preset_name() for cfg in configs]
    return {
        "dimension": dimension,
        "count": len(configs),
        "datasets": sorted({cfg.dataset for cfg in configs}),
        "alpha_values": sorted({_serialize(cfg.alpha) for cfg in configs}),
        "mu_values": sorted({_serialize(cfg.fedprox_mu) for cfg in configs}),
        "sample_presets": presets[:5],
    }


def main() -> None:
    base_dir = Path.cwd()
    output_dir = base_dir / "analysis" / "cic_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = ComparisonMatrix(
        dataset="cic",
        data_path="data/cic/cic_ids2017_multiclass.csv",
    )

    dimensions: List[str] = [
        "aggregation",
        "attack",
        "heterogeneity",
        "privacy",
        "personalization",
    ]

    plan = {
        "dataset": matrix.dataset,
        "data_path": matrix.data_path,
        "num_clients": matrix.num_clients,
        "num_rounds": matrix.num_rounds,
        "seeds": matrix.seeds,
        "alpha_values": [_serialize(v) for v in matrix.alpha_values],
        "fedprox_mu_values": [_serialize(v) for v in matrix.fedprox_mu_values],
        "dimensions": [summarize_dimension(matrix, dim) for dim in dimensions],
    }

    manifest_path = output_dir / "cic_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(plan, fh, indent=2)

    print(f"Wrote CIC manifest to {manifest_path}")


if __name__ == "__main__":
    main()
