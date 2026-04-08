"""Unit tests for runs census pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.runs_census import (
    build_dedup_map,
    classify_run_family,
    infer_quality_state,
    reliability_grade,
    run_pipeline,
)


def _write_run(
    base_dir: Path,
    run_id: str,
    config: dict,
    metrics: pd.DataFrame,
) -> None:
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config))
    metrics.to_csv(run_dir / "metrics.csv", index=False)


def test_classify_run_family_precedence() -> None:
    assert classify_run_family("dscic_comp_fedavg_alpha0.5") == "ds_prefixed"
    assert classify_run_family("unsw_simple_abc_comp_krum") == "simple"
    assert classify_run_family("comp_fedavg_alpha1.0_seed42") == "comp_short"
    assert classify_run_family("unsw_p123_comp_fedavg_alpha1.0") == "comp_other"
    assert classify_run_family("archive_20260107") == "other"


def test_infer_quality_state_complete_vs_truncated() -> None:
    complete_df = pd.DataFrame({"round": [1, 2, 3], "global_macro_f1_test": [0.5, 0.6, 0.7]})
    truncated_df = pd.DataFrame({"round": [1, 2], "global_macro_f1_test": [0.5, 0.6]})

    complete = infer_quality_state(
        has_config=True,
        has_metrics=True,
        config_error=False,
        metrics_error=False,
        metrics_df=complete_df,
        target_rounds=3,
    )
    truncated = infer_quality_state(
        has_config=True,
        has_metrics=True,
        config_error=False,
        metrics_error=False,
        metrics_df=truncated_df,
        target_rounds=3,
    )

    assert complete[0] == "complete_or_exceeds_target"
    assert truncated[0] == "truncated_before_target"


def test_build_dedup_map_prefers_complete_run(tmp_path: Path) -> None:
    registry = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "dataset": "cic",
                "aggregation": "fedavg",
                "alpha": 1.0,
                "adversary_fraction": 0.0,
                "adversary_mode": "none",
                "attack_mode": "none",
                "fedprox_mu": 0.0,
                "dp_enabled": False,
                "dp_noise_multiplier": 0.0,
                "personalization_epochs": 0,
                "seed": 42,
                "source_dataset": None,
                "source_datasets": "",
                "quality_state": "truncated_before_target",
                "max_round": 8.0,
                "parsed_row_count": 8,
                "run_mtime": 10.0,
                "adv_percent": 0.0,
            },
            {
                "run_id": "run_b",
                "dataset": "cic",
                "aggregation": "fedavg",
                "alpha": 1.0,
                "adversary_fraction": 0.0,
                "adversary_mode": "none",
                "attack_mode": "none",
                "fedprox_mu": 0.0,
                "dp_enabled": False,
                "dp_noise_multiplier": 0.0,
                "personalization_epochs": 0,
                "seed": 42,
                "source_dataset": None,
                "source_datasets": "",
                "quality_state": "complete_or_exceeds_target",
                "max_round": 20.0,
                "parsed_row_count": 20,
                "run_mtime": 5.0,
                "adv_percent": 0.0,
            },
        ]
    )

    dedup = build_dedup_map(registry)
    canonical = dedup[dedup["is_canonical"]]

    assert len(canonical) == 1
    assert canonical.iloc[0]["run_id"] == "run_b"


def test_reliability_grade_thresholds() -> None:
    assert reliability_grade(12) == "A"
    assert reliability_grade(5) == "B"
    assert reliability_grade(3) == "C"
    assert reliability_grade(2) == "D"


def test_run_pipeline_writes_expected_artifacts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    common = {
        "dataset": "cic",
        "aggregation": "fedavg",
        "alpha": 1.0,
        "adversary_fraction": 0.0,
        "fedprox_mu": 0.0,
        "dp_enabled": False,
        "dp_noise_multiplier": 0.0,
        "personalization_epochs": 0,
        "num_clients": 10,
        "num_rounds": 3,
        "seed": 42,
    }

    metrics_ok = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "global_macro_f1_test": [0.4, 0.5, 0.6],
            "n_test_total": [100, 100, 100],
        }
    )
    _write_run(runs_dir, "comp_fedavg_alpha1.0_seed42", common, metrics_ok)

    metrics_short = pd.DataFrame(
        {
            "round": [1, 2],
            "global_macro_f1_test": [0.4, 0.5],
            "n_test_total": [100, 100],
        }
    )
    short_cfg = dict(common)
    short_cfg["seed"] = 43
    _write_run(runs_dir, "comp_fedavg_alpha1.0_seed43", short_cfg, metrics_short)

    output_dir = tmp_path / "out"
    summary = run_pipeline(runs_dir=runs_dir, output_dir=output_dir, write_parquet=False)

    expected_files = {
        "runs_registry.csv",
        "schema_drift_config.csv",
        "schema_drift_metrics.csv",
        "runs_quality_states.csv",
        "runs_dedup_map.csv",
        "coverage_confirmatory.csv",
        "coverage_exploratory.csv",
        "gap_inventory.csv",
        "claim_ledger.csv",
        "runs_data_dictionary.md",
    }

    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    assert summary["runs_registry_rows"] == 2


def test_run_pipeline_preserves_heterogeneity_cells_in_gap_inventory(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    metrics = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "global_macro_f1_test": [0.21, 0.24, 0.27],
            "n_test_total": [100, 100, 100],
        }
    )

    for seed in [42, 43, 44]:
        config = {
            "dataset": "cic",
            "aggregation": "fedavg",
            "alpha": 0.02,
            "adversary_fraction": 0.0,
            "fedprox_mu": 0.0,
            "dp_enabled": False,
            "dp_noise_multiplier": 0.0,
            "personalization_epochs": 0,
            "num_clients": 10,
            "num_rounds": 3,
            "seed": seed,
        }
        _write_run(
            runs_dir,
            f"comp_fedavg_alpha0.02_seed{seed}",
            config,
            metrics,
        )

    output_dir = tmp_path / "out"
    run_pipeline(runs_dir=runs_dir, output_dir=output_dir, write_parquet=False)

    gap_inventory = pd.read_csv(output_dir / "gap_inventory.csv")
    heterogeneity_cell = gap_inventory[
        (gap_inventory["slice"] == "heterogeneity_fedavg")
        & (gap_inventory["dataset"] == "cic")
        & (gap_inventory["aggregation"] == "fedavg")
        & (gap_inventory["alpha"] == 0.02)
    ]

    assert len(heterogeneity_cell) == 1
    assert heterogeneity_cell.iloc[0]["cell_status"] == "exploratory_only"
