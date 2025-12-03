#!/usr/bin/env python3
"""Unit tests for dataset filtering in generate_thesis_plots.py"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from generate_thesis_plots import load_experiment_results


@pytest.fixture
def temp_runs_dir():
    """Create temporary runs directory with mixed dataset experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir)

        # Create Edge-IIoT experiment
        edge_dir = runs_dir / "comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42"
        edge_dir.mkdir(parents=True)

        edge_config = {
            "aggregation": "fedavg",
            "alpha": 1.0,
            "dataset": "edge-iiotset-full",
            "seed": 42,
        }
        (edge_dir / "config.json").write_text(json.dumps(edge_config))

        edge_metrics = pd.DataFrame({"round": [1, 2, 3], "loss": [0.5, 0.4, 0.3], "accuracy": [0.8, 0.85, 0.9]})
        edge_metrics.to_csv(edge_dir / "metrics.csv", index=False)

        # Create client metrics for compute_server_macro_f1_from_clients
        (edge_dir / "client_0_metrics.csv").write_text("round,macro_f1_after\n1,0.80\n2,0.85\n3,0.90\n")

        # Create CIC-IDS2017 experiment
        cic_dir = runs_dir / "comp_krum_alpha0.5_adv0_dp0_pers0_mu0.0_seed42"
        cic_dir.mkdir(parents=True)

        cic_config = {
            "aggregation": "krum",
            "alpha": 0.5,
            "dataset": "cic-ids2017",
            "seed": 42,
        }
        (cic_dir / "config.json").write_text(json.dumps(cic_config))

        cic_metrics = pd.DataFrame({"round": [1, 2, 3], "loss": [0.6, 0.5, 0.4], "accuracy": [0.75, 0.80, 0.85]})
        cic_metrics.to_csv(cic_dir / "metrics.csv", index=False)

        (cic_dir / "client_0_metrics.csv").write_text("round,macro_f1_after\n1,0.75\n2,0.80\n3,0.85\n")

        # Create another Edge-IIoT experiment
        edge2_dir = runs_dir / "comp_bulyan_alpha0.02_adv0_dp0_pers0_mu0.0_seed43"
        edge2_dir.mkdir(parents=True)

        edge2_config = {
            "aggregation": "bulyan",
            "alpha": 0.02,
            "dataset": "edge-iiotset-full",
            "seed": 43,
        }
        (edge2_dir / "config.json").write_text(json.dumps(edge2_config))

        edge2_metrics = pd.DataFrame({"round": [1, 2], "loss": [0.7, 0.6], "accuracy": [0.7, 0.75]})
        edge2_metrics.to_csv(edge2_dir / "metrics.csv", index=False)

        (edge2_dir / "client_0_metrics.csv").write_text("round,macro_f1_after\n1,0.70\n2,0.75\n")

        yield runs_dir


def test_load_all_datasets_no_filter(temp_runs_dir):
    """Test loading all datasets when no filter is specified."""
    df = load_experiment_results(temp_runs_dir, dataset_filter=None)

    assert len(df) > 0, "Should load experiments when no filter specified"
    assert "dataset" in df.columns, "Dataset column should be present"

    datasets = df["dataset"].unique()
    assert "edge-iiotset-full" in datasets, "Should include Edge-IIoT experiments"
    assert "cic-ids2017" in datasets, "Should include CIC-IDS2017 experiments"


def test_filter_edge_iiot_only(temp_runs_dir):
    """Test filtering to load only Edge-IIoT dataset experiments."""
    df = load_experiment_results(temp_runs_dir, dataset_filter="edge-iiotset-full")

    assert len(df) > 0, "Should load Edge-IIoT experiments"
    assert all(df["dataset"] == "edge-iiotset-full"), "All experiments should be edge-iiotset-full"

    # Should have 2 Edge-IIoT experiments (seed 42 and 43)
    unique_runs = df["run_dir"].unique()
    assert len(unique_runs) == 2, "Should load exactly 2 Edge-IIoT experiments"

    # Verify aggregation methods
    aggregations = df["aggregation"].unique()
    assert "fedavg" in aggregations, "Should include FedAvg experiment"
    assert "bulyan" in aggregations, "Should include Bulyan experiment"
    assert "krum" not in aggregations, "Should NOT include Krum (CIC-IDS2017)"


def test_filter_cic_only(temp_runs_dir):
    """Test filtering to load only CIC-IDS2017 dataset experiments."""
    df = load_experiment_results(temp_runs_dir, dataset_filter="cic-ids2017")

    assert len(df) > 0, "Should load CIC-IDS2017 experiments"
    assert all(df["dataset"] == "cic-ids2017"), "All experiments should be cic-ids2017"

    # Should have 1 CIC-IDS2017 experiment
    unique_runs = df["run_dir"].unique()
    assert len(unique_runs) == 1, "Should load exactly 1 CIC-IDS2017 experiment"

    # Verify aggregation method
    assert all(df["aggregation"] == "krum"), "Should only include Krum experiment"


def test_filter_nonexistent_dataset(temp_runs_dir):
    """Test filtering with dataset that doesn't exist returns empty DataFrame."""
    df = load_experiment_results(temp_runs_dir, dataset_filter="nonexistent-dataset")

    assert df.empty, "Should return empty DataFrame for nonexistent dataset"


def test_dataset_column_preserved(temp_runs_dir):
    """Test that dataset column is preserved in loaded data."""
    df = load_experiment_results(temp_runs_dir, dataset_filter="edge-iiotset-full")

    assert "dataset" in df.columns, "Dataset column should be in DataFrame"
    assert all(df["dataset"].notna()), "All rows should have dataset value"


def test_config_fields_loaded(temp_runs_dir):
    """Test that config fields are properly loaded into DataFrame."""
    df = load_experiment_results(temp_runs_dir, dataset_filter="edge-iiotset-full")

    required_fields = ["aggregation", "alpha", "seed", "dataset"]
    for field in required_fields:
        assert field in df.columns, f"Config field '{field}' should be in DataFrame"


def test_metrics_computed(temp_runs_dir):
    """Test that macro_f1 metric is computed from client metrics."""
    df = load_experiment_results(temp_runs_dir, dataset_filter="edge-iiotset-full")

    assert "macro_f1" in df.columns, "macro_f1 should be computed"
    assert all(df["macro_f1"].notna()), "macro_f1 should be computed for all rounds"
