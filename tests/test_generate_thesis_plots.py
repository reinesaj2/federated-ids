#!/usr/bin/env python3
"""Unit tests for generate_thesis_plots.py"""

import json
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-config"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.generate_thesis_plots import (
    _render_cosine_plot,
    _render_l2_plot,
    _render_macro_f1_plot,
    _render_timing_plot,
    _render_privacy_curve,
    collect_personalization_deltas,
    compute_confidence_interval,
    compute_server_macro_f1_from_clients,
    perform_statistical_tests,
    plot_privacy_utility,
    summarize_personalization_deltas,
)


def test_compute_confidence_interval_basic():
    """Test CI computation with basic dataset."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

    assert mean == 3.0
    assert lower < mean
    assert upper > mean
    assert upper - lower > 0  # CI should have non-zero width


def test_compute_confidence_interval_single_value():
    """Test CI computation with single value (edge case)."""
    data = np.array([5.0])
    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

    # With single value, CI is undefined but function should not crash
    assert mean == 5.0
    # CI will be NaN or inf for single value - just check it doesn't crash


def test_compute_server_macro_f1_from_clients():
    """Test server-level macro-F1 aggregation from client CSVs."""
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create mock client CSV files
        for client_id in range(3):
            client_data = pd.DataFrame(
                {
                    "client_id": [client_id] * 5,
                    "round": [0, 1, 2, 3, 4],
                    "macro_f1_after": [0.8, 0.82, 0.85, 0.87, 0.9],
                    "macro_f1_argmax": [0.79, 0.81, 0.84, 0.86, 0.89],
                }
            )
            client_data.to_csv(run_dir / f"client_{client_id}_metrics.csv", index=False)

        # Test round 0
        f1 = compute_server_macro_f1_from_clients(run_dir, 0)
        assert f1 is not None
        assert 0.75 <= f1 <= 0.85  # Should be around 0.8

        # Test round 4
        f1 = compute_server_macro_f1_from_clients(run_dir, 4)
        assert f1 is not None
        assert 0.85 <= f1 <= 0.95  # Should be around 0.9

        # Test non-existent round
        f1 = compute_server_macro_f1_from_clients(run_dir, 999)
        assert f1 is None


def test_compute_server_macro_f1_missing_data():
    """Test macro-F1 computation when no client CSVs exist."""
    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        f1 = compute_server_macro_f1_from_clients(run_dir, 0)
        assert f1 is None


def test_l2_spurious_zeros_detection():
    """Test detection of spurious L2 zeros (regression test for median=0 artifact)."""
    # Create dataset with suspiciously many zeros
    l2_data = pd.Series([0.0, 0.0, 0.0, 0.001, 0.002])
    zero_count = (l2_data == 0.0).sum()

    # Should detect that >50% are zeros
    assert zero_count > len(l2_data) * 0.5

    # Create dataset with normal small values
    l2_data_normal = pd.Series([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    zero_count_normal = (l2_data_normal == 0.0).sum()

    # Should NOT detect zeros in normal data
    assert zero_count_normal == 0


def test_perform_statistical_tests_ttest():
    """Test t-test for 2 groups."""
    df = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    result = perform_statistical_tests(df, "group", "value")

    assert result["test"] == "t_test"
    assert "p_value" in result
    assert "statistic" in result
    assert result["p_value"] < 0.05  # Groups are significantly different


def test_perform_statistical_tests_anova():
    """Test ANOVA for >2 groups."""
    df = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C"], "value": [1.0, 2.0, 4.0, 5.0, 7.0, 8.0]})

    result = perform_statistical_tests(df, "group", "value")

    assert result["test"] == "anova"
    assert "p_value" in result
    assert "statistic" in result
    assert "pairwise" in result
    assert len(result["pairwise"]) == 3  # 3 pairwise comparisons for 3 groups


def test_perform_statistical_tests_insufficient_data():
    """Test statistical tests with insufficient data."""
    df = pd.DataFrame({"group": ["A"], "value": [1.0]})

    result = perform_statistical_tests(df, "group", "value")

    assert result["test"] == "insufficient_data"
    assert result["p_value"] is None


def test_render_macro_f1_plot_success():
    """Test macro-F1 renderer with valid data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum"],
            "seed": [1, 2, 1, 2],
            "macro_f1": [0.85, 0.87, 0.90, 0.92],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    assert len(ax.patches) > 0  # Bars were drawn
    assert ax.get_title() == "Detection Performance (Macro-F1, 95% CI)"
    plt.close(fig)


def test_render_macro_f1_plot_missing_column():
    """Test macro-F1 renderer with missing data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame({"aggregation": ["fedavg"], "seed": [1]})
    fig, ax = plt.subplots()

    result = _render_macro_f1_plot(ax, final_rounds, ["fedavg"])

    assert result is False
    assert len(ax.patches) == 0
    plt.close(fig)


def test_render_timing_plot_success():
    """Test timing renderer with valid data."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum"],
            "seed": [1, 2, 1, 2],
            "t_aggregate_ms": [10.0, 12.0, 15.0, 17.0],
        }
    )
    fig, ax = plt.subplots()

    result = _render_timing_plot(ax, df, ["fedavg", "krum"])

    assert result is True
    assert len(ax.patches) > 0
    assert ax.get_title() == "Aggregation Time (95% CI)"
    plt.close(fig)


def test_render_timing_plot_missing_column():
    """Test timing renderer with missing data."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"aggregation": ["fedavg"], "seed": [1]})
    fig, ax = plt.subplots()

    result = _render_timing_plot(ax, df, ["fedavg"])

    assert result is False
    plt.close(fig)


def test_render_l2_plot_success():
    """Test L2 renderer with valid data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "krum", "median"],
            "seed": [1, 1, 1],
            "l2_to_benign_mean": [0.001, 0.002, 0.0015],
        }
    )
    fig, ax = plt.subplots()

    result = _render_l2_plot(ax, final_rounds, ["fedavg", "krum", "median"])

    assert result is True
    assert ax.get_title() == "Model Drift (L2 Distance)"
    plt.close(fig)


def test_render_cosine_plot_success():
    """Test cosine renderer with valid data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "krum"],
            "seed": [1, 1],
            "cos_to_benign_mean": [0.99, 0.98],
        }
    )
    fig, ax = plt.subplots()

    result = _render_cosine_plot(ax, final_rounds, ["fedavg", "krum"])

    assert result is True
    assert ax.get_title() == "Model Alignment (Cosine Similarity)"
    plt.close(fig)


def _write_config(path: Path, seed: int, dp_enabled: bool, dp_noise: float) -> None:
    config = {
        "aggregation": "fedavg",
        "alpha": 0.5,
        "adversary_fraction": 0.0,
        "dp_enabled": dp_enabled,
        "dp_noise_multiplier": dp_noise,
        "personalization_epochs": 0,
        "num_clients": 2,
        "num_rounds": 3,
        "seed": seed,
    }
    path.write_text(json.dumps(config))


@pytest.mark.parametrize("dp_noise", [0.7])
def test_privacy_utility_curve_outputs(tmp_path, dp_noise):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    dp_run_seed42 = runs_dir / "comp_fedavg_alpha0.5_dp1_seed42"
    dp_run_seed43 = runs_dir / "comp_fedavg_alpha0.5_dp1_seed43"
    baseline_run = runs_dir / "comp_fedavg_alpha0.5_dp0_seed42"

    for run_dir in [dp_run_seed42, dp_run_seed43, baseline_run]:
        run_dir.mkdir()

    _write_config(dp_run_seed42 / "config.json", seed=42, dp_enabled=True, dp_noise=dp_noise)
    _write_config(dp_run_seed43 / "config.json", seed=43, dp_enabled=True, dp_noise=dp_noise)
    _write_config(baseline_run / "config.json", seed=42, dp_enabled=False, dp_noise=0.0)

    metrics_df = pd.DataFrame(
        [
            {"round": 1, "dp_enabled": True, "dp_noise_multiplier": dp_noise, "seed": 42},
            {"round": 3, "dp_enabled": True, "dp_noise_multiplier": dp_noise, "seed": 42},
        ]
    )
    metrics_df.to_csv(dp_run_seed42 / "metrics.csv", index=False)
    metrics_df.assign(seed=43).to_csv(dp_run_seed43 / "metrics.csv", index=False)
    metrics_df.assign(dp_enabled=False, dp_noise_multiplier=0.0).to_csv(baseline_run / "metrics.csv", index=False)

    for run_dir, epsilon, f1_values in [
        (dp_run_seed42, 1.5, [0.82, 0.80]),
        (dp_run_seed43, 1.5, [0.78, 0.76]),
        (baseline_run, float("nan"), [0.90, 0.88]),
    ]:
        for idx, f1 in enumerate(f1_values):
            client_df = pd.DataFrame(
                [
                    {"round": 1, "macro_f1_after": f1 - 0.05, "dp_epsilon": epsilon / 3},
                    {"round": 3, "macro_f1_after": f1, "dp_epsilon": epsilon},
                ]
            )
            (run_dir / f"client_{idx}_metrics.csv").write_text(client_df.to_csv(index=False))

    df = pd.DataFrame(
        [
            {
                "dp_enabled": True,
                "dp_noise_multiplier": dp_noise,
                "seed": 42,
                "round": 3,
                "run_dir": str(dp_run_seed42),
            },
            {
                "dp_enabled": True,
                "dp_noise_multiplier": dp_noise,
                "seed": 43,
                "round": 3,
                "run_dir": str(dp_run_seed43),
            },
            {
                "dp_enabled": False,
                "dp_noise_multiplier": 0.0,
                "seed": 42,
                "round": 3,
                "run_dir": str(baseline_run),
            },
        ]
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    plot_privacy_utility(df, output_dir, runs_dir=runs_dir)

    curve_path = output_dir / "privacy_utility_curve.png"
    summary_path = output_dir / "privacy_utility_curve.csv"

    assert curve_path.exists()
    assert summary_path.exists()

    summary_df = pd.read_csv(summary_path)

    assert set(summary_df["is_baseline"].unique()) == {0, 1}

    dp_row = summary_df.loc[summary_df["is_baseline"] == 0].iloc[0]
    baseline_row = summary_df.loc[summary_df["is_baseline"] == 1].iloc[0]

    assert pytest.approx(dp_row["epsilon"], rel=1e-3) == 1.5
    assert dp_row["n"] == 2
    assert pytest.approx(dp_row["macro_f1_mean"], rel=1e-3) == 0.79

    assert pd.isna(baseline_row["epsilon"])
    assert baseline_row["n"] == 1
    assert pytest.approx(baseline_row["macro_f1_mean"], rel=1e-3) == 0.89


def test_privacy_curve_includes_dataset_and_missing_baseline_epsilon(tmp_path):
    dp_df = pd.DataFrame(
        {
            "epsilon": [1.0, 2.0],
            "macro_f1": [0.8, 0.82],
            "dp_noise_multiplier": [0.5, 0.7],
            "dataset": ["unsw", "cic"],
            "seed": [42, 43],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "macro_f1": [0.9, 0.88],
            "dp_noise_multiplier": [0.0, 0.0],
            "dataset": ["unsw", "cic"],
            "seed": [44, 45],
        }
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _render_privacy_curve(dp_df, baseline_df, output_dir)

    summary_df = pd.read_csv(output_dir / "privacy_utility_curve.csv")

    assert "dataset" in summary_df.columns
    assert set(summary_df["dataset"].dropna()) == {"unsw", "cic"}

    baseline_rows = summary_df[summary_df["is_baseline"] == 1]
    assert not baseline_rows.empty
    assert baseline_rows["epsilon"].isna().all()

    assert (summary_df["ci_lower"] >= 0.0).all()
    assert (summary_df["ci_upper"] <= 1.0).all()


def _write_personalization_run(run_dir: Path, dataset: str, personalization_epochs: int, seed: int, deltas: list[float]) -> None:
    run_dir.mkdir(parents=True)
    config = {
        "aggregation": "fedavg",
        "alpha": 0.5,
        "adversary_fraction": 0.0,
        "dp_enabled": False,
        "dp_noise_multiplier": 0.0,
        "personalization_epochs": personalization_epochs,
        "num_clients": len(deltas),
        "num_rounds": 1,
        "seed": seed,
        "dataset": dataset,
    }
    (run_dir / "config.json").write_text(json.dumps(config))

    metrics_df = pd.DataFrame([{"round": 1, "aggregation": "fedavg", "personalization_epochs": personalization_epochs, "seed": seed}])
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    for idx, delta in enumerate(deltas):
        client_df = pd.DataFrame(
            [
                {
                    "round": 1,
                    "macro_f1_global": 0.6,
                    "macro_f1_personalized": 0.6 + delta,
                    "macro_f1_after": 0.6,
                    "client_id": idx,
                }
            ]
        )
        client_df.to_csv(run_dir / f"client_{idx}_metrics.csv", index=False)


def test_personalization_summary_per_dataset(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    _write_personalization_run(runs_dir / "unsw_pers3_seed42", "unsw", personalization_epochs=3, seed=42, deltas=[0.1, 0.05])
    _write_personalization_run(runs_dir / "unsw_pers3_seed43", "unsw", personalization_epochs=3, seed=43, deltas=[0.08])
    _write_personalization_run(runs_dir / "cic_pers3_seed42", "cic", personalization_epochs=3, seed=42, deltas=[0.02, 0.03])

    delta_df = collect_personalization_deltas(runs_dir)
    summary_df = summarize_personalization_deltas(delta_df)

    assert set(summary_df["dataset"]) == {"unsw", "cic"}

    unsw_row = summary_df[(summary_df["dataset"] == "unsw") & (summary_df["personalization_epochs"] == 3)].iloc[0]
    cic_row = summary_df[(summary_df["dataset"] == "cic") & (summary_df["personalization_epochs"] == 3)].iloc[0]

    assert pytest.approx(unsw_row["gain_mean"], rel=1e-3) == 0.07666666666666666
    assert unsw_row["n_clients"] == 3

    assert pytest.approx(cic_row["gain_mean"], rel=1e-3) == 0.025
    assert cic_row["n_clients"] == 2

    assert (summary_df["ci_lower"] <= summary_df["gain_mean"]).all()
    assert (summary_df["ci_upper"] >= summary_df["gain_mean"]).all()


def test_compute_f1_degradation_bounded():
    """Test F1 degradation formula is bounded in [0, 100]."""
    # Normal case: degradation from 0.9 to 0.7
    baseline_f1 = 0.9
    attack_f1 = 0.7
    degradation = max(0.0, (baseline_f1 - attack_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0.0

    assert 0.0 <= degradation <= 100.0
    assert pytest.approx(degradation, rel=1e-3) == 22.222

    # Edge case: improvement (should be 0, not negative)
    baseline_f1 = 0.8
    attack_f1 = 0.9
    degradation = max(0.0, (baseline_f1 - attack_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0.0

    assert degradation == 0.0

    # Edge case: complete degradation
    baseline_f1 = 0.9
    attack_f1 = 0.0
    degradation = max(0.0, (baseline_f1 - attack_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0.0

    assert pytest.approx(degradation, rel=1e-3) == 100.0

    # Edge case: zero baseline (avoid division by zero)
    baseline_f1 = 0.0
    attack_f1 = 0.5
    degradation = max(0.0, (baseline_f1 - attack_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0.0

    assert degradation == 0.0


def test_attack_resilience_csv_structure():
    """Test attack resilience CSV has correct structure and bounded degradation."""
    import matplotlib.pyplot as plt

    from scripts.generate_thesis_plots import plot_attack_resilience

    with TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "runs"
        runs_dir.mkdir()
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Create mock attack experiment data
        for agg in ["fedavg", "krum", "median"]:
            for adv_frac in [0.0, 0.1, 0.3]:
                for seed in [42, 43, 44]:
                    run_name = f"comp_{agg}_alpha0.5_adv{int(adv_frac*100)}_dp0_pers0_seed{seed}"
                    run_dir = runs_dir / run_name
                    run_dir.mkdir(parents=True)

                    # Config
                    config = {
                        "aggregation": agg,
                        "adversary_fraction": adv_frac,
                        "alpha": 0.5,
                        "seed": seed,
                        "num_clients": 6,
                        "num_rounds": 20,
                        "dataset": "unsw",
                    }
                    (run_dir / "config.json").write_text(json.dumps(config))

                    # Server metrics
                    server_df = pd.DataFrame([{"round": 20, "aggregation": agg, "adversary_fraction": adv_frac}])
                    server_df.to_csv(run_dir / "metrics.csv", index=False)

                    # Client metrics (simulate degradation for FedAvg, resilience for robust methods)
                    for client_id in range(6):
                        if agg == "fedavg":
                            # FedAvg degrades significantly
                            macro_f1 = 0.9 - (adv_frac * 0.4)  # 0.9 -> 0.78 -> 0.66
                        else:
                            # Robust methods maintain performance
                            macro_f1 = 0.9 - (adv_frac * 0.05)  # 0.9 -> 0.885 -> 0.87

                        client_df = pd.DataFrame([{"round": 20, "macro_f1_after": macro_f1}])
                        (run_dir / f"client_{client_id}_metrics.csv").write_text(client_df.to_csv(index=False))

        # Load data
        from scripts.generate_thesis_plots import load_experiment_results

        df = load_experiment_results(runs_dir)

        # Generate plot
        plot_attack_resilience(df, output_dir)

        # Check CSV exists
        csv_path = output_dir / "attack_resilience_stats.csv"
        assert csv_path.exists()

        # Load and validate CSV structure
        stats_df = pd.read_csv(csv_path)

        # Required columns
        assert "aggregation" in stats_df.columns
        assert "adversary_fraction" in stats_df.columns
        assert "macro_f1_mean" in stats_df.columns
        assert "ci_lower" in stats_df.columns
        assert "ci_upper" in stats_df.columns
        assert "n" in stats_df.columns
        assert "degradation_pct" in stats_df.columns

        # Check degradation is bounded [0, 100]
        assert (stats_df["degradation_pct"] >= 0.0).all()
        assert (stats_df["degradation_pct"] <= 100.0).all()

        # Check CIs are valid
        assert (stats_df["ci_lower"] <= stats_df["macro_f1_mean"]).all()
        assert (stats_df["ci_upper"] >= stats_df["macro_f1_mean"]).all()

        plt.close("all")


def test_extract_threat_model_metadata():
    """Test extraction of threat model metadata from config."""
    config = {
        "dataset": "unsw",
        "num_clients": 6,
        "alpha": 0.5,
        "adversary_fraction": 0.1,
        "aggregation": "median",
        "seed": 42,
        "num_rounds": 20,
    }

    # Extract metadata
    dataset = config.get("dataset", "unknown")
    num_clients = config.get("num_clients", 0)
    alpha = config.get("alpha", 1.0)
    adv_frac = config.get("adversary_fraction", 0.0)
    seed = config.get("seed", 0)

    # Validate extraction
    assert dataset == "unsw"
    assert num_clients == 6
    assert alpha == 0.5
    assert adv_frac == 0.1
    assert seed == 42

    # Build subtitle
    subtitle = f"Dataset: {dataset.upper()} | Clients: {num_clients} | α={alpha} (Dirichlet) | " f"Attack: grad_ascent | Seeds: n=3"

    assert "UNSW" in subtitle
    assert "Clients: 6" in subtitle
    assert "α=0.5" in subtitle
    assert "grad_ascent" in subtitle


def test_attack_resilience_ci_computation():
    """Test CI computation for attack resilience across seeds."""
    # Simulate macro-F1 values across 3 seeds for same config
    f1_values = np.array([0.85, 0.87, 0.86])

    mean, lower, upper = compute_confidence_interval(f1_values, confidence=0.95)

    # Check mean
    assert pytest.approx(mean, rel=1e-3) == 0.8600

    # Check CIs are reasonable
    assert lower < mean < upper
    assert (upper - lower) < 0.1  # CI width should be reasonable


def test_attack_resilience_plot_artifacts():
    """Test that plot_attack_resilience generates all required artifacts."""
    import matplotlib.pyplot as plt

    from scripts.generate_thesis_plots import plot_attack_resilience

    with TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "runs"
        runs_dir.mkdir()
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Create minimal mock data
        for agg in ["fedavg", "krum"]:
            for adv_frac in [0.0, 0.1]:
                for seed in [42, 43]:
                    run_name = f"comp_{agg}_alpha0.5_adv{int(adv_frac*100)}_dp0_pers0_seed{seed}"
                    run_dir = runs_dir / run_name
                    run_dir.mkdir(parents=True)

                    config = {
                        "aggregation": agg,
                        "adversary_fraction": adv_frac,
                        "alpha": 0.5,
                        "seed": seed,
                        "num_clients": 6,
                    }
                    (run_dir / "config.json").write_text(json.dumps(config))

                    server_df = pd.DataFrame([{"round": 10, "aggregation": agg}])
                    server_df.to_csv(run_dir / "metrics.csv", index=False)

                    for client_id in range(2):
                        client_df = pd.DataFrame([{"round": 10, "macro_f1_after": 0.85}])
                        (run_dir / f"client_{client_id}_metrics.csv").write_text(client_df.to_csv(index=False))

        from scripts.generate_thesis_plots import load_experiment_results

        df = load_experiment_results(runs_dir)
        plot_attack_resilience(df, output_dir)

        # Check all artifacts exist
        assert (output_dir / "attack_resilience.png").exists()
        assert (output_dir / "attack_resilience.pdf").exists()
        assert (output_dir / "attack_resilience_stats.csv").exists()

        plt.close("all")


def test_render_macro_f1_plot_insufficient_data_issue77():
    """Test Issue #77 fix: ANOVA annotation skipped when macro_f1 data is sparse."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "fedavg", "krum", "krum", "bulyan", "median"],
            "seed": [42, 43, 44, 42, 43, 42, 42],
            "macro_f1": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.85],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum", "bulyan", "median"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    texts = [t.get_text() for t in ax.texts]

    anova_p_texts = [t for t in texts if "ANOVA p=" in t]
    assert len(anova_p_texts) == 0, "ANOVA p-value should not appear with insufficient data"

    insufficient_texts = [t for t in texts if "insufficient" in t.lower()]
    assert len(insufficient_texts) > 0, "Should display message about insufficient data"
    plt.close(fig)


def test_render_macro_f1_plot_sufficient_data_for_anova():
    """Test Issue #77 fix: ANOVA annotation displayed when sufficient macro_f1 data."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "fedavg", "krum", "krum", "krum"],
            "seed": [42, 43, 44, 42, 43, 44],
            "macro_f1": [0.85, 0.86, 0.85, 0.90, 0.91, 0.89],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    texts = [t.get_text() for t in ax.texts]

    anova_texts = [t for t in texts if "ANOVA" in t and "p=" in t]
    assert len(anova_texts) > 0, "ANOVA annotation should appear with sufficient data"

    insufficient_texts = [t for t in texts if "insufficient" in t.lower()]
    assert len(insufficient_texts) == 0, "Should not display insufficient data warning"
    plt.close(fig)


def test_render_macro_f1_plot_edge_case_exactly_three_points():
    """Test Issue #77 fix: Edge case with exactly 3 data points (minimum for ANOVA)."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "krum", "bulyan"],
            "seed": [42, 42, 42],
            "macro_f1": [0.85, 0.90, 0.88],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum", "bulyan"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    texts = [t.get_text() for t in ax.texts]

    annotation_texts = [t for t in texts if "ANOVA p=" in t or "inconclusive" in t.lower() or "insufficient" in t.lower()]
    assert len(annotation_texts) >= 1, "Should have ANOVA p-value, inconclusive, or insufficient data annotation. " f"Got texts: {texts}"
    plt.close(fig)


def test_render_macro_f1_plot_two_data_points_below_threshold():
    """Test Issue #77 fix: With 2 points (below minimum 3), ANOVA skipped."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "krum"],
            "seed": [42, 42],
            "macro_f1": [0.85, 0.90],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    texts = [t.get_text() for t in ax.texts]

    anova_texts = [t for t in texts if "ANOVA" in t and "p=" in t]
    assert len(anova_texts) == 0, "ANOVA should not run with < 3 data points"

    insufficient_texts = [t for t in texts if "insufficient" in t.lower()]
    assert len(insufficient_texts) > 0, "Should display insufficient data message"
    plt.close(fig)


def test_perform_statistical_tests_all_identical_values():
    """Test Issue #77: ANOVA skipped when all values are identical."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "fedavg", "krum", "krum", "krum"],
            "macro_f1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = perform_statistical_tests(df, "aggregation", "macro_f1")

    assert "test" in result
    # Issue #77: Identical values should return "skipped_identical", not run ANOVA
    assert result["test"] == "skipped_identical"
    assert result.get("p_value") is None
    assert "precision_info" in result
    assert result["precision_info"]["is_identical"] is True


def test_perform_statistical_tests_precision_artifact():
    """Test Issue #77: Near-perfect values (precision artifact scenario)."""
    # Values differ by tiny amounts but all near 1.0 (like 0.999995-0.999999)
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "fedavg", "krum", "krum", "krum", "bulyan", "bulyan", "bulyan"],
            "macro_f1": [0.999997, 0.999996, 0.999998, 0.999991, 0.999989, 0.999992, 0.999994, 0.999993, 0.999995],
        }
    )

    result = perform_statistical_tests(df, "aggregation", "macro_f1")

    assert result["test"] == "anova"
    assert result.get("p_value") is not None
    assert "bonferroni_corrected_p" in result
    assert "precision_info" in result

    precision_info = result["precision_info"]
    assert precision_info["is_near_perfect"] is True
    assert precision_info["precision_artifact"] is True
    assert precision_info["max_value"] > 0.999
    assert precision_info["min_value"] > 0.999


def test_perform_statistical_tests_bonferroni_correction():
    """Test Issue #77: Bonferroni correction applied for multiple comparisons."""
    df = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum", "bulyan", "bulyan", "median", "median"],
            "macro_f1": [0.85, 0.86, 0.90, 0.91, 0.88, 0.89, 0.87, 0.88],
        }
    )

    result = perform_statistical_tests(df, "aggregation", "macro_f1")

    assert result["test"] == "anova"
    assert "bonferroni_corrected_p" in result
    assert "num_comparisons" in result

    # With 4 groups, we have 4 choose 2 = 6 pairwise comparisons
    assert result["num_comparisons"] == 6
    assert result["bonferroni_corrected_p"] >= result["p_value"]


def test_render_macro_f1_plot_precision_artifact_issue77():
    """Test Issue #77: Precision artifact displays 6-decimal F1 and notes ceiling effect."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum", "bulyan", "bulyan"],
            "seed": [42, 43, 42, 43, 42, 43],
            "macro_f1": [0.999997, 0.999996, 0.999991, 0.999992, 0.999994, 0.999993],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum", "bulyan"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    texts = [t.get_text() for t in ax.texts]

    # Should show precision annotation (6-decimal F1) OR ANOVA with precision note
    precision_texts = [t for t in texts if "F1=" in t and "0.999" in t]
    anova_texts = [t for t in texts if "ANOVA p=" in t]
    ceiling_texts = [t for t in texts if "ceiling" in t.lower() or "±" in t]

    # At least one should appear
    assert (
        len(precision_texts) > 0 or len(anova_texts) > 0 or len(ceiling_texts) > 0
    ), f"Should show precision annotation. Got texts: {texts}"
    plt.close(fig)


def test_render_macro_f1_plot_identical_values_issue77():
    """Test Issue #77: Identical values show 'all methods identical' annotation."""
    import matplotlib.pyplot as plt

    final_rounds = pd.DataFrame(
        {
            "aggregation": ["fedavg", "fedavg", "krum", "krum", "bulyan", "bulyan"],
            "seed": [42, 43, 42, 43, 42, 43],
            "macro_f1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    fig, ax = plt.subplots()
    available_methods = ["fedavg", "krum", "bulyan"]

    result = _render_macro_f1_plot(ax, final_rounds, available_methods)

    assert result is True
    texts = [t.get_text() for t in ax.texts]

    # Should show "identical" annotation, not ANOVA
    identical_texts = [t for t in texts if "identical" in t.lower()]
    anova_texts = [t for t in texts if "ANOVA p=" in t]

    assert len(identical_texts) > 0, f"Should show identical annotation. Got texts: {texts}"
    assert len(anova_texts) == 0, "Should not show ANOVA for identical values"
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
