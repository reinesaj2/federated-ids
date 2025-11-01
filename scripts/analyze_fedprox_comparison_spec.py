from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_fedprox_comparison import (
    RunMetrics,
    aggregate_run_metrics,
    collect_run_metrics,
    compute_weighted_macro_f1,
    compute_paired_statistics,
    ensure_minimum_samples,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_run_artifact(
    root: Path,
    alpha: float,
    mu: float,
    seed: int,
    client_rows: list[list[dict]],
    server_rows: list[dict],
) -> Path:
    artifact_dir = root / f"fedprox-nightly-alpha{alpha}-mu{mu}-seed{seed}-abc123"
    run_dir = artifact_dir / "runs" / f"nightly_fedprox_alpha{alpha}_mu{mu}_seed{seed}"
    for client_id, rows in enumerate(client_rows):
        _write_csv(run_dir / f"client_{client_id}_metrics.csv", rows)
    _write_csv(run_dir / "metrics.csv", server_rows)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "alpha": alpha,
                "mu": mu,
                "seed": seed,
                "algorithm": "FedProx" if mu > 0 else "FedAvg",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_collect_run_metrics_loads_macro_f1_and_times(tmp_path: Path) -> None:
    run_a = _write_run_artifact(
        tmp_path,
        alpha=0.1,
        mu=0.0,
        seed=0,
        client_rows=[
            [
                {
                    "round": 1,
                    "dataset_size": 100,
                    "macro_f1_after": 0.8,
                },
                {
                    "round": 2,
                    "dataset_size": 100,
                    "macro_f1_after": 0.82,
                },
            ],
            [
                {
                    "round": 1,
                    "dataset_size": 50,
                    "macro_f1_after": 0.7,
                },
                {
                    "round": 2,
                    "dataset_size": 50,
                    "macro_f1_after": 0.75,
                },
            ],
        ],
        server_rows=[
            {"round": 1, "t_aggregate_ms": 100.0},
            {"round": 2, "t_aggregate_ms": 110.0},
        ],
    )

    run_b = _write_run_artifact(
        tmp_path,
        alpha=0.1,
        mu=0.1,
        seed=0,
        client_rows=[
            [
                {
                    "round": 1,
                    "dataset_size": 80,
                    "macro_f1_after": 0.85,
                },
                {
                    "round": 2,
                    "dataset_size": 80,
                    "macro_f1_after": 0.9,
                },
            ],
        ],
        server_rows=[
            {"round": 1, "t_aggregate_ms": 150.0},
            {"round": 2, "t_aggregate_ms": 160.0},
        ],
    )

    metrics = collect_run_metrics(tmp_path)
    assert len(metrics) == 2
    by_run = {m.mu: m for m in metrics}
    fedavg_metrics = by_run[0.0]
    fedprox_metrics = by_run[0.1]

    assert fedavg_metrics.run_dir == run_a
    assert fedavg_metrics.alpha == pytest.approx(0.1)
    assert fedavg_metrics.algorithm == "FedAvg"
    assert fedavg_metrics.seed == 0
    assert fedavg_metrics.mean_aggregation_time_ms == pytest.approx(105.0)
    assert fedavg_metrics.weighted_macro_f1 == pytest.approx(
        compute_weighted_macro_f1(
            [
                {"macro_f1_after": 0.82, "dataset_size": 100},
                {"macro_f1_after": 0.75, "dataset_size": 50},
            ]
        )
    )

    assert fedprox_metrics.algorithm == "FedProx"
    assert fedprox_metrics.mean_aggregation_time_ms == pytest.approx(155.0)
    assert fedprox_metrics.weighted_macro_f1 == pytest.approx(0.9)


def test_aggregate_run_metrics_computes_mean_and_ci(tmp_path: Path) -> None:
    run_metrics = [
        RunMetrics(
            alpha=0.1,
            mu=0.0,
            seed=0,
            algorithm="FedAvg",
            weighted_macro_f1=0.80,
            mean_aggregation_time_ms=100.0,
            rounds=20,
            run_dir=tmp_path,
            final_l2_distance=math.nan,
            final_cosine_similarity=math.nan,
        ),
        RunMetrics(
            alpha=0.1,
            mu=0.0,
            seed=1,
            algorithm="FedAvg",
            weighted_macro_f1=0.84,
            mean_aggregation_time_ms=120.0,
            rounds=20,
            run_dir=tmp_path,
            final_l2_distance=math.nan,
            final_cosine_similarity=math.nan,
        ),
        RunMetrics(
            alpha=0.1,
            mu=0.1,
            seed=0,
            algorithm="FedProx",
            weighted_macro_f1=0.86,
            mean_aggregation_time_ms=150.0,
            rounds=20,
            run_dir=tmp_path,
            final_l2_distance=math.nan,
            final_cosine_similarity=math.nan,
        ),
        RunMetrics(
            alpha=0.1,
            mu=0.1,
            seed=1,
            algorithm="FedProx",
            weighted_macro_f1=0.90,
            mean_aggregation_time_ms=180.0,
            rounds=20,
            run_dir=tmp_path,
            final_l2_distance=math.nan,
            final_cosine_similarity=math.nan,
        ),
    ]

    aggregated = aggregate_run_metrics(run_metrics, metrics=("weighted_macro_f1", "mean_aggregation_time_ms"))
    macro_f1_rows = aggregated[aggregated["metric"] == "weighted_macro_f1"]
    fedavg_row = macro_f1_rows[(macro_f1_rows["mu"] == 0.0)].iloc[0]
    fedprox_row = macro_f1_rows[(macro_f1_rows["mu"] == 0.1)].iloc[0]

    assert fedavg_row["mean"] == pytest.approx(0.82)
    assert fedavg_row["n"] == 2
    assert fedavg_row["ci_upper"] > fedavg_row["mean"]
    assert fedavg_row["ci_lower"] < fedavg_row["mean"]

    assert fedprox_row["mean"] == pytest.approx(0.88)
    assert fedprox_row["n"] == 2
    assert fedprox_row["ci_upper"] > fedprox_row["mean"]
    assert fedprox_row["ci_lower"] < fedprox_row["mean"]


def test_compute_paired_statistics_returns_effect_size(tmp_path: Path) -> None:
    runs: list[RunMetrics] = []
    for seed, fedavg_score, fedprox_score in [
        (0, 0.80, 0.86),
        (1, 0.82, 0.90),
        (2, 0.79, 0.84),
        (3, 0.81, 0.83),
        (4, 0.80, 0.88),
    ]:
        runs.append(
            RunMetrics(
                alpha=0.1,
                mu=0.0,
                seed=seed,
                algorithm="FedAvg",
                weighted_macro_f1=fedavg_score,
                mean_aggregation_time_ms=100 + seed,
                rounds=20,
                run_dir=tmp_path,
                final_l2_distance=math.nan,
                final_cosine_similarity=math.nan,
            )
        )
        runs.append(
            RunMetrics(
                alpha=0.1,
                mu=0.1,
                seed=seed,
                algorithm="FedProx",
                weighted_macro_f1=fedprox_score,
                mean_aggregation_time_ms=120 + seed,
                rounds=20,
                run_dir=tmp_path,
                final_l2_distance=math.nan,
                final_cosine_similarity=math.nan,
            )
        )

    stats_rows = compute_paired_statistics(runs, metric_name="weighted_macro_f1")
    assert len(stats_rows) == 1
    row = stats_rows[0]

    assert row["alpha"] == pytest.approx(0.1)
    assert row["mu"] == pytest.approx(0.1)
    assert row["metric"] == "weighted_macro_f1"
    assert row["n"] == 5
    assert row["mean_diff"] > 0
    assert row["p_value"] < 0.05
    assert row["effect_size"] > 0
    assert row["ci_lower"] < row["ci_upper"]


def test_ensure_minimum_samples_raises_when_insufficient_runs(tmp_path: Path) -> None:
    runs = [
        RunMetrics(
            alpha=0.1,
            mu=0.0,
            seed=0,
            algorithm="FedAvg",
            weighted_macro_f1=0.8,
            mean_aggregation_time_ms=100.0,
            rounds=10,
            run_dir=tmp_path,
            final_l2_distance=math.nan,
            final_cosine_similarity=math.nan,
        )
    ]

    with pytest.raises(ValueError, match="alpha=0.1 mu=0.0 algorithm=FedAvg"):
        ensure_minimum_samples(runs, minimum=5)


def test_compute_paired_statistics_uses_statistical_utils(tmp_path: Path) -> None:
    """Verify paired statistics use statistical_utils for p-values and effect sizes."""
    runs: list[RunMetrics] = []
    for seed, fedavg_score, fedprox_score in [
        (0, 0.80, 0.86),
        (1, 0.82, 0.90),
        (2, 0.79, 0.84),
        (3, 0.81, 0.83),
        (4, 0.80, 0.88),
    ]:
        runs.append(
            RunMetrics(
                alpha=0.1,
                mu=0.0,
                seed=seed,
                algorithm="FedAvg",
                weighted_macro_f1=fedavg_score,
                mean_aggregation_time_ms=100 + seed,
                rounds=20,
                run_dir=tmp_path,
                final_l2_distance=math.nan,
                final_cosine_similarity=math.nan,
            )
        )
        runs.append(
            RunMetrics(
                alpha=0.1,
                mu=0.1,
                seed=seed,
                algorithm="FedProx",
                weighted_macro_f1=fedprox_score,
                mean_aggregation_time_ms=120 + seed,
                rounds=20,
                run_dir=tmp_path,
                final_l2_distance=math.nan,
                final_cosine_similarity=math.nan,
            )
        )

    stats_rows = compute_paired_statistics(runs, metric_name="weighted_macro_f1")
    assert len(stats_rows) == 1
    row = stats_rows[0]

    assert "p_value" in row
    assert "effect_size" in row
    assert not pd.isna(row["p_value"])
    assert not pd.isna(row["effect_size"])


def test_compute_paired_statistics_aggregation_time_metric(tmp_path: Path) -> None:
    """Verify effect sizes computed for aggregation time metric."""
    runs: list[RunMetrics] = []
    for seed, fedavg_time, fedprox_time in [
        (0, 100.0, 120.0),
        (1, 102.0, 122.0),
        (2, 98.0, 118.0),
        (3, 101.0, 121.0),
        (4, 99.0, 119.0),
    ]:
        runs.append(
            RunMetrics(
                alpha=0.1,
                mu=0.0,
                seed=seed,
                algorithm="FedAvg",
                weighted_macro_f1=0.8,
                mean_aggregation_time_ms=fedavg_time,
                rounds=20,
                run_dir=tmp_path,
                final_l2_distance=math.nan,
                final_cosine_similarity=math.nan,
            )
        )
        runs.append(
            RunMetrics(
                alpha=0.1,
                mu=0.1,
                seed=seed,
                algorithm="FedProx",
                weighted_macro_f1=0.85,
                mean_aggregation_time_ms=fedprox_time,
                rounds=20,
                run_dir=tmp_path,
                final_l2_distance=math.nan,
                final_cosine_similarity=math.nan,
            )
        )

    stats_rows = compute_paired_statistics(runs, metric_name="mean_aggregation_time_ms")
    assert len(stats_rows) == 1
    row = stats_rows[0]

    assert row["metric"] == "mean_aggregation_time_ms"
    assert row["p_value"] < 0.05
    assert not pd.isna(row["effect_size"])


def _create_cli_artifacts(root: Path, seeds: range) -> None:
    for seed in seeds:
        client_rows = [
            [
                {
                    "round": 1,
                    "dataset_size": 100,
                    "macro_f1_after": 0.75 + 0.01 * seed,
                },
                {
                    "round": 2,
                    "dataset_size": 100,
                    "macro_f1_after": 0.76 + 0.01 * seed,
                },
            ]
        ]
        server_rows = [
            {
                "round": 1,
                "t_aggregate_ms": 100.0 + seed,
                "final_l2_distance": 2.0,
                "final_cosine_similarity": 0.5,
            },
            {
                "round": 2,
                "t_aggregate_ms": 110.0 + seed,
                "final_l2_distance": 1.0 + 0.1 * seed,
                "final_cosine_similarity": 0.6,
            },
        ]
        _write_run_artifact(root, 0.1, 0.0, seed, client_rows, server_rows)

        prox_client_rows = [
            [
                {
                    "round": 1,
                    "dataset_size": 120,
                    "macro_f1_after": 0.80 + 0.01 * seed,
                },
                {
                    "round": 2,
                    "dataset_size": 120,
                    "macro_f1_after": 0.82 + 0.01 * seed,
                },
            ]
        ]
        prox_server_rows = [
            {
                "round": 1,
                "t_aggregate_ms": 130.0 + seed,
                "final_l2_distance": 1.5,
                "final_cosine_similarity": 0.55,
            },
            {
                "round": 2,
                "t_aggregate_ms": 140.0 + seed,
                "final_l2_distance": 0.9 + 0.05 * seed,
                "final_cosine_similarity": 0.7,
            },
        ]
        _write_run_artifact(root, 0.1, 0.1, seed, prox_client_rows, prox_server_rows)


def test_cli_invocation_succeeds_and_populates_summary(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    _create_cli_artifacts(artifacts_dir, range(5))
    output_dir = tmp_path / "summary"

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "analyze_fedprox_comparison.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--artifacts_dir",
            str(artifacts_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(f"CLI execution failed: {result.stderr}")

    summary_path = output_dir / "fedprox_comparison_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    convergence = summary["raw_analysis_results"]["convergence_analysis"]
    sample_key = next(iter(convergence))
    sample_entry = convergence[sample_key]
    assert not math.isnan(sample_entry["final_l2_distance"])
    assert not math.isnan(sample_entry["final_cosine_similarity"])


def test_cli_fails_when_seed_threshold_not_met(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    _create_cli_artifacts(artifacts_dir, range(2))
    output_dir = tmp_path / "summary"

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "analyze_fedprox_comparison.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--artifacts_dir",
            str(artifacts_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "require at least" in result.stderr
