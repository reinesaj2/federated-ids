from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_fedprox_comparison import (
    RunMetrics,
    aggregate_run_metrics,
    collect_run_metrics,
    compute_weighted_macro_f1,
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
