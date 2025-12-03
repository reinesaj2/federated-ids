from __future__ import annotations

from pathlib import Path

import pytest

from scripts.comparative_analysis import DATASET_DEFAULT_PATHS, ExperimentConfig
from scripts.run_experiments_optimized import (
    apply_config_filters,
    build_s3_sync_plan,
    determine_timeout_seconds,
    parse_runner_args,
)


def _make_config(aggregation: str = "fedavg", seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        aggregation=aggregation,
        alpha=1.0,
        adversary_fraction=0.0,
        dp_enabled=False,
        dp_noise_multiplier=0.0,
        personalization_epochs=0,
        num_clients=6,
        num_rounds=20,
        seed=seed,
        fedprox_mu=0.0,
        dataset="edge-iiotset-nightly",
        data_path=DATASET_DEFAULT_PATHS["edge-iiotset-nightly"],
    )


def test_parse_runner_args_uses_env_overrides(monkeypatch):
    env = {
        "EDGE_DIMENSION": "aggregation",
        "EDGE_WORKERS": "4",
        "EDGE_DATASET": "edge-iiotset-nightly",
        "EDGE_DATASET_TYPE": "full",
        "EDGE_STRATEGY": "krum",
        "EDGE_SEED": "43",
        "EDGE_CLIENT_TIMEOUT_SEC": "1800",
        "EDGE_S3_PREFIX": "s3://bucket/path",
    }

    args = parse_runner_args(argv=[], env=env)

    assert args.dimension == "aggregation"
    assert args.workers == 4
    assert args.dataset == "edge-iiotset-nightly"
    assert args.dataset_type == "full"
    assert args.strategy == "krum"
    assert args.seed == 43
    assert args.client_timeout_sec == 1800
    assert args.s3_sync_prefix == "s3://bucket/path"


def test_apply_config_filters_limits_to_strategy_and_seed():
    configs = [
        _make_config("fedavg", 42),
        _make_config("krum", 42),
        _make_config("krum", 43),
        _make_config("bulyan", 43),
    ]

    filtered = apply_config_filters(configs, strategy="krum", seed=43, preset=None)

    assert len(filtered) == 1
    assert filtered[0].aggregation == "krum"
    assert filtered[0].seed == 43


def test_determine_timeout_seconds_prefers_override():
    config = _make_config()

    timeout = determine_timeout_seconds(config, dataset_type="full", timeout_override=900)

    assert timeout == 900


@pytest.mark.parametrize(
    ("prefix", "preset", "expected"),
    [
        ("s3://bucket/base", "ds1", "s3://bucket/base/ds1"),
        ("s3://bucket/base/", "ds1", "s3://bucket/base/ds1"),
    ],
)
def test_build_s3_sync_plan_normalizes_prefix(prefix: str, preset: str, expected: str):
    run_dir = Path("/tmp/run-dir")

    plan = build_s3_sync_plan(run_dir, prefix, preset)

    assert len(plan) == 1
    assert plan[0][0] == run_dir
    assert plan[0][1] == expected
