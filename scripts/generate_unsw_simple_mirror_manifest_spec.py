from __future__ import annotations

import pytest

from scripts.generate_unsw_simple_mirror_manifest import (
    build_run_name,
    build_unsw_simple_mirror_manifest,
    is_bulyan_feasible,
)

AGGREGATION_FEDAVG = "fedavg"
AGGREGATION_BULYAN = "bulyan"
ALPHA_NON_IID = 0.1
ADVERSARY_FRACTION_LOW = 0.1
ADVERSARY_FRACTION_HIGH = 0.3
ADVERSARY_FRACTION_BORDER = 0.2
ADVERSARY_FRACTION_NONE = 0.0
DP_DISABLED = False
DP_NOISE = 0.0
PERSONALIZATION_EPOCHS = 0
NUM_CLIENTS_TEN = 10
NUM_CLIENTS_ELEVEN = 11
NUM_ROUNDS = 20
SEED = 42
FEDPROX_MU = 0.0
DATA_PATH = "data/unsw/UNSW_NB15_full.csv"
MODEL_ARCH = "simple"
DIGEST = "abc123def4"


def _base_config(dataset: str) -> dict:
    return {
        "aggregation": AGGREGATION_FEDAVG,
        "alpha": ALPHA_NON_IID,
        "adversary_fraction": ADVERSARY_FRACTION_LOW,
        "dp_enabled": DP_DISABLED,
        "dp_noise_multiplier": DP_NOISE,
        "personalization_epochs": PERSONALIZATION_EPOCHS,
        "num_clients": NUM_CLIENTS_TEN,
        "num_rounds": NUM_ROUNDS,
        "seed": SEED,
        "fedprox_mu": FEDPROX_MU,
        "dataset": dataset,
        "data_path": DATA_PATH,
    }


@pytest.mark.parametrize(
    "num_clients,adversary_fraction,expected",
    [
        (NUM_CLIENTS_ELEVEN, ADVERSARY_FRACTION_HIGH, False),
        (NUM_CLIENTS_ELEVEN, ADVERSARY_FRACTION_BORDER, True),
        (NUM_CLIENTS_TEN, ADVERSARY_FRACTION_NONE, True),
    ],
)
def test_is_bulyan_feasible_threshold(num_clients: int, adversary_fraction: float, expected: bool) -> None:
    assert is_bulyan_feasible(num_clients, adversary_fraction) is expected


def test_build_run_name_includes_core_fields() -> None:
    config = {
        "aggregation": AGGREGATION_FEDAVG,
        "alpha": ALPHA_NON_IID,
        "adversary_fraction": ADVERSARY_FRACTION_BORDER,
        "dp_enabled": DP_DISABLED,
        "dp_noise_multiplier": DP_NOISE,
        "personalization_epochs": PERSONALIZATION_EPOCHS,
        "fedprox_mu": FEDPROX_MU,
        "seed": SEED,
    }

    expected = "unsw_simple_abc123def4_comp_fedavg_alpha0.1_adv20_dp0_pers0_mu0.0_seed42"
    assert build_run_name(config, DIGEST) == expected


def test_build_unsw_simple_mirror_manifest_dedupes_sources() -> None:
    configs = [_base_config("cic"), _base_config("edge-iiotset-full")]
    entries = build_unsw_simple_mirror_manifest(configs, DATA_PATH, MODEL_ARCH)

    assert len(entries) == 1

    entry = entries[0]
    expected = {
        "dataset": "unsw",
        "data_path": DATA_PATH,
        "model_arch": MODEL_ARCH,
        "source_datasets": ["cic", "edge-iiotset-full"],
    }
    assert {key: entry[key] for key in expected} == expected
