import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

import train_centralized

MODEL_SIMPLE = "simple"
MODEL_ENCODER = "encoder"
WEIGHT_DECAY_DEFAULT_SIMPLE = 0.0
WEIGHT_DECAY_DEFAULT_ENCODER = 1e-4
WEIGHT_DECAY_EXPLICIT = 5e-4


@pytest.mark.parametrize(
    ("model_name", "expected_optimizer", "expected_weight_decay"),
    [
        (MODEL_SIMPLE, torch.optim.Adam, WEIGHT_DECAY_DEFAULT_SIMPLE),
        (MODEL_ENCODER, torch.optim.AdamW, WEIGHT_DECAY_DEFAULT_ENCODER),
    ],
)
def test_resolve_optimizer_config_defaults(
    model_name: str,
    expected_optimizer: type[torch.optim.Optimizer],
    expected_weight_decay: float,
) -> None:
    optimizer_cls, weight_decay = train_centralized.resolve_optimizer_config(model_name, None)

    assert optimizer_cls is expected_optimizer
    assert weight_decay == expected_weight_decay


@pytest.mark.parametrize(
    ("model_name", "expected_optimizer"),
    [
        (MODEL_SIMPLE, torch.optim.Adam),
        (MODEL_ENCODER, torch.optim.AdamW),
    ],
)
def test_resolve_optimizer_config_explicit_weight_decay(
    model_name: str,
    expected_optimizer: type[torch.optim.Optimizer],
) -> None:
    optimizer_cls, weight_decay = train_centralized.resolve_optimizer_config(model_name, WEIGHT_DECAY_EXPLICIT)

    assert optimizer_cls is expected_optimizer
    assert weight_decay == WEIGHT_DECAY_EXPLICIT
