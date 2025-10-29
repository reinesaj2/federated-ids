from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import SimpleNet, get_parameters, set_parameters, train_epoch


def test_train_epoch_uses_adamw_defaults(monkeypatch):
    torch.manual_seed(123)
    data = torch.randn(20, 4)
    labels = torch.randint(0, 2, (20,))
    loader = DataLoader(TensorDataset(data, labels), batch_size=5)

    captured = {}

    class DummyOptimizer:
        def __init__(self, params, lr=None, weight_decay=None):  # noqa: D401
            captured["params"] = list(params)
            captured["lr"] = lr
            captured["weight_decay"] = weight_decay

        def zero_grad(self):
            pass

        def step(self):
            pass

    def fake_adamw(params, lr=None, weight_decay=None):
        return DummyOptimizer(params, lr=lr, weight_decay=weight_decay)

    def adam_fail(*args, **kwargs):  # pragma: no cover - ensures Adam unused
        raise AssertionError("torch.optim.Adam should not be used")

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)
    monkeypatch.setattr(torch.optim, "Adam", adam_fail)

    model = SimpleNet(num_features=4, num_classes=2)
    set_parameters(model, get_parameters(model))

    loss = train_epoch(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        lr=0.001,
    )

    assert isinstance(loss, float)
    assert captured["lr"] == pytest.approx(0.001)
    assert captured["weight_decay"] == pytest.approx(1e-4)
    assert len(captured["params"]) > 0
