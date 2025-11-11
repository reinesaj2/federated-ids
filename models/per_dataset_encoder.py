from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class PerDatasetEncoderConfig:
    dataset_name: str
    input_dim: int
    num_classes: int
    encoder_hidden: List[int]
    latent_dim: int
    shared_hidden: List[int]
    dropout: float = 0.0


def _build_mlp(
    input_dim: int,
    hidden_layers: Iterable[int],
    dropout: float,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden
    return nn.Sequential(*layers)


class PerDatasetEncoderNet(nn.Module):
    def __init__(self, config: PerDatasetEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = _build_mlp(config.input_dim, config.encoder_hidden, config.dropout)
        encoder_out_dim = config.encoder_hidden[-1] if config.encoder_hidden else config.input_dim

        self.latent_projection = nn.Sequential(
            nn.Linear(encoder_out_dim, config.latent_dim),
            nn.BatchNorm1d(config.latent_dim),
            nn.ReLU(),
        )

        shared_layers: List[nn.Module] = []
        prev = config.latent_dim
        for hidden in config.shared_hidden:
            shared_layers.append(nn.Linear(prev, hidden))
            shared_layers.append(nn.ReLU())
            if config.dropout > 0:
                shared_layers.append(nn.Dropout(config.dropout))
            prev = hidden
        shared_layers.append(nn.Linear(prev, config.num_classes))
        self.shared_head = nn.Sequential(*shared_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder:
            x = self.encoder(x)
        z = self.latent_projection(x)
        return self.shared_head(z)


DEFAULT_ENCODER_LAYOUTS = {
    "unsw": {
        "encoder_hidden": [512, 256],
        "latent_dim": 192,
        "shared_hidden": [128, 64],
        "dropout": 0.2,
    },
    "cic": {
        "encoder_hidden": [768, 384, 192],
        "latent_dim": 256,
        "shared_hidden": [192, 96],
        "dropout": 0.25,
    },
}

FALLBACK_LAYOUT = {
    "encoder_hidden": [256, 128],
    "latent_dim": 128,
    "shared_hidden": [64],
    "dropout": 0.1,
}


def get_default_encoder_config(
    dataset_name: str,
    input_dim: int,
    num_classes: int,
    latent_dim: int | None = None,
) -> PerDatasetEncoderConfig:
    layout = DEFAULT_ENCODER_LAYOUTS.get(dataset_name.lower(), FALLBACK_LAYOUT)
    encoder_hidden = layout["encoder_hidden"]
    shared_hidden = layout["shared_hidden"]
    dropout = layout["dropout"]
    final_latent = latent_dim or layout["latent_dim"]

    return PerDatasetEncoderConfig(
        dataset_name=dataset_name,
        input_dim=input_dim,
        num_classes=num_classes,
        encoder_hidden=list(encoder_hidden),
        latent_dim=final_latent,
        shared_hidden=list(shared_hidden),
        dropout=dropout,
    )
