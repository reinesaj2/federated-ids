from __future__ import annotations

import torch

from models.per_dataset_encoder import (
    DEFAULT_ENCODER_LAYOUTS,
    PerDatasetEncoderNet,
    get_default_encoder_config,
)


def test_edge_config_exists_in_default_layouts():
    assert "edge" in DEFAULT_ENCODER_LAYOUTS
    edge_layout = DEFAULT_ENCODER_LAYOUTS["edge"]
    assert edge_layout["encoder_hidden"] == [512, 384, 256]
    assert edge_layout["latent_dim"] == 192
    assert edge_layout["shared_hidden"] == [128, 64]
    assert edge_layout["dropout"] == 0.3


def test_get_default_encoder_config_returns_edge_config():
    input_dim = 61
    num_classes = 15
    config = get_default_encoder_config("edge", input_dim, num_classes)

    assert config.dataset_name == "edge"
    assert config.input_dim == 61
    assert config.num_classes == 15
    assert config.encoder_hidden == [512, 384, 256]
    assert config.latent_dim == 192
    assert config.shared_hidden == [128, 64]
    assert config.dropout == 0.3


def test_edge_encoder_net_forward_pass_shape():
    input_dim = 61
    num_classes = 15
    batch_size = 32

    config = get_default_encoder_config("edge", input_dim, num_classes)
    model = PerDatasetEncoderNet(config)

    x = torch.randn(batch_size, input_dim)
    output = model(x)

    assert output.shape == (batch_size, num_classes)


def test_edge_encoder_net_has_sufficient_capacity():
    input_dim = 61
    num_classes = 15

    config = get_default_encoder_config("edge", input_dim, num_classes)
    model = PerDatasetEncoderNet(config)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 200000


def test_edge_config_case_insensitive():
    config_lower = get_default_encoder_config("edge", 61, 15)
    config_upper = get_default_encoder_config("EDGE", 61, 15)
    config_mixed = get_default_encoder_config("Edge-IIoTset", 61, 15)

    assert config_lower.encoder_hidden == config_upper.encoder_hidden
    assert config_lower.encoder_hidden == config_mixed.encoder_hidden


def test_edge_encoder_net_applies_dropout():
    config = get_default_encoder_config("edge", 61, 15)
    model = PerDatasetEncoderNet(config)

    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) > 0
    assert all(layer.p == 0.3 for layer in dropout_layers)
