from __future__ import annotations

import torch

from models.per_dataset_encoder import (
    PerDatasetEncoderConfig,
    PerDatasetEncoderNet,
    get_default_encoder_config,
)


def test_per_dataset_encoder_forward_shape():
    config = PerDatasetEncoderConfig(
        dataset_name="unsw",
        input_dim=128,
        num_classes=5,
        encoder_hidden=[256, 128],
        latent_dim=64,
        shared_hidden=[64],
        dropout=0.1,
    )
    model = PerDatasetEncoderNet(config)
    x = torch.randn(32, 128)
    logits = model(x)
    assert logits.shape == (32, 5)


def test_get_default_encoder_config_provides_dataset_overrides():
    cfg_unsw = get_default_encoder_config("unsw", input_dim=196, num_classes=10)
    cfg_cic = get_default_encoder_config("cic", input_dim=220, num_classes=12)
    cfg_unknown = get_default_encoder_config("mystery", input_dim=64, num_classes=3)

    assert cfg_unsw.encoder_hidden != cfg_cic.encoder_hidden
    assert cfg_unknown.encoder_hidden == [256, 128]
    assert cfg_unknown.latent_dim == 128
