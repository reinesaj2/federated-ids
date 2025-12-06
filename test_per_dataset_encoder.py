from __future__ import annotations

import pytest
import torch

from models.per_dataset_encoder import (
    PerDatasetEncoderConfig,
    PerDatasetEncoderNet,
    get_default_encoder_config,
)

TEST_BATCH_SIZE = 32
TEST_INPUT_DIM_SMALL = 128
TEST_INPUT_DIM_MEDIUM = 196
TEST_INPUT_DIM_LARGE = 220
TEST_NUM_CLASSES_SMALL = 5
TEST_NUM_CLASSES_MEDIUM = 10
TEST_NUM_CLASSES_LARGE = 12


def test_per_dataset_encoder_forward_shape():
    config = PerDatasetEncoderConfig(
        dataset_name="unsw",
        input_dim=TEST_INPUT_DIM_SMALL,
        num_classes=TEST_NUM_CLASSES_SMALL,
        encoder_hidden=[256, 128],
        latent_dim=64,
        shared_hidden=[64],
        dropout=0.1,
    )
    model = PerDatasetEncoderNet(config)
    x = torch.randn(TEST_BATCH_SIZE, TEST_INPUT_DIM_SMALL)
    logits = model(x)
    assert logits.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES_SMALL)


def test_per_dataset_encoder_empty_encoder_hidden():
    config = PerDatasetEncoderConfig(
        dataset_name="test",
        input_dim=TEST_INPUT_DIM_SMALL,
        num_classes=TEST_NUM_CLASSES_SMALL,
        encoder_hidden=[],
        latent_dim=64,
        shared_hidden=[32],
        dropout=0.0,
    )
    model = PerDatasetEncoderNet(config)
    x = torch.randn(TEST_BATCH_SIZE, TEST_INPUT_DIM_SMALL)
    logits = model(x)
    assert logits.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES_SMALL)


def test_per_dataset_encoder_single_layer_encoder():
    config = PerDatasetEncoderConfig(
        dataset_name="test",
        input_dim=TEST_INPUT_DIM_SMALL,
        num_classes=TEST_NUM_CLASSES_SMALL,
        encoder_hidden=[256],
        latent_dim=128,
        shared_hidden=[64],
        dropout=0.1,
    )
    model = PerDatasetEncoderNet(config)
    x = torch.randn(TEST_BATCH_SIZE, TEST_INPUT_DIM_SMALL)
    logits = model(x)
    assert logits.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES_SMALL)


def test_per_dataset_encoder_zero_dropout():
    config = PerDatasetEncoderConfig(
        dataset_name="test",
        input_dim=TEST_INPUT_DIM_SMALL,
        num_classes=TEST_NUM_CLASSES_SMALL,
        encoder_hidden=[128, 64],
        latent_dim=32,
        shared_hidden=[16],
        dropout=0.0,
    )
    model = PerDatasetEncoderNet(config)
    x = torch.randn(TEST_BATCH_SIZE, TEST_INPUT_DIM_SMALL)
    logits = model(x)
    assert logits.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES_SMALL)


@pytest.mark.parametrize(
    "batch_size,input_dim,num_classes",
    [
        (1, 64, 2),
        (16, 128, 5),
        (64, 256, 10),
        (128, 512, 15),
    ],
)
def test_per_dataset_encoder_various_dimensions(batch_size: int, input_dim: int, num_classes: int):
    config = PerDatasetEncoderConfig(
        dataset_name="test",
        input_dim=input_dim,
        num_classes=num_classes,
        encoder_hidden=[256, 128],
        latent_dim=64,
        shared_hidden=[32],
        dropout=0.1,
    )
    model = PerDatasetEncoderNet(config)
    if batch_size == 1:
        model.eval()
    x = torch.randn(batch_size, input_dim)
    logits = model(x)
    assert logits.shape == (batch_size, num_classes)


def test_per_dataset_encoder_gradient_flow():
    config = PerDatasetEncoderConfig(
        dataset_name="test",
        input_dim=TEST_INPUT_DIM_SMALL,
        num_classes=TEST_NUM_CLASSES_SMALL,
        encoder_hidden=[256, 128],
        latent_dim=64,
        shared_hidden=[32],
        dropout=0.0,
    )
    model = PerDatasetEncoderNet(config)
    x = torch.randn(TEST_BATCH_SIZE, TEST_INPUT_DIM_SMALL, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_get_default_encoder_config_unsw():
    cfg = get_default_encoder_config("unsw", input_dim=TEST_INPUT_DIM_MEDIUM, num_classes=TEST_NUM_CLASSES_MEDIUM)
    assert (
        cfg.dataset_name,
        cfg.input_dim,
        cfg.num_classes,
        cfg.encoder_hidden,
        cfg.latent_dim,
        cfg.shared_hidden,
        cfg.dropout,
    ) == ("unsw", TEST_INPUT_DIM_MEDIUM, TEST_NUM_CLASSES_MEDIUM, [512, 256], 192, [128, 64], 0.2)


def test_get_default_encoder_config_cic():
    cfg = get_default_encoder_config("cic", input_dim=TEST_INPUT_DIM_LARGE, num_classes=TEST_NUM_CLASSES_LARGE)
    assert (
        cfg.dataset_name,
        cfg.input_dim,
        cfg.num_classes,
        cfg.encoder_hidden,
        cfg.latent_dim,
        cfg.shared_hidden,
        cfg.dropout,
    ) == ("cic", TEST_INPUT_DIM_LARGE, TEST_NUM_CLASSES_LARGE, [768, 384, 192], 256, [192, 96], 0.25)


def test_get_default_encoder_config_unknown_dataset_uses_fallback():
    cfg = get_default_encoder_config("mystery", input_dim=64, num_classes=3)
    assert (cfg.encoder_hidden, cfg.latent_dim, cfg.shared_hidden, cfg.dropout) == ([256, 128], 128, [64], 0.1)


def test_get_default_encoder_config_latent_override():
    override_latent = 512
    cfg = get_default_encoder_config(
        "unsw", input_dim=TEST_INPUT_DIM_MEDIUM, num_classes=TEST_NUM_CLASSES_MEDIUM, latent_dim=override_latent
    )
    assert cfg.latent_dim == override_latent


def test_get_default_encoder_config_case_insensitive():
    cfg_lower = get_default_encoder_config("unsw", input_dim=TEST_INPUT_DIM_MEDIUM, num_classes=TEST_NUM_CLASSES_MEDIUM)
    cfg_upper = get_default_encoder_config("UNSW", input_dim=TEST_INPUT_DIM_MEDIUM, num_classes=TEST_NUM_CLASSES_MEDIUM)
    assert cfg_lower.encoder_hidden == cfg_upper.encoder_hidden
    assert cfg_lower.latent_dim == cfg_upper.latent_dim
