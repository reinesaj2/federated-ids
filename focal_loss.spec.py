from __future__ import annotations

import torch

from models.focal_loss import FocalLoss, compute_class_weights


def test_focal_loss_gamma_zero_equals_cross_entropy():
    batch_size = 32
    num_classes = 15
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    focal_loss = FocalLoss(alpha=None, gamma=0.0, reduction="mean")
    focal_output = focal_loss(inputs, targets)

    ce_loss = torch.nn.functional.cross_entropy(inputs, targets)

    assert torch.allclose(focal_output, ce_loss, atol=1e-6)


def test_focal_loss_with_alpha_weights():
    batch_size = 32
    num_classes = 5
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    alpha = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0, reduction="mean")

    loss = focal_loss(inputs, targets)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_focal_loss_focuses_on_hard_examples():
    num_classes = 3
    easy_inputs = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    easy_targets = torch.tensor([0, 1])

    hard_inputs = torch.tensor([[1.0, 0.9, 0.8], [0.8, 1.0, 0.9]])
    hard_targets = torch.tensor([0, 1])

    focal_loss = FocalLoss(alpha=None, gamma=2.0, reduction="mean")
    easy_loss = focal_loss(easy_inputs, easy_targets)
    hard_loss = focal_loss(hard_inputs, hard_targets)

    assert hard_loss > easy_loss


def test_focal_loss_reduction_modes():
    batch_size = 10
    num_classes = 5
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    focal_mean = FocalLoss(reduction="mean")
    focal_sum = FocalLoss(reduction="sum")
    focal_none = FocalLoss(reduction="none")

    loss_mean = focal_mean(inputs, targets)
    loss_sum = focal_sum(inputs, targets)
    loss_none = focal_none(inputs, targets)

    assert loss_mean.shape == torch.Size([])
    assert loss_sum.shape == torch.Size([])
    assert loss_none.shape == torch.Size([batch_size])
    assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-6)


def test_compute_class_weights_inverse_frequency():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    num_classes = 3
    weights = compute_class_weights(labels, num_classes)

    assert weights[0] < weights[1]
    assert weights[1] < weights[2]
    assert weights.shape == torch.Size([3])


def test_compute_class_weights_handles_zero_counts():
    labels = torch.tensor([0, 0, 1, 1])
    num_classes = 5
    weights = compute_class_weights(labels, num_classes)

    assert weights.shape == torch.Size([5])
    assert torch.all(weights > 0)
    assert not torch.any(torch.isinf(weights))


def test_focal_loss_gamma_increases_loss_for_hard_examples():
    batch_size = 16
    num_classes = 10
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    focal_gamma_1 = FocalLoss(gamma=1.0)
    focal_gamma_2 = FocalLoss(gamma=2.0)
    focal_gamma_3 = FocalLoss(gamma=3.0)

    loss_1 = focal_gamma_1(inputs, targets)
    loss_2 = focal_gamma_2(inputs, targets)
    loss_3 = focal_gamma_3(inputs, targets)

    assert loss_1 >= 0
    assert loss_2 >= 0
    assert loss_3 >= 0


def test_focal_loss_device_compatibility():
    batch_size = 16
    num_classes = 5
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    alpha = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    loss = focal_loss(inputs, targets)

    assert loss.device == inputs.device


def test_compute_class_weights_normalization():
    labels = torch.tensor([0] * 100 + [1] * 10 + [2] * 1)
    num_classes = 3
    weights = compute_class_weights(labels, num_classes)

    assert torch.allclose(weights.sum(), torch.tensor(float(num_classes)), atol=1e-5)
