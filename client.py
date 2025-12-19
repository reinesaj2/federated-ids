import argparse
import json
import logging
import os
import random
from typing import List, Tuple, Optional

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
)
from confusion_matrix_utils import compute_confusion_matrix

from data_preprocessing import (
    create_synthetic_classification_loaders,
    load_unsw_nb15,
    load_cic_ids2017,
    load_edge_iiotset,
    prepare_partitions_from_dataframe,
    numpy_to_loaders,
    infer_class_names_from_series,
)
from client_metrics import (
    ClientMetricsLogger,
    ClientFitTimer,
    calculate_weight_norms,
    calculate_weight_update_norm,
    analyze_data_distribution,
    create_label_histogram_json,
)
from privacy_accounting import DPAccountant
from secure_aggregation import generate_client_mask_sequence, mask_updates
from logging_utils import configure_logging, get_logger
from models.per_dataset_encoder import get_default_encoder_config, PerDatasetEncoderNet
from models.focal_loss import FocalLoss, compute_class_weights


DEFAULT_CLIENT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
ENCODER_DATASETS = {"unsw", "cic", "edge"}


def create_adamw_optimizer(parameters, lr: float, weight_decay: float = DEFAULT_WEIGHT_DECAY) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimpleNet(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve_model_arch(dataset_name: str, model_arch: str) -> str:
    arch = model_arch.lower()
    if arch == "auto":
        return "per_dataset_encoder" if dataset_name.lower() in ENCODER_DATASETS else "simple"
    return arch


def create_model(
    dataset_name: str,
    num_features: int,
    num_classes: int,
    model_arch: str,
    encoder_latent_dim: int,
) -> tuple[nn.Module, dict[str, str | int]]:
    resolved_arch = _resolve_model_arch(dataset_name, model_arch)
    if resolved_arch == "per_dataset_encoder":
        latent_override = encoder_latent_dim if encoder_latent_dim > 0 else None
        config = get_default_encoder_config(dataset_name.lower(), num_features, num_classes, latent_dim=latent_override)
        model = PerDatasetEncoderNet(config)
        metadata = {
            "model_arch": "per_dataset_encoder",
            "latent_dim": config.latent_dim,
            "encoder_hidden": "x".join(str(h) for h in config.encoder_hidden) if config.encoder_hidden else "",
        }
        return model, metadata

    return SimpleNet(num_features=num_features, num_classes=num_classes), {"model_arch": "simple"}


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state_dict = {}
    for (name, old_tensor), param in zip(state_dict.items(), parameters):
        new_state_dict[name] = torch.tensor(param, dtype=old_tensor.dtype)
    model.load_state_dict(new_state_dict, strict=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    global_params: Optional[List[np.ndarray]] = None,
    fedprox_mu: float = 0.0,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    loss_fn: Optional[nn.Module] = None,
) -> float:
    model.train()
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

    # Convert global parameters to tensors if FedProx is enabled
    global_tensors = None
    if fedprox_mu > 0.0 and global_params is not None:
        global_tensors = [torch.tensor(param, dtype=torch.float32).to(device) for param in global_params]

    # Use AdamW for all cases - proximal term is optimizer-agnostic
    # per research in docs/FEDPROX_OPTIMIZER_RESEARCH.md
    optimizer = create_adamw_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)

        # Add FedProx proximal term: mu/2 * ||w - w_global||^2
        if fedprox_mu > 0.0 and global_tensors is not None:
            prox_term = torch.tensor(0.0, device=device)
            for param, global_param in zip(model.parameters(), global_tensors):
                prox_term += torch.sum((param - global_param) ** 2)
            loss = loss + (fedprox_mu / 2.0) * prox_term

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)
            # store softmax probabilities and labels
            probs = torch.softmax(preds, dim=1).detach().cpu().numpy()
            probs_list.append(probs)
            labels_list.append(yb.detach().cpu().numpy())
    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    if probs_list:
        all_probs = np.concatenate(probs_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
    else:
        all_probs = np.zeros((0, 0), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int64)
    return float(avg_loss), float(acc), all_probs, all_labels


def _select_tau_for_target_fpr(
    y_true_val: np.ndarray,
    attack_probs_val: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float,
) -> float:
    """
    Select threshold tau to achieve target FPR on benign class.

    Args:
        y_true_val: Binary labels on validation set (0=benign, 1=attack)
        attack_probs_val: P(attack) scores on validation set
        thresholds: Candidate thresholds from precision_recall_curve
        target_fpr: Desired false positive rate (e.g., 0.10)

    Returns:
        Threshold tau achieving closest FPR to target
    """
    best_tau = 0.5
    best_fpr_diff = float("inf")

    for tau in thresholds:
        y_pred_val = (attack_probs_val >= tau).astype(int)
        benign_mask_val = y_true_val == 0

        if benign_mask_val.sum() == 0:
            continue

        fp_val = ((y_pred_val == 1) & benign_mask_val).sum()
        tn_val = ((y_pred_val == 0) & benign_mask_val).sum()
        fpr_val = fp_val / max(fp_val + tn_val, 1)
        fpr_diff = abs(fpr_val - target_fpr)

        if fpr_diff < best_fpr_diff:
            best_fpr_diff = fpr_diff
            best_tau = float(tau)

    return best_tau


def _select_tau_for_max_f1(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Select threshold tau to maximize F1 score.

    Args:
        precision: Precision values from precision_recall_curve
        recall: Recall values from precision_recall_curve
        thresholds: Candidate thresholds from precision_recall_curve

    Returns:
        Threshold tau maximizing F1 score
    """
    denom = np.maximum(precision + recall, 1e-12)
    f1_curve = 2 * precision * recall / denom

    if f1_curve.size == 0:
        return 0.5

    best_idx = int(np.argmax(f1_curve))
    # precision_recall_curve returns thresholds of length len(precision)-1
    tau_idx = max(0, min(best_idx - 1, thresholds.size - 1))
    return float(thresholds[tau_idx])


def select_threshold_tau(
    y_true_val: np.ndarray,
    attack_probs_val: np.ndarray,
    tau_mode: str,
    target_fpr: float,
) -> float:
    """
    Select threshold tau on validation set based on mode.

    Args:
        y_true_val: Binary labels on validation set (0=benign, 1=attack)
        attack_probs_val: P(attack) scores on validation set
        tau_mode: "low_fpr" (target FPR) or "max_f1" (maximize F1)
        target_fpr: Target false positive rate for low_fpr mode

    Returns:
        Selected threshold tau in [0, 1]
    """
    if attack_probs_val.size == 0:
        return 0.5

    # Compute PR curve on validation subset
    precision, recall, thresholds = precision_recall_curve(y_true_val, attack_probs_val)

    if thresholds.size == 0:
        return 0.5

    if tau_mode == "low_fpr":
        return _select_tau_for_target_fpr(y_true_val, attack_probs_val, thresholds, target_fpr)
    else:  # max_f1 mode
        return _select_tau_for_max_f1(precision, recall, thresholds)


class TorchClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        metrics_logger: ClientMetricsLogger,
        fit_timer: ClientFitTimer,
        data_stats: dict,
        runtime_config: dict,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.metrics_logger = metrics_logger
        self.fit_timer = fit_timer
        self.data_stats = data_stats
        self.model.to(self.device)
        self.round_num = 0
        self.runtime_config = runtime_config
        self.secure_aggregation_enabled = bool(runtime_config.get("secure_aggregation", False))
        self._secure_aggregation_seed_override = runtime_config.get("secure_aggregation_seed")
        self._secure_pairwise_seeds_override = runtime_config.get("secure_pairwise_seeds")
        self._last_secure_seed: Optional[int] = None
        self._last_pairwise_seeds: dict[str, int] = {}
        dp_delta = float(runtime_config.get("dp_delta", 1e-5))
        self.dp_accountant = DPAccountant(delta=dp_delta)
        self._dp_enabled_previous = bool(runtime_config.get("dp_enabled", False))
        self.logger = get_logger("client")
        self.class_names = class_names or []
        self.loss_fn = self._create_loss_function()

    def _create_loss_function(self) -> nn.Module:
        use_focal_loss = bool(self.runtime_config.get("use_focal_loss", False))
        if not use_focal_loss:
            return nn.CrossEntropyLoss()

        focal_gamma = float(self.runtime_config.get("focal_gamma", 2.0))
        n_classes = int(self.data_stats.get("n_classes", 2))

        all_labels = []
        for _, yb in self.train_loader:
            all_labels.append(yb)
        all_labels_tensor = torch.cat(all_labels)

        class_weights = compute_class_weights(all_labels_tensor, n_classes)
        class_weights = class_weights.to(self.device)

        return FocalLoss(alpha=class_weights, gamma=focal_gamma)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.round_num += 1

        secure_requested = bool(config.get("secure_aggregation", self.secure_aggregation_enabled))
        secure_seed = config.get("secure_aggregation_seed")
        if secure_seed is None:
            secure_seed = self._secure_aggregation_seed_override
        if secure_seed is not None:
            try:
                secure_seed = int(secure_seed)
            except (TypeError, ValueError):
                secure_seed = None
        if secure_requested and secure_seed is None:
            base_seed = int(os.environ.get("SEED", "42"))
            secure_seed = base_seed * 1000 + int(self.metrics_logger.client_id) * 97 + self.round_num
        secure_aggregation_active = secure_requested and secure_seed is not None
        self._last_secure_seed = secure_seed if secure_aggregation_active else None
        pairwise_config = config.get("secure_pairwise_seeds")
        if pairwise_config is None:
            pairwise_config = self._secure_pairwise_seeds_override
        pairwise_seeds: dict[str, int] = {}
        if pairwise_config:
            parsed_pairs = pairwise_config
            if isinstance(pairwise_config, str):
                try:
                    parsed_pairs = json.loads(pairwise_config)
                except json.JSONDecodeError:
                    parsed_pairs = {}
            if isinstance(parsed_pairs, dict):
                for peer_id, seed_value in parsed_pairs.items():
                    try:
                        pairwise_seeds[str(peer_id)] = int(seed_value)
                    except (TypeError, ValueError):
                        continue
        self._last_pairwise_seeds = pairwise_seeds if secure_aggregation_active else {}
        if secure_aggregation_active:
            try:
                self.logger.info(
                    "secure_aggregation_enabled",
                    extra={
                        "client_id": self.metrics_logger.client_id,
                        "round": self.round_num,
                    },
                )
            except Exception:
                pass

        # Get training hyperparameters
        epochs = int(config.get("epoch", self.runtime_config.get("local_epochs", 1)))
        lr = float(config.get("lr", self.runtime_config.get("lr", DEFAULT_CLIENT_LR)))
        weight_decay = float(config.get("weight_decay", self.runtime_config.get("weight_decay", DEFAULT_WEIGHT_DECAY)))
        batch_size = self.train_loader.batch_size or 32

        # Set initial parameters and capture before metrics
        set_parameters(self.model, parameters)
        weights_before = get_parameters(self.model)

        # Evaluate before training (optional - can be slow)
        loss_before, acc_before = None, None
        macro_f1_before = None
        try:
            if len(self.test_loader.dataset) > 0:
                loss_before, acc_before, probs_before, labels_before = evaluate(self.model, self.test_loader, self.device)
                # macro-F1 from hard predictions
                if probs_before.size > 0:
                    preds_before = np.argmax(probs_before, axis=1)
                    macro_f1_before = float(f1_score(labels_before, preds_before, average="macro"))
        except Exception:
            # Skip evaluation if it fails (e.g., empty test set)
            pass

        weight_norm_before = calculate_weight_norms(weights_before)

        # Time the training
        with self.fit_timer.time_fit():
            epochs_completed = 0
            attack_mode = str(self.runtime_config.get("adversary_mode", "none"))
            topk_fraction = float(self.runtime_config.get("adversary_topk_fraction", 0.1))
            target_class = int(self.runtime_config.get("adversary_target_class", 0))
            for epoch in range(epochs):
                if attack_mode == "grad_ascent":
                    # Perform gradient ascent by negating the loss
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    # Get FedProx parameters (server config takes precedence)
                    fedprox_mu = float(config.get("fedprox_mu", self.runtime_config.get("fedprox_mu", 0.0)))
                    global_tensors = None
                    if fedprox_mu > 0.0:
                        global_tensors = [torch.tensor(param, dtype=torch.float32).to(self.device) for param in parameters]
                    optimizer = create_adamw_optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        optimizer.zero_grad()
                        preds = self.model(xb)
                        loss = -criterion(preds, yb)

                        # Add FedProx proximal term (even for adversarial training)
                        if fedprox_mu > 0.0 and global_tensors is not None:
                            prox_term = 0.0
                            for param, global_param in zip(self.model.parameters(), global_tensors):
                                prox_term += torch.sum((param - global_param) ** 2)
                            loss = loss + (fedprox_mu / 2.0) * prox_term

                        loss.backward()

                        # Apply gradient clipping ONLY to adversarial clients
                        if attack_mode in ["grad_ascent", "label_flip", "sign_flip_topk"]:
                            clip_factor = float(self.runtime_config.get("adversary_clip_factor", 2.0))
                            if clip_factor > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_factor, norm_type=2.0)

                        optimizer.step()
                elif attack_mode == "label_flip":
                    # Train on intentionally wrong labels: rotate class index by +1
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    n_classes = max(int(self.data_stats.get("n_classes", 2)), 2)

                    # Get FedProx parameters (server config takes precedence)
                    fedprox_mu = float(config.get("fedprox_mu", self.runtime_config.get("fedprox_mu", 0.0)))
                    global_tensors = None
                    if fedprox_mu > 0.0:
                        global_tensors = [torch.tensor(param, dtype=torch.float32).to(self.device) for param in parameters]
                    optimizer = create_adamw_optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        with torch.no_grad():
                            flipped = (yb + 1) % n_classes
                        optimizer.zero_grad()
                        preds = self.model(xb)
                        loss = criterion(preds, flipped)

                        # Add FedProx proximal term
                        if fedprox_mu > 0.0 and global_tensors is not None:
                            prox_term = 0.0
                            for param, global_param in zip(self.model.parameters(), global_tensors):
                                prox_term += torch.sum((param - global_param) ** 2)
                            loss = loss + (fedprox_mu / 2.0) * prox_term

                        loss.backward()

                        # Apply gradient clipping ONLY to adversarial clients
                        if attack_mode in ["grad_ascent", "label_flip", "sign_flip_topk"]:
                            clip_factor = float(self.runtime_config.get("adversary_clip_factor", 2.0))
                            if clip_factor > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_factor, norm_type=2.0)

                        optimizer.step()
                elif attack_mode == "sign_flip_topk":
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    fedprox_mu = float(config.get("fedprox_mu", self.runtime_config.get("fedprox_mu", 0.0)))
                    global_tensors = None
                    if fedprox_mu > 0.0:
                        global_tensors = [torch.tensor(param, dtype=torch.float32).to(self.device) for param in parameters]
                    optimizer = create_adamw_optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        optimizer.zero_grad()
                        preds = self.model(xb)
                        loss = criterion(preds, yb)

                        if fedprox_mu > 0.0 and global_tensors is not None:
                            prox_term = 0.0
                            for param, global_param in zip(self.model.parameters(), global_tensors):
                                prox_term += torch.sum((param - global_param) ** 2)
                            loss = loss + (fedprox_mu / 2.0) * prox_term

                        loss.backward()

                        grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
                        if grads:
                            flat = torch.cat(grads)
                            k = max(1, int(topk_fraction * flat.numel()))
                            _, idx = torch.topk(flat.abs(), k)
                            flat[idx] = -flat[idx]
                            start = 0
                            for p in self.model.parameters():
                                if p.grad is None:
                                    continue
                                grad_len = p.grad.numel()
                                reshaped = flat[start : start + grad_len].reshape_as(p.grad)
                                p.grad.copy_(reshaped)
                                start += grad_len

                        clip_factor = float(self.runtime_config.get("adversary_clip_factor", 2.0))
                        if clip_factor > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_factor, norm_type=2.0)

                        optimizer.step()
                elif attack_mode == "targeted_label":
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    fedprox_mu = float(self.runtime_config.get("fedprox_mu", 0.0))
                    optimizer = create_adamw_optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
                    n_classes = max(int(self.data_stats.get("n_classes", 2)), 2)

                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        optimizer.zero_grad()
                        target = torch.full_like(yb, fill_value=target_class % n_classes)
                        preds = self.model(xb)
                        loss = criterion(preds, target)

                        loss.backward()

                        clip_factor = float(self.runtime_config.get("adversary_clip_factor", 2.0))
                        if clip_factor > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_factor, norm_type=2.0)

                        optimizer.step()
                else:
                    fedprox_mu = float(self.runtime_config.get("fedprox_mu", 0.0))
                    train_epoch(
                        self.model,
                        self.train_loader,
                        self.device,
                        lr,
                        global_params=parameters if fedprox_mu > 0.0 else None,
                        fedprox_mu=fedprox_mu,
                        weight_decay=weight_decay,
                        loss_fn=self.loss_fn,
                    )
                epochs_completed += 1

        # Capture after metrics
        weights_after = get_parameters(self.model)

        # Initialize DP metrics (None by default)
        dp_epsilon = None
        dp_delta = None
        dp_sigma = None
        dp_clip_norm = None
        dp_sample_rate = None
        dp_total_steps: Optional[int] = None
        dp_enabled = bool(self.runtime_config.get("dp_enabled", False))

        # Differential Privacy: clip update and add Gaussian noise (if enabled)
        if dp_enabled != self._dp_enabled_previous:
            self.dp_accountant.reset()
        self._dp_enabled_previous = dp_enabled
        try:
            if dp_enabled:
                clip = float(self.runtime_config.get("dp_clip", 1.0))
                noise_mult = float(self.runtime_config.get("dp_noise_multiplier", 0.0))
                sample_rate = float(self.runtime_config.get("dp_sample_rate", 1.0))
                dp_sample_rate = sample_rate
                # Build update (delta)
                deltas: List[np.ndarray] = [wa - wb for wb, wa in zip(weights_before, weights_after)]
                # Compute global L2 norm of concatenated delta
                flat = np.concatenate([d.reshape(-1) for d in deltas]) if deltas else np.zeros(1, dtype=np.float32)
                l2 = float(np.linalg.norm(flat))
                scale = 1.0
                if l2 > 0.0:
                    scale = min(1.0, clip / l2)
                clipped: List[np.ndarray] = [d * scale for d in deltas]
                # Noise scale
                sigma = noise_mult * clip
                # RNG seed prioritizes config seed, then env SEED, finally round number
                dp_seed = int(self.runtime_config.get("dp_seed", -1))
                if dp_seed < 0:
                    dp_seed = int(os.environ.get("SEED", "42"))
                rng = np.random.default_rng(dp_seed + self.round_num)
                noisy: List[np.ndarray] = [c + rng.normal(loc=0.0, scale=sigma, size=c.shape).astype(c.dtype) for c in clipped]
                # Reconstruct noisy weights as weights_before + noisy_delta
                weights_after = [wb + nd for wb, nd in zip(weights_before, noisy)]

                if noise_mult > 0.0:
                    self.dp_accountant.step(noise_multiplier=noise_mult, sample_rate=sample_rate)
                    dp_epsilon = self.dp_accountant.get_epsilon()
                dp_delta = self.dp_accountant.delta
                dp_sigma = noise_mult
                dp_clip_norm = clip
        except Exception:
            # Fail-open: if DP step errors, proceed with original weights_after
            pass
        if dp_enabled and dp_delta is None:
            dp_delta = self.dp_accountant.delta
        if dp_enabled:
            dp_total_steps = self.dp_accountant.get_total_steps()
        weight_norm_after = calculate_weight_norms(weights_after)
        weight_update_norm = calculate_weight_update_norm(weights_before, weights_after)
        # Compute a simple gradient norm proxy: norm of (weights_after - weights_before) / lr
        grad_norm_l2 = None
        try:
            if lr > 0:
                scaled = [(wa - wb) / lr for wb, wa in zip(weights_before, weights_after)]
                grad_norm_l2 = calculate_weight_norms(scaled)
        except Exception:
            pass

        # Evaluate after training (optional)
        loss_after, acc_after = None, None
        macro_f1_after = None
        micro_f1_after = None
        macro_f1_argmax = None
        benign_fpr_argmax = None
        f1_per_class_after_json = None
        f1_per_class_holdout_json = None
        precision_per_class_json = None
        recall_per_class_json = None
        fpr_after = None
        pr_auc_after = None
        threshold_tau = None
        f1_bin_tau = None
        benign_fpr_bin_tau = None
        tau_bin = None
        confusion_matrix_counts_json = None
        confusion_matrix_normalized_json = None
        confusion_matrix_class_names_json = None
        confusion_matrix_counts_holdout_json = None
        confusion_matrix_normalized_holdout_json = None
        confusion_matrix_class_names_holdout_json = None
        macro_f1_global_holdout = None
        micro_f1_global_holdout = None
        try:
            if len(self.test_loader.dataset) > 0:
                loss_after, acc_after, probs_after, labels_after = evaluate(self.model, self.test_loader, self.device)
                if probs_after.size > 0:
                    preds_after = np.argmax(probs_after, axis=1)
                    macro_f1_after = float(f1_score(labels_after, preds_after, average="macro"))
                    micro_f1_after = float(f1_score(labels_after, preds_after, average="micro"))
                    # Argmax metrics
                    macro_f1_argmax = macro_f1_after
                    benign_idx = 0
                    if np.sum(labels_after == benign_idx) > 0:
                        benign_recall = float(np.sum((labels_after == benign_idx) & (preds_after == benign_idx))) / float(
                            np.sum(labels_after == benign_idx)
                        )
                        benign_fpr_argmax = float(max(0.0, 1.0 - benign_recall))
                    # Per-class F1
                    num_classes = probs_after.shape[1]
                    f1s = []
                    for c in range(num_classes):
                        f1s.append(
                            float(
                                f1_score(
                                    labels_after,
                                    preds_after,
                                    labels=list(range(num_classes)),
                                    average=None,
                                    zero_division=0,
                                )[c]
                            )
                        )
                    import json as _json

                    f1_per_class_after_json = _json.dumps({str(i): f for i, f in enumerate(f1s)})
                    f1_per_class_holdout_json = f1_per_class_after_json
                    # Per-class Precision
                    precisions = []
                    for c in range(num_classes):
                        precisions.append(
                            float(
                                precision_score(
                                    labels_after,
                                    preds_after,
                                    labels=list(range(num_classes)),
                                    average=None,
                                    zero_division=0,
                                )[c]
                            )
                        )
                    precision_per_class_json = _json.dumps({str(i): p for i, p in enumerate(precisions)})
                    # Per-class Recall
                    recalls = []
                    for c in range(num_classes):
                        recalls.append(
                            float(
                                recall_score(
                                    labels_after,
                                    preds_after,
                                    labels=list(range(num_classes)),
                                    average=None,
                                    zero_division=0,
                                )[c]
                            )
                        )
                    recall_per_class_json = _json.dumps({str(i): r for i, r in enumerate(recalls)})
                    # BENIGN is class 0 by construction in preprocessing
                    if num_classes >= 2:
                        benign_idx = 0
                        benign_probs = probs_after[:, benign_idx]
                        # Attack-vs-BENIGN probabilities (1 - P(benign))
                        attack_probs = 1.0 - benign_probs
                        # PR-AUC (average precision) on full test
                        y_true_bin_full = (labels_after != benign_idx).astype(int)
                        pr_auc_after = float(average_precision_score(y_true_bin_full, attack_probs))

                        # Select tau on validation subset based on mode
                        tau_mode = str(self.runtime_config.get("tau_mode", "low_fpr"))
                        target_fpr = float(self.runtime_config.get("target_fpr", 0.10))

                        n = attack_probs.shape[0]
                        if n > 1:
                            # Split into validation subset for tau selection
                            rng = np.random.default_rng(int(os.environ.get("SEED", "42")) + self.round_num)
                            n_val = max(1, int(0.4 * n))
                            val_idx = rng.choice(n, size=n_val, replace=False)
                            y_true_bin_val = y_true_bin_full[val_idx]
                            attack_probs_val = attack_probs[val_idx]

                            threshold_tau = select_threshold_tau(y_true_bin_val, attack_probs_val, tau_mode, target_fpr)
                        else:
                            threshold_tau = 0.5

                        tau_bin = threshold_tau

                        # Apply chosen tau to full test
                        y_pred_attack_full = (attack_probs >= threshold_tau).astype(int)
                        benign_mask_full = labels_after == benign_idx
                        fp = int(np.sum(y_pred_attack_full[benign_mask_full] == 1))
                        tn = int(np.sum(y_pred_attack_full[benign_mask_full] == 0))
                        fpr_after = float(fp / max(fp + tn, 1))
                        benign_fpr_bin_tau = fpr_after
                        # Binary F1 at tau on full test
                        from sklearn.metrics import f1_score as _f1_bin

                        f1_bin_tau = float(_f1_bin(y_true_bin_full, y_pred_attack_full))

                    cm_counts = compute_confusion_matrix(labels_after, preds_after, num_classes, normalize=False)
                    cm_normalized = compute_confusion_matrix(labels_after, preds_after, num_classes, normalize=True)
                    cm_class_names = self.class_names or [f"class_{i}" for i in range(num_classes)]
                    confusion_matrix_counts_json = _json.dumps(cm_counts.tolist())
                    confusion_matrix_normalized_json = _json.dumps(cm_normalized.tolist())
                    confusion_matrix_class_names_json = _json.dumps(cm_class_names)
                    confusion_matrix_counts_holdout_json = confusion_matrix_counts_json
                    confusion_matrix_normalized_holdout_json = confusion_matrix_normalized_json
                    confusion_matrix_class_names_holdout_json = confusion_matrix_class_names_json
                    macro_f1_global_holdout = macro_f1_after
                    micro_f1_global_holdout = micro_f1_after
        except Exception:
            pass

        num_examples = len(self.train_loader.dataset)
        weights_to_send = weights_after
        secure_metrics: dict[str, float | int | bool | None] = {"secure_aggregation": False}
        effective_pairwise = self._last_pairwise_seeds if secure_aggregation_active else {}
        if secure_aggregation_active and self._last_secure_seed is not None:
            shapes = [tuple(w.shape) for w in weights_after]
            mask_sequence = generate_client_mask_sequence(
                str(self.metrics_logger.client_id),
                shapes,
                int(self._last_secure_seed),
                effective_pairwise,
            )
            weights_to_send = [mask_updates(w, mask) for w, mask in zip(weights_after, mask_sequence)]
            secure_metrics = {
                "secure_aggregation": True,
                "secure_aggregation_seed": int(self._last_secure_seed),
                "secure_aggregation_mask_checksum": float(sum(float(mask.sum()) for mask in mask_sequence)),
            }

        secure_flag = bool(secure_metrics.get("secure_aggregation", False))
        secure_seed = secure_metrics.get("secure_aggregation_seed")
        secure_mask_checksum = secure_metrics.get("secure_aggregation_mask_checksum")

        # Get timing
        t_fit_ms = self.fit_timer.get_last_fit_time_ms()

        # Log metrics
        self.metrics_logger.log_round_metrics(
            round_num=self.round_num,
            dataset_size=self.data_stats["dataset_size"],
            n_classes=self.data_stats["n_classes"],
            loss_before=loss_before,
            acc_before=acc_before,
            loss_after=loss_after,
            acc_after=acc_after,
            macro_f1_before=macro_f1_before,
            macro_f1_after=macro_f1_after,
            macro_f1_argmax=macro_f1_argmax,
            benign_fpr_argmax=benign_fpr_argmax,
            f1_per_class_after_json=f1_per_class_after_json,
            precision_per_class_json=precision_per_class_json,
            recall_per_class_json=recall_per_class_json,
            confusion_matrix_counts_json=confusion_matrix_counts_json,
            confusion_matrix_normalized_json=confusion_matrix_normalized_json,
            confusion_matrix_class_names_json=confusion_matrix_class_names_json,
            fpr_after=fpr_after,
            pr_auc_after=pr_auc_after,
            threshold_tau=threshold_tau,
            f1_bin_tau=f1_bin_tau,
            benign_fpr_bin_tau=benign_fpr_bin_tau,
            tau_bin=tau_bin,
            seed=(int(os.environ.get("SEED", str(config.get("seed", 0)))) if isinstance(config, dict) else None),
            weight_norm_before=weight_norm_before,
            weight_norm_after=weight_norm_after,
            weight_update_norm=weight_update_norm,
            grad_norm_l2=grad_norm_l2,
            t_fit_ms=t_fit_ms,
            epochs_completed=epochs_completed,
            lr=lr,
            batch_size=batch_size,
            dp_epsilon=dp_epsilon,
            dp_delta=dp_delta,
            dp_sigma=dp_sigma,
            dp_clip_norm=dp_clip_norm,
            dp_sample_rate=dp_sample_rate,
            dp_total_steps=dp_total_steps,
            dp_enabled_flag=dp_enabled,
            attack_mode=attack_mode,
            macro_f1_global_holdout=macro_f1_global_holdout,
            micro_f1_global_holdout=micro_f1_global_holdout,
            f1_per_class_holdout_json=f1_per_class_holdout_json,
            confusion_matrix_counts_holdout_json=confusion_matrix_counts_holdout_json,
            confusion_matrix_normalized_holdout_json=confusion_matrix_normalized_holdout_json,
            confusion_matrix_class_names_holdout_json=confusion_matrix_class_names_holdout_json,
            secure_aggregation_flag=secure_flag,
            secure_aggregation_seed=secure_seed if isinstance(secure_seed, (int, float)) else None,
            secure_aggregation_mask_checksum=(float(secure_mask_checksum) if isinstance(secure_mask_checksum, (int, float)) else None),
        )

        metrics_payload: dict[str, float | int | bool | None] = {"epochs_completed": epochs_completed}
        metrics_payload.update(secure_metrics)
        if dp_enabled:
            metrics_payload.update(
                {
                    "dp_enabled": True,
                    "dp_epsilon": dp_epsilon,
                    "dp_delta": dp_delta,
                    "dp_sigma": dp_sigma,
                    "dp_clip_norm": dp_clip_norm,
                }
            )
        else:
            metrics_payload["dp_enabled"] = False
        metrics_payload["attack_mode"] = attack_mode

        # Personalization: post-FL local fine-tuning (if enabled)
        personalization_epochs = int(self.runtime_config.get("personalization_epochs", 0))

        # Early exit if personalization disabled or no test data
        if personalization_epochs == 0 or len(self.test_loader.dataset) == 0:
            # Restore weights after DP modifications before returning
            set_parameters(self.model, weights_after)
            return weights_to_send, num_examples, metrics_payload

        # Save global model performance (already computed above)
        macro_f1_global = macro_f1_after
        benign_fpr_global = benign_fpr_bin_tau

        # Debug: Compute weight norm before personalization
        import os as _debug_os

        debug_enabled = _debug_os.environ.get("DEBUG_PERSONALIZATION", "0") == "1"
        weights_before_pers = get_parameters(self.model)
        if debug_enabled:
            norm_before = float(np.sqrt(sum(np.sum(w**2) for w in weights_before_pers)))
            cid = self.metrics_logger.client_id
            logging.getLogger("client").info(
                "personalization_start",
                extra={
                    "client_id": cid,
                    "round": self.round_num,
                    "epochs": personalization_epochs,
                    "global_f1": macro_f1_global,
                    "weight_norm": norm_before,
                    "train_size": len(self.train_loader.dataset),
                    "test_size": len(self.test_loader.dataset),
                },
            )
            # Preserve human-readable debug prints for tests and CLI
            print(
                f"[Client {cid}] Personalization R{self.round_num}: "
                f"Starting with {personalization_epochs} epochs, "
                f"global F1={macro_f1_global:.4f}, "
                f"weight_norm={norm_before:.4f}"
            )
            print(f"[Client {cid}] Train size: {len(self.train_loader.dataset)}, " f"Test size: {len(self.test_loader.dataset)}")

        # Fine-tune on local train data
        for epoch_idx in range(personalization_epochs):
            train_epoch(
                self.model,
                self.train_loader,
                self.device,
                lr,
                global_params=None,
                fedprox_mu=0.0,
                loss_fn=self.loss_fn,
            )
            if debug_enabled and epoch_idx == 0:
                # Check if weights changed after first epoch
                weights_after_first = get_parameters(self.model)
                norm_after_first = float(np.sqrt(sum(np.sum(w**2) for w in weights_after_first)))
                weight_delta = float(np.sqrt(sum(np.sum((w1 - w2) ** 2) for w1, w2 in zip(weights_after_first, weights_before_pers))))
                cid = self.metrics_logger.client_id
                logging.getLogger("client").info(
                    "personalization_epoch",
                    extra={
                        "client_id": cid,
                        "round": self.round_num,
                        "epoch": 1,
                        "weight_norm": norm_after_first,
                        "delta": weight_delta,
                    },
                )
                print(f"[Client {cid}] After epoch 1: " f"weight_norm={norm_after_first:.4f}, " f"delta={weight_delta:.6f}")

        # Evaluate personalized model
        try:
            _, _, probs_pers, labels_pers = evaluate(self.model, self.test_loader, self.device)

            macro_f1_personalized = None
            benign_fpr_personalized = None
            personalization_gain = None

            if probs_pers.size == 0:
                # No predictions, skip metrics
                pass
            else:
                preds_pers = np.argmax(probs_pers, axis=1)
                macro_f1_personalized = float(f1_score(labels_pers, preds_pers, average="macro"))

                # Compute FPR for personalized model using same threshold
                if probs_pers.shape[1] >= 2 and threshold_tau is not None:
                    benign_idx = 0
                    benign_probs_pers = probs_pers[:, benign_idx]
                    attack_probs_pers = 1.0 - benign_probs_pers
                    y_pred_attack_pers = (attack_probs_pers >= threshold_tau).astype(int)
                    benign_mask_pers = labels_pers == benign_idx
                    fp_pers = int(np.sum(y_pred_attack_pers[benign_mask_pers] == 1))
                    tn_pers = int(np.sum(y_pred_attack_pers[benign_mask_pers] == 0))
                    benign_fpr_personalized = float(fp_pers / max(fp_pers + tn_pers, 1))

                # Compute improvement gain
                if macro_f1_global is not None and macro_f1_personalized is not None:
                    personalization_gain = macro_f1_personalized - macro_f1_global

                if debug_enabled:
                    cid = self.metrics_logger.client_id
                    logging.getLogger("client").info(
                        "personalization_results",
                        extra={
                            "client_id": cid,
                            "round": self.round_num,
                            "global_f1": macro_f1_global,
                            "personalized_f1": macro_f1_personalized,
                            "gain": personalization_gain,
                        },
                    )
                    print(
                        f"[Client {cid}] Personalization results: "
                        f"global_F1={macro_f1_global:.4f}, "
                        f"personalized_F1={macro_f1_personalized:.4f}, "
                        f"gain={personalization_gain:.6f}"
                    )
                    if abs(personalization_gain) < 0.001:
                        logging.getLogger("client").warning(
                            "personalization_low_gain",
                            extra={
                                "client_id": cid,
                                "round": self.round_num,
                                "hint": "Check distribution, epochs, learning rate",
                            },
                        )
                        print(
                            f"[Client {cid}] WARNING: Near-zero gain detected! "
                            f"Possible causes: (1) train/test same distribution, "
                            f"(2) insufficient personalization epochs, "
                            f"(3) learning rate too low"
                        )

            # Log personalization metrics
            self.metrics_logger.log_personalization_metrics(
                round_num=self.round_num,
                macro_f1_global=macro_f1_global,
                macro_f1_personalized=macro_f1_personalized,
                benign_fpr_global=benign_fpr_global,
                benign_fpr_personalized=benign_fpr_personalized,
                personalization_gain=personalization_gain,
            )
            metrics_payload.update(
                {
                    "macro_f1_global": macro_f1_global,
                    "macro_f1_personalized": macro_f1_personalized,
                    "benign_fpr_global": benign_fpr_global,
                    "benign_fpr_personalized": benign_fpr_personalized,
                    "personalization_gain": personalization_gain,
                }
            )
        except Exception as e:
            # If personalization evaluation fails, log warning but continue
            logging.getLogger("client").warning("personalization_eval_failed", extra={"error": str(e)})

        # CRITICAL: Restore global model weights before returning to server
        set_parameters(self.model, weights_after)
        return weights_to_send, num_examples, metrics_payload

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc, _probs, _labels = evaluate(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        return loss, num_examples, {"accuracy": acc}


def main() -> None:
    configure_logging()
    logger = get_logger("client")
    parser = argparse.ArgumentParser(description="Flower client for Federated IDS demo")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "unsw", "cic", "edge-iiotset-quick", "edge-iiotset-nightly", "edge-iiotset-full"],
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="auto",
        choices=["auto", "simple", "per_dataset_encoder"],
        help="Model architecture to use (auto selects per-dataset encoders for CIC/UNSW)",
    )
    parser.add_argument(
        "--encoder_latent_dim",
        type=int,
        default=0,
        help="Override latent dimension for per-dataset encoders (0 keeps dataset default)",
    )
    parser.add_argument("--data_path", type=str, default="", help="Path to dataset CSV for unsw/cic")
    parser.add_argument(
        "--partition_strategy",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "protocol"],
    )
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--client_id", type=int, default=0, help="Client index in [0, num_clients)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha for non-IID")
    parser.add_argument(
        "--protocol_col",
        type=str,
        default="",
        help="Protocol column for protocol partitioning",
    )
    parser.add_argument(
        "--protocol_mapping_path",
        type=str,
        default="",
        help="Optional JSON file mapping protocol names to client IDs for protocol partitioning",
    )
    parser.add_argument(
        "--leakage_safe",
        action="store_true",
        help="Fit preprocessor on train-only and drop identifier/time-like columns",
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for synthetic dataset (default: 2 for binary)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adversary_mode",
        type=str,
        default="none",
        choices=["none", "label_flip", "grad_ascent"],
        help="Adversarial client behavior for robustness smoke tests",
    )
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=DEFAULT_CLIENT_LR)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay for local optimizer",
    )
    parser.add_argument("--use_focal_loss", action="store_true", help="Use FocalLoss instead of CrossEntropyLoss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="FocalLoss gamma parameter (focus on hard examples)")
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.01,
        help="FedProx proximal term coefficient (mu)",
    )
    parser.add_argument(
        "--secure_aggregation",
        action="store_true",
        help="Enable secure aggregation masking (deterministic additive shares; requires server support)",
    )
    parser.add_argument(
        "--dp_enabled",
        action="store_true",
        help="Enable client-side DP (clip + Gaussian noise on updates)",
    )
    parser.add_argument(
        "--dp_clip",
        type=float,
        default=1.0,
        help="L2 clipping threshold for client update (DP)",
    )
    parser.add_argument(
        "--dp_noise_multiplier",
        type=float,
        default=0.0,
        help="Gaussian noise multiplier (sigma) relative to clip for DP",
    )
    parser.add_argument(
        "--dp_seed",
        type=int,
        default=-1,
        help="Optional DP RNG seed (defaults to SEED if < 0)",
    )
    parser.add_argument(
        "--dp_delta",
        type=float,
        default=1e-5,
        help="Target delta for (epsilon, delta)-DP accountant",
    )
    parser.add_argument(
        "--dp_sample_rate",
        type=float,
        default=1.0,
        help="Sample rate used by DP accountant (default assumes full participation)",
    )
    parser.add_argument(
        "--tau_mode",
        type=str,
        default="low_fpr",
        choices=["low_fpr", "max_f1"],
        help="Threshold selection mode: low_fpr (target FPR) or max_f1 (maximize F1)",
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.10,
        help="Target false positive rate for low_fpr tau mode",
    )
    parser.add_argument(
        "--personalization_epochs",
        type=int,
        default=0,
        help="Number of local fine-tuning epochs after FL rounds for personalization (0=disabled)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory for metrics logging",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Initialize metrics logging
    client_metrics_path = os.path.join(args.logdir, f"client_{args.client_id}_metrics.csv")
    metrics_logger = ClientMetricsLogger(client_metrics_path, args.client_id)
    fit_timer = ClientFitTimer()
    logger.info(
        "metrics_init",
        extra={"client_id": args.client_id, "metrics_path": client_metrics_path},
    )

    class_names: list[str] = []
    protocol_mapping: dict[str, int] | None = None
    model_metadata: dict[str, str | int] = {}

    if args.dataset == "synthetic":
        train_loader, test_loader = create_synthetic_classification_loaders(
            num_samples=args.samples,
            num_features=args.features,
            batch_size=args.batch_size,
            seed=args.seed,
            num_classes=args.num_classes,
        )
        # Analyze data distribution for synthetic data
        synthetic_labels = np.random.randint(0, args.num_classes, size=args.samples)  # Approximate for metrics
        data_stats = analyze_data_distribution(synthetic_labels)
        label_hist_json = create_label_histogram_json(synthetic_labels)
        num_classes_global = args.num_classes
        class_names = ["BENIGN", "ATTACK"] if args.num_classes == 2 else [f"CLASS_{i}" for i in range(args.num_classes)]
        model, model_metadata = create_model(
            dataset_name=args.dataset,
            num_features=args.features,
            num_classes=num_classes_global,
            model_arch=args.model_arch,
            encoder_latent_dim=args.encoder_latent_dim,
        )
    else:
        if not args.data_path:
            raise SystemExit("--data_path is required for dataset unsw/cic/edge-iiotset")
        if args.dataset == "unsw":
            df, label_col, proto_col = load_unsw_nb15(args.data_path)
        elif args.dataset == "cic":
            df, label_col, proto_col = load_cic_ids2017(args.data_path)
        elif args.dataset.startswith("edge-iiotset"):
            df, label_col, proto_col = load_edge_iiotset(args.data_path, use_multiclass=True)
        else:
            raise SystemExit(f"Unknown dataset: {args.dataset}")
        chosen_proto_col = args.protocol_col or (proto_col or "")
        class_names = infer_class_names_from_series(df[label_col])
        if args.protocol_mapping_path:
            try:
                with open(args.protocol_mapping_path) as f:
                    raw_mapping = json.load(f)
            except OSError as exc:
                raise SystemExit(f"Failed to read protocol_mapping_path: {exc}") from exc
            if not isinstance(raw_mapping, dict):
                raise SystemExit("protocol_mapping_path must contain a JSON object mapping protocol names to client IDs")
            protocol_mapping = {}
            for proto_name, client_idx in raw_mapping.items():
                try:
                    normalized_proto = str(proto_name).strip().upper()
                    normalized_idx = int(client_idx)
                except (TypeError, ValueError):
                    continue
                if normalized_idx < 0 or normalized_idx >= args.num_clients:
                    raise SystemExit(f"protocol mapping client id {normalized_idx} must be in [0, {args.num_clients})")
                protocol_mapping[normalized_proto] = normalized_idx
        pre, X_parts, y_parts, num_classes_global = prepare_partitions_from_dataframe(
            df=df,
            label_col=label_col,
            partition_strategy=args.partition_strategy,
            num_clients=args.num_clients,
            seed=args.seed,
            alpha=args.alpha,
            protocol_col=chosen_proto_col if chosen_proto_col else None,
            leakage_safe=bool(args.leakage_safe),
            protocol_mapping=protocol_mapping,
        )
        if args.client_id < 0 or args.client_id >= len(X_parts):
            raise SystemExit(f"client_id must be in [0, {len(X_parts)})")
        X_client = X_parts[args.client_id]
        y_client = y_parts[args.client_id]
        # Warn if shard contains only a single class
        if len(np.unique(y_client)) <= 1:
            logger.warning(
                "single_class_shard",
                extra={
                    "client_id": args.client_id,
                    "num_classes_global": num_classes_global,
                },
            )
        train_loader, test_loader = numpy_to_loaders(X_client, y_client, batch_size=args.batch_size, seed=args.seed)
        num_features = X_client.shape[1]
        model, model_metadata = create_model(
            dataset_name=args.dataset,
            num_features=num_features,
            num_classes=num_classes_global,
            model_arch=args.model_arch,
            encoder_latent_dim=args.encoder_latent_dim,
        )
        # Analyze actual data distribution
        data_stats = analyze_data_distribution(y_client)
        label_hist_json = create_label_histogram_json(y_client)

    logger.info(
        "model_architecture",
        extra={
            "client_id": args.client_id,
            "model_arch": model_metadata.get("model_arch", "unknown"),
            "latent_dim": model_metadata.get("latent_dim"),
            "encoder_hidden": model_metadata.get("encoder_hidden"),
        },
    )

    # Model validation guard: assert output features match global num_classes
    model_output_features = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "net.4" in name:  # Last layer of SimpleNet
            model_output_features = module.out_features
            break

    if model_output_features is None:
        # Fallback: inspect the last layer
        last_layer = list(model.modules())[-1]
        if isinstance(last_layer, torch.nn.Linear):
            model_output_features = last_layer.out_features

    if model_output_features is not None:
        assert model_output_features == num_classes_global, (
            f"Model output features ({model_output_features}) must match "
            f"global num_classes ({num_classes_global}). "
            f"Label distribution: {label_hist_json}"
        )
        logger.info(
            "model_validation",
            extra={
                "client_id": args.client_id,
                "out_features": model_output_features,
                "num_classes_global": num_classes_global,
            },
        )
        logger.info(
            "label_histogram",
            extra={"client_id": args.client_id, "histogram": label_hist_json},
        )
    else:
        logger.warning(
            "model_validation_skipped",
            extra={"client_id": args.client_id},
        )
        logger.info(
            "label_histogram",
            extra={"client_id": args.client_id, "histogram": label_hist_json},
        )

    client = TorchClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        metrics_logger=metrics_logger,
        fit_timer=fit_timer,
        data_stats=data_stats,
        runtime_config={
            "adversary_mode": args.adversary_mode,
            "local_epochs": args.local_epochs,
            "lr": args.lr,
            "weight_decay": float(os.environ.get("D2_WEIGHT_DECAY", str(args.weight_decay))),
            "fedprox_mu": float(os.environ.get("D2_FEDPROX_MU", str(args.fedprox_mu))),
            # Privacy/robustness toggles
            "secure_aggregation": bool(
                args.secure_aggregation or os.environ.get("D2_SECURE_AGG", "0").lower() not in ("0", "false", "no", "")
            ),
            "secure_aggregation_seed": None,
            "secure_pairwise_seeds": None,
            "dp_enabled": bool(args.dp_enabled or os.environ.get("D2_DP_ENABLED", "0").lower() not in ("0", "false", "no", "")),
            "dp_clip": float(os.environ.get("D2_DP_CLIP", str(args.dp_clip))),
            "dp_noise_multiplier": float(os.environ.get("D2_DP_NOISE_MULTIPLIER", str(args.dp_noise_multiplier))),
            "dp_seed": int(os.environ.get("D2_DP_SEED", str(args.dp_seed))),
            "dp_delta": float(os.environ.get("D2_DP_DELTA", str(args.dp_delta))),
            "dp_sample_rate": float(os.environ.get("D2_DP_SAMPLE_RATE", str(args.dp_sample_rate))),
            # Adversarial gradient clipping
            "adversary_clip_factor": float(os.environ.get("D2_ADVERSARY_CLIP_FACTOR", "2.0")),
            # Threshold selection
            "tau_mode": args.tau_mode,
            "target_fpr": args.target_fpr,
            # Personalization
            "personalization_epochs": args.personalization_epochs,
            # FocalLoss
            "use_focal_loss": args.use_focal_loss,
            "focal_gamma": args.focal_gamma,
        },
        class_names=class_names,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
