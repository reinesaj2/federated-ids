import argparse
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

from data_preprocessing import (
    create_synthetic_classification_loaders,
    load_unsw_nb15,
    load_cic_ids2017,
    prepare_partitions_from_dataframe,
    numpy_to_loaders,
    numpy_to_train_val_test_loaders,
)
from client_metrics import (
    ClientMetricsLogger,
    ClientFitTimer,
    calculate_weight_norms,
    calculate_weight_update_norm,
    analyze_data_distribution,
    create_label_histogram_json,
)


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
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    num_batches = 0

    # Convert global parameters to tensors if FedProx is enabled
    global_tensors = None
    if fedprox_mu > 0.0 and global_params is not None:
        global_tensors = [
            torch.tensor(param, dtype=torch.float32).to(device)
            for param in global_params
        ]

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)

        # Add FedProx proximal term: mu/2 * ||w - w_global||^2
        if fedprox_mu > 0.0 and global_tensors is not None:
            prox_term = 0.0
            for param, global_param in zip(model.parameters(), global_tensors):
                prox_term += torch.sum((param - global_param) ** 2)
            loss = loss + (fedprox_mu / 2.0) * prox_term

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
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


def _select_tau_for_max_f1(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray
) -> float:
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
        return _select_tau_for_target_fpr(
            y_true_val, attack_probs_val, thresholds, target_fpr
        )
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

    def get_parameters(self, config):  # type: ignore[override]
        return get_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        self.round_num += 1

        # Get training hyperparameters
        epochs = int(config.get("epoch", self.runtime_config.get("local_epochs", 1)))
        lr = float(config.get("lr", self.runtime_config.get("lr", 0.01)))
        batch_size = self.train_loader.batch_size or 32

        # Set initial parameters and capture before metrics
        set_parameters(self.model, parameters)
        weights_before = get_parameters(self.model)

        # Evaluate before training (optional - can be slow)
        loss_before, acc_before = None, None
        macro_f1_before = None
        try:
            if len(self.test_loader.dataset) > 0:
                loss_before, acc_before, probs_before, labels_before = evaluate(
                    self.model, self.test_loader, self.device
                )
                # macro-F1 from hard predictions
                if probs_before.size > 0:
                    preds_before = np.argmax(probs_before, axis=1)
                    macro_f1_before = float(
                        f1_score(labels_before, preds_before, average="macro")
                    )
        except Exception:
            # Skip evaluation if it fails (e.g., empty test set)
            pass

        weight_norm_before = calculate_weight_norms(weights_before)

        # Time the training
        with self.fit_timer.time_fit():
            epochs_completed = 0
            for epoch in range(epochs):
                mode = str(self.runtime_config.get("adversary_mode", "none"))
                if mode == "grad_ascent":
                    # Perform gradient ascent by negating the loss
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

                    # Get FedProx parameters
                    fedprox_mu = float(self.runtime_config.get("fedprox_mu", 0.0))
                    global_tensors = None
                    if fedprox_mu > 0.0:
                        global_tensors = [
                            torch.tensor(param, dtype=torch.float32).to(self.device)
                            for param in parameters
                        ]

                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        optimizer.zero_grad()
                        preds = self.model(xb)
                        loss = -criterion(preds, yb)

                        # Add FedProx proximal term (even for adversarial training)
                        if fedprox_mu > 0.0 and global_tensors is not None:
                            prox_term = 0.0
                            for param, global_param in zip(
                                self.model.parameters(), global_tensors
                            ):
                                prox_term += torch.sum((param - global_param) ** 2)
                            loss = loss + (fedprox_mu / 2.0) * prox_term

                        loss.backward()
                        optimizer.step()
                elif mode == "label_flip":
                    # Train on intentionally wrong labels: rotate class index by +1
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    n_classes = max(int(self.data_stats.get("n_classes", 2)), 2)

                    # Get FedProx parameters
                    fedprox_mu = float(self.runtime_config.get("fedprox_mu", 0.0))
                    global_tensors = None
                    if fedprox_mu > 0.0:
                        global_tensors = [
                            torch.tensor(param, dtype=torch.float32).to(self.device)
                            for param in parameters
                        ]

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
                            for param, global_param in zip(
                                self.model.parameters(), global_tensors
                            ):
                                prox_term += torch.sum((param - global_param) ** 2)
                            loss = loss + (fedprox_mu / 2.0) * prox_term

                        loss.backward()
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
                    )
                epochs_completed += 1

        # Capture after metrics
        weights_after = get_parameters(self.model)
        # Differential Privacy: clip update and add Gaussian noise (if enabled)
        try:
            dp_enabled = bool(self.runtime_config.get("dp_enabled", False))
            if dp_enabled:
                clip = float(self.runtime_config.get("dp_clip", 1.0))
                noise_mult = float(self.runtime_config.get("dp_noise_multiplier", 0.0))
                # Build update (delta)
                deltas: List[np.ndarray] = [
                    wa - wb for wb, wa in zip(weights_before, weights_after)
                ]
                # Compute global L2 norm of concatenated delta
                flat = (
                    np.concatenate([d.reshape(-1) for d in deltas])
                    if deltas
                    else np.zeros(1, dtype=np.float32)
                )
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
                noisy: List[np.ndarray] = [
                    c + rng.normal(loc=0.0, scale=sigma, size=c.shape).astype(c.dtype)
                    for c in clipped
                ]
                # Reconstruct noisy weights as weights_before + noisy_delta
                weights_after = [wb + nd for wb, nd in zip(weights_before, noisy)]
        except Exception:
            # Fail-open: if DP step errors, proceed with original weights_after
            pass
        weight_norm_after = calculate_weight_norms(weights_after)
        weight_update_norm = calculate_weight_update_norm(weights_before, weights_after)
        # Compute a simple gradient norm proxy: norm of (weights_after - weights_before) / lr
        grad_norm_l2 = None
        try:
            if lr > 0:
                scaled = [
                    (wa - wb) / lr for wb, wa in zip(weights_before, weights_after)
                ]
                grad_norm_l2 = calculate_weight_norms(scaled)
        except Exception:
            pass

        # Evaluate after training (optional)
        loss_after, acc_after = None, None
        macro_f1_after = None
        macro_f1_argmax = None
        benign_fpr_argmax = None
        f1_per_class_after_json = None
        precision_per_class_json = None
        recall_per_class_json = None
        fpr_after = None
        pr_auc_after = None
        threshold_tau = None
        f1_bin_tau = None
        benign_fpr_bin_tau = None
        tau_bin = None
        try:
            if len(self.test_loader.dataset) > 0:
                loss_after, acc_after, probs_after, labels_after = evaluate(
                    self.model, self.test_loader, self.device
                )
                if probs_after.size > 0:
                    preds_after = np.argmax(probs_after, axis=1)
                    macro_f1_after = float(
                        f1_score(labels_after, preds_after, average="macro")
                    )
                    # Argmax metrics
                    macro_f1_argmax = macro_f1_after
                    benign_idx = 0
                    if np.sum(labels_after == benign_idx) > 0:
                        benign_recall = float(
                            np.sum(
                                (labels_after == benign_idx)
                                & (preds_after == benign_idx)
                            )
                        ) / float(np.sum(labels_after == benign_idx))
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

                    f1_per_class_after_json = _json.dumps(
                        {str(i): f for i, f in enumerate(f1s)}
                    )
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
                    precision_per_class_json = _json.dumps(
                        {str(i): p for i, p in enumerate(precisions)}
                    )
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
                    recall_per_class_json = _json.dumps(
                        {str(i): r for i, r in enumerate(recalls)}
                    )
                    # BENIGN is class 0 by construction in preprocessing
                    if num_classes >= 2:
                        benign_idx = 0
                        benign_probs = probs_after[:, benign_idx]
                        # Attack-vs-BENIGN probabilities (1 - P(benign))
                        attack_probs = 1.0 - benign_probs
                        # PR-AUC (average precision) on full test
                        y_true_bin_full = (labels_after != benign_idx).astype(int)
                        pr_auc_after = float(
                            average_precision_score(y_true_bin_full, attack_probs)
                        )

                        # Select tau on validation subset based on mode
                        tau_mode = str(self.runtime_config.get("tau_mode", "low_fpr"))
                        target_fpr = float(self.runtime_config.get("target_fpr", 0.10))

                        n = attack_probs.shape[0]
                        if n > 1:
                            # Split into validation subset for tau selection
                            rng = np.random.default_rng(
                                int(os.environ.get("SEED", "42")) + self.round_num
                            )
                            n_val = max(1, int(0.4 * n))
                            val_idx = rng.choice(n, size=n_val, replace=False)
                            y_true_bin_val = y_true_bin_full[val_idx]
                            attack_probs_val = attack_probs[val_idx]

                            threshold_tau = select_threshold_tau(
                                y_true_bin_val, attack_probs_val, tau_mode, target_fpr
                            )
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
        except Exception:
            pass

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
            fpr_after=fpr_after,
            pr_auc_after=pr_auc_after,
            threshold_tau=threshold_tau,
            f1_bin_tau=f1_bin_tau,
            benign_fpr_bin_tau=benign_fpr_bin_tau,
            tau_bin=tau_bin,
            seed=(
                int(os.environ.get("SEED", str(config.get("seed", 0))))
                if isinstance(config, dict)
                else None
            ),
            weight_norm_before=weight_norm_before,
            weight_norm_after=weight_norm_after,
            weight_update_norm=weight_update_norm,
            grad_norm_l2=grad_norm_l2,
            t_fit_ms=t_fit_ms,
            epochs_completed=epochs_completed,
            lr=lr,
            batch_size=batch_size,
        )

        # Personalization: post-FL local fine-tuning (if enabled)
        personalization_epochs = int(
            self.runtime_config.get("personalization_epochs", 0)
        )

        # Early exit if personalization disabled or no test data
        if personalization_epochs == 0 or len(self.test_loader.dataset) == 0:
            num_examples = len(self.train_loader.dataset)
            metrics = {}
            return weights_after, num_examples, metrics

        # Save global model performance (already computed above)
        macro_f1_global = macro_f1_after
        benign_fpr_global = benign_fpr_bin_tau

        # Fine-tune on local train data
        for _ in range(personalization_epochs):
            train_epoch(
                self.model,
                self.train_loader,
                self.device,
                lr,
                global_params=None,
                fedprox_mu=0.0,
            )

        # Evaluate personalized model
        try:
            _, _, probs_pers, labels_pers = evaluate(
                self.model, self.test_loader, self.device
            )

            macro_f1_personalized = None
            benign_fpr_personalized = None
            personalization_gain = None

            if probs_pers.size == 0:
                # No predictions, skip metrics
                pass
            else:
                preds_pers = np.argmax(probs_pers, axis=1)
                macro_f1_personalized = float(
                    f1_score(labels_pers, preds_pers, average="macro")
                )

                # Compute FPR for personalized model using same threshold
                if probs_pers.shape[1] >= 2 and threshold_tau is not None:
                    benign_idx = 0
                    benign_probs_pers = probs_pers[:, benign_idx]
                    attack_probs_pers = 1.0 - benign_probs_pers
                    y_pred_attack_pers = (attack_probs_pers >= threshold_tau).astype(
                        int
                    )
                    benign_mask_pers = labels_pers == benign_idx
                    fp_pers = int(np.sum(y_pred_attack_pers[benign_mask_pers] == 1))
                    tn_pers = int(np.sum(y_pred_attack_pers[benign_mask_pers] == 0))
                    benign_fpr_personalized = float(fp_pers / max(fp_pers + tn_pers, 1))

                # Compute improvement gain
                if macro_f1_global is not None and macro_f1_personalized is not None:
                    personalization_gain = macro_f1_personalized - macro_f1_global

            # Log personalization metrics
            self.metrics_logger.log_personalization_metrics(
                round_num=self.round_num,
                macro_f1_global=macro_f1_global,
                macro_f1_personalized=macro_f1_personalized,
                benign_fpr_global=benign_fpr_global,
                benign_fpr_personalized=benign_fpr_personalized,
                personalization_gain=personalization_gain,
            )
        except Exception as e:
            # If personalization evaluation fails, log warning but continue
            print(f"[Client] Warning: Personalization evaluation failed: {e}")

        # CRITICAL: Restore global model weights before returning to server
        set_parameters(self.model, weights_after)

        num_examples = len(self.train_loader.dataset)
        metrics = {}
        return weights_after, num_examples, metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        set_parameters(self.model, parameters)
        loss, acc, _probs, _labels = evaluate(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        return loss, num_examples, {"accuracy": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower client for Federated IDS demo")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument(
        "--dataset", type=str, default="synthetic", choices=["synthetic", "unsw", "cic"]
    )
    parser.add_argument(
        "--data_path", type=str, default="", help="Path to dataset CSV for unsw/cic"
    )
    parser.add_argument(
        "--partition_strategy",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "protocol"],
    )
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument(
        "--client_id", type=int, default=0, help="Client index in [0, num_clients)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Dirichlet alpha for non-IID"
    )
    parser.add_argument(
        "--protocol_col",
        type=str,
        default="",
        help="Protocol column for protocol partitioning",
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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.01,
        help="FedProx proximal term coefficient (mu)",
    )
    parser.add_argument(
        "--secure_aggregation",
        action="store_true",
        help="Enable secure aggregation mode (stub toggle; no functional change)",
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

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Initialize metrics logging
    client_metrics_path = os.path.join(
        args.logdir, f"client_{args.client_id}_metrics.csv"
    )
    metrics_logger = ClientMetricsLogger(client_metrics_path, args.client_id)
    fit_timer = ClientFitTimer()
    print(f"[Client {args.client_id}] Logging metrics to: {client_metrics_path}")

    if args.dataset == "synthetic":
        train_loader, test_loader = create_synthetic_classification_loaders(
            num_samples=args.samples,
            num_features=args.features,
            batch_size=args.batch_size,
            seed=args.seed,
            num_classes=args.num_classes,
        )
        model = SimpleNet(num_features=args.features, num_classes=args.num_classes)
        # Analyze data distribution for synthetic data
        synthetic_labels = np.random.randint(
            0, args.num_classes, size=args.samples
        )  # Approximate for metrics
        data_stats = analyze_data_distribution(synthetic_labels)
        label_hist_json = create_label_histogram_json(synthetic_labels)
        num_classes_global = args.num_classes
    else:
        if not args.data_path:
            raise SystemExit("--data_path is required for dataset unsw/cic")
        if args.dataset == "unsw":
            df, label_col, proto_col = load_unsw_nb15(args.data_path)
        else:
            df, label_col, proto_col = load_cic_ids2017(args.data_path)
        chosen_proto_col = args.protocol_col or (proto_col or "")
        pre, X_parts, y_parts, num_classes_global = prepare_partitions_from_dataframe(
            df=df,
            label_col=label_col,
            partition_strategy=args.partition_strategy,
            num_clients=args.num_clients,
            seed=args.seed,
            alpha=args.alpha,
            protocol_col=chosen_proto_col if chosen_proto_col else None,
            leakage_safe=bool(args.leakage_safe),
        )
        if args.client_id < 0 or args.client_id >= len(X_parts):
            raise SystemExit(f"client_id must be in [0, {len(X_parts)})")
        X_client = X_parts[args.client_id]
        y_client = y_parts[args.client_id]
        # Warn if shard contains only a single class
        if len(np.unique(y_client)) <= 1:
            print(
                f"[Client {args.client_id}] Warning: single-class shard detected; "
                f"using global num_classes={num_classes_global}"
            )
        train_loader, test_loader = numpy_to_loaders(
            X_client, y_client, batch_size=args.batch_size, seed=args.seed
        )
        num_features = X_client.shape[1]
        model = SimpleNet(num_features=num_features, num_classes=num_classes_global)
        # Analyze actual data distribution
        data_stats = analyze_data_distribution(y_client)
        label_hist_json = create_label_histogram_json(y_client)

    # Model validation guard: assert output features match global num_classes
    model_output_features = None
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear) and "net.4" in name
        ):  # Last layer of SimpleNet
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
        print(
            f"[Client {args.client_id}] Model validation passed: "
            f"out_features={model_output_features}, num_classes_global={num_classes_global}"
        )
        print(f"[Client {args.client_id}] Label histogram: {label_hist_json}")
    else:
        print(
            f"[Client {args.client_id}] Warning: Could not validate model output features"
        )
        print(f"[Client {args.client_id}] Label histogram: {label_hist_json}")

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
            "fedprox_mu": float(os.environ.get("D2_FEDPROX_MU", str(args.fedprox_mu))),
            # Privacy/robustness toggles
            "secure_aggregation": bool(
                args.secure_aggregation
                or os.environ.get("D2_SECURE_AGG", "0").lower()
                not in ("0", "false", "no", "")
            ),
            "dp_enabled": bool(
                args.dp_enabled
                or os.environ.get("D2_DP_ENABLED", "0").lower()
                not in ("0", "false", "no", "")
            ),
            "dp_clip": float(os.environ.get("D2_DP_CLIP", str(args.dp_clip))),
            "dp_noise_multiplier": float(
                os.environ.get("D2_DP_NOISE_MULTIPLIER", str(args.dp_noise_multiplier))
            ),
            "dp_seed": int(os.environ.get("D2_DP_SEED", str(args.dp_seed))),
            # Threshold selection
            "tau_mode": args.tau_mode,
            "target_fpr": args.target_fpr,
            # Personalization
            "personalization_epochs": args.personalization_epochs,
        },
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
