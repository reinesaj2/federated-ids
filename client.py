import argparse
import os
import random
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

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
    model: nn.Module, loader: DataLoader, device: torch.device, lr: float
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    num_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
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
            for epoch in range(epochs):
                mode = str(self.runtime_config.get("adversary_mode", "none"))
                if mode == "grad_ascent":
                    # Perform gradient ascent by negating the loss
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        optimizer.zero_grad()
                        preds = self.model(xb)
                        loss = -criterion(preds, yb)
                        loss.backward()
                        optimizer.step()
                elif mode == "label_flip":
                    # Train on intentionally wrong labels: rotate class index by +1
                    self.model.train()
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    n_classes = max(int(self.data_stats.get("n_classes", 2)), 2)
                    for xb, yb in self.train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        with torch.no_grad():
                            flipped = (yb + 1) % n_classes
                        optimizer.zero_grad()
                        preds = self.model(xb)
                        loss = criterion(preds, flipped)
                        loss.backward()
                        optimizer.step()
                else:
                    train_epoch(self.model, self.train_loader, self.device, lr)
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
        except Exception:
            # Fail-open: if DP step errors, proceed with original weights_after
            pass
        weight_norm_after = calculate_weight_norms(weights_after)
        weight_update_norm = calculate_weight_update_norm(weights_before, weights_after)
        # Compute a simple gradient norm proxy: norm of (weights_after - weights_before) / lr
        grad_norm_l2 = None
        try:
            if lr > 0:
                scaled = [ (wa - wb) / lr for wb, wa in zip(weights_before, weights_after) ]
                grad_norm_l2 = calculate_weight_norms(scaled)
        except Exception:
            pass

        # Evaluate after training (optional)
        loss_after, acc_after = None, None
        macro_f1_after = None
        macro_f1_argmax = None
        benign_fpr_argmax = None
        f1_per_class_after_json = None
        fpr_after = None
        pr_auc_after = None
        threshold_tau = None
        f1_bin_tau = None
        benign_fpr_bin_tau = None
        tau_bin = None
        try:
            if len(self.test_loader.dataset) > 0:
                loss_after, acc_after, probs_after, labels_after = evaluate(self.model, self.test_loader, self.device)
                if probs_after.size > 0:
                    preds_after = np.argmax(probs_after, axis=1)
                    macro_f1_after = float(f1_score(labels_after, preds_after, average="macro"))
                    # Argmax metrics
                    macro_f1_argmax = macro_f1_after
                    benign_idx = 0
                    if np.sum(labels_after == benign_idx) > 0:
                        benign_recall = float(
                            np.sum((labels_after == benign_idx) & (preds_after == benign_idx))
                        ) / float(np.sum(labels_after == benign_idx))
                        benign_fpr_argmax = float(max(0.0, 1.0 - benign_recall))
                    # Per-class F1
                    num_classes = probs_after.shape[1]
                    f1s = []
                    for c in range(num_classes):
                        f1s.append(float(f1_score(labels_after, preds_after == c, labels=[c], average="macro")))
                    import json as _json
                    f1_per_class_after_json = _json.dumps({str(i): f for i, f in enumerate(f1s)})
                    # BENIGN is class 0 by construction in preprocessing
                    if num_classes >= 2:
                        benign_idx = 0
                        benign_probs = probs_after[:, benign_idx]
                        # Attack-vs-BENIGN probabilities (1 - P(benign))
                        attack_probs = 1.0 - benign_probs
                        # PR-AUC (average precision) on full test
                        y_true_bin_full = (labels_after != benign_idx).astype(int)
                        pr_auc_after = float(average_precision_score(y_true_bin_full, attack_probs))

                        # Select a single tau on a validation subset, reuse for full test logging
                        n = attack_probs.shape[0]
                        if n > 1:
                            rng = np.random.default_rng(int(os.environ.get("SEED", "42")) + self.round_num)
                            n_val = max(1, int(0.4 * n))
                            val_idx = rng.choice(n, size=n_val, replace=False)
                            y_true_bin_val = y_true_bin_full[val_idx]
                            attack_probs_val = attack_probs[val_idx]
                            # Compute PR curve on validation subset
                            precision, recall, thresholds = precision_recall_curve(y_true_bin_val, attack_probs_val)
                            denom = np.maximum(precision + recall, 1e-12)
                            f1_curve = 2 * precision * recall / denom
                            if thresholds.size > 0 and f1_curve.size > 0:
                                best_idx = int(np.argmax(f1_curve))
                                # precision_recall_curve returns thresholds of length len(precision)-1
                                tau_idx = max(0, min(best_idx - 1, thresholds.size - 1))
                                threshold_tau = float(thresholds[tau_idx])
                            else:
                                threshold_tau = 0.5
                        else:
                            threshold_tau = 0.5
                        tau_bin = threshold_tau

                        # Apply chosen tau to full test
                        y_pred_attack_full = (attack_probs >= threshold_tau).astype(int)
                        benign_mask_full = (labels_after == benign_idx)
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
            fpr_after=fpr_after,
            pr_auc_after=pr_auc_after,
            threshold_tau=threshold_tau,
            f1_bin_tau=f1_bin_tau,
            benign_fpr_bin_tau=benign_fpr_bin_tau,
            tau_bin=tau_bin,
            seed=int(os.environ.get("SEED", str(config.get("seed", 0)))) if isinstance(config, dict) else None,
            weight_norm_before=weight_norm_before,
            weight_norm_after=weight_norm_after,
            weight_update_norm=weight_update_norm,
            grad_norm_l2=grad_norm_l2,
            t_fit_ms=t_fit_ms,
            epochs_completed=epochs_completed,
            lr=lr,
            batch_size=batch_size,
        )

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
    client_metrics_path = os.path.join(args.logdir, f"client_{args.client_id}_metrics.csv")
    metrics_logger = ClientMetricsLogger(client_metrics_path, args.client_id)
    fit_timer = ClientFitTimer()
    print(f"[Client {args.client_id}] Logging metrics to: {client_metrics_path}")

    if args.dataset == "synthetic":
        train_loader, test_loader = create_synthetic_classification_loaders(
            num_samples=args.samples,
            num_features=args.features,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model = SimpleNet(num_features=args.features, num_classes=2)
        # Analyze data distribution for synthetic data
        synthetic_labels = np.random.randint(0, 2, size=args.samples)  # Approximate for metrics
        data_stats = analyze_data_distribution(synthetic_labels)
        label_hist_json = create_label_histogram_json(synthetic_labels)
        num_classes_global = 2
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
            print(f"[Client {args.client_id}] Warning: single-class shard detected; using global num_classes={num_classes_global}")
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
        print(f"[Client {args.client_id}] Model validation passed: "
              f"out_features={model_output_features}, num_classes_global={num_classes_global}")
        print(f"[Client {args.client_id}] Label histogram: {label_hist_json}")
    else:
        print(f"[Client {args.client_id}] Warning: Could not validate model output features")
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
            # Privacy/robustness toggles
            "secure_aggregation": bool(
                args.secure_aggregation or os.environ.get("D2_SECURE_AGG", "0").lower() not in ("0", "false", "no", "")
            ),
            "dp_enabled": bool(
                args.dp_enabled or os.environ.get("D2_DP_ENABLED", "0").lower() not in ("0", "false", "no", "")
            ),
            "dp_clip": float(os.environ.get("D2_DP_CLIP", str(args.dp_clip))),
            "dp_noise_multiplier": float(
                os.environ.get("D2_DP_NOISE_MULTIPLIER", str(args.dp_noise_multiplier))
            ),
            "dp_seed": int(os.environ.get("D2_DP_SEED", str(args.dp_seed))),
        },
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
