import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preprocessing import (
    fit_preprocessor_global,
    load_cic_ids2017,
    load_edge_iiotset,
    load_unsw_nb15,
    numpy_to_train_val_test_loaders,
)


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


def train(model: nn.Module, loader: DataLoader, device: torch.device, lr: float) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


def evaluate_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += float(loss.item())
            n += 1
            probs = torch.softmax(preds, dim=1).detach().cpu().numpy()
            probs_list.append(probs)
            labels_list.append(yb.detach().cpu().numpy())
    avg_loss = total_loss / max(n, 1)
    if probs_list:
        all_probs = np.concatenate(probs_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
    else:
        all_probs = np.zeros((0, 0), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int64)
    return avg_loss, all_probs, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized baseline training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["unsw", "cic", "edge-iiotset-quick", "edge-iiotset-nightly", "edge-iiotset-full"],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="./logs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset == "unsw":
        df, label_col, _ = load_unsw_nb15(args.data_path)
    elif args.dataset == "cic":
        df, label_col, _ = load_cic_ids2017(args.data_path)
    elif args.dataset.startswith("edge-iiotset"):
        df, label_col, _ = load_edge_iiotset(args.data_path, use_multiclass=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    pre, X_all, y_all = fit_preprocessor_global(df, label_col)
    num_features = X_all.shape[1]
    num_classes = int(len(np.unique(y_all)))

    train_loader, val_loader, test_loader = numpy_to_train_val_test_loaders(
        X_all, y_all, batch_size=args.batch_size, seed=args.seed, splits=(0.7, 0.15, 0.15)
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SimpleNet(num_features=num_features, num_classes=num_classes).to(device)

    for _ in range(args.epochs):
        train(model, train_loader, device, args.lr)

    # Evaluate on val and test
    from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

    val_loss, val_probs, val_labels = evaluate_probs(model, val_loader, device)
    test_loss, test_probs, test_labels = evaluate_probs(model, test_loader, device)

    preds_test = np.argmax(test_probs, axis=1)
    macro_f1 = float(f1_score(test_labels, preds_test, average="macro")) if test_probs.size > 0 else 0.0

    # Per-class support and F1
    labels_unique, counts = np.unique(test_labels, return_counts=True)
    support = counts / counts.sum() if counts.sum() > 0 else counts
    per_class_f1 = []
    for c in labels_unique:
        mask = test_labels == c
        f1_c = float(f1_score(test_labels, preds_test == c, labels=[c], average="macro")) if mask.sum() > 0 else 0.0
        per_class_f1.append((int(c), f1_c))

    os.makedirs(args.logdir, exist_ok=True)
    # Write class-level CSVs
    sup_path = os.path.join(args.logdir, "per_class_support.csv")
    with open(sup_path, "w") as f:
        f.write("label,support_fraction\n")
        for c, frac in zip(labels_unique, support, strict=False):
            f.write(f"{int(c)},{float(frac)}\n")

    f1_path = os.path.join(args.logdir, "per_class_f1.csv")
    with open(f1_path, "w") as f:
        f.write("label,f1\n")
        for c, f1c in per_class_f1:
            f.write(f"{c},{f1c}\n")

    rare_path = os.path.join(args.logdir, "rare_classes_f1.csv")
    with open(rare_path, "w") as f:
        f.write("label,f1\n")
        for c, frac in zip(labels_unique, support, strict=False):
            if float(frac) <= 0.05:
                f1c = next((v for k, v in per_class_f1 if k == int(c)), 0.0)
                f.write(f"{int(c)},{f1c}\n")

    # Binary BENIGN vs attack metrics
    benign_idx = 0 if num_classes >= 2 else 0
    attack_probs_val = 1.0 - val_probs[:, benign_idx]
    attack_probs_test = 1.0 - test_probs[:, benign_idx]
    y_val_bin = (val_labels != benign_idx).astype(int)
    y_test_bin = (test_labels != benign_idx).astype(int)
    pr_auc = float(average_precision_score(y_test_bin, attack_probs_test)) if attack_probs_test.size > 0 else 0.0
    precision, recall, thresholds = precision_recall_curve(y_val_bin, attack_probs_val)
    denom = np.maximum(precision + recall, 1e-12)
    f1_curve = 2 * precision * recall / denom
    best_idx = int(np.argmax(f1_curve))
    tau = float(thresholds[best_idx - 1]) if best_idx > 0 and best_idx - 1 < len(thresholds) else 0.5
    y_pred_attack = (attack_probs_test >= tau).astype(int)
    benign_mask = y_test_bin == 0
    fp = int(np.sum(y_pred_attack[benign_mask] == 1))
    tn = int(np.sum(y_pred_attack[benign_mask] == 0))
    fpr = float(fp / max(fp + tn, 1))

    os.makedirs(args.logdir, exist_ok=True)
    out_path = os.path.join(args.logdir, "centralized_metrics.csv")
    with open(out_path, "w") as f:
        f.write("macro_f1,pr_auc,fpr,threshold_tau\n")
        f.write(f"{macro_f1},{pr_auc},{fpr},{tau}\n")
    print(f"[Centralized] Wrote metrics to: {out_path}")


if __name__ == "__main__":
    main()
