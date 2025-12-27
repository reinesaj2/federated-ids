# Centralized Baseline Training with PerDatasetEncoderNet

This document describes how to train a centralized baseline model using the same
architecture as federated experiments, enabling fair comparison for thesis
Success Metric 1.1.

## Problem Statement

The existing `scripts/train_centralized.py` uses `SimpleNet` (3-layer MLP with
7,648 parameters), while federated experiments use `PerDatasetEncoderNet`
(5-layer architecture with ~500K parameters for CIC-IDS2017). This architectural
mismatch invalidates direct performance comparisons.

## Solution: Modified Centralized Training Script

Preferred implementation: extend `scripts/train_centralized.py` with `--model encoder` so the centralized
baseline uses the same `PerDatasetEncoderNet` architecture as federated runs.

Create `scripts/train_centralized_encoder.py` with the same model architecture
used in federated training.

### Architecture Comparison

| Component         | SimpleNet      | PerDatasetEncoderNet (CIC) |
| ----------------- | -------------- | -------------------------- |
| Encoder layers    | None           | 768 -> 384 -> 192          |
| Latent projection | None           | 192 -> 256                 |
| Shared head       | 64 -> 32 -> 15 | 128 -> 64 -> 15            |
| BatchNorm         | No             | Yes                        |
| Dropout           | No             | 0.25                       |
| Total params      | ~7.6K          | ~500K                      |

### CIC-IDS2017 Architecture Details

```
Input (80 features)
    |
    v
[Encoder]
    Linear(80, 768) -> BatchNorm1d(768) -> ReLU -> Dropout(0.25)
    Linear(768, 384) -> BatchNorm1d(384) -> ReLU -> Dropout(0.25)
    Linear(384, 192) -> BatchNorm1d(192) -> ReLU -> Dropout(0.25)
    |
    v
[Latent Projection]
    Linear(192, 256) -> BatchNorm1d(256) -> ReLU
    |
    v
[Shared Classification Head]
    Linear(256, 128) -> ReLU -> Dropout(0.25)
    Linear(128, 64) -> ReLU -> Dropout(0.25)
    Linear(64, 15)
    |
    v
Output (15 classes)
```

## Implementation

### Option A: Create New Script

Create `scripts/train_centralized_encoder.py`:

```python
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
from models.per_dataset_encoder import (
    PerDatasetEncoderNet,
    get_default_encoder_config,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb)
            probs = torch.softmax(preds, dim=1).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(yb.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralized baseline with PerDatasetEncoderNet"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["unsw", "cic", "edge-iiotset-quick", "edge-iiotset-nightly", "edge-iiotset-full"],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="./logs/centralized_encoder")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    if args.dataset == "unsw":
        df, label_col, _ = load_unsw_nb15(args.data_path)
        dataset_key = "unsw"
    elif args.dataset == "cic":
        df, label_col, _ = load_cic_ids2017(args.data_path)
        dataset_key = "cic"
    elif args.dataset.startswith("edge-iiotset"):
        df, label_col, _ = load_edge_iiotset(args.data_path, use_multiclass=True)
        dataset_key = "edge"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Preprocess
    pre, X_all, y_all = fit_preprocessor_global(df, label_col)
    num_features = X_all.shape[1]
    num_classes = int(len(np.unique(y_all)))

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {X_all.shape[0]:,}")
    print(f"Features: {num_features}")
    print(f"Classes: {num_classes}")

    # Create data loaders
    train_loader, val_loader, test_loader = numpy_to_train_val_test_loaders(
        X_all, y_all, batch_size=args.batch_size, seed=args.seed, splits=(0.7, 0.15, 0.15)
    )

    # Create model with same architecture as federated
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = get_default_encoder_config(
        dataset_name=dataset_key,
        input_dim=num_features,
        num_classes=num_classes,
    )
    model = PerDatasetEncoderNet(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Device: {device}")

    # Training setup (matches federated client settings)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}")

    # Evaluation
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
    )

    test_probs, test_labels = evaluate(model, test_loader, device)
    preds_test = np.argmax(test_probs, axis=1)

    # Macro F1
    macro_f1 = float(f1_score(test_labels, preds_test, average="macro"))

    # Per-class F1
    per_class_f1 = f1_score(test_labels, preds_test, average=None)

    # Confusion matrix
    cm = confusion_matrix(test_labels, preds_test)

    # Binary metrics (BENIGN vs attack)
    benign_idx = 0
    attack_probs = 1.0 - test_probs[:, benign_idx]
    y_test_bin = (test_labels != benign_idx).astype(int)

    pr_auc = float(average_precision_score(y_test_bin, attack_probs))

    # Optimal threshold from validation set
    val_probs, val_labels = evaluate(model, val_loader, device)
    val_attack_probs = 1.0 - val_probs[:, benign_idx]
    y_val_bin = (val_labels != benign_idx).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_val_bin, val_attack_probs)
    f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1_curve))
    tau = float(thresholds[best_idx - 1]) if best_idx > 0 else 0.5

    # FPR at optimal threshold
    y_pred_attack = (attack_probs >= tau).astype(int)
    benign_mask = y_test_bin == 0
    fp = int(np.sum(y_pred_attack[benign_mask] == 1))
    tn = int(np.sum(y_pred_attack[benign_mask] == 0))
    fpr = float(fp / max(fp + tn, 1))

    # Save results
    os.makedirs(args.logdir, exist_ok=True)

    # Main metrics
    metrics_path = os.path.join(args.logdir, "centralized_encoder_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("macro_f1,pr_auc,fpr,threshold_tau,num_params\n")
        f.write(f"{macro_f1},{pr_auc},{fpr},{tau},{param_count}\n")

    # Per-class F1
    f1_path = os.path.join(args.logdir, "per_class_f1.csv")
    with open(f1_path, "w") as f:
        f.write("class,f1\n")
        for c, f1c in enumerate(per_class_f1):
            f.write(f"{c},{f1c}\n")

    # Confusion matrix
    cm_path = os.path.join(args.logdir, "confusion_matrix.csv")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    # Class distribution
    labels_unique, counts = np.unique(test_labels, return_counts=True)
    dist_path = os.path.join(args.logdir, "class_distribution.csv")
    with open(dist_path, "w") as f:
        f.write("class,count,fraction\n")
        total = counts.sum()
        for c, cnt in zip(labels_unique, counts):
            f.write(f"{c},{cnt},{cnt/total}\n")

    print(f"\n{'='*60}")
    print("CENTRALIZED BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"FPR:          {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"Threshold:    {tau:.4f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.logdir}/")


if __name__ == "__main__":
    main()
```

### Option B: Modify Existing Script (Implemented)

Add a `--model` flag and `--weight_decay` option to `scripts/train_centralized.py`:

```python
parser.add_argument(
    "--model",
    type=str,
    default="simple",
    choices=["simple", "encoder"],
    help="Model architecture: simple (3-layer MLP) or encoder (PerDatasetEncoderNet)",
)
parser.add_argument("--weight_decay", type=float, default=None)
```

Then conditionally create the model:

```python
if args.model == "encoder":
    from models.per_dataset_encoder import (
        PerDatasetEncoderNet,
        get_default_encoder_config,
    )
    config = get_default_encoder_config(
        dataset_name=dataset_key,
        input_dim=num_features,
        num_classes=num_classes,
    )
    model = PerDatasetEncoderNet(config).to(device)
    optimizer_cls = torch.optim.AdamW
    default_weight_decay = 1e-4
else:
    model = SimpleNet(num_features=num_features, num_classes=num_classes).to(device)
    optimizer_cls = torch.optim.Adam
    default_weight_decay = 0.0

weight_decay = default_weight_decay if args.weight_decay is None else args.weight_decay
```

## Usage

### Running the Centralized Baseline

```bash
cd /Users/abrahamreines/Documents/Thesis/federated-ids

python scripts/train_centralized.py \
  --dataset cic \
  --data_path data/cic/cic_ids2017_multiclass.csv \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 42 \
  --model encoder \
  --logdir ./logs/centralized_cic_baseline
```

### Cluster Execution (Slurm)

Use the dedicated Slurm script to run the centralized encoder baseline on the cluster:

```bash
cd /scratch/$USER/federated-ids
sbatch scripts/slurm/centralized_baseline_encoder.sbatch
```

### Expected Runtime

| Hardware          | Dataset Size | Epochs | Estimated Time |
| ----------------- | ------------ | ------ | -------------- |
| Apple M1/M2 (MPS) | 2.8M samples | 20     | 15-30 minutes  |
| NVIDIA GPU (CUDA) | 2.8M samples | 20     | 10-20 minutes  |
| CPU only          | 2.8M samples | 20     | 2-4 hours      |

### Output Files

```
logs/centralized_cic_baseline/
    centralized_encoder_metrics.csv   # macro_f1, pr_auc, fpr, tau
    per_class_f1.csv                  # F1 score for each of 15 classes
    confusion_matrix.csv              # 15x15 confusion matrix
    class_distribution.csv            # Test set class distribution
```

## Interpreting Results for Thesis

### Success Metric 1.1 Comparison

After training, compare federated results to centralized baseline:

```
Performance Ratio = Federated_Macro_F1 / Centralized_Macro_F1
```

Target: Ratio >= 0.95 (federated achieves at least 95% of centralized performance)

### Example Comparison Table

| Configuration        | Macro F1 | Ratio vs Centralized |
| -------------------- | -------- | -------------------- |
| Centralized baseline | 0.85     | 1.00 (reference)     |
| FedAvg (IID)         | 0.82     | 0.96                 |
| FedAvg (alpha=0.5)   | 0.78     | 0.92                 |
| Bulyan (alpha=0.1)   | 0.75     | 0.88                 |

### Per-Class Analysis

Compare per-class F1 between centralized and federated to identify which attack
types suffer most from distributed training:

```python
import pandas as pd

central = pd.read_csv("logs/centralized_cic_baseline/per_class_f1.csv")
federated = pd.read_csv("cluster-runs/dscic_comp_fedavg_.../per_class_f1.csv")

comparison = central.merge(federated, on="class", suffixes=("_central", "_fed"))
comparison["ratio"] = comparison["f1_fed"] / comparison["f1_central"]
print(comparison.sort_values("ratio"))
```

## Hyperparameter Alignment

Ensure these match between centralized and federated training:

| Parameter     | Centralized          | Federated (per client) |
| ------------- | -------------------- | ---------------------- |
| Optimizer     | AdamW                | AdamW                  |
| Learning rate | 1e-3                 | 1e-3                   |
| Weight decay  | 1e-4                 | 1e-4                   |
| Batch size    | 64                   | 64                     |
| Loss function | CrossEntropyLoss     | CrossEntropyLoss       |
| Architecture  | PerDatasetEncoderNet | PerDatasetEncoderNet   |

## Reproducibility

For reproducible results across runs:

1. Set all random seeds:

   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

2. Use the same data split seed as federated experiments

3. Document exact software versions:
   ```bash
   pip freeze > requirements_centralized.txt
   ```

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
--batch_size 32
```

### Slow Training on CPU

Use a smaller subset for initial validation:

```python
# Add to script for debugging
X_all = X_all[:100000]
y_all = y_all[:100000]
```

### NaN Loss

Check for inf values in data:

```python
print(f"Inf values: {np.isinf(X_all).sum()}")
print(f"NaN values: {np.isnan(X_all).sum()}")
```

## References

- Model architecture: `models/per_dataset_encoder.py`
- Data preprocessing: `data_preprocessing.py`
- Federated client training: `client.py` (train_epoch function)
- Default hyperparameters: `client.py` (DEFAULT_CLIENT_LR, DEFAULT_WEIGHT_DECAY)
