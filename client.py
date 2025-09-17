import argparse
import os
import random
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preprocessing import (
    create_synthetic_classification_loaders,
    load_unsw_nb15,
    load_cic_ids2017,
    prepare_partitions_from_dataframe,
    numpy_to_loaders,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
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
    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return float(avg_loss), float(acc)


class TorchClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def get_parameters(self, config):  # type: ignore[override]
        return get_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        set_parameters(self.model, parameters)
        epochs = int(config.get("epoch", 1))
        lr = float(config.get("lr", 0.01))
        for _ in range(epochs):
            train_epoch(self.model, self.train_loader, self.device, lr)
        num_examples = len(self.train_loader.dataset)
        metrics = {}
        return get_parameters(self.model), num_examples, metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.device)
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
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.dataset == "synthetic":
        train_loader, test_loader = create_synthetic_classification_loaders(
            num_samples=args.samples,
            num_features=args.features,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model = SimpleNet(num_features=args.features, num_classes=2)
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

    client = TorchClient(
        model=model, train_loader=train_loader, test_loader=test_loader, device=device
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
