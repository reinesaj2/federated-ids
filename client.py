import argparse
import os
import random
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preprocessing import create_synthetic_classification_loaders


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


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lr: float) -> float:
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> None:
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
    parser = argparse.ArgumentParser(description="Flower client for Federated IDS demo (synthetic fallback)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, test_loader = create_synthetic_classification_loaders(
        num_samples=args.samples,
        num_features=args.features,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = SimpleNet(num_features=args.features, num_classes=2)

    client = TorchClient(model=model, train_loader=train_loader, test_loader=test_loader, device=device)

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
