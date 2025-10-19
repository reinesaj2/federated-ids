#!/usr/bin/env python3
"""
Analyze gradient norms for adversarial attacks with different clipping factors.

Helps determine optimal adversary_clip_factor for bounding unrealistic attack strength.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data_preprocessing import create_synthetic_classification_loaders  # noqa: E402
from client import SimpleNet, DEFAULT_CLIENT_LR, DEFAULT_WEIGHT_DECAY, create_adamw_optimizer  # noqa: E402


def measure_gradient_norms(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    mode: str = "normal",
    num_rounds: int = 5,
) -> dict:
    """
    Measure gradient norms for different adversarial modes.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        device: Device to run on
        mode: "normal", "label_flip", or "grad_ascent"
        num_rounds: Number of training rounds to measure

    Returns:
        Dictionary with gradient norm statistics
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_adamw_optimizer(model.parameters(), lr=DEFAULT_CLIENT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)

    norms_per_round = []

    for round_num in range(num_rounds):
        model.train()
        round_norms = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)

            if mode == "normal":
                loss = criterion(preds, yb)
            elif mode == "label_flip":
                # Intentionally wrong labels (rotate by +1)
                n_classes = max(int(preds.shape[1]), 2)
                wrong_labels = (yb + 1) % n_classes
                loss = criterion(preds, wrong_labels)
            elif mode == "grad_ascent":
                # Gradient ascent: maximize loss
                loss = -criterion(preds, yb)
            else:
                loss = criterion(preds, yb)

            loss.backward()

            # Measure gradient norm before any clipping
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += float((param.grad.data**2).sum())
            total_norm = float(np.sqrt(total_norm))
            round_norms.append(total_norm)

            optimizer.step()

        norms_per_round.append(round_norms)

    # Flatten all norms
    all_norms = np.concatenate(norms_per_round)

    return {
        "mode": mode,
        "count": len(all_norms),
        "mean": float(np.mean(all_norms)),
        "median": float(np.median(all_norms)),
        "std": float(np.std(all_norms)),
        "min": float(np.min(all_norms)),
        "max": float(np.max(all_norms)),
        "p25": float(np.percentile(all_norms, 25)),
        "p75": float(np.percentile(all_norms, 75)),
        "p95": float(np.percentile(all_norms, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze gradient norms for adversarial attacks")
    parser.add_argument("--dataset", type=str, default="unsw", choices=["unsw", "synthetic"])
    parser.add_argument("--num_clients", type=int, default=6)
    parser.add_argument("--num_rounds", type=int, default=5, help="Training rounds to measure")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_features", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 70)
    print("GRADIENT NORM ANALYSIS FOR ADVERSARIAL ATTACKS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Num rounds: {args.num_rounds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print()

    # Create training data
    train_loader, _ = create_synthetic_classification_loaders(
        num_samples=args.num_samples,
        num_features=args.num_features,
        batch_size=args.batch_size,
        seed=args.seed,
        num_classes=2,
    )

    # Create model
    model = SimpleNet(num_features=args.num_features, num_classes=2).to(device)

    # Measure gradient norms for each mode
    modes = ["normal", "label_flip", "grad_ascent"]
    results = {}

    for mode in modes:
        print(f"Measuring {mode} gradients...")
        stats = measure_gradient_norms(model, train_loader, device, mode, args.num_rounds)
        results[mode] = stats
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Median: {stats['median']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  P95: {stats['p95']:.3f}")
        print()

    # Analysis and recommendations
    print("=" * 70)
    print("CLIPPING FACTOR RECOMMENDATIONS")
    print("=" * 70)

    normal_mean = results["normal"]["mean"]
    gradascent_mean = results["grad_ascent"]["mean"]
    gradascent_p95 = results["grad_ascent"]["p95"]

    ratio = gradascent_mean / normal_mean if normal_mean > 0 else 1.0

    print(f"Normal gradient mean: {normal_mean:.3f}")
    print(f"Grad-ascent mean: {gradascent_mean:.3f}")
    print(f"Grad-ascent P95: {gradascent_p95:.3f}")
    print(f"Ratio (grad-ascent / normal): {ratio:.1f}x")
    print()

    # Recommend clipping factor based on P95 of grad-ascent divided by P95 of normal
    normal_p95 = results["normal"]["p95"]
    if normal_p95 > 0:
        clip_factor_for_p95 = gradascent_p95 / normal_p95
        print(f"Suggested clip_factor (P95-based): {clip_factor_for_p95:.1f}x")
        print(f"  This would clip grad-ascent at 100x * {clip_factor_for_p95:.1f} = {100 * clip_factor_for_p95:.0f}")
        print()

    # Conservative recommendations
    print("Recommended settings:")
    print("  Conservative (clip more aggressively):")
    print("    --adversary_clip_factor 2.0  (clips at 200)")
    print("  Moderate (balanced):")
    print("    --adversary_clip_factor 3.0  (clips at 300)")
    print("  Permissive (allows larger attacks):")
    print("    --adversary_clip_factor 5.0  (clips at 500)")
    print()


if __name__ == "__main__":
    main()
