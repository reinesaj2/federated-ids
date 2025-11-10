from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size == 0 or y_pred.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.float64)

    valid_mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    if not np.all(valid_mask):
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1.0)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    return cm


def render_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Render confusion matrix as heatmap and save to file.
    """
    import matplotlib.pyplot as plt  # Imported lazily to avoid heavy dependency for call sites that only compute matrices
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    if normalize and not np.allclose(cm.sum(axis=1), 1.0):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_normalized = cm / row_sums
    else:
        cm_normalized = cm

    vmin = 0.0
    vmax = 1.0 if normalize else cm.max()

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage" if normalize else "Count"},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_confusion_matrices(cms: List[np.ndarray]) -> np.ndarray:
    """
    Aggregate multiple confusion matrices by summing.
    """
    if len(cms) == 0:
        raise ValueError("Cannot aggregate empty list of confusion matrices")

    first_shape = cms[0].shape
    for i, cm in enumerate(cms[1:], start=1):
        if cm.shape != first_shape:
            raise ValueError(f"Confusion matrix shape mismatch: cms[0].shape={first_shape}, cms[{i}].shape={cm.shape}")
    return np.sum(cms, axis=0)
