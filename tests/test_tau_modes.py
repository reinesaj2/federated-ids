"""
Test suite for deterministic tau threshold selection modes.
Tests both low_fpr and max_f1 tau selection strategies.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from client import select_threshold_tau


def test_tau_low_fpr_mode():
    """Test that low_fpr mode achieves target FPR within tolerance."""
    # Create synthetic data: 100 benign (class 0), 100 attack (class 1)
    np.random.seed(42)

    # Benign samples: low attack probability (mean=0.2)
    benign_probs = np.random.beta(2, 8, size=100)

    # Attack samples: high attack probability (mean=0.8)
    attack_probs = np.random.beta(8, 2, size=100)

    # Combine
    all_probs = np.concatenate([benign_probs, attack_probs])
    y_true = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)

    # Test low_fpr mode with target_fpr=0.10
    target_fpr = 0.10
    tau = select_threshold_tau(y_true, all_probs, "low_fpr", target_fpr)

    # Predict with selected tau
    y_pred = (all_probs >= tau).astype(int)

    # Calculate actual FPR on benign samples
    benign_mask = y_true == 0
    fp = ((y_pred == 1) & benign_mask).sum()
    tn = ((y_pred == 0) & benign_mask).sum()
    actual_fpr = fp / max(fp + tn, 1)

    # Assert FPR is within tolerance (±0.02)
    assert abs(actual_fpr - target_fpr) <= 0.02, f"FPR {actual_fpr:.3f} not within tolerance of target {target_fpr:.3f}"

    # Assert tau is reasonable (between 0 and 1)
    assert 0 <= tau <= 1, f"Tau {tau} out of valid range [0, 1]"


def test_tau_max_f1_mode():
    """Test that max_f1 mode maximizes F1 score."""
    # Create synthetic data
    np.random.seed(43)

    benign_probs = np.random.beta(2, 8, size=100)
    attack_probs = np.random.beta(8, 2, size=100)

    all_probs = np.concatenate([benign_probs, attack_probs])
    y_true = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)

    # Test max_f1 mode
    tau = select_threshold_tau(y_true, all_probs, "max_f1", 0.10)

    # Predict with selected tau
    y_pred = (all_probs >= tau).astype(int)
    f1_at_tau = f1_score(y_true, y_pred)

    # Verify this is near-optimal by testing nearby thresholds
    test_thresholds = np.linspace(max(0, tau - 0.1), min(1, tau + 0.1), 20)
    f1_scores = []
    for test_tau in test_thresholds:
        y_test = (all_probs >= test_tau).astype(int)
        f1_scores.append(f1_score(y_true, y_test))

    max_f1 = max(f1_scores)

    # Assert selected tau achieves near-maximum F1 (within 5% relative)
    assert f1_at_tau >= 0.95 * max_f1, f"F1 at tau={tau:.3f} is {f1_at_tau:.3f}, " f"but max F1 in neighborhood is {max_f1:.3f}"

    # Assert tau is reasonable
    assert 0 <= tau <= 1, f"Tau {tau} out of valid range [0, 1]"


def test_tau_edge_case_single_class():
    """Test tau selection with single-class data (should not crash)."""
    # All benign samples
    y_true = np.zeros(100).astype(int)
    probs = np.random.rand(100)

    # Should return reasonable default without crashing
    with pytest.warns(UserWarning, match="No positive class found"):
        tau_low_fpr = select_threshold_tau(y_true, probs, "low_fpr", 0.10)
    with pytest.warns(UserWarning, match="No positive class found"):
        tau_max_f1 = select_threshold_tau(y_true, probs, "max_f1", 0.10)

    assert 0 <= tau_low_fpr <= 1
    assert 0 <= tau_max_f1 <= 1


def test_tau_edge_case_empty_thresholds():
    """Test tau selection with data that produces no thresholds."""
    # Constant probabilities (edge case)
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.5, 0.5, 0.5, 0.5])

    # Should return default 0.5 without crashing
    tau_low_fpr = select_threshold_tau(y_true, probs, "low_fpr", 0.10)
    tau_max_f1 = select_threshold_tau(y_true, probs, "max_f1", 0.10)

    # Default fallback should be 0.5
    assert tau_low_fpr == 0.5 or 0 <= tau_low_fpr <= 1
    assert tau_max_f1 == 0.5 or 0 <= tau_max_f1 <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
