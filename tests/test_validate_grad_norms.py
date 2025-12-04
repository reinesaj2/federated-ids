"""
Unit tests for gradient norm validation script.

Tests verify:
1. Valid gradient norms pass validation
2. Missing column detection
3. Invalid value detection (NaN, Inf, negative)
4. Range validation
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_grad_norms import validate_grad_norms


def test_valid_grad_norms():
    """Verify that valid gradient norms pass validation."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0],
                "round": [1, 2, 3],
                "grad_norm_l2": [1.5, 2.3, 1.8],
                "loss_after": [0.5, 0.4, 0.3],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert success
        assert len(errors) == 0


def test_missing_column():
    """Verify detection when grad_norm_l2 column is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame({"client_id": [0, 0], "round": [1, 2], "loss_after": [0.5, 0.4]})
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert not success
        assert len(errors) == 1
        assert "not found" in errors[0]


def test_all_nan_values():
    """Verify detection when all gradient norms are NaN."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0],
                "round": [1, 2, 3],
                "grad_norm_l2": [float("nan"), float("nan"), float("nan")],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert not success
        assert any("No gradient norm values found" in e for e in errors)


def test_infinite_values():
    """Verify detection of infinite gradient norms."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0],
                "round": [1, 2, 3],
                "grad_norm_l2": [1.5, float("inf"), 1.8],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert not success
        assert any("Infinite" in e for e in errors)


def test_negative_values():
    """Verify detection of negative gradient norms."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0],
                "round": [1, 2, 3],
                "grad_norm_l2": [1.5, -2.3, 1.8],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert not success
        assert any("Negative" in e for e in errors)


def test_all_zero_values():
    """Verify warning when all gradient norms are zero."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0],
                "round": [1, 2, 3],
                "grad_norm_l2": [0.0, 0.0, 0.0],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert not success
        assert any("zero" in e.lower() for e in errors)


def test_suspiciously_large_values():
    """Verify detection of unreasonably large gradient norms."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0],
                "round": [1, 2, 3],
                "grad_norm_l2": [50000.0, 60000.0, 55000.0],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert not success
        assert any("large" in e.lower() for e in errors)


def test_mixed_valid_and_nan():
    """Verify that some valid values pass even with some NaN values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {
                "client_id": [0, 0, 0, 0],
                "round": [1, 2, 3, 4],
                "grad_norm_l2": [1.5, float("nan"), 2.3, 1.8],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        # Should pass as long as we have some valid values
        assert success
        assert len(errors) == 0


def test_realistic_gradient_norms():
    """Verify realistic gradient norm values pass validation."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Realistic gradient norms from actual training
        df = pd.DataFrame(
            {
                "client_id": [0] * 10,
                "round": list(range(1, 11)),
                "grad_norm_l2": [5.2, 4.8, 4.1, 3.7, 3.2, 2.9, 2.6, 2.3, 2.1, 1.9],
            }
        )
        df.to_csv(f.name, index=False)

        success, errors = validate_grad_norms(Path(f.name))

        assert success
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
