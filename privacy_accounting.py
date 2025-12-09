"""
Differential privacy accounting for federated learning.

Prefers Opacus RDP accounting when available; falls back to a lightweight analytic
approximation when Opacus (and its torch dependency) are not installed.

References:
- Mironov (2017): Rényi Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
- McMahan et al. (2017): Learning Differentially Private Language Models
"""

import importlib.util
import logging
import math
import os

logger = logging.getLogger(__name__)
_USE_OPACUS = os.getenv("FEDIDS_USE_OPACUS", "").lower() in {"1", "true", "yes"}


def _load_opacus():
    if not _USE_OPACUS:
        return None, None

    opacus_spec = importlib.util.find_spec("opacus")
    torch_spec = importlib.util.find_spec("torch")
    if opacus_spec is None or torch_spec is None:
        logger.warning("Opacus/torch not available; using analytic DP fallback")
        return None, None

    try:
        from opacus.accountants.rdp import RDPAccountant as _RDPAccountant
        from opacus.accountants.utils import get_noise_multiplier as _get_noise_multiplier
        return _RDPAccountant, _get_noise_multiplier
    except Exception as exc:  # pragma: no cover - defensive fallback path
        logger.warning("Opacus import failed (%s); using analytic DP fallback", exc)
        return None, None


RDPAccountant, get_noise_multiplier = _load_opacus()


def compute_epsilon(
    noise_multiplier: float,
    delta: float,
    num_steps: int,
    sample_rate: float = 1.0,
) -> float:
    """
    Compute epsilon privacy budget for given noise and steps.

    Args:
        noise_multiplier: Gaussian noise multiplier (sigma)
        delta: Target delta for (epsilon, delta)-DP
        num_steps: Number of training steps/rounds
        sample_rate: Sampling rate per step (1.0 for full batch)

    Returns:
        Epsilon privacy budget consumed

    Example:
        >>> epsilon = compute_epsilon(noise_multiplier=1.0, delta=1e-5, num_steps=10)
        >>> print(f"Privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)")
    """
    if noise_multiplier <= 0.0:
        return float("inf")

    if RDPAccountant is None:
        if delta <= 0.0 or delta >= 1.0:
            return float("inf")
        epsilon = sample_rate * math.sqrt(2 * num_steps * math.log(1 / delta)) / noise_multiplier
        logger.warning("Opacus not available; using analytic DP epsilon approximation")
        return float(epsilon)

    accountant = RDPAccountant()
    for _ in range(num_steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    return float(accountant.get_epsilon(delta=delta))


def compute_noise_multiplier_for_target_epsilon(
    target_epsilon: float,
    delta: float,
    num_steps: int,
    sample_rate: float = 1.0,
) -> float:
    """
    Compute required noise multiplier to achieve target epsilon.

    Uses binary search to find sigma that achieves target_epsilon within tolerance.

    Args:
        target_epsilon: Desired epsilon privacy budget
        delta: Target delta for (epsilon, delta)-DP
        num_steps: Number of training steps/rounds
        sample_rate: Sampling rate per step (1.0 for full batch)

    Returns:
        Noise multiplier (sigma) required to achieve target_epsilon

    Example:
        >>> sigma = compute_noise_multiplier_for_target_epsilon(
        ...     target_epsilon=5.0, delta=1e-5, num_steps=20
        ... )
        >>> print(f"Use --dp_noise_multiplier={sigma:.2f}")
    """
    if target_epsilon <= 0.0:
        return float("inf")

    if get_noise_multiplier is None:
        if delta <= 0.0 or delta >= 1.0:
            return float("inf")
        sigma = sample_rate * math.sqrt(2 * num_steps * math.log(1 / delta)) / target_epsilon
        logger.warning("Opacus not available; using analytic noise multiplier approximation")
        return float(sigma)

    # Use Opacus utility to compute noise multiplier; treat num_steps as epochs
    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=num_steps,
    )
    return float(sigma)


class DPAccountant:
    """
    Stateful privacy accountant for tracking epsilon across FL rounds.

    This class maintains privacy budget state across multiple federated learning rounds,
    allowing clients to track cumulative privacy loss.

    Example:
        >>> accountant = DPAccountant(delta=1e-5)
        >>> for round_num in range(10):
        ...     accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        ...     epsilon = accountant.get_epsilon()
        ...     print(f"Round {round_num}: ε={epsilon:.2f}")
    """

    def __init__(self, delta: float = 1e-5) -> None:
        """
        Initialize privacy accountant.

        Args:
            delta: Target delta for (epsilon, delta)-DP
        """
        if RDPAccountant is None:
            raise RuntimeError("Opacus not available; DPAccountant requires RDPAccountant from opacus (set FEDIDS_USE_OPACUS=1 when installed)")
        self.delta = delta
        self.accountant = RDPAccountant()
        self._total_steps = 0

    def step(self, noise_multiplier: float, sample_rate: float = 1.0) -> None:
        """
        Record a single training step with given noise.

        Args:
            noise_multiplier: Gaussian noise multiplier (sigma) for this step
            sample_rate: Sampling rate for this step
        """
        self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        self._total_steps += 1

    def get_epsilon(self) -> float:
        """
        Get current cumulative epsilon privacy budget.

        Returns:
            Epsilon value for (epsilon, delta)-DP
        """
        if self._total_steps == 0:
            return 0.0
        return float(self.accountant.get_epsilon(delta=self.delta))

    def get_total_steps(self) -> int:
        """
        Get total number of steps recorded.

        Returns:
            Number of training steps
        """
        return self._total_steps

    def reset(self) -> None:
        """
        Reset privacy accountant state.

        Clears all accumulated privacy loss. Useful for starting new experiments.
        """
        self.accountant = RDPAccountant()
        self._total_steps = 0

    def get_privacy_summary(self) -> dict[str, float | int]:
        """
        Get summary of current privacy state.

        Returns:
            Dictionary with epsilon, delta, and total_steps
        """
        return {
            "epsilon": self.get_epsilon(),
            "delta": self.delta,
            "total_steps": self._total_steps,
        }
