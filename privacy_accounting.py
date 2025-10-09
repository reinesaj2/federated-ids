"""
Differential privacy accounting for federated learning.

Implements epsilon-delta privacy budget tracking using Rényi Differential Privacy (RDP)
composition from Opacus library.

References:
- Mironov (2017): Rényi Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
- McMahan et al. (2017): Learning Differentially Private Language Models
"""

from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


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

    accountant = RDPAccountant()

    for _ in range(num_steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    epsilon = accountant.get_epsilon(delta=delta)
    return float(epsilon)


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
    # Use Opacus utility to compute noise multiplier
    # Note: get_noise_multiplier expects epochs, we treat num_steps as epochs
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

    def get_privacy_summary(self) -> dict[str, float]:
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
