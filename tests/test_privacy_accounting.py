"""
Unit tests for differential privacy accounting.

Following TDD approach: write tests first, then implement.
"""

from privacy_accounting import (
    DPAccountant,
    compute_epsilon,
    compute_noise_multiplier_for_target_epsilon,
)


class TestComputeEpsilon:
    """Test epsilon computation from noise multiplier."""

    def test_zero_noise_gives_infinite_epsilon(self):
        """Zero noise provides no privacy (infinite epsilon)."""
        epsilon = compute_epsilon(noise_multiplier=0.0, delta=1e-5, num_steps=1, sample_rate=1.0)
        assert epsilon == float("inf")

    def test_high_noise_gives_low_epsilon(self):
        """High noise should give strong privacy (low epsilon)."""
        epsilon = compute_epsilon(noise_multiplier=10.0, delta=1e-5, num_steps=1, sample_rate=1.0)
        assert epsilon < 1.0
        assert epsilon > 0.0

    def test_epsilon_increases_with_more_steps(self):
        """Privacy budget degrades with more training rounds."""
        eps_1_step = compute_epsilon(noise_multiplier=1.0, delta=1e-5, num_steps=1, sample_rate=1.0)
        eps_10_steps = compute_epsilon(noise_multiplier=1.0, delta=1e-5, num_steps=10, sample_rate=1.0)
        assert eps_10_steps > eps_1_step

    def test_known_value_validation(self):
        """Validate against known DP-SGD values from literature.

        From Abadi et al. 2016, sigma=1.0, delta=1e-5, T=1 step
        should give epsilon approximately 1.0-2.0 (order of magnitude check).
        """
        epsilon = compute_epsilon(noise_multiplier=1.0, delta=1e-5, num_steps=1, sample_rate=1.0)
        # Sanity check: epsilon should be in reasonable range
        assert 0.5 < epsilon < 5.0


class TestComputeNoiseMuiltiplier:
    """Test noise multiplier computation from target epsilon."""

    def test_target_epsilon_1_0(self):
        """Compute sigma for target epsilon=1.0."""
        sigma = compute_noise_multiplier_for_target_epsilon(target_epsilon=1.0, delta=1e-5, num_steps=10)
        assert sigma > 0.0
        # Verify: computing epsilon with this sigma should give ~1.0
        eps = compute_epsilon(noise_multiplier=sigma, delta=1e-5, num_steps=10, sample_rate=1.0)
        assert abs(eps - 1.0) < 0.5  # Allow some tolerance

    def test_target_epsilon_5_0(self):
        """Compute sigma for target epsilon=5.0."""
        sigma = compute_noise_multiplier_for_target_epsilon(target_epsilon=5.0, delta=1e-5, num_steps=20)
        assert sigma > 0.0
        # Verify: computing epsilon with this sigma should give ~5.0
        eps = compute_epsilon(noise_multiplier=sigma, delta=1e-5, num_steps=20, sample_rate=1.0)
        assert abs(eps - 5.0) < 1.0


class TestDPAccountant:
    """Test stateful privacy accountant for tracking across rounds."""

    def test_single_step(self):
        """Single training step should consume privacy budget."""
        accountant = DPAccountant(delta=1e-5)
        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        epsilon = accountant.get_epsilon()
        assert epsilon > 0.0
        assert epsilon < float("inf")

    def test_multiple_steps_accumulate(self):
        """Multiple steps should accumulate privacy loss."""
        accountant = DPAccountant(delta=1e-5)
        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        eps_1 = accountant.get_epsilon()

        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        eps_2 = accountant.get_epsilon()

        assert eps_2 > eps_1

    def test_reset_clears_history(self):
        """Reset should clear accumulated privacy loss."""
        accountant = DPAccountant(delta=1e-5)
        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        eps_before = accountant.get_epsilon()
        assert eps_before > 0.0

        accountant.reset()
        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        eps_after = accountant.get_epsilon()

        # After reset, single step should have lower epsilon than 2 steps
        assert eps_after < eps_before

    def test_get_total_steps(self):
        """Should track total number of steps."""
        accountant = DPAccountant(delta=1e-5)
        assert accountant.get_total_steps() == 0

        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        assert accountant.get_total_steps() == 1

        accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        assert accountant.get_total_steps() == 2
