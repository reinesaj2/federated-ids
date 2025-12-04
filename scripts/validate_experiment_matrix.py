#!/usr/bin/env python3
"""
Pre-flight validation for experiment matrix feasibility.

Validates Byzantine resilience constraints for robust aggregation algorithms:
- Bulyan: n >= 4f + 3
- Krum: n >= 2f + 3
- Median: n >= 2f + 1

Prevents mathematically impossible experiments from blocking nightly CI runs.
"""

import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExperimentConstraints:
    """Byzantine resilience requirements for robust aggregation."""

    aggregation: str
    n_clients: int
    adversary_fraction: float

    def validate(self) -> Tuple[bool, str]:
        """Validate this experiment configuration.

        Returns:
            (is_valid, reason): True if feasible, (False, error_msg) if impossible
        """
        f = int(self.n_clients * self.adversary_fraction)

        if self.aggregation == "bulyan":
            required = 4 * f + 3
            if self.n_clients < required:
                msg = f"Bulyan requires n >= 4f+3; " f"got n={self.n_clients}, f={f}, required={required}"
                return (False, msg)

        elif self.aggregation == "krum":
            required = 2 * f + 3
            if self.n_clients < required:
                msg = f"Krum requires n >= 2f+3; got n={self.n_clients}, f={f}, required={required}"
                return (False, msg)

        elif self.aggregation == "median":
            required = 2 * f + 1
            if self.n_clients < required:
                msg = f"Median requires n >= 2f+1; " f"got n={self.n_clients}, f={f}, required={required}"
                return (False, msg)

        elif self.aggregation == "fedavg":
            # FedAvg has no Byzantine constraint
            pass

        return True, "OK"


class ExperimentMatrixValidator:
    """Validates all experiments in the comparison matrix for feasibility."""

    def __init__(self, n_clients: int = 15):
        """Initialize validator with matrix parameters.

        Args:
            n_clients: Number of clients in federated system
        """
        self.n_clients = n_clients
        self.aggregations = ["fedavg", "krum", "bulyan", "median"]
        self.adversary_fractions = [0.0, 0.1, 0.2]

    def validate_all(self) -> Tuple[int, int, List[str]]:
        """Validate all experiments in matrix.

        Returns:
            (viable_count, impossible_count, impossible_configs): Summary and list
                of infeasible experiment configurations
        """
        viable = []
        impossible = []

        for agg in self.aggregations:
            for adv_frac in self.adversary_fractions:
                config = ExperimentConstraints(
                    aggregation=agg,
                    n_clients=self.n_clients,
                    adversary_fraction=adv_frac,
                )
                is_valid, reason = config.validate()

                config_name = f"comp_{agg}_alpha*_adv{int(adv_frac * 100)}_*"
                if is_valid:
                    viable.append(config_name)
                else:
                    impossible.append(f"SKIP (impossible): {config_name} - {reason}")

        return len(viable), len(impossible), impossible

    def print_summary(self) -> None:
        """Print validation summary to stdout."""
        viable, impossible, impossible_configs = self.validate_all()

        print("=" * 70)
        print("EXPERIMENT MATRIX VALIDATION")
        print("=" * 70)
        print(f"Configuration: {self.n_clients} clients")
        print(f"Adversary fractions: {self.adversary_fractions}")
        print(f"Aggregation methods: {self.aggregations}")
        print()

        if impossible_configs:
            print("IMPOSSIBLE EXPERIMENTS (will be skipped):")
            for config in impossible_configs:
                print(f"  {config}")
            print()

        print("SUMMARY:")
        print(f"  Viable experiments: {viable}")
        print(f"  Impossible: {impossible}")
        total = viable + impossible
        print(f"  Total in matrix: {total}")
        print()

        if impossible > 0:
            print("NOTE: Impossible experiments are filtered out before execution.")
            print("Target completion: {}/{} viable experiments".format(viable, viable))
        else:
            print("SUCCESS: All experiments in matrix are feasible!")

        print("=" * 70)


def main() -> int:
    """Main entry point."""
    validator = ExperimentMatrixValidator(n_clients=15)
    validator.print_summary()

    # Return success (we don't fail; we just inform)
    return 0


if __name__ == "__main__":
    sys.exit(main())
