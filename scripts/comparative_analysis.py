#!/usr/bin/env python3
"""
Comparative Analysis Framework for Federated IDS

Orchestrates systematic comparison experiments across multiple dimensions:
- Aggregation methods (FedAvg, Krum, Bulyan, Median)
- Data heterogeneity (IID vs Non-IID)
- Attack resilience (benign vs adversarial clients)
- Privacy impact (DP enabled/disabled)
- Personalization benefit (global vs personalized)

Generates reproducible results for thesis validation.
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single comparative experiment."""

    aggregation: str
    alpha: float
    adversary_fraction: float
    dp_enabled: bool
    dp_noise_multiplier: float
    personalization_epochs: int
    num_clients: int
    num_rounds: int
    seed: int
    dataset: str = "unsw"
    data_path: str = "data/unsw/unsw_nb15_sample.csv"

    def to_preset_name(self) -> str:
        """Generate unique preset name for this configuration."""
        parts = [
            f"comp_{self.aggregation}",
            f"alpha{self.alpha}",
            f"adv{int(self.adversary_fraction * 100)}",
            f"dp{int(self.dp_enabled)}",
            f"pers{self.personalization_epochs}",
            f"seed{self.seed}",
        ]
        return "_".join(parts)


@dataclass
class ComparisonMatrix:
    """Defines the full comparison experiment matrix."""

    aggregation_methods: List[str] = field(
        default_factory=lambda: ["fedavg", "krum", "bulyan", "median"]
    )
    alpha_values: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.1])
    adversary_fractions: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    dp_configs: List[Dict] = field(
        default_factory=lambda: [
            {"enabled": False, "noise": 0.0},
            {"enabled": True, "noise": 0.5},
            {"enabled": True, "noise": 1.0},
        ]
    )
    personalization_epochs: List[int] = field(default_factory=lambda: [0, 5])
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    num_clients: int = 6
    num_rounds: int = 20

    def generate_configs(
        self, filter_dimension: Optional[str] = None
    ) -> List[ExperimentConfig]:
        """Generate all experiment configurations.

        Args:
            filter_dimension: If specified, only vary this dimension (e.g., 'aggregation')
        """
        configs = []

        # Base config for filtered experiments
        if filter_dimension == "aggregation":
            # Only vary aggregation method, keep other params fixed
            for agg in self.aggregation_methods:
                for seed in self.seeds:
                    configs.append(
                        ExperimentConfig(
                            aggregation=agg,
                            alpha=1.0,
                            adversary_fraction=0.0,
                            dp_enabled=False,
                            dp_noise_multiplier=0.0,
                            personalization_epochs=0,
                            num_clients=self.num_clients,
                            num_rounds=self.num_rounds,
                            seed=seed,
                        )
                    )
        elif filter_dimension == "heterogeneity":
            # Only vary alpha, keep aggregation=fedavg
            for alpha in self.alpha_values:
                for seed in self.seeds:
                    configs.append(
                        ExperimentConfig(
                            aggregation="fedavg",
                            alpha=alpha,
                            adversary_fraction=0.0,
                            dp_enabled=False,
                            dp_noise_multiplier=0.0,
                            personalization_epochs=0,
                            num_clients=self.num_clients,
                            num_rounds=self.num_rounds,
                            seed=seed,
                        )
                    )
        elif filter_dimension == "attack":
            # Vary adversary fraction with robust aggregation
            for agg in ["fedavg", "krum", "median"]:
                for adv_frac in self.adversary_fractions:
                    for seed in self.seeds:
                        configs.append(
                            ExperimentConfig(
                                aggregation=agg,
                                alpha=0.5,
                                adversary_fraction=adv_frac,
                                dp_enabled=False,
                                dp_noise_multiplier=0.0,
                                personalization_epochs=0,
                                num_clients=self.num_clients,
                                num_rounds=self.num_rounds,
                                seed=seed,
                            )
                        )
        elif filter_dimension == "privacy":
            # Vary DP settings
            for dp_config in self.dp_configs:
                for seed in self.seeds:
                    configs.append(
                        ExperimentConfig(
                            aggregation="fedavg",
                            alpha=0.5,
                            adversary_fraction=0.0,
                            dp_enabled=dp_config["enabled"],
                            dp_noise_multiplier=dp_config["noise"],
                            personalization_epochs=0,
                            num_clients=self.num_clients,
                            num_rounds=self.num_rounds,
                            seed=seed,
                        )
                    )
        elif filter_dimension == "personalization":
            # Vary personalization epochs
            for pers_epochs in self.personalization_epochs:
                for seed in self.seeds:
                    configs.append(
                        ExperimentConfig(
                            aggregation="fedavg",
                            alpha=0.5,
                            adversary_fraction=0.0,
                            dp_enabled=False,
                            dp_noise_multiplier=0.0,
                            personalization_epochs=pers_epochs,
                            num_clients=self.num_clients,
                            num_rounds=self.num_rounds,
                            seed=seed,
                        )
                    )
        else:
            # Full factorial experiment (WARNING: very large)
            for agg in self.aggregation_methods:
                for alpha in self.alpha_values:
                    for adv_frac in self.adversary_fractions:
                        for dp_config in self.dp_configs:
                            for pers in self.personalization_epochs:
                                for seed in self.seeds:
                                    configs.append(
                                        ExperimentConfig(
                                            aggregation=agg,
                                            alpha=alpha,
                                            adversary_fraction=adv_frac,
                                            dp_enabled=dp_config["enabled"],
                                            dp_noise_multiplier=dp_config["noise"],
                                            personalization_epochs=pers,
                                            num_clients=self.num_clients,
                                            num_rounds=self.num_rounds,
                                            seed=seed,
                                        )
                                    )

        return configs


def run_federated_experiment(config: ExperimentConfig, base_dir: Path) -> Dict:
    """Run a single federated learning experiment.

    Returns:
        Dictionary with experiment results and metadata
    """
    preset = config.to_preset_name()
    run_dir = base_dir / "runs" / preset
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config metadata
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Calculate adversary client count
    num_adversaries = int(config.adversary_fraction * config.num_clients)

    # Start server
    port = 8080
    server_log = run_dir / "server.log"
    server_cmd = [
        "python",
        "server.py",
        "--rounds",
        str(config.num_rounds),
        "--aggregation",
        config.aggregation,
        "--server_address",
        f"localhost:{port}",
        "--logdir",
        str(run_dir),
        "--min_fit_clients",
        str(config.num_clients),
        "--min_eval_clients",
        str(config.num_clients),
        "--min_available_clients",
        str(config.num_clients),
    ]

    with open(server_log, "w") as log:
        server_proc = subprocess.Popen(
            server_cmd, stdout=log, stderr=subprocess.STDOUT, cwd=base_dir
        )

    # Wait for server startup
    time.sleep(3)

    # Start clients
    client_procs = []
    for client_id in range(config.num_clients):
        # Determine if this client is adversarial
        adversary_mode = (
            "grad_ascent" if client_id < num_adversaries else "none"
        )

        client_log = run_dir / f"client_{client_id}.log"
        client_cmd = [
            "python",
            "client.py",
            "--server_address",
            f"localhost:{port}",
            "--dataset",
            config.dataset,
            "--data_path",
            config.data_path,
            "--partition_strategy",
            "dirichlet" if config.alpha < 1.0 else "iid",
            "--alpha",
            str(config.alpha),
            "--num_clients",
            str(config.num_clients),
            "--client_id",
            str(client_id),
            "--seed",
            str(config.seed),
            "--local_epochs",
            "1",
            "--adversary_mode",
            adversary_mode,
            "--personalization_epochs",
            str(config.personalization_epochs),
            "--logdir",
            str(run_dir),
        ]

        if config.dp_enabled:
            client_cmd.extend(
                [
                    "--dp_enabled",
                    "--dp_noise_multiplier",
                    str(config.dp_noise_multiplier),
                ]
            )

        with open(client_log, "w") as log:
            proc = subprocess.Popen(
                client_cmd, stdout=log, stderr=subprocess.STDOUT, cwd=base_dir
            )
            client_procs.append(proc)

    # Wait for all clients to complete
    for proc in client_procs:
        proc.wait()

    # Wait for server to complete
    server_proc.wait()

    # Collect results
    metrics_file = run_dir / "metrics.csv"
    results = {
        "preset": preset,
        "config": config.__dict__,
        "run_dir": str(run_dir),
        "metrics_exist": metrics_file.exists(),
        "server_exit_code": server_proc.returncode,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run comparative analysis experiments"
    )
    parser.add_argument(
        "--dimension",
        type=str,
        choices=[
            "aggregation",
            "heterogeneity",
            "attack",
            "privacy",
            "personalization",
            "full",
        ],
        default="aggregation",
        help="Comparison dimension to explore",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparative_analysis",
        help="Directory for analysis results",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configs without running experiments",
    )

    args = parser.parse_args()

    base_dir = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment matrix
    matrix = ComparisonMatrix()
    configs = matrix.generate_configs(
        filter_dimension=None if args.dimension == "full" else args.dimension
    )

    print(f"Generated {len(configs)} experiment configurations for dimension: {args.dimension}")

    if args.dry_run:
        for i, config in enumerate(configs):
            print(f"{i+1}. {config.to_preset_name()}")
        return

    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.to_preset_name()}")
        result = run_federated_experiment(config, base_dir)
        results.append(result)
        print(f"  Completed with exit code: {result['server_exit_code']}")

    # Save experiment manifest
    manifest_path = output_dir / f"experiment_manifest_{args.dimension}.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {"dimension": args.dimension, "total_experiments": len(results), "results": results},
            f,
            indent=2,
        )

    print(f"\nExperiment manifest saved to: {manifest_path}")
    print(f"Total experiments run: {len(results)}")


if __name__ == "__main__":
    main()
