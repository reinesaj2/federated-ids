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
import errno
import json
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Default baseline values for controlled experiments
DEFAULT_ALPHA_IID = 1.0  # Alpha=1.0 means IID (uniform Dirichlet)
DEFAULT_ALPHA_NON_IID = 0.5  # Moderate non-IID for multi-dimensional experiments
DEFAULT_AGGREGATION = "fedavg"  # Baseline aggregation
# Attack dimension: compare all robust aggregation methods
ATTACK_AGGREGATIONS = ["fedavg", "krum", "bulyan", "median"]


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
    fedprox_mu: float = 0.0
    dataset: str = "unsw"
    data_path: str = "data/unsw/UNSW_NB15_training-set.csv"

    @classmethod
    def with_dataset(cls, dataset: str, **kwargs):
        """Create config with dataset-specific defaults."""
        dataset_paths = {
            "unsw": "data/unsw/UNSW_NB15_training-set.csv",
            "cic": "data/cic/cic_ids2017_multiclass.csv",
        }
        if dataset not in dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(dataset_paths.keys())}")
        return cls(dataset=dataset, data_path=dataset_paths[dataset], **kwargs)

    def to_preset_name(self) -> str:
        """Generate unique preset name for this configuration."""
        parts = [
            f"comp_{self.aggregation}",
            f"alpha{self.alpha}",
            f"adv{int(self.adversary_fraction * 100)}",
            f"dp{int(self.dp_enabled)}",
            f"pers{self.personalization_epochs}",
            f"mu{self.fedprox_mu}",
            f"seed{self.seed}",
        ]
        if self.dataset != "unsw":
            parts.append(f"dataset{self.dataset.lower()}")
        return "_".join(parts)


@dataclass
class ComparisonMatrix:
    """Defines the full comparison experiment matrix.

    Expanded grids per Issue #44 acceptance criteria for thesis validation.
    """

    aggregation_methods: List[str] = field(default_factory=lambda: ["fedavg", "krum", "bulyan", "median"])
    alpha_values: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf')])
    adversary_fractions: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    dp_configs: List[Dict] = field(
        default_factory=lambda: [
            {"enabled": False, "noise": 0.0},
            {"enabled": True, "noise": 0.1},
            {"enabled": True, "noise": 0.3},
            {"enabled": True, "noise": 0.5},
            {"enabled": True, "noise": 0.7},
            {"enabled": True, "noise": 1.0},
            {"enabled": True, "noise": 1.5},
            {"enabled": True, "noise": 2.0},
        ]
    )
    personalization_epochs: List[int] = field(default_factory=lambda: [0, 3, 5])
    fedprox_mu_values: List[float] = field(default_factory=lambda: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2])
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    num_clients: int = 6
    num_rounds: int = 20
    dataset: str = "unsw"
    data_path: Optional[str] = None

    def _base_config(self, seed: int) -> Dict:
        """Get baseline config with fixed parameters for controlled experiments."""
        base = {
            "aggregation": DEFAULT_AGGREGATION,
            "alpha": DEFAULT_ALPHA_IID,
            "adversary_fraction": 0.0,
            "dp_enabled": False,
            "dp_noise_multiplier": 0.0,
            "personalization_epochs": 0,
            "fedprox_mu": 0.0,
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "seed": seed,
            "dataset": self.dataset,
        }
        if self.data_path:
            base["data_path"] = self.data_path
        return base

    def _create_config(self, base: Dict, **overrides) -> ExperimentConfig:
        """Create config with overrides applied to base."""
        return ExperimentConfig(**{**base, **overrides})

    def _generate_aggregation_configs(self) -> List[ExperimentConfig]:
        """Generate configs varying only aggregation method."""
        configs = []
        for agg in self.aggregation_methods:
            for seed in self.seeds:
                configs.append(self._create_config(self._base_config(seed), aggregation=agg))
        return configs

    def _generate_heterogeneity_configs(self) -> List[ExperimentConfig]:
        """Generate configs varying only alpha (data heterogeneity)."""
        configs = []
        for alpha in self.alpha_values:
            for seed in self.seeds:
                configs.append(self._create_config(self._base_config(seed), alpha=alpha))
        return configs

    def _generate_heterogeneity_fedprox_configs(self) -> List[ExperimentConfig]:
        """Generate FedProx configs for heterogeneity comparison.

        Tests FedProx algorithm across different alpha values (data heterogeneity)
        and mu values (proximal term strength) to evaluate heterogeneity mitigation.
        """
        configs = []
        for alpha in self.alpha_values:
            for mu in self.fedprox_mu_values:
                for seed in self.seeds:
                    configs.append(self._create_config(self._base_config(seed), alpha=alpha, fedprox_mu=mu))
        return configs

    def _generate_attack_configs(self) -> List[ExperimentConfig]:
        """Generate configs for attack resilience comparison.

        Uses all robust aggregation methods including Bulyan.
        Uses alpha=0.5 for moderate non-IID setting.
        Uses num_clients=11 to meet Bulyan's n >= 4f + 3 requirement
        (allows f=2 Byzantine tolerance: 11 >= 4*2 + 3).
        """
        configs = []
        for agg in ATTACK_AGGREGATIONS:
            for adv_frac in self.adversary_fractions:
                for seed in self.seeds:
                    configs.append(
                        self._create_config(
                            self._base_config(seed),
                            aggregation=agg,
                            alpha=DEFAULT_ALPHA_NON_IID,
                            adversary_fraction=adv_frac,
                            num_clients=11,  # Bulyan requires n >= 4f + 3
                        )
                    )
        return configs

    def _generate_privacy_configs(self) -> List[ExperimentConfig]:
        """Generate configs varying DP settings.

        Uses alpha=0.5 for moderate non-IID to test privacy impact under
        realistic heterogeneous conditions.
        """
        configs = []
        for dp_config in self.dp_configs:
            for seed in self.seeds:
                configs.append(
                    self._create_config(
                        self._base_config(seed),
                        alpha=DEFAULT_ALPHA_NON_IID,
                        dp_enabled=dp_config["enabled"],
                        dp_noise_multiplier=dp_config["noise"],
                    )
                )
        return configs

    def _generate_personalization_configs(self) -> List[ExperimentConfig]:
        """Generate configs varying personalization epochs.

        Uses alpha=0.5 to test personalization benefit under non-IID conditions
        where it is expected to provide the most value.
        """
        configs = []
        for pers_epochs in self.personalization_epochs:
            for seed in self.seeds:
                configs.append(
                    self._create_config(
                        self._base_config(seed),
                        alpha=DEFAULT_ALPHA_NON_IID,
                        personalization_epochs=pers_epochs,
                    )
                )
        return configs

    def _generate_full_factorial_configs(self) -> List[ExperimentConfig]:
        """Generate full factorial experiment matrix (WARNING: very large)."""
        configs = []
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

    def generate_configs(self, filter_dimension: Optional[str] = None) -> List[ExperimentConfig]:
        """Generate experiment configurations for specified dimension.

        Args:
            filter_dimension: Dimension to vary. Options:
                - 'aggregation': Compare aggregation methods
                - 'heterogeneity': Compare IID vs Non-IID
                - 'heterogeneity_fedprox': Compare FedProx across heterogeneity levels
                - 'attack': Compare attack resilience
                - 'privacy': Compare privacy-utility tradeoff
                - 'personalization': Compare personalization benefit
                - None: Full factorial (all combinations)

        Returns:
            List of experiment configurations
        """
        dimension_map = {
            "aggregation": self._generate_aggregation_configs,
            "heterogeneity": self._generate_heterogeneity_configs,
            "heterogeneity_fedprox": self._generate_heterogeneity_fedprox_configs,
            "attack": self._generate_attack_configs,
            "privacy": self._generate_privacy_configs,
            "personalization": self._generate_personalization_configs,
        }

        if filter_dimension is None:
            return self._generate_full_factorial_configs()

        generator = dimension_map.get(filter_dimension)
        if generator is None:
            raise ValueError(
                f"Invalid dimension: {filter_dimension}. " f"Must be one of {list(dimension_map.keys())} or None for full factorial."
            )

        return generator()


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available for use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except PermissionError:
            if port < 1024:
                return False
            return True
        except OSError as exc:
            if exc.errno == errno.EPERM and port >= 1024:
                return True
            return False


def find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


@contextmanager
def managed_subprocess(cmd: List[str], log_file: Path, cwd: Path, timeout: int = 600):
    """Context manager for subprocess with proper cleanup.

    Args:
        cmd: Command and arguments
        log_file: Path to log file for stdout/stderr
        cwd: Working directory
        timeout: Timeout in seconds for process wait

    Yields:
        subprocess.Popen object
    """
    proc = None
    try:
        with open(log_file, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
        yield proc
    finally:
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def run_federated_experiment(
    config: ExperimentConfig, base_dir: Path, port_start: int = 8080, server_timeout: int = 300, client_timeout: int = 900
) -> Dict:
    """Run a single federated learning experiment with proper error handling.

    Args:
        config: Experiment configuration
        base_dir: Base directory for project
        port_start: Starting port for server (will find next available)
        server_timeout: Timeout in seconds for server process (default: 300)
        client_timeout: Timeout in seconds for client processes (default: 900)

    Returns:
        Dictionary with experiment results and metadata

    Raises:
        RuntimeError: If experiment fails to complete
    """
    preset = config.to_preset_name()
    run_dir = base_dir / "runs" / preset
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config metadata
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Find available port
    port = find_available_port(port_start)

    # Calculate adversary client count
    num_adversaries = int(config.adversary_fraction * config.num_clients)

    # Build server command
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
        "--fedprox_mu",
        str(config.fedprox_mu),
    ]

    client_procs = []
    try:
        # Start server with managed subprocess
        with managed_subprocess(server_cmd, server_log, base_dir, timeout=server_timeout) as server_proc:
            # Wait for server startup with basic health check
            max_retries = 10
            for _ in range(max_retries):
                time.sleep(0.5)
                if is_port_available(port):
                    continue  # Port still available, server not ready
                break
            else:
                raise RuntimeError(f"Server failed to bind to port {port} after {max_retries * 0.5}s")

            # Start clients
            for client_id in range(config.num_clients):
                adversary_mode = "grad_ascent" if client_id < num_adversaries else "none"

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
                    proc = subprocess.Popen(client_cmd, stdout=log, stderr=subprocess.STDOUT, cwd=base_dir)
                    client_procs.append(proc)

            # Wait for all clients to complete with timeout
            for proc in client_procs:
                try:
                    proc.wait(timeout=client_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise RuntimeError("Client process timed out")

            # Server will complete after all clients finish
            server_exit_code = server_proc.returncode

    finally:
        # Ensure all client processes are terminated
        for proc in client_procs:
            if proc.poll() is None:  # Still running
                proc.kill()
                proc.wait()

    # Collect results
    metrics_file = run_dir / "metrics.csv"
    results = {
        "preset": preset,
        "config": asdict(config),
        "run_dir": str(run_dir),
        "metrics_exist": metrics_file.exists(),
        "server_exit_code": server_exit_code,
        "port": port,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run comparative analysis experiments")
    parser.add_argument(
        "--dimension",
        type=str,
        choices=[
            "aggregation",
            "heterogeneity",
            "heterogeneity_fedprox",
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
    parser.add_argument(
        "--server_timeout",
        type=int,
        default=300,
        help="Server process timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--client_timeout",
        type=int,
        default=900,
        help="Client process timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=6,
        help="Number of clients per experiment (default: 6)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=20,
        help="Number of FL rounds per experiment (default: 20)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["unsw", "cic"],
        default="unsw",
        help="Dataset to use (unsw=UNSW-NB15, cic=CIC-IDS2017)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Override default dataset path",
    )
    parser.add_argument(
        "--aggregation-methods",
        type=str,
        help="Comma-separated list of aggregation methods to evaluate.",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        help="Comma-separated list of alpha values to evaluate.",
    )
    parser.add_argument(
        "--fedprox-mu-values",
        type=str,
        help="Comma-separated list of FedProx mu values to evaluate.",
    )
    parser.add_argument(
        "--adversary-fractions",
        type=str,
        help="Comma-separated list of adversary fractions to evaluate.",
    )
    parser.add_argument(
        "--dp-noise-multipliers",
        type=str,
        help="Comma-separated list of DP noise multipliers to evaluate.",
    )
    parser.add_argument(
        "--personalization-epochs",
        type=str,
        help="Comma-separated list of personalization epochs to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of random seeds to evaluate.",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Zero-based split index when dividing experiment configs across jobs.",
    )
    parser.add_argument(
        "--split-total",
        type=int,
        default=1,
        help="Total number of splits when dividing experiment configs across jobs.",
    )

    args = parser.parse_args()

    base_dir = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment matrix with dataset configuration
    dataset_paths = {
        "unsw": "data/unsw/UNSW_NB15_training-set.csv",
        "cic": "data/cic/cic_ids2017_multiclass.csv",
    }
    data_path = args.data_path if args.data_path else dataset_paths[args.dataset]

    matrix = ComparisonMatrix(dataset=args.dataset, data_path=data_path)
    matrix.num_clients = args.num_clients
    matrix.num_rounds = args.num_rounds
    if args.aggregation_methods:
        matrix.aggregation_methods = [method.strip() for method in args.aggregation_methods.split(",") if method.strip()]
    if args.alpha_values:
        matrix.alpha_values = [
            float("inf") if value.strip().lower() in {"inf", "infinity"} else float(value.strip())
            for value in args.alpha_values.split(",")
            if value.strip()
        ]
    if args.fedprox_mu_values:
        matrix.fedprox_mu_values = [float(value.strip()) for value in args.fedprox_mu_values.split(",") if value.strip()]
    if args.adversary_fractions:
        matrix.adversary_fractions = [float(value.strip()) for value in args.adversary_fractions.split(",") if value.strip()]
    if args.dp_noise_multipliers:
        matrix.dp_configs = [
            {"enabled": float(value.strip()) > 0.0, "noise": float(value.strip())}
            for value in args.dp_noise_multipliers.split(",")
            if value.strip()
        ]
    if args.personalization_epochs:
        matrix.personalization_epochs = [int(value.strip()) for value in args.personalization_epochs.split(",") if value.strip()]
    if args.seeds:
        matrix.seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if args.split_total < 1:
        raise ValueError("--split-total must be >= 1")
    if args.split_index < 0 or args.split_index >= args.split_total:
        raise ValueError("--split-index must satisfy 0 <= split_index < split_total")

    configs = matrix.generate_configs(filter_dimension=None if args.dimension == "full" else args.dimension)
    if args.split_total > 1:
        configs = configs[args.split_index :: args.split_total]

    print(f"Generated {len(configs)} experiment configurations for dimension: {args.dimension}")
    print(f"Dataset: {args.dataset} ({data_path})")
    if args.split_total > 1:
        print(f"Split {args.split_index + 1}/{args.split_total}")

    if args.dry_run:
        for i, config in enumerate(configs):
            print(f"{i + 1}. {config.to_preset_name()}")
        return

    # Run experiments with progress monitoring
    results = []
    successful_experiments = 0
    failed_experiments = 0

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Running: {config.to_preset_name()}")
        print(f"  Progress: {successful_experiments} successful, {failed_experiments} failed")

        try:
            result = run_federated_experiment(config, base_dir, server_timeout=args.server_timeout, client_timeout=args.client_timeout)
            results.append(result)

            if result.get('metrics_exist', False):
                successful_experiments += 1
                print(f"  SUCCESS: Exit code {result['server_exit_code']}, metrics generated")
            else:
                failed_experiments += 1
                print(f"  WARNING: Exit code {result['server_exit_code']}, no metrics generated")

        except subprocess.TimeoutExpired as e:
            failed_experiments += 1
            print(f"  TIMEOUT: {e}")
            results.append({"preset": config.to_preset_name(), "error": f"Timeout: {e}"})
        except Exception as e:
            failed_experiments += 1
            print(f"  FAILED: {e}")
            results.append({"preset": config.to_preset_name(), "error": str(e)})

    print("\nEXPERIMENT SUMMARY:")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Successful: {successful_experiments}")
    print(f"  Failed: {failed_experiments}")
    print(f"  Success rate: {successful_experiments / len(configs) * 100:.1f}%")

    # Save experiment manifest
    manifest_name = f"experiment_manifest_{args.dimension}"
    if args.split_total > 1:
        manifest_name += f"_split{args.split_index + 1}of{args.split_total}"
    manifest_path = output_dir / f"{manifest_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {"dimension": args.dimension, "total_experiments": len(results), "results": results},
            f,
            indent=2,
        )

    print(f"\nExperiment manifest saved to: {manifest_path}")
    print(f"Total experiments run: {len(results)}")
    failed = sum(1 for r in results if "error" in r)
    if failed:
        print(f"Failed experiments: {failed}")


if __name__ == "__main__":
    main()
