import argparse
import os
import random
from typing import Literal

import flwr as fl
import numpy as np

from robust_aggregation import AggregationMethod


AggregationChoice = Literal["fedavg", "median", "krum", "bulyan"]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower server for Federated IDS demo")
    parser.add_argument("--rounds", type=int, default=2, help="Number of FL rounds")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="fedavg",
        choices=["fedavg", "median", "krum", "bulyan"],
        help="Aggregation rule (baseline FedAvg; others are stubs)",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Host:port for the Flower server",
    )
    args = parser.parse_args()

    seed = int(os.environ.get("SEED", "42"))
    _set_global_seed(seed)

    agg_method = AggregationMethod.from_string(args.aggregation)
    print(f"[Server] Using aggregation method: {agg_method.value} (FedAvg baseline currently)")

    # Baseline FedAvg strategy; robust hooks are defined but not yet activated
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"epoch": 1, "lr": 0.01, "seed": seed},
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
