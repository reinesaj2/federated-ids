import argparse
import os
import random
from typing import Literal, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from robust_aggregation import (
    AggregationMethod,
    aggregate_weights,
    aggregate_weighted_mean,
)


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
        "--byzantine_f",
        type=int,
        default=-1,
        help="Assumed number of Byzantine clients for robust aggregation (negative to auto)",
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
    print(f"[Server] Using aggregation method: {agg_method.value}")

    class RobustStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):  # type: ignore[override]
            if len(results) == 0:
                return None, {}
            # Convert client parameters to numpy lists per client
            client_weights: List[List[np.ndarray]] = []
            sample_counts: List[int] = []
            for _, fit_res in results:
                nds = parameters_to_ndarrays(fit_res.parameters)
                client_weights.append(nds)
                sample_counts.append(int(fit_res.num_examples))
            # Aggregate
            f_arg = args.byzantine_f if args.byzantine_f >= 0 else None
            if agg_method == AggregationMethod.FED_AVG:
                aggregated = aggregate_weighted_mean(client_weights, sample_counts)
            else:
                aggregated = aggregate_weights(
                    client_weights, agg_method, byzantine_f=f_arg
                )
            parameters = ndarrays_to_parameters(aggregated)
            # Optionally aggregate metrics (e.g., mean)
            metrics = {}
            return parameters, metrics

    def _on_fit_config(rnd: int):
        return {"epoch": 1, "lr": 0.01, "seed": seed}

    def _eval_metrics_agg(results: List[Tuple[int, dict]]):
        # results: list of (num_examples, metrics)
        if not results:
            return {}
        total = sum(n for n, _ in results)
        mean_accuracy = sum(n * m.get("accuracy", 0.0) for n, m in results) / max(
            total, 1
        )
        return {"accuracy": float(mean_accuracy)}

    strategy = RobustStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=_on_fit_config,
        evaluate_metrics_aggregation_fn=_eval_metrics_agg,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
