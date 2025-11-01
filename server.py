import argparse
import os
import random
import time

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from logging_utils import configure_logging, get_logger
from robust_aggregation import (
    AggregationMethod,
    aggregate_weighted_mean,
    aggregate_weights,
)
from server_metrics import (
    AggregationTimer,
    ServerMetricsLogger,
    calculate_robustness_metrics,
)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    configure_logging()
    logger = get_logger("server")
    parser = argparse.ArgumentParser(description="Flower server for Federated IDS demo")
    parser.add_argument("--rounds", type=int, default=2, help="Number of FL rounds")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="fedavg",
        choices=["fedavg", "median", "krum", "bulyan"],
        help="Aggregation rule (FedAvg uses sample-weighted mean; median/krum/bulyan implemented, experimental)",
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
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory for metrics logging",
    )
    parser.add_argument(
        "--secure_aggregation",
        action="store_true",
        help="Enable secure aggregation mode (stub; placeholder for crypto/masking)",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=2,
        help="Minimum number of clients for fit rounds",
    )
    parser.add_argument(
        "--min_eval_clients",
        type=int,
        default=2,
        help="Minimum number of clients for evaluation rounds",
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required",
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=1.0,
        help="Fraction of clients participating in fit per round",
    )
    parser.add_argument(
        "--fraction_eval",
        type=float,
        default=1.0,
        help="Fraction of clients participating in evaluation per round",
    )
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.0,
        help="FedProx proximal term coefficient (mu) for heterogeneity mitigation",
    )
    args = parser.parse_args()

    seed = int(os.environ.get("SEED", "42"))
    _set_global_seed(seed)

    agg_method = AggregationMethod.from_string(args.aggregation)
    logger.info(
        "server_config",
        extra={
            "aggregation": agg_method.value,
        },
    )
    secure_agg_enabled = bool(args.secure_aggregation or os.environ.get("D2_SECURE_AGG", "0").lower() not in ("0", "false", "no", ""))
    logger.info(
        "secure_aggregation_mode",
        extra={"enabled": bool(secure_agg_enabled), "note": "stub"},
    )

    # Initialize metrics logging
    metrics_path = os.path.join(args.logdir, "metrics.csv")
    metrics_logger = ServerMetricsLogger(metrics_path)
    agg_timer = AggregationTimer()
    logger.info("metrics_init", extra={"metrics_path": metrics_path})

    class RobustStrategy(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.round_start_time: float | None = None

        def configure_fit(self, server_round: int, parameters, client_manager):
            # Track round start time
            self.round_start_time = time.perf_counter()
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
        ) -> tuple[Parameters | None, dict[str, Scalar]]:
            if len(results) == 0:
                return None, {}

            # Convert client parameters to numpy lists per client
            client_weights: list[list[np.ndarray]] = []
            sample_counts: list[int] = []
            for _, fit_res in results:
                nds = parameters_to_ndarrays(fit_res.parameters)
                client_weights.append(nds)
                sample_counts.append(int(fit_res.num_examples))

            # Time the aggregation
            with agg_timer.time_aggregation():
                f_arg = args.byzantine_f if args.byzantine_f >= 0 else None
                if agg_method == AggregationMethod.FED_AVG:
                    aggregated = aggregate_weighted_mean(client_weights, sample_counts)
                else:
                    aggregated = aggregate_weights(client_weights, agg_method, byzantine_f=f_arg)

            # Calculate metrics
            t_aggregate_ms = agg_timer.get_last_aggregation_time_ms()
            t_round_ms = None
            if self.round_start_time is not None:
                t_round_ms = (time.perf_counter() - self.round_start_time) * 1000.0

            # For research metrics, estimate benign mean as simple average (excluding outliers)
            # This is a simplified approach for demo purposes
            benign_mean = self._estimate_benign_mean(client_weights)
            robustness_metrics = calculate_robustness_metrics(client_weights, benign_mean, aggregated)

            # Compute pairwise dispersion metrics
            def _flatten(update: list[np.ndarray]) -> np.ndarray:
                return np.concatenate([u.reshape(-1) for u in update])

            pairwise_cos = []
            l2_disp = []
            try:
                flats = [_flatten(u) for u in client_weights]
                if len(flats) >= 2:
                    import numpy as _np

                    # pairwise cosine
                    norms = [_np.linalg.norm(v) for v in flats]
                    for i in range(len(flats)):
                        for j in range(i + 1, len(flats)):
                            denom = max(norms[i] * norms[j], 1e-12)
                            pairwise_cos.append(float(_np.dot(flats[i], flats[j]) / denom))
                    # l2 dispersion: distance to mean
                    mean_vec = sum(flats) / len(flats)
                    for v in flats:
                        l2_disp.append(float(_np.linalg.norm(v - mean_vec)))
            except Exception:
                pass

            # Log metrics
            metrics_logger.log_round_metrics(
                round_num=server_round,
                agg_method=agg_method,
                n_clients=len(client_weights),
                byzantine_f=f_arg,
                l2_to_benign_mean=robustness_metrics["l2_to_benign_mean"],
                cos_to_benign_mean=robustness_metrics["cos_to_benign_mean"],
                coord_median_agree_pct=robustness_metrics["coord_median_agree_pct"],
                update_norm_mean=robustness_metrics["update_norm_mean"],
                update_norm_std=robustness_metrics["update_norm_std"],
                t_aggregate_ms=t_aggregate_ms,
                t_round_ms=t_round_ms,
                pairwise_cosine_mean=(float(sum(pairwise_cos) / len(pairwise_cos)) if pairwise_cos else None),
                pairwise_cosine_std=(float(_np.array(pairwise_cos).std()) if pairwise_cos else None),
                l2_dispersion_mean=(float(sum(l2_disp) / len(l2_disp)) if l2_disp else None),
                l2_dispersion_std=(float(_np.array(l2_disp).std()) if l2_disp else None),
            )

            parameters = ndarrays_to_parameters(aggregated)
            metrics: dict[str, Scalar] = {}
            return parameters, metrics

        def _estimate_benign_mean(self, client_weights: list[list[np.ndarray]]) -> list[np.ndarray]:
            """Estimate benign mean by using a simple average (FedAvg).

            This provides a stable, independent reference point to compare
            robust aggregation methods against. For FedAvg itself, we use
            the same computation but avoid circular reference.
            """
            if not client_weights:
                return []

            # Use simple unweighted average as the reference for what a 'benign'
            # model should look like. This avoids circular reference when the
            # aggregation method is FedAvg itself.
            num_clients = len(client_weights)
            if num_clients == 0:
                return []

            # Compute element-wise average across all clients
            num_layers = len(client_weights[0])
            benign_mean = []

            for layer_idx in range(num_layers):
                # Stack all client layers for this layer index
                layer_stack = np.stack([client[layer_idx] for client in client_weights], axis=0)
                # Compute mean across clients (axis=0)
                layer_mean = np.mean(layer_stack, axis=0)
                benign_mean.append(layer_mean)

            return benign_mean

    def _on_fit_config(rnd: int):
        # Pass through seed and default hyperparameters; rounds can adjust epochs if desired
        return {"epoch": 1, "lr": 0.01, "seed": seed, "fedprox_mu": args.fedprox_mu}

    def _eval_metrics_agg(results: list[tuple[int, dict]]):
        # results: list of (num_examples, metrics)
        if not results:
            return {}
        total = sum(n for n, _ in results)
        mean_accuracy = sum(n * m.get("accuracy", 0.0) for n, m in results) / max(total, 1)
        return {"accuracy": float(mean_accuracy)}

    strategy = RobustStrategy(
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=_on_fit_config,
        evaluate_metrics_aggregation_fn=_eval_metrics_agg,
        fraction_evaluate=args.fraction_eval,
    )

    logger.info(
        "strategy_config",
        extra={
            "min_fit_clients": args.min_fit_clients,
            "min_eval_clients": args.min_eval_clients,
            "min_available_clients": args.min_available_clients,
            "fraction_fit": args.fraction_fit,
            "fraction_eval": args.fraction_eval,
        },
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
