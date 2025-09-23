import argparse
import os
import random
import time
from typing import Literal, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from robust_aggregation import (
    AggregationMethod,
    aggregate_weights,
    aggregate_weighted_mean,
)
from server_metrics import (
    ServerMetricsLogger,
    AggregationTimer,
    calculate_robustness_metrics,
)
from model_utils import save_global_model, save_final_model
from client import SimpleNet


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
        "--save_models",
        type=str,
        default="final",
        choices=["none", "final", "all"],
        help="Model saving strategy: none (no models), final (only final model), all (every round)",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=20,
        help="Number of input features for model architecture",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes for model architecture",
    )
    args = parser.parse_args()

    seed = int(os.environ.get("SEED", "42"))
    _set_global_seed(seed)

    agg_method = AggregationMethod.from_string(args.aggregation)
    print(f"[Server] Using aggregation method: {agg_method.value}")
    secure_agg_enabled = bool(
        args.secure_aggregation or os.environ.get("D2_SECURE_AGG", "0").lower() not in ("0", "false", "no", "")
    )
    print(f"[Server] Secure aggregation mode: {'ON' if secure_agg_enabled else 'OFF'} (stub)")

    # Initialize metrics logging
    metrics_path = os.path.join(args.logdir, "metrics.csv")
    metrics_logger = ServerMetricsLogger(metrics_path)
    agg_timer = AggregationTimer()
    print(f"[Server] Logging metrics to: {metrics_path}")

    class RobustStrategy(fl.server.strategy.FedAvg):
        def __init__(self, save_models: str, logdir: str, num_rounds: int,
                     num_features: int, num_classes: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.round_start_time: Optional[float] = None
            self.save_models = save_models
            self.logdir = logdir
            self.num_rounds = num_rounds
            self.num_features = num_features
            self.num_classes = num_classes

        def configure_fit(self, server_round: int, parameters, client_manager):
            # Track round start time
            self.round_start_time = time.perf_counter()
            return super().configure_fit(server_round, parameters, client_manager)

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

            # Time the aggregation
            with agg_timer.time_aggregation():
                f_arg = args.byzantine_f if args.byzantine_f >= 0 else None
                if agg_method == AggregationMethod.FED_AVG:
                    aggregated = aggregate_weighted_mean(client_weights, sample_counts)
                else:
                    aggregated = aggregate_weights(
                        client_weights, agg_method, byzantine_f=f_arg
                    )

            # Calculate metrics
            t_aggregate_ms = agg_timer.get_last_aggregation_time_ms()
            t_round_ms = None
            if self.round_start_time is not None:
                t_round_ms = (time.perf_counter() - self.round_start_time) * 1000.0

            # For research metrics, estimate benign mean as simple average (excluding outliers)
            # This is a simplified approach for demo purposes
            benign_mean = self._estimate_benign_mean(client_weights)
            robustness_metrics = calculate_robustness_metrics(
                client_weights, benign_mean, aggregated
            )

            # Compute pairwise dispersion metrics
            def _flatten(update: List[np.ndarray]) -> np.ndarray:
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
                round_num=rnd,
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

            # Save global model if enabled
            if self.save_models != "none":
                self._save_global_model_if_needed(aggregated, rnd)

            metrics = {}
            return parameters, metrics

        def _save_global_model_if_needed(self, aggregated_weights: List[np.ndarray], round_num: int) -> None:
            """Save global model based on save_models strategy."""
            should_save = False

            if self.save_models == "all":
                should_save = True
            elif self.save_models == "final" and round_num == self.num_rounds:
                should_save = True

            if not should_save:
                return

            try:
                # Create model and set weights
                model = SimpleNet(num_features=self.num_features, num_classes=self.num_classes)

                # Convert aggregated weights back to state dict format
                state_dict = model.state_dict()
                new_state_dict = {}
                for (name, _), param in zip(state_dict.items(), aggregated_weights):
                    new_state_dict[name] = torch.tensor(param)
                model.load_state_dict(new_state_dict)

                # Save the model
                if round_num == self.num_rounds:
                    # Save final model
                    save_final_model(
                        model,
                        self.logdir,
                        metadata={
                            'final_round': round_num,
                            'save_strategy': self.save_models,
                            'num_features': self.num_features,
                            'num_classes': self.num_classes
                        }
                    )
                    print(f"[Server] Saved final global model: {self.logdir}/final_global_model.pth")
                else:
                    # Save round model
                    save_global_model(
                        model,
                        self.logdir,
                        round_num,
                        metadata={
                            'save_strategy': self.save_models,
                            'num_features': self.num_features,
                            'num_classes': self.num_classes
                        }
                    )
                    print(f"[Server] Saved global model for round {round_num}: {self.logdir}/global_model_round_{round_num}.pth")

            except Exception as e:
                print(f"[Server] Warning: Failed to save global model for round {round_num}: {e}")

        def _estimate_benign_mean(
            self, client_weights: List[List[np.ndarray]]
        ) -> List[np.ndarray]:
            """Estimate benign mean by using simple average (placeholder for research)."""
            if not client_weights:
                return []

            # Simple approach: use median aggregation as benign estimate
            # This assumes majority of clients are benign
            from robust_aggregation import _median_aggregate

            return _median_aggregate(client_weights)

    def _on_fit_config(rnd: int):
        # Pass through seed and default hyperparameters; rounds can adjust epochs if desired
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
        save_models=args.save_models,
        logdir=args.logdir,
        num_rounds=args.rounds,
        num_features=args.num_features,
        num_classes=args.num_classes,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=_on_fit_config,
        evaluate_metrics_aggregation_fn=_eval_metrics_agg,
        fraction_evaluate=args.fraction_eval,
    )

    print(f"[Server] Strategy configuration:")
    print(f"  min_fit_clients: {args.min_fit_clients}")
    print(f"  min_eval_clients: {args.min_eval_clients}")
    print(f"  min_available_clients: {args.min_available_clients}")
    print(f"  fraction_fit: {args.fraction_fit}")
    print(f"  fraction_eval: {args.fraction_eval}")

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
