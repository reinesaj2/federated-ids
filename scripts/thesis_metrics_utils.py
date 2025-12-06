from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from metric_validation import MetricValidator
from plot_metrics_utils import compute_confidence_interval


def compute_server_macro_f1_from_clients(run_dir: Path, round_num: int) -> Optional[float]:
    """Compute server-level macro-F1 by averaging client macro-F1 scores for a given round."""
    client_f1_scores = []

    for client_csv in run_dir.glob("client_*_metrics.csv"):
        try:
            client_df = pd.read_csv(client_csv)
            round_data = client_df[client_df["round"] == round_num]
            if not round_data.empty:
                f1_value = None
                if "macro_f1_after" in round_data.columns:
                    f1_value = round_data["macro_f1_after"].iloc[0]
                elif "macro_f1_argmax" in round_data.columns:
                    f1_value = round_data["macro_f1_argmax"].iloc[0]

                if f1_value is not None and not pd.isna(f1_value):
                    client_f1_scores.append(float(f1_value))
        except Exception:
            continue

    if not client_f1_scores:
        return None

    mean_f1 = float(np.mean(client_f1_scores))
    return float(np.clip(mean_f1, 0.0, 1.0))


def load_experiment_results(runs_dir: Path) -> pd.DataFrame:
    """Load all experiment results for a given dimension."""
    all_data = []
    patterns = ["comp_*", "d2_*"]

    for pattern in patterns:
        for run_dir in runs_dir.glob(pattern):
            config = {}
            config_file = run_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

            metrics_file = run_dir / "metrics.csv"
            if not metrics_file.exists():
                continue

            df = pd.read_csv(metrics_file)

            macro_f1_values = []
            for idx, row in df.iterrows():
                round_num = row.get("round", idx)
                macro_f1 = compute_server_macro_f1_from_clients(run_dir, round_num)
                clipped_macro_f1 = float(np.clip(macro_f1, 0.0, 1.0)) if macro_f1 is not None else np.nan
                macro_f1_values.append(clipped_macro_f1)

            df["macro_f1"] = macro_f1_values

            for key, value in config.items():
                df[key] = value

            if "aggregation" not in df.columns:
                df["aggregation"] = config.get("aggregation", "fedavg")
            if "seed" not in df.columns:
                df["seed"] = config.get("seed", 42)

            df["run_dir"] = str(run_dir)
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    if "macro_f1" in combined_df.columns:
        combined_df["macro_f1"] = pd.to_numeric(combined_df["macro_f1"], errors="coerce").clip(0.0, 1.0)

    validator = MetricValidator()
    warnings = validator.validate_plot_metrics(combined_df, "experiment_data")
    if warnings:
        for warning in warnings[:5]:
            print(f"Metric validation warning: {warning}")

    return combined_df


def compute_attack_resilience_stats(final_rounds: pd.DataFrame, available_methods: list[str]) -> pd.DataFrame:
    """
    Compute attack resilience summary statistics with macro-F1 clipping.

    Returns DataFrame with columns:
    aggregation, adversary_fraction, macro_f1_mean, ci_lower, ci_upper, n, degradation_pct
    """
    if "macro_f1" not in final_rounds.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for agg in available_methods:
        agg_data = final_rounds[final_rounds["aggregation"] == agg]
        benign_f1 = agg_data[agg_data["adversary_fraction"] == 0.0]["macro_f1"].dropna()
        benign_mean = float(benign_f1.mean()) if len(benign_f1) > 0 else 0.0

        for adv_frac in sorted(agg_data["adversary_fraction"].unique()):
            frac_data = agg_data[agg_data["adversary_fraction"] == adv_frac]["macro_f1"].dropna()
            if len(frac_data) == 0:
                continue

            if len(frac_data) >= 2:
                mean, ci_lower, ci_upper = compute_confidence_interval(frac_data.values)
            else:
                mean = float(frac_data.iloc[0])
                ci_lower = mean
                ci_upper = mean

            mean = float(np.clip(mean, 0.0, 1.0))
            ci_lower = float(np.clip(ci_lower, 0.0, 1.0))
            ci_upper = float(np.clip(ci_upper, 0.0, 1.0))

            degradation_pct = max(0.0, (benign_mean - mean) / benign_mean * 100) if benign_mean > 0 else 0.0

            rows.append(
                {
                    "aggregation": agg,
                    "adversary_fraction": adv_frac,
                    "macro_f1_mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n": len(frac_data),
                    "degradation_pct": degradation_pct,
                }
            )

    return pd.DataFrame(rows)
