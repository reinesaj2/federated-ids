#!/usr/bin/env python3
"""
Robust IIoT Data Loader

Properly loads and merges:
- Server metrics (L2, dispersion, timing)
- Client metrics (F1 scores, personalization)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run_config(run_dir: Path) -> dict:
    """Parse experiment directory name."""
    name = run_dir.name
    config = {}

    # Dataset
    config["dataset"] = "edge-iiotset" if "edge-iiotset" in name else "unknown"

    # Aggregation
    for agg in ["fedavg", "krum", "bulyan", "median", "fedprox"]:
        if f"comp_{agg}" in name or f"_{agg}_" in name:
            config["aggregation"] = agg
            break
    else:
        config["aggregation"] = "unknown"

    # Alpha
    alpha_match = re.search(r"alpha([0-9.]+|inf)", name)
    if alpha_match:
        alpha_str = alpha_match.group(1)
        config["alpha"] = float("inf") if alpha_str == "inf" else float(alpha_str)

    # Adversary percentage
    adv_match = re.search(r"adv(\d+)", name)
    config["adv_pct"] = int(adv_match.group(1)) if adv_match else 0

    # Personalization
    pers_match = re.search(r"pers(\d+)", name)
    config["pers_epochs"] = int(pers_match.group(1)) if pers_match else 0

    # Seed
    seed_match = re.search(r"seed(\d+)", name)
    config["seed"] = int(seed_match.group(1)) if seed_match else 42

    config["run_id"] = run_dir.name

    return config


def _load_server_metrics_for_run(run_dir: Path) -> pd.DataFrame | None:
    """Load server metrics CSV for a single run."""
    server_csv = run_dir / "metrics.csv"
    if not server_csv.exists():
        return None

    try:
        return pd.read_csv(server_csv)
    except Exception as e:
        print(f"Warning: Failed to load {server_csv}: {e}")
        return None


def _load_client_metrics_for_run(run_dir: Path) -> dict[int, list[dict]]:
    """Load all client metrics CSVs for a run, grouped by round."""
    client_data_by_round: dict[int, list[dict]] = {}

    for client_csv in run_dir.glob("client_*_metrics.csv"):
        try:
            df_client = pd.read_csv(client_csv)

            for _, row in df_client.iterrows():
                round_num = int(row["round"])
                if round_num not in client_data_by_round:
                    client_data_by_round[round_num] = []

                # Use macro_f1_before as primary source (always populated)
                # Fall back to macro_f1_global for personalization experiments
                f1_global = row.get("macro_f1_before", np.nan)
                if pd.isna(f1_global):
                    f1_global = row.get("macro_f1_global", np.nan)

                f1_pers = row.get("macro_f1_personalized", row.get("macro_f1_after", np.nan))

                client_data_by_round[round_num].append(
                    {
                        "macro_f1_global": f1_global,
                        "macro_f1_personalized": f1_pers,
                    }
                )
        except Exception as e:
            print(f"Warning: Failed to load {client_csv}: {e}")
            continue

    return client_data_by_round


def _create_merged_records(
    config: dict,
    df_server: pd.DataFrame,
    client_data_by_round: dict[int, list[dict]],
) -> list[dict]:
    """Merge server and client metrics by round."""
    records = []

    for _, row in df_server.iterrows():
        round_num = int(row["round"])

        record = {**config}
        record.update(
            {
                "round": round_num,
                "l2_to_benign_mean": row.get("l2_to_benign_mean", np.nan),
                "l2_dispersion_mean": row.get("l2_dispersion_mean", np.nan),
                "t_aggregate_ms": row.get("t_aggregate_ms", np.nan),
            }
        )

        # Average client F1 scores for this round
        if round_num in client_data_by_round:
            client_round_data = client_data_by_round[round_num]

            f1_global_vals = [c["macro_f1_global"] for c in client_round_data if not np.isnan(c["macro_f1_global"])]
            f1_pers_vals = [c["macro_f1_personalized"] for c in client_round_data if not np.isnan(c["macro_f1_personalized"])]

            record["macro_f1_global"] = np.mean(f1_global_vals) if f1_global_vals else np.nan
            record["macro_f1_personalized"] = np.mean(f1_pers_vals) if f1_pers_vals else np.nan
        else:
            record["macro_f1_global"] = np.nan
            record["macro_f1_personalized"] = np.nan

        records.append(record)

    return records


def load_iiot_data(runs_dir: Path, run_pattern: str = "dsedge-iiotset-nightly*") -> pd.DataFrame:
    """
    Load all IIoT experiments into unified DataFrame.

    Args:
        runs_dir: Directory containing experiment runs
        run_pattern: Glob pattern for run directories

    Returns:
        DataFrame with columns:
        - Config: aggregation, alpha, adv_pct, pers_epochs, seed, run_id
        - Per-round: round
        - Server metrics: l2_to_benign_mean, l2_dispersion_mean, t_aggregate_ms
        - Client metrics (averaged): macro_f1_global, macro_f1_personalized
    """
    all_records = []

    for run_dir in runs_dir.glob(run_pattern):
        if not run_dir.is_dir():
            continue

        config = parse_run_config(run_dir)

        # Load server and client metrics
        df_server = _load_server_metrics_for_run(run_dir)
        if df_server is None:
            continue

        client_data_by_round = _load_client_metrics_for_run(run_dir)

        # Merge and collect records
        records = _create_merged_records(config, df_server, client_data_by_round)
        all_records.extend(records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Compute personalization gain
    df["personalization_gain"] = df["macro_f1_personalized"] - df["macro_f1_global"]

    return df


if __name__ == "__main__":
    # Test loading
    runs_dir = Path("runs")
    df = load_iiot_data(runs_dir)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df.head())

    print(f"\nAggregators: {sorted(df['aggregation'].unique())}")
    print(f"Alpha values: {sorted(df['alpha'].unique())}")
    print(f"Attack levels: {sorted(df['adv_pct'].unique())}")

    # Check F1 data availability
    print(f"\nF1 global non-null: {df['macro_f1_global'].notna().sum()} / {len(df)}")
    print(f"L2 distance non-null: {df['l2_to_benign_mean'].notna().sum()} / {len(df)}")
