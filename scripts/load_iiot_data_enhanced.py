#!/usr/bin/env python3
"""
Enhanced IIoT Data Loader with Per-Class F1 Extraction

Extends load_iiot_data to include:
- Per-class F1 scores for minority attack analysis
- Holdout set metrics
- Support for statistical comparisons
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run_config(run_dir: Path) -> dict:
    """Parse experiment directory name."""
    name = run_dir.name
    config = {}

    config["dataset"] = "edge-iiotset" if "edge-iiotset" in name else "unknown"

    for agg in ["fedavg", "krum", "bulyan", "median", "fedprox"]:
        if f"comp_{agg}" in name or f"_{agg}_" in name:
            config["aggregation"] = agg
            break
    else:
        config["aggregation"] = "unknown"

    alpha_match = re.search(r"alpha([0-9.]+|inf)", name)
    if alpha_match:
        alpha_str = alpha_match.group(1)
        config["alpha"] = float("inf") if alpha_str == "inf" else float(alpha_str)

    adv_match = re.search(r"adv(\d+)", name)
    config["adv_pct"] = int(adv_match.group(1)) if adv_match else 0

    pers_match = re.search(r"pers(\d+)", name)
    config["pers_epochs"] = int(pers_match.group(1)) if pers_match else 0

    seed_match = re.search(r"seed(\d+)", name)
    config["seed"] = int(seed_match.group(1)) if seed_match else 42

    config["run_id"] = run_dir.name

    return config


def _parse_json_dict(value: str) -> dict:
    """Parse JSON dictionary string, handling various formats."""
    if pd.isna(value) or value == "":
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


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
    """
    Load all client metrics CSVs for a run, grouped by round.

    Now includes per-class F1 scores and holdout metrics.
    """
    client_data_by_round: dict[int, list[dict]] = {}

    for client_csv in run_dir.glob("client_*_metrics.csv"):
        try:
            df_client = pd.read_csv(client_csv)

            for _, row in df_client.iterrows():
                round_num = int(row["round"])
                if round_num not in client_data_by_round:
                    client_data_by_round[round_num] = []

                f1_global = row.get("macro_f1_before", np.nan)
                if pd.isna(f1_global):
                    f1_global = row.get("macro_f1_global", np.nan)

                f1_pers = row.get("macro_f1_personalized", row.get("macro_f1_after", np.nan))

                f1_per_class = _parse_json_dict(row.get("f1_per_class_after", "{}"))
                f1_per_class_holdout = _parse_json_dict(row.get("f1_per_class_holdout", "{}"))

                class_names_str = row.get("confusion_matrix_class_names", "[]")
                class_names = json.loads(class_names_str) if isinstance(class_names_str, str) else []

                client_data_by_round[round_num].append(
                    {
                        "macro_f1_global": f1_global,
                        "macro_f1_personalized": f1_pers,
                        "f1_per_class": f1_per_class,
                        "f1_per_class_holdout": f1_per_class_holdout,
                        "class_names": class_names,
                        "macro_f1_global_holdout": row.get("macro_f1_global_holdout", np.nan),
                    }
                )
        except Exception as e:
            print(f"Warning: Failed to load {client_csv}: {e}")
            continue

    return client_data_by_round


def _merge_per_class_f1(client_round_data: list[dict]) -> dict[str, float]:
    """
    Merge per-class F1 scores across clients for a round.

    Returns average F1 for each class across all clients.
    """
    all_class_f1s = {}

    for client_data in client_round_data:
        f1_dict = client_data.get("f1_per_class", {})
        for class_id, f1_value in f1_dict.items():
            if class_id not in all_class_f1s:
                all_class_f1s[class_id] = []
            try:
                all_class_f1s[class_id].append(float(f1_value))
            except (ValueError, TypeError):
                continue

    merged = {}
    for class_id, values in all_class_f1s.items():
        if values:
            merged[class_id] = float(np.mean(values))

    return merged


def _create_merged_records(
    config: dict,
    df_server: pd.DataFrame,
    client_data_by_round: dict[int, list[dict]],
) -> list[dict]:
    """Merge server and client metrics by round, including per-class F1."""
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

        if round_num in client_data_by_round:
            client_round_data = client_data_by_round[round_num]

            f1_global_vals = [
                c["macro_f1_global"]
                for c in client_round_data
                if not np.isnan(c["macro_f1_global"])
            ]
            f1_pers_vals = [
                c["macro_f1_personalized"]
                for c in client_round_data
                if not np.isnan(c["macro_f1_personalized"])
            ]
            f1_holdout_vals = [
                c["macro_f1_global_holdout"]
                for c in client_round_data
                if not np.isnan(c.get("macro_f1_global_holdout", np.nan))
            ]

            record["macro_f1_global"] = np.mean(f1_global_vals) if f1_global_vals else np.nan
            record["macro_f1_personalized"] = np.mean(f1_pers_vals) if f1_pers_vals else np.nan
            record["macro_f1_global_holdout"] = np.mean(f1_holdout_vals) if f1_holdout_vals else np.nan

            per_class_f1 = _merge_per_class_f1(client_round_data)
            record["f1_per_class"] = per_class_f1

            if client_round_data and client_round_data[0].get("class_names"):
                record["class_names"] = client_round_data[0]["class_names"]
        else:
            record["macro_f1_global"] = np.nan
            record["macro_f1_personalized"] = np.nan
            record["macro_f1_global_holdout"] = np.nan
            record["f1_per_class"] = {}
            record["class_names"] = []

        records.append(record)

    return records


def load_iiot_data_enhanced(runs_dir: Path, run_pattern: str = "dsedge-iiotset-nightly*") -> pd.DataFrame:
    """
    Load all IIoT experiments into unified DataFrame with per-class F1.

    Args:
        runs_dir: Directory containing experiment runs
        run_pattern: Glob pattern for run directories

    Returns:
        DataFrame with columns:
        - Config: aggregation, alpha, adv_pct, pers_epochs, seed, run_id
        - Per-round: round
        - Server metrics: l2_to_benign_mean, l2_dispersion_mean, t_aggregate_ms
        - Client metrics (averaged): macro_f1_global, macro_f1_personalized
        - Per-class: f1_per_class (dict), class_names (list)
    """
    all_records = []

    for run_dir in runs_dir.glob(run_pattern):
        if not run_dir.is_dir():
            continue

        config = parse_run_config(run_dir)

        df_server = _load_server_metrics_for_run(run_dir)
        if df_server is None:
            continue

        client_data_by_round = _load_client_metrics_for_run(run_dir)

        records = _create_merged_records(config, df_server, client_data_by_round)
        all_records.extend(records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    df["personalization_gain"] = df["macro_f1_personalized"] - df["macro_f1_global"]

    return df


def expand_per_class_f1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand f1_per_class dict into separate rows for analysis.

    Args:
        df: DataFrame with f1_per_class column containing dicts

    Returns:
        DataFrame with one row per (experiment config, round, class)
        Columns: all original columns + class_id, class_name, class_f1
    """
    expanded_records = []

    for _, row in df.iterrows():
        f1_dict = row.get("f1_per_class", {})
        class_names = row.get("class_names", [])

        if not f1_dict:
            continue

        for class_id_str, f1_value in f1_dict.items():
            class_id = int(class_id_str)
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"

            record = row.to_dict()
            record["class_id"] = class_id
            record["class_name"] = class_name
            record["class_f1"] = float(f1_value)

            expanded_records.append(record)

    if not expanded_records:
        return pd.DataFrame()

    return pd.DataFrame(expanded_records)


if __name__ == "__main__":
    runs_dir = Path("runs")
    df = load_iiot_data_enhanced(runs_dir)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nAggregators: {sorted(df['aggregation'].unique())}")
    print(f"Alpha values: {sorted(df['alpha'].unique())}")
    print(f"Attack levels: {sorted(df['adv_pct'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    print("\nExpanding per-class F1...")
    df_expanded = expand_per_class_f1(df)
    print(f"Expanded to {len(df_expanded)} records")
    print(f"Classes: {sorted(df_expanded['class_name'].unique())}")
