#!/usr/bin/env python3
"""Comprehensive runs census and claim-building pipeline for publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CORE_DATASETS = ["cic", "unsw", "edge-iiotset-full"]
BASELINE_AGGREGATIONS = ["fedavg", "krum", "bulyan", "median"]
ATTACK_LEVELS = [10, 20, 30]
METRIC_CANDIDATE_COLUMNS = [
    "global_macro_f1_test",
    "global_macro_f1_val",
    "macro_f1_after",
    "macro_f1_global",
    "macro_f1",
]
QUALITY_RANK = {
    "complete_or_exceeds_target": 0,
    "truncated_before_target": 1,
    "empty_metrics": 2,
    "no_round_column": 3,
    "round_non_numeric": 4,
    "unreadable_metrics": 5,
    "bad_config_json": 6,
    "missing_core_file": 7,
}


@dataclass(frozen=True)
class Signature:
    """Schema signature."""

    signature_id: str
    key_count: int
    keys_json: str
    runs_count: int


@dataclass(frozen=True)
class ClaimSpec:
    """Claim requirement specification."""

    claim_id: str
    description: str
    slice_name: str
    min_grade: str
    min_fraction: float


def classify_run_family(run_id: str) -> str:
    """Classify run naming family with deterministic precedence."""
    if run_id.startswith("dscic_") or run_id.startswith("dsunsw_") or run_id.startswith("dsedge-"):
        return "ds_prefixed"
    if "_simple_" in run_id:
        return "simple"
    if run_id.startswith("comp_"):
        return "comp_short"
    if "_comp_" in run_id:
        return "comp_other"
    return "other"


def parse_source_override_type(run_id: str) -> str | None:
    """Extract source override from run name when present."""
    match = re.search(r"_src([A-Za-z0-9._-]+)", run_id)
    return match.group(1) if match else None


def signature_id(items: tuple[str, ...]) -> str:
    """Build a stable short signature id from ordered items."""
    payload = "|".join(items)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:12]


def normalize_alpha(value: Any) -> float:
    """Normalize alpha values from mixed json types."""
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_float(value: Any) -> float:
    """Normalize float values from mixed json types."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_int(value: Any) -> int | None:
    """Normalize integer values from mixed json types."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mode_value(config: dict[str, Any], primary: str, secondary: str) -> str:
    """Get mode field with fallback and default."""
    value = config.get(primary)
    if value is None:
        value = config.get(secondary)
    if value is None:
        return "none"
    return str(value)


def format_alpha(value: float) -> str:
    """Format alpha for joining and output."""
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:g}"


def compute_final_metric(df: pd.DataFrame) -> float:
    """Compute final-round metric from prioritized candidate columns."""
    if "round" not in df.columns or df.empty:
        return float("nan")
    rounds = pd.to_numeric(df["round"], errors="coerce")
    if rounds.dropna().empty:
        return float("nan")
    max_round = rounds.max()
    final_df = df[rounds == max_round]
    for column in METRIC_CANDIDATE_COLUMNS:
        if column not in final_df.columns:
            continue
        values = pd.to_numeric(final_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        return float(values.mean())
    return float("nan")


def reliability_grade(n_seeds: int) -> str:
    """Compute reliability grade from seed count."""
    if n_seeds >= 10:
        return "A"
    if n_seeds >= 5:
        return "B"
    if n_seeds >= 3:
        return "C"
    return "D"


def compute_ci_bounds(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Compute t-based confidence interval bounds around the mean."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return float("nan"), float("nan")
    mean = float(values.mean())
    n = len(values)
    if n < 2:
        return mean, mean
    sem = float(values.std(ddof=1) / math.sqrt(n))
    if sem == 0:
        return mean, mean
    from scipy import stats  # local import keeps import cost low for non-analysis tooling

    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * sem
    return mean - margin, mean + margin


def infer_quality_state(
    has_config: bool,
    has_metrics: bool,
    config_error: bool,
    metrics_error: bool,
    metrics_df: pd.DataFrame | None,
    target_rounds: int | None,
) -> tuple[str, float | None, int]:
    """Infer terminal quality state for a run."""
    if not (has_config and has_metrics):
        return "missing_core_file", None, 0
    if config_error:
        return "bad_config_json", None, 0
    if metrics_error or metrics_df is None:
        return "unreadable_metrics", None, 0
    if metrics_df.empty:
        return "empty_metrics", None, 0
    if "round" not in metrics_df.columns:
        return "no_round_column", None, len(metrics_df)

    rounds = pd.to_numeric(metrics_df["round"], errors="coerce").dropna()
    if rounds.empty:
        return "round_non_numeric", None, len(metrics_df)

    max_round = float(rounds.max())
    if target_rounds and max_round < target_rounds:
        return "truncated_before_target", max_round, len(metrics_df)
    return "complete_or_exceeds_target", max_round, len(metrics_df)


def build_signature_table(counter: dict[tuple[str, ...], int]) -> pd.DataFrame:
    """Build signature dataframe from a tuple->count mapping."""
    rows: list[Signature] = []
    for keys, count in counter.items():
        rows.append(
            Signature(
                signature_id=signature_id(keys),
                key_count=len(keys),
                keys_json=json.dumps(list(keys)),
                runs_count=count,
            )
        )
    return pd.DataFrame(rows).sort_values("runs_count", ascending=False).reset_index(drop=True)


def scan_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan run directories and return registry + schema drift tables."""
    registry_rows: list[dict[str, Any]] = []
    config_signature_counts: dict[tuple[str, ...], int] = {}
    metrics_signature_counts: dict[tuple[str, ...], int] = {}

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        has_config = config_path.exists()
        has_metrics = metrics_path.exists()

        row: dict[str, Any] = {
            "run_id": run_id,
            "run_path": str(run_dir),
            "family": classify_run_family(run_id),
            "is_partial": "__partial_" in run_id,
            "source_override_type": parse_source_override_type(run_id),
            "has_source_override": parse_source_override_type(run_id) is not None,
            "has_config": has_config,
            "has_metrics": has_metrics,
            "run_mtime": run_dir.stat().st_mtime,
            "config_size_bytes": config_path.stat().st_size if has_config else 0,
            "metrics_size_bytes": metrics_path.stat().st_size if has_metrics else 0,
        }

        config: dict[str, Any] = {}
        config_error = False
        if has_config:
            try:
                config = json.loads(config_path.read_text())
                config_signature = tuple(sorted(config.keys()))
                config_signature_counts[config_signature] = config_signature_counts.get(config_signature, 0) + 1
                row["config_signature_id"] = signature_id(config_signature)
            except (OSError, json.JSONDecodeError):
                config_error = True
                row["config_signature_id"] = None
        else:
            row["config_signature_id"] = None

        metrics_df: pd.DataFrame | None = None
        metrics_error = False
        if has_metrics:
            try:
                metrics_df = pd.read_csv(metrics_path, engine="python", on_bad_lines="skip")
                metrics_signature = tuple(metrics_df.columns)
                metrics_signature_counts[metrics_signature] = metrics_signature_counts.get(metrics_signature, 0) + 1
                row["metrics_signature_id"] = signature_id(metrics_signature)
            except Exception:
                metrics_error = True
                row["metrics_signature_id"] = None
        else:
            row["metrics_signature_id"] = None

        target_rounds = normalize_int(config.get("num_rounds")) if config else None
        quality_state, max_round, parsed_row_count = infer_quality_state(
            has_config=has_config,
            has_metrics=has_metrics,
            config_error=config_error,
            metrics_error=metrics_error,
            metrics_df=metrics_df,
            target_rounds=target_rounds,
        )

        source_datasets = config.get("source_datasets") if isinstance(config.get("source_datasets"), list) else []
        source_datasets_normalized = ",".join(sorted(str(item) for item in source_datasets))

        row.update(
            {
                "dataset": str(config.get("dataset")) if config else None,
                "aggregation": str(config.get("aggregation")) if config else None,
                "alpha": normalize_alpha(config.get("alpha")) if config else float("nan"),
                "adversary_fraction": normalize_float(config.get("adversary_fraction")) if config else float("nan"),
                "adv_percent": normalize_float(config.get("adversary_fraction")) * 100 if config else float("nan"),
                "fedprox_mu": normalize_float(config.get("fedprox_mu")) if config else float("nan"),
                "dp_enabled": bool(config.get("dp_enabled")) if config else False,
                "dp_noise_multiplier": normalize_float(config.get("dp_noise_multiplier")) if config else float("nan"),
                "personalization_epochs": normalize_int(config.get("personalization_epochs")) if config else None,
                "seed": normalize_int(config.get("seed")) if config else None,
                "num_clients": normalize_int(config.get("num_clients")) if config else None,
                "num_rounds": target_rounds,
                "adversary_mode": mode_value(config, "adversary_mode", "attack_mode") if config else "none",
                "attack_mode": mode_value(config, "attack_mode", "adversary_mode") if config else "none",
                "source_dataset": str(config.get("source_dataset")) if config.get("source_dataset") is not None else None,
                "source_datasets": source_datasets_normalized,
                "quality_state": quality_state,
                "max_round": max_round,
                "parsed_row_count": parsed_row_count,
                "final_metric": compute_final_metric(metrics_df) if metrics_df is not None else float("nan"),
            }
        )

        registry_rows.append(row)

    registry_df = pd.DataFrame(registry_rows)
    config_signatures_df = build_signature_table(config_signature_counts)
    metrics_signatures_df = build_signature_table(metrics_signature_counts)
    return registry_df, config_signatures_df, metrics_signatures_df


def build_canonical_key(row: pd.Series) -> tuple[Any, ...]:
    """Create canonical dedup key for a run row."""
    return (
        row.get("dataset"),
        row.get("aggregation"),
        format_alpha(row.get("alpha")) if pd.notna(row.get("alpha")) else "nan",
        row.get("adversary_fraction"),
        row.get("adversary_mode"),
        row.get("attack_mode"),
        row.get("fedprox_mu"),
        row.get("dp_enabled"),
        row.get("dp_noise_multiplier"),
        row.get("personalization_epochs"),
        row.get("seed"),
        row.get("source_dataset"),
        row.get("source_datasets"),
    )


def build_dedup_map(registry_df: pd.DataFrame) -> pd.DataFrame:
    """Select one canonical run per dedup key with deterministic ordering."""
    dedup_candidates = registry_df[
        registry_df["quality_state"].isin(
            {
                "complete_or_exceeds_target",
                "truncated_before_target",
                "empty_metrics",
                "no_round_column",
                "round_non_numeric",
                "unreadable_metrics",
            }
        )
    ].copy()

    dedup_candidates["canonical_key"] = dedup_candidates.apply(build_canonical_key, axis=1)
    dedup_candidates["canonical_key_id"] = dedup_candidates["canonical_key"].apply(
        lambda key: signature_id(tuple(str(item) for item in key))
    )
    dedup_candidates["quality_rank"] = dedup_candidates["quality_state"].map(QUALITY_RANK).fillna(999)
    dedup_candidates["max_round_sort"] = dedup_candidates["max_round"].fillna(-1)
    dedup_candidates["parsed_row_count_sort"] = dedup_candidates["parsed_row_count"].fillna(-1)

    sort_cols = [
        "canonical_key_id",
        "quality_rank",
        "max_round_sort",
        "parsed_row_count_sort",
        "run_mtime",
        "run_id",
    ]
    dedup_candidates = dedup_candidates.sort_values(
        by=sort_cols,
        ascending=[True, True, False, False, False, True],
    )

    canonical_map = (
        dedup_candidates.groupby("canonical_key_id", as_index=False)
        .first()[["canonical_key_id", "run_id"]]
        .rename(columns={"run_id": "canonical_run_id"})
    )

    dedup_map = dedup_candidates.merge(canonical_map, on="canonical_key_id", how="left")
    dedup_map["is_canonical"] = dedup_map["run_id"] == dedup_map["canonical_run_id"]
    group_sizes = dedup_map.groupby("canonical_key_id")["run_id"].transform("count")
    dedup_map["duplicate_group_size"] = group_sizes

    dedup_map = dedup_map[
        [
            "run_id",
            "canonical_run_id",
            "is_canonical",
            "canonical_key_id",
            "duplicate_group_size",
            "dataset",
            "aggregation",
            "alpha",
            "adv_percent",
            "attack_mode",
            "fedprox_mu",
            "seed",
            "quality_state",
            "quality_rank",
            "max_round",
            "parsed_row_count",
        ]
    ]

    return dedup_map.sort_values(["canonical_key_id", "is_canonical", "run_id"], ascending=[True, False, True]).reset_index(drop=True)


def summarize_coverage(df: pd.DataFrame, slice_name: str, group_cols: list[str]) -> pd.DataFrame:
    """Summarize coverage and metric quality for a slice."""
    if df.empty:
        return pd.DataFrame(columns=["slice", *group_cols, "n_runs", "n_seeds", "metric_mean", "ci_low", "ci_high", "reliability_grade"])

    grouped = df.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        metric_mean = float(pd.to_numeric(group["final_metric"], errors="coerce").dropna().mean())
        ci_low, ci_high = compute_ci_bounds(group["final_metric"])
        n_seeds = int(group["seed"].dropna().nunique())
        row = {
            "slice": slice_name,
            "n_runs": int(group["run_id"].nunique()),
            "n_seeds": n_seeds,
            "metric_mean": metric_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reliability_grade": reliability_grade(n_seeds),
        }
        for idx, col in enumerate(group_cols):
            row[col] = key[idx]
        rows.append(row)

    return pd.DataFrame(rows)


def build_coverage_tables(canonical_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build confirmatory, exploratory, and all-coverage tables."""
    evidence = canonical_runs[
        (canonical_runs["quality_state"] == "complete_or_exceeds_target") & (canonical_runs["final_metric"].notna())
    ].copy()

    baseline_df = evidence[
        (evidence["alpha"] == 1.0)
        & (evidence["adv_percent"] == 0.0)
        & (evidence["fedprox_mu"] == 0.0)
        & (evidence["aggregation"].isin(BASELINE_AGGREGATIONS))
        & (evidence["dataset"].isin(CORE_DATASETS))
    ]
    baseline_cov = summarize_coverage(baseline_df, "baseline_iid", ["dataset", "aggregation"])

    attack_collapsed_df = evidence[
        (evidence["alpha"] == 0.5)
        & (evidence["fedprox_mu"] == 0.0)
        & (evidence["adv_percent"].isin(ATTACK_LEVELS))
        & (evidence["aggregation"].isin(BASELINE_AGGREGATIONS))
        & (evidence["dataset"].isin(CORE_DATASETS))
    ]
    attack_collapsed_cov = summarize_coverage(
        attack_collapsed_df,
        "attack_alpha0.5_collapsed",
        ["dataset", "aggregation", "adv_percent"],
    )

    attack_mode_df = attack_collapsed_df[attack_collapsed_df["attack_mode"].notna()]
    attack_mode_cov = summarize_coverage(
        attack_mode_df,
        "attack_alpha0.5_by_mode",
        ["dataset", "aggregation", "adv_percent", "attack_mode"],
    )

    heterogeneity_df = evidence[
        (evidence["aggregation"] == "fedavg")
        & (evidence["adv_percent"] == 0.0)
        & (evidence["fedprox_mu"] == 0.0)
        & (evidence["dataset"].isin(CORE_DATASETS))
    ]
    heterogeneity_cov = summarize_coverage(heterogeneity_df, "heterogeneity_fedavg", ["dataset", "alpha"])

    fedprox_df = evidence[(evidence["fedprox_mu"] > 0.0) & (evidence["adv_percent"] == 0.0) & (evidence["dataset"].isin(CORE_DATASETS))]
    fedprox_cov = summarize_coverage(fedprox_df, "fedprox_nonzero_mu", ["dataset", "alpha", "fedprox_mu"])

    coverage_all = pd.concat(
        [baseline_cov, attack_collapsed_cov, attack_mode_cov, heterogeneity_cov, fedprox_cov],
        ignore_index=True,
        sort=False,
    )

    confirmatory = coverage_all[coverage_all["reliability_grade"].isin(["A", "B"])].copy()
    exploratory = coverage_all[~coverage_all["reliability_grade"].isin(["A", "B"])].copy()

    return confirmatory, exploratory, coverage_all


def build_gap_inventory(coverage_all: pd.DataFrame) -> pd.DataFrame:
    """Build expected-vs-observed gap inventory for confirmatory slices."""
    expected_rows: list[dict[str, Any]] = []

    for dataset in CORE_DATASETS:
        for aggregation in BASELINE_AGGREGATIONS:
            expected_rows.append(
                {
                    "slice": "baseline_iid",
                    "dataset": dataset,
                    "aggregation": aggregation,
                    "adv_percent": float("nan"),
                    "alpha": float("nan"),
                    "fedprox_mu": float("nan"),
                    "attack_mode": None,
                }
            )

    for dataset in CORE_DATASETS:
        for aggregation in BASELINE_AGGREGATIONS:
            for adv_percent in ATTACK_LEVELS:
                expected_rows.append(
                    {
                        "slice": "attack_alpha0.5_collapsed",
                        "dataset": dataset,
                        "aggregation": aggregation,
                        "adv_percent": float(adv_percent),
                        "alpha": float("nan"),
                        "fedprox_mu": float("nan"),
                        "attack_mode": None,
                    }
                )

    for dataset in CORE_DATASETS:
        for alpha in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, float("inf")]:
            expected_rows.append(
                {
                    "slice": "heterogeneity_fedavg",
                    "dataset": dataset,
                    "aggregation": "fedavg",
                    "adv_percent": float("nan"),
                    "alpha": alpha,
                    "fedprox_mu": float("nan"),
                    "attack_mode": None,
                }
            )

    expected = pd.DataFrame(expected_rows)
    join_cols = ["slice", "dataset", "aggregation", "adv_percent", "alpha", "fedprox_mu", "attack_mode"]

    observed = coverage_all.copy()
    for col in ["adv_percent", "alpha", "fedprox_mu"]:
        if col in observed.columns:
            observed[col] = pd.to_numeric(observed[col], errors="coerce")
    for col in ["adv_percent", "alpha", "fedprox_mu"]:
        if col in expected.columns:
            expected[col] = pd.to_numeric(expected[col], errors="coerce")

    gap = expected.merge(
        observed[
            [
                "slice",
                "dataset",
                "aggregation",
                "adv_percent",
                "alpha",
                "fedprox_mu",
                "attack_mode",
                "n_seeds",
                "reliability_grade",
                "metric_mean",
            ]
        ],
        on=join_cols,
        how="left",
    )

    gap["cell_status"] = "missing"
    gap.loc[gap["n_seeds"].notna() & gap["reliability_grade"].isin(["A", "B"]), "cell_status"] = "claim_eligible"
    gap.loc[gap["n_seeds"].notna() & gap["reliability_grade"].isin(["C", "D"]), "cell_status"] = "exploratory_only"

    return gap.sort_values(["slice", "dataset", "aggregation", "adv_percent", "alpha"]).reset_index(drop=True)


def support_level(required: int, available: int, min_fraction: float) -> str:
    """Infer support level from required/available evidence counts."""
    if required == 0:
        return "unsupported"
    if available == required:
        return "supported"
    if available >= math.ceil(required * min_fraction):
        return "directional"
    if available > 0:
        return "exploratory"
    return "unsupported"


def build_claim_ledger(gap_inventory: pd.DataFrame) -> pd.DataFrame:
    """Build claim ledger mapping claims to coverage evidence."""
    specs = [
        ClaimSpec(
            claim_id="C1",
            description="Robust aggregation improves resilience over FedAvg under adversarial participation.",
            slice_name="attack_alpha0.5_collapsed",
            min_grade="B",
            min_fraction=0.7,
        ),
        ClaimSpec(
            claim_id="C2",
            description="FedAvg performance changes across heterogeneity levels in core datasets.",
            slice_name="heterogeneity_fedavg",
            min_grade="B",
            min_fraction=0.7,
        ),
        ClaimSpec(
            claim_id="C3",
            description="IID benign baseline supports cross-aggregator comparison in core datasets.",
            slice_name="baseline_iid",
            min_grade="B",
            min_fraction=0.8,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        subset = gap_inventory[gap_inventory["slice"] == spec.slice_name]
        required = int(len(subset))
        available = int((subset["cell_status"] == "claim_eligible").sum())
        exploratory = int((subset["cell_status"] == "exploratory_only").sum())
        missing = int((subset["cell_status"] == "missing").sum())
        rows.append(
            {
                "claim_id": spec.claim_id,
                "description": spec.description,
                "slice": spec.slice_name,
                "required_cells": required,
                "claim_eligible_cells": available,
                "exploratory_cells": exploratory,
                "missing_cells": missing,
                "support_level": support_level(required, available, spec.min_fraction),
            }
        )

    return pd.DataFrame(rows)


def write_data_dictionary(path: Path) -> None:
    """Write data dictionary for output artifacts."""
    content = """# Runs Census Data Dictionary

## runs_registry
- `run_id`: directory name under `runs/`.
- `family`: naming family (`ds_prefixed`, `simple`, `comp_short`, `comp_other`, `other`).
- `is_partial`: whether run name contains `__partial_`.
- `has_config`, `has_metrics`: core file existence flags.
- `config_signature_id`, `metrics_signature_id`: schema signature identifiers.
- `dataset`, `aggregation`, `alpha`, `adversary_fraction`, `adv_percent`, `fedprox_mu`, `seed`: normalized config fields.
- `quality_state`: terminal quality state.
- `max_round`, `parsed_row_count`: parsed metrics integrity indicators.
- `final_metric`: final-round macro-F1-like metric from prioritized columns.

## schema_drift_config / schema_drift_metrics
- `signature_id`: schema fingerprint.
- `key_count`: number of keys/columns.
- `keys_json`: ordered key list.
- `runs_count`: number of runs with this schema.

## runs_dedup_map
- `canonical_key_id`: dedup key fingerprint.
- `canonical_run_id`: selected run for key.
- `is_canonical`: whether row is selected representative.
- `duplicate_group_size`: size of duplicate cluster.

## coverage_confirmatory / coverage_exploratory
- `slice`: analysis slice identifier.
- `n_runs`, `n_seeds`: evidence counts.
- `metric_mean`, `ci_low`, `ci_high`: summary statistics.
- `reliability_grade`: grade from seed count (`A>=10`, `B>=5`, `C>=3`, `D<3`).

## gap_inventory
- `cell_status`: `claim_eligible`, `exploratory_only`, or `missing`.

## claim_ledger
- `support_level`: `supported`, `directional`, `exploratory`, `unsupported`.
"""
    path.write_text(content)


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> None:
    """Write dataframe to CSV and optionally Parquet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path is None:
        return
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # Parquet optional; CSV is canonical fallback.
        pass


def run_pipeline(runs_dir: Path, output_dir: Path, write_parquet: bool) -> dict[str, int]:
    """Execute full runs census pipeline and persist all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_df, config_sig_df, metrics_sig_df = scan_runs(runs_dir)
    dedup_map_df = build_dedup_map(registry_df)

    canonical_run_ids = set(dedup_map_df[dedup_map_df["is_canonical"]]["run_id"].tolist())
    canonical_runs = registry_df[registry_df["run_id"].isin(canonical_run_ids)].copy()

    confirmatory_cov_df, exploratory_cov_df, coverage_all_df = build_coverage_tables(canonical_runs)
    gap_inventory_df = build_gap_inventory(coverage_all_df)
    claim_ledger_df = build_claim_ledger(gap_inventory_df)

    write_table(
        registry_df,
        output_dir / "runs_registry.csv",
        output_dir / "runs_registry.parquet" if write_parquet else None,
    )
    write_table(config_sig_df, output_dir / "schema_drift_config.csv")
    write_table(metrics_sig_df, output_dir / "schema_drift_metrics.csv")
    write_table(
        registry_df[["run_id", "quality_state", "max_round", "parsed_row_count", "dataset", "aggregation"]],
        output_dir / "runs_quality_states.csv",
    )
    write_table(dedup_map_df, output_dir / "runs_dedup_map.csv")
    write_table(confirmatory_cov_df, output_dir / "coverage_confirmatory.csv")
    write_table(exploratory_cov_df, output_dir / "coverage_exploratory.csv")
    write_table(gap_inventory_df, output_dir / "gap_inventory.csv")
    write_table(claim_ledger_df, output_dir / "claim_ledger.csv")

    write_data_dictionary(output_dir / "runs_data_dictionary.md")

    return {
        "runs_registry_rows": len(registry_df),
        "canonical_runs": len(canonical_runs),
        "config_schema_signatures": len(config_sig_df),
        "metrics_schema_signatures": len(metrics_sig_df),
        "confirmatory_cells": len(confirmatory_cov_df),
        "exploratory_cells": len(exploratory_cov_df),
        "gap_inventory_rows": len(gap_inventory_df),
        "claim_count": len(claim_ledger_df),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Build comprehensive runs census and claim ledger artifacts.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Path to runs directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "runs_census",
        help="Output directory for census artifacts.",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Attempt writing parquet alongside CSV for major tables.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    summary = run_pipeline(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        write_parquet=args.write_parquet,
    )
    print("Runs census complete:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
