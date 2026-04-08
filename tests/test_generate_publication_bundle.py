"""Unit tests for publication bundle generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_publication_bundle import (
    build_edge_baseline_attack_deltas,
    build_edge_per_class_profiles,
    build_edge_practitioner_walkthrough_candidates,
    build_publication_claim_matrix,
    find_missing_edge_clean_baselines,
    parse_named_metric_series,
    summarize_edge_attack_modes,
)


def _write_run(
    runs_dir: Path,
    run_id: str,
    final_metrics: dict[str, float],
    holdout_rows: list[dict[str, object]],
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    pd.DataFrame([final_metrics]).to_csv(run_dir / "metrics.csv", index=False)
    for client_id, row in enumerate(holdout_rows):
        pd.DataFrame([row]).to_csv(run_dir / f"client_{client_id}_metrics.csv", index=False)


def test_parse_named_metric_series_supports_dict_and_list_payloads() -> None:
    class_names = json.dumps(["BENIGN", "ATTACK"])

    dict_payload = json.dumps({"0": 0.8, "1": 0.2})
    list_payload = json.dumps([0.8, 0.2])

    assert parse_named_metric_series(dict_payload, class_names) == {"BENIGN": 0.8, "ATTACK": 0.2}
    assert parse_named_metric_series(list_payload, class_names) == {"BENIGN": 0.8, "ATTACK": 0.2}


def test_find_missing_edge_clean_baselines_requires_exact_family() -> None:
    canonical_runs = pd.DataFrame(
        [
            {
                "run_id": "fedavg_nearby",
                "dataset": "edge-iiotset-full",
                "aggregation": "fedavg",
                "alpha": 0.5,
                "adversary_fraction": 0.0,
                "attack_mode": "none",
                "adversary_mode": "none",
                "fedprox_mu": 0.01,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
            {
                "run_id": "median_attack",
                "dataset": "edge-iiotset-full",
                "aggregation": "median",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
        ]
    )

    gaps = find_missing_edge_clean_baselines(canonical_runs)

    assert set(gaps["aggregation"]) == {"fedavg", "krum", "median"}
    assert set(gaps["status"]) == {"missing_exact_family"}


def test_summarize_edge_attack_modes_averages_raw_run_metrics(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    _write_run(
        runs_dir,
        "fedavg_signflip_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.1,
            "l2_to_benign_mean": 400.0,
            "cos_to_benign_mean": 0.2,
            "pairwise_cosine_mean": 0.1,
            "update_norm_mean": 35.0,
        },
        [],
    )
    _write_run(
        runs_dir,
        "fedavg_signflip_seed43",
        {
            "round": 20,
            "global_macro_f1_test": 0.2,
            "l2_to_benign_mean": 420.0,
            "cos_to_benign_mean": 0.3,
            "pairwise_cosine_mean": 0.2,
            "update_norm_mean": 37.0,
        },
        [],
    )
    _write_run(
        runs_dir,
        "fedavg_targeted_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.4,
            "l2_to_benign_mean": 1.1,
            "cos_to_benign_mean": 0.99,
            "pairwise_cosine_mean": 0.95,
            "update_norm_mean": 15.0,
        },
        [],
    )
    _write_run(
        runs_dir,
        "fedavg_targeted_seed43",
        {
            "round": 20,
            "global_macro_f1_test": 0.42,
            "l2_to_benign_mean": 1.3,
            "cos_to_benign_mean": 0.98,
            "pairwise_cosine_mean": 0.94,
            "update_norm_mean": 16.0,
        },
        [],
    )

    canonical_runs = pd.DataFrame(
        [
            {
                "run_id": "fedavg_signflip_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "fedavg",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "sign_flip_topk",
                "adversary_mode": "sign_flip_topk",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
            {
                "run_id": "fedavg_signflip_seed43",
                "dataset": "edge-iiotset-full",
                "aggregation": "fedavg",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "sign_flip_topk",
                "adversary_mode": "sign_flip_topk",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 43,
            },
            {
                "run_id": "fedavg_targeted_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "fedavg",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
            {
                "run_id": "fedavg_targeted_seed43",
                "dataset": "edge-iiotset-full",
                "aggregation": "fedavg",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 43,
            },
        ]
    )

    summary = summarize_edge_attack_modes(canonical_runs=canonical_runs, runs_dir=runs_dir)

    sign_flip = summary[
        (summary["aggregation"] == "fedavg") & (summary["attack_mode"] == "sign_flip_topk")
    ].iloc[0]
    targeted = summary[
        (summary["aggregation"] == "fedavg") & (summary["attack_mode"] == "targeted_label")
    ].iloc[0]

    assert sign_flip["macro_f1_mean"] == 0.15
    assert sign_flip["l2_to_benign_mean_mean"] == 410.0
    assert targeted["macro_f1_mean"] == 0.41
    assert targeted["cos_to_benign_mean_mean"] == 0.985


def test_build_edge_per_class_profiles_averages_clients_and_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    holdout_rows_a = [
        {
            "round": 20,
            "macro_f1_global_holdout": 0.4,
            "f1_per_class_holdout": json.dumps({"0": 0.9, "1": 0.1}),
            "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "ATTACK"]),
        },
        {
            "round": 20,
            "macro_f1_global_holdout": 0.5,
            "f1_per_class_holdout": json.dumps({"0": 0.7, "1": 0.3}),
            "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "ATTACK"]),
        },
    ]
    holdout_rows_b = [
        {
            "round": 20,
            "macro_f1_global_holdout": 0.45,
            "f1_per_class_holdout": json.dumps([0.8, 0.2]),
            "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "ATTACK"]),
        }
    ]

    _write_run(
        runs_dir,
        "median_targeted_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.4,
            "l2_to_benign_mean": 1.5,
            "cos_to_benign_mean": 0.99,
            "pairwise_cosine_mean": 0.95,
            "update_norm_mean": 14.0,
        },
        holdout_rows_a,
    )
    _write_run(
        runs_dir,
        "median_targeted_seed43",
        {
            "round": 20,
            "global_macro_f1_test": 0.42,
            "l2_to_benign_mean": 1.4,
            "cos_to_benign_mean": 0.98,
            "pairwise_cosine_mean": 0.94,
            "update_norm_mean": 15.0,
        },
        holdout_rows_b,
    )

    canonical_runs = pd.DataFrame(
        [
            {
                "run_id": "median_targeted_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "median",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
            {
                "run_id": "median_targeted_seed43",
                "dataset": "edge-iiotset-full",
                "aggregation": "median",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 43,
            },
        ]
    )

    profiles = build_edge_per_class_profiles(canonical_runs=canonical_runs, runs_dir=runs_dir)
    benign = profiles[
        (profiles["aggregation"] == "median")
        & (profiles["attack_mode"] == "targeted_label")
        & (profiles["class_name"] == "BENIGN")
    ].iloc[0]
    attack = profiles[
        (profiles["aggregation"] == "median")
        & (profiles["attack_mode"] == "targeted_label")
        & (profiles["class_name"] == "ATTACK")
    ].iloc[0]

    assert benign["per_class_f1_mean"] == 0.8
    assert attack["per_class_f1_mean"] == 0.2


def test_build_edge_baseline_attack_deltas_uses_exact_clean_family(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    _write_run(
        runs_dir,
        "median_clean_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.68,
            "l2_to_benign_mean": 0.0,
            "cos_to_benign_mean": 1.0,
            "pairwise_cosine_mean": 1.0,
            "update_norm_mean": 10.0,
        },
        [
            {
                "round": 20,
                "macro_f1_global_holdout": 0.68,
                "f1_per_class_holdout": json.dumps({"0": 0.95, "1": 0.85}),
                "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "ATTACK"]),
            }
        ],
    )
    _write_run(
        runs_dir,
        "median_targeted_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.43,
            "l2_to_benign_mean": 1.5,
            "cos_to_benign_mean": 0.99,
            "pairwise_cosine_mean": 0.95,
            "update_norm_mean": 15.0,
        },
        [
            {
                "round": 20,
                "macro_f1_global_holdout": 0.43,
                "f1_per_class_holdout": json.dumps({"0": 0.90, "1": 0.25}),
                "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "ATTACK"]),
            }
        ],
    )

    canonical_runs = pd.DataFrame(
        [
            {
                "run_id": "median_clean_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "median",
                "alpha": 0.5,
                "adversary_fraction": 0.0,
                "attack_mode": "none",
                "adversary_mode": "none",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
            {
                "run_id": "median_targeted_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "median",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
        ]
    )

    deltas = build_edge_baseline_attack_deltas(canonical_runs=canonical_runs, runs_dir=runs_dir)
    attack_row = deltas[
        (deltas["aggregation"] == "median")
        & (deltas["attack_mode"] == "targeted_label")
        & (deltas["class_name"] == "ATTACK")
    ].iloc[0]

    assert attack_row["baseline_per_class_f1"] == 0.85
    assert attack_row["attack_per_class_f1"] == 0.25
    assert attack_row["delta"] == -0.6


def test_build_edge_practitioner_walkthrough_candidates_finds_green_light_blind_spot(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    _write_run(
        runs_dir,
        "krum_clean_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.65,
            "l2_to_benign_mean": 0.0,
            "cos_to_benign_mean": 1.0,
            "pairwise_cosine_mean": 1.0,
            "update_norm_mean": 10.0,
        },
        [
            {
                "round": 20,
                "macro_f1_global_holdout": 0.65,
                "f1_per_class_holdout": json.dumps({"0": 0.9, "1": 0.6}),
                "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "MITM"]),
            }
        ],
    )
    _write_run(
        runs_dir,
        "krum_targeted_seed42",
        {
            "round": 20,
            "global_macro_f1_test": 0.5,
            "l2_to_benign_mean": 2.0,
            "cos_to_benign_mean": 0.99,
            "pairwise_cosine_mean": 0.94,
            "update_norm_mean": 12.0,
        },
        [
            {
                "round": 20,
                "macro_f1_global_holdout": 0.5,
                "f1_per_class_holdout": json.dumps({"0": 0.8, "1": 0.0}),
                "confusion_matrix_class_names_holdout": json.dumps(["BENIGN", "MITM"]),
            }
        ],
    )

    canonical_runs = pd.DataFrame(
        [
            {
                "run_id": "krum_clean_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "krum",
                "alpha": 0.5,
                "adversary_fraction": 0.0,
                "attack_mode": "none",
                "adversary_mode": "none",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
            {
                "run_id": "krum_targeted_seed42",
                "dataset": "edge-iiotset-full",
                "aggregation": "krum",
                "alpha": 0.5,
                "adversary_fraction": 0.3,
                "attack_mode": "targeted_label",
                "adversary_mode": "targeted_label",
                "fedprox_mu": 0.0,
                "num_clients": 10,
                "num_rounds": 20,
                "seed": 42,
            },
        ]
    )

    candidates = build_edge_practitioner_walkthrough_candidates(
        canonical_runs=canonical_runs,
        runs_dir=runs_dir,
    )
    candidate = candidates.iloc[0]

    assert candidate["aggregation"] == "krum"
    assert candidate["attack_mode"] == "targeted_label"
    assert candidate["class_name"] == "MITM"
    assert candidate["macro_f1"] == 0.5
    assert candidate["attack_per_class_f1"] == 0.0
    assert candidate["baseline_per_class_f1"] == 0.6


def test_build_publication_claim_matrix_downgrades_unfinished_per_class_claims() -> None:
    raid_summary = pd.DataFrame(
        [
            {
                "dataset": "cic",
                "aggregation": "median",
                "benign_macro_f1": 0.49,
                "attack30_macro_f1": 0.31,
                "delta_vs_fedavg_attack30": 0.03,
            },
            {
                "dataset": "edge-iiotset-full",
                "aggregation": "median",
                "benign_macro_f1": 0.68,
                "attack30_macro_f1": 0.43,
                "delta_vs_fedavg_attack30": 0.22,
            },
            {
                "dataset": "unsw",
                "aggregation": "median",
                "benign_macro_f1": 0.46,
                "attack30_macro_f1": 0.35,
                "delta_vs_fedavg_attack30": 0.07,
            }
        ]
    )
    edge_attack_summary = pd.DataFrame(
        [
            {
                "aggregation": "fedavg",
                "attack_mode": "sign_flip_topk",
                "macro_f1_mean": 0.05,
                "l2_to_benign_mean_mean": 400.0,
            },
            {
                "aggregation": "fedavg",
                "attack_mode": "targeted_label",
                "macro_f1_mean": 0.41,
                "l2_to_benign_mean_mean": 1.2,
            },
        ]
    )
    edge_baseline_gaps = pd.DataFrame(
        [
            {
                "aggregation": "fedavg",
                "status": "missing_exact_family",
                "available_exact_runs": 0,
                "available_nearby_runs": 0,
            }
        ]
    )

    claim_matrix = build_publication_claim_matrix(
        raid_summary=raid_summary,
        edge_attack_summary=edge_attack_summary,
        edge_baseline_gaps=edge_baseline_gaps,
    )

    raid_claim = claim_matrix[claim_matrix["claim_id"] == "RAID-R1"].iloc[0]
    ccs_gap_claim = claim_matrix[claim_matrix["claim_id"] == "CCS-C3"].iloc[0]

    assert raid_claim["claim_state"] == "main_safe"
    assert ccs_gap_claim["claim_state"] == "secondary_safe"
