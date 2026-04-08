#!/usr/bin/env python3
"""Generate a dual-track publication bundle for RAID and CCS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CORE_DATASETS = ("cic", "edge-iiotset-full", "unsw")
CORE_AGGREGATIONS = ("fedavg", "krum", "median")
EDGE_ATTACK_MODES = ("label_flip", "sign_flip_topk", "targeted_label")
DATASET_LABELS = {
    "cic": "CIC-IDS2017",
    "edge-iiotset-full": "Edge-IIoTset",
    "unsw": "UNSW-NB15",
}


def effective_attack_mode(row: pd.Series) -> str:
    """Return the attack mode recorded for a run."""
    attack_mode = str(row.get("attack_mode", "") or "").strip()
    adversary_mode = str(row.get("adversary_mode", "") or "").strip()
    return attack_mode or adversary_mode or "none"


def parse_named_metric_series(raw_values: str, raw_names: str) -> dict[str, float]:
    """Map a serialized per-class metric payload to class names."""
    if not raw_values or not raw_names:
        return {}

    values = json.loads(raw_values)
    class_names = json.loads(raw_names)

    if isinstance(values, dict):
        indexed_values = {int(index): float(value) for index, value in values.items()}
    else:
        indexed_values = {index: float(value) for index, value in enumerate(values)}

    return {
        str(class_name): indexed_values[index]
        for index, class_name in enumerate(class_names)
        if index in indexed_values
    }


def load_canonical_runs(runs_registry_path: Path, dedup_map_path: Path) -> pd.DataFrame:
    """Load canonical runs from the census artifacts."""
    runs_registry = pd.read_csv(runs_registry_path)
    dedup_map = pd.read_csv(dedup_map_path)

    canonical = runs_registry.merge(
        dedup_map[["run_id", "is_canonical", "canonical_run_id"]],
        on="run_id",
        how="left",
    )
    canonical["is_canonical"] = canonical["is_canonical"].where(
        canonical["is_canonical"].notna(), False
    ).astype(bool)
    canonical = canonical[canonical["is_canonical"]].copy()
    canonical["effective_attack_mode"] = canonical.apply(effective_attack_mode, axis=1)
    return canonical


def load_final_metrics(run_dir: Path) -> dict[str, float] | None:
    """Load the final metrics row for a run."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None

    final_row = metrics.iloc[-1]
    return {
        "round": float(final_row.get("round", 0.0)),
        "macro_f1": float(final_row.get("global_macro_f1_test", 0.0)),
        "l2_to_benign_mean": float(final_row.get("l2_to_benign_mean", 0.0)),
        "cos_to_benign_mean": float(final_row.get("cos_to_benign_mean", 0.0)),
        "pairwise_cosine_mean": float(final_row.get("pairwise_cosine_mean", 0.0)),
        "update_norm_mean": float(final_row.get("update_norm_mean", 0.0)),
    }


def load_run_per_class_means(run_dir: Path) -> dict[str, float]:
    """Average per-class holdout F1 across clients for a run."""
    client_paths = sorted(run_dir.glob("client_*_metrics.csv"))
    per_client_rows: list[dict[str, float]] = []

    for client_path in client_paths:
        metrics = pd.read_csv(client_path)
        if metrics.empty:
            continue
        final_row = metrics.iloc[-1]
        class_metrics = parse_named_metric_series(
            str(final_row.get("f1_per_class_holdout", "")),
            str(final_row.get("confusion_matrix_class_names_holdout", "")),
        )
        if class_metrics:
            per_client_rows.append(class_metrics)

    if not per_client_rows:
        return {}

    class_frame = pd.DataFrame(per_client_rows)
    return {column: float(class_frame[column].mean()) for column in class_frame.columns}


def _filter_edge_attack_family(canonical_runs: pd.DataFrame, *, adversary_fraction: float) -> pd.DataFrame:
    filtered = canonical_runs[
        (canonical_runs["dataset"] == "edge-iiotset-full")
        & (canonical_runs["aggregation"].isin(CORE_AGGREGATIONS))
        & (canonical_runs["alpha"] == 0.5)
        & (canonical_runs["fedprox_mu"] == 0.0)
        & (canonical_runs["num_clients"] == 10)
        & (canonical_runs["num_rounds"] == 20)
        & (canonical_runs["adversary_fraction"] == adversary_fraction)
    ].copy()
    filtered["effective_attack_mode"] = filtered.apply(effective_attack_mode, axis=1)
    return filtered


def find_missing_edge_clean_baselines(canonical_runs: pd.DataFrame) -> pd.DataFrame:
    """Identify missing exact Edge baseline families needed by CCS."""
    exact_baselines = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.0)
    nearby = canonical_runs[
        (canonical_runs["dataset"] == "edge-iiotset-full")
        & (canonical_runs["aggregation"].isin(CORE_AGGREGATIONS))
        & (canonical_runs["alpha"] == 0.5)
        & (canonical_runs["adversary_fraction"] == 0.0)
        & (canonical_runs["num_clients"] == 10)
        & (canonical_runs["num_rounds"] == 20)
    ].copy()
    nearby["effective_attack_mode"] = nearby.apply(effective_attack_mode, axis=1)

    columns = [
        "aggregation",
        "status",
        "available_exact_runs",
        "available_nearby_runs",
        "nearby_fedprox_mu_values",
    ]
    rows: list[dict[str, object]] = []
    for aggregation in CORE_AGGREGATIONS:
        exact_group = exact_baselines[
            (exact_baselines["aggregation"] == aggregation)
            & (exact_baselines["effective_attack_mode"] == "none")
        ]
        nearby_group = nearby[nearby["aggregation"] == aggregation]
        if len(exact_group) > 0:
            continue

        rows.append(
            {
                "aggregation": aggregation,
                "status": "missing_exact_family",
                "available_exact_runs": len(exact_group),
                "available_nearby_runs": len(nearby_group),
                "nearby_fedprox_mu_values": ",".join(
                    sorted({f"{float(mu):g}" for mu in nearby_group["fedprox_mu"].dropna().tolist()})
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def summarize_edge_attack_modes(canonical_runs: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """Summarize Edge attack-mode damage and stealth from raw canonical runs."""
    attack_runs = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.3)
    attack_runs = attack_runs[attack_runs["effective_attack_mode"].isin(EDGE_ATTACK_MODES)].copy()

    columns = [
        "aggregation",
        "attack_mode",
        "n_runs",
        "n_seeds",
        "macro_f1_mean",
        "macro_f1_std",
        "l2_to_benign_mean_mean",
        "cos_to_benign_mean_mean",
        "pairwise_cosine_mean_mean",
        "update_norm_mean_mean",
        "final_round_mean",
    ]
    rows: list[dict[str, object]] = []
    for (aggregation, attack_mode), group in attack_runs.groupby(
        ["aggregation", "effective_attack_mode"], sort=False
    ):
        metrics_rows: list[dict[str, float]] = []
        for run_id in group["run_id"]:
            metrics = load_final_metrics(runs_dir / str(run_id))
            if metrics is not None:
                metrics_rows.append(metrics)
        if not metrics_rows:
            continue

        metrics_frame = pd.DataFrame(metrics_rows)
        rows.append(
            {
                "aggregation": aggregation,
                "attack_mode": attack_mode,
                "n_runs": len(metrics_frame),
                "n_seeds": int(group["seed"].nunique()),
                "macro_f1_mean": round(float(metrics_frame["macro_f1"].mean()), 12),
                "macro_f1_std": round(float(metrics_frame["macro_f1"].std(ddof=0)), 12),
                "l2_to_benign_mean_mean": round(float(metrics_frame["l2_to_benign_mean"].mean()), 12),
                "cos_to_benign_mean_mean": round(float(metrics_frame["cos_to_benign_mean"].mean()), 12),
                "pairwise_cosine_mean_mean": round(float(metrics_frame["pairwise_cosine_mean"].mean()), 12),
                "update_norm_mean_mean": round(float(metrics_frame["update_norm_mean"].mean()), 12),
                "final_round_mean": round(float(metrics_frame["round"].mean()), 12),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_edge_per_class_profiles(canonical_runs: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """Build per-class holdout F1 profiles for Edge 30% attack-mode runs."""
    attack_runs = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.3)
    attack_runs = attack_runs[attack_runs["effective_attack_mode"].isin(EDGE_ATTACK_MODES)].copy()

    columns = [
        "aggregation",
        "attack_mode",
        "class_name",
        "per_class_f1_mean",
        "n_runs",
        "n_seeds",
    ]
    rows: list[dict[str, object]] = []
    for (aggregation, attack_mode), group in attack_runs.groupby(
        ["aggregation", "effective_attack_mode"], sort=False
    ):
        per_run_profiles: list[dict[str, float]] = []
        for run_id in group["run_id"]:
            profile = load_run_per_class_means(runs_dir / str(run_id))
            if profile:
                per_run_profiles.append(profile)
        if not per_run_profiles:
            continue

        profile_frame = pd.DataFrame(per_run_profiles)
        for class_name in profile_frame.columns:
            rows.append(
                {
                    "aggregation": aggregation,
                    "attack_mode": attack_mode,
                    "class_name": class_name,
                    "per_class_f1_mean": round(float(profile_frame[class_name].mean()), 12),
                    "n_runs": len(profile_frame),
                    "n_seeds": int(group["seed"].nunique()),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def build_edge_baseline_attack_deltas(canonical_runs: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """Compute exact Edge baseline-vs-attack per-class deltas."""
    baseline_runs = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.0)
    baseline_runs = baseline_runs[baseline_runs["effective_attack_mode"] == "none"].copy()
    attack_runs = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.3)
    attack_runs = attack_runs[attack_runs["effective_attack_mode"].isin(EDGE_ATTACK_MODES)].copy()

    columns = [
        "aggregation",
        "attack_mode",
        "class_name",
        "baseline_per_class_f1",
        "attack_per_class_f1",
        "delta",
        "baseline_runs",
        "attack_runs",
    ]
    rows: list[dict[str, object]] = []
    for aggregation in CORE_AGGREGATIONS:
        baseline_group = baseline_runs[baseline_runs["aggregation"] == aggregation]
        baseline_profiles = [
            load_run_per_class_means(runs_dir / str(run_id))
            for run_id in baseline_group["run_id"].tolist()
        ]
        baseline_profiles = [profile for profile in baseline_profiles if profile]
        if not baseline_profiles:
            continue
        baseline_frame = pd.DataFrame(baseline_profiles)

        for attack_mode, attack_group in attack_runs[attack_runs["aggregation"] == aggregation].groupby(
            "effective_attack_mode", sort=False
        ):
            attack_profiles = [
                load_run_per_class_means(runs_dir / str(run_id))
                for run_id in attack_group["run_id"].tolist()
            ]
            attack_profiles = [profile for profile in attack_profiles if profile]
            if not attack_profiles:
                continue
            attack_frame = pd.DataFrame(attack_profiles)
            class_names = sorted(set(baseline_frame.columns).union(set(attack_frame.columns)))
            for class_name in class_names:
                baseline_value = float(baseline_frame[class_name].mean()) if class_name in baseline_frame else float("nan")
                attack_value = float(attack_frame[class_name].mean()) if class_name in attack_frame else float("nan")
                rows.append(
                    {
                        "aggregation": aggregation,
                        "attack_mode": attack_mode,
                        "class_name": class_name,
                        "baseline_per_class_f1": round(baseline_value, 12),
                        "attack_per_class_f1": round(attack_value, 12),
                        "delta": round(attack_value - baseline_value, 12),
                        "baseline_runs": len(baseline_frame),
                        "attack_runs": len(attack_frame),
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def build_edge_practitioner_walkthrough_candidates(
    canonical_runs: pd.DataFrame,
    runs_dir: Path,
    *,
    macro_floor: float = 0.45,
    l2_ceiling: float = 3.0,
    cos_floor: float = 0.98,
    baseline_floor: float = 0.5,
    blind_threshold: float = 0.1,
) -> pd.DataFrame:
    """Find exact Edge runs that look deployable in aggregate while hiding a class failure."""
    baseline_runs = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.0)
    baseline_runs = baseline_runs[baseline_runs["effective_attack_mode"] == "none"].copy()
    attack_runs = _filter_edge_attack_family(canonical_runs, adversary_fraction=0.3)
    attack_runs = attack_runs[attack_runs["effective_attack_mode"].isin(EDGE_ATTACK_MODES)].copy()

    baseline_means: dict[str, dict[str, float]] = {}
    for aggregation, group in baseline_runs.groupby("aggregation", sort=False):
        profiles = [
            load_run_per_class_means(runs_dir / str(run_id))
            for run_id in group["run_id"].tolist()
        ]
        profiles = [profile for profile in profiles if profile]
        if not profiles:
            continue
        frame = pd.DataFrame(profiles)
        baseline_means[aggregation] = {column: float(frame[column].mean()) for column in frame.columns}

    columns = [
        "run_id",
        "aggregation",
        "attack_mode",
        "seed",
        "macro_f1",
        "l2_to_benign_mean",
        "cos_to_benign_mean",
        "pairwise_cosine_mean",
        "update_norm_mean",
        "class_name",
        "attack_per_class_f1",
        "baseline_per_class_f1",
        "delta",
        "selection_rule",
    ]
    rows: list[dict[str, object]] = []

    for attack_row in attack_runs.itertuples(index=False):
        baseline_profile = baseline_means.get(str(attack_row.aggregation))
        if baseline_profile is None:
            continue

        metrics = load_final_metrics(runs_dir / str(attack_row.run_id))
        profile = load_run_per_class_means(runs_dir / str(attack_row.run_id))
        if metrics is None or not profile:
            continue

        if (
            metrics["macro_f1"] < macro_floor
            or metrics["l2_to_benign_mean"] > l2_ceiling
            or metrics["cos_to_benign_mean"] < cos_floor
        ):
            continue

        for class_name, attack_value in profile.items():
            baseline_value = baseline_profile.get(class_name)
            if baseline_value is None:
                continue
            if baseline_value < baseline_floor or attack_value > blind_threshold:
                continue
            rows.append(
                {
                    "run_id": str(attack_row.run_id),
                    "aggregation": str(attack_row.aggregation),
                    "attack_mode": str(attack_row.effective_attack_mode),
                    "seed": int(attack_row.seed),
                    "macro_f1": round(metrics["macro_f1"], 12),
                    "l2_to_benign_mean": round(metrics["l2_to_benign_mean"], 12),
                    "cos_to_benign_mean": round(metrics["cos_to_benign_mean"], 12),
                    "pairwise_cosine_mean": round(metrics["pairwise_cosine_mean"], 12),
                    "update_norm_mean": round(metrics["update_norm_mean"], 12),
                    "class_name": class_name,
                    "attack_per_class_f1": round(float(attack_value), 12),
                    "baseline_per_class_f1": round(float(baseline_value), 12),
                    "delta": round(float(attack_value - baseline_value), 12),
                    "selection_rule": (
                        f"macro_f1>={macro_floor:g}, l2<={l2_ceiling:g}, cos>={cos_floor:g}, "
                        f"baseline>={baseline_floor:g}, attack<={blind_threshold:g}"
                    ),
                }
            )

    candidates = pd.DataFrame(rows, columns=columns)
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["macro_f1", "attack_per_class_f1", "baseline_per_class_f1", "l2_to_benign_mean"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def build_raid_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    """Build the RAID-safe cross-dataset backbone summary."""
    benign = coverage[
        (coverage["slice"] == "baseline_iid")
        & (coverage["dataset"].isin(CORE_DATASETS))
        & (coverage["aggregation"].isin(CORE_AGGREGATIONS))
    ].copy()
    attack = coverage[
        (coverage["slice"] == "attack_alpha0.5_collapsed")
        & (coverage["dataset"].isin(CORE_DATASETS))
        & (coverage["aggregation"].isin(CORE_AGGREGATIONS))
    ].copy()

    benign = benign[["dataset", "aggregation", "metric_mean"]].rename(
        columns={"metric_mean": "benign_macro_f1"}
    )
    attack30 = attack[attack["adv_percent"] == 30.0][["dataset", "aggregation", "metric_mean"]].rename(
        columns={"metric_mean": "attack30_macro_f1"}
    )
    attack_pivot = attack.pivot_table(
        index=["dataset", "adv_percent"],
        columns="aggregation",
        values="metric_mean",
    )

    rows: list[dict[str, object]] = []
    for dataset in CORE_DATASETS:
        if (dataset, 30.0) not in attack_pivot.index:
            continue
        fedavg30 = float(attack_pivot.loc[(dataset, 30.0), "fedavg"])
        for aggregation in CORE_AGGREGATIONS:
            benign_row = benign[
                (benign["dataset"] == dataset) & (benign["aggregation"] == aggregation)
            ].iloc[0]
            attack30_row = attack30[
                (attack30["dataset"] == dataset) & (attack30["aggregation"] == aggregation)
            ].iloc[0]
            attack_wins = int(
                sum(
                    float(attack_pivot.loc[(dataset, adv_percent), aggregation])
                    > float(attack_pivot.loc[(dataset, adv_percent), "fedavg"])
                    for adv_percent in (10.0, 20.0, 30.0)
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "aggregation": aggregation,
                    "benign_macro_f1": float(benign_row["benign_macro_f1"]),
                    "attack30_macro_f1": float(attack30_row["attack30_macro_f1"]),
                    "delta_vs_fedavg_attack30": float(attack30_row["attack30_macro_f1"]) - fedavg30,
                    "attack_wins_vs_fedavg": attack_wins,
                }
            )

    return pd.DataFrame(rows)


def build_publication_claim_matrix(
    raid_summary: pd.DataFrame,
    edge_attack_summary: pd.DataFrame,
    edge_baseline_gaps: pd.DataFrame,
    edge_baseline_attack_deltas: pd.DataFrame | None = None,
    *,
    heterogeneity_supported: bool = False,
) -> pd.DataFrame:
    """Build the safe-claims matrix for the RAID and CCS tracks."""
    median_rows = raid_summary[raid_summary["aggregation"] == "median"]
    krum_rows = raid_summary[raid_summary["aggregation"] == "krum"]
    missing_edge_baselines = not edge_baseline_gaps.empty

    median_backbone_supported = (
        len(median_rows) == len(CORE_DATASETS)
        and bool((median_rows["delta_vs_fedavg_attack30"] > 0).all())
        and bool((median_rows["benign_macro_f1"] >= 0.45).all())
    )
    krum_dataset_dependent = (
        len(krum_rows) == len(CORE_DATASETS)
        and bool((krum_rows["delta_vs_fedavg_attack30"] > 0).sum() >= 2)
        and bool((krum_rows["delta_vs_fedavg_attack30"] < 0).sum() >= 1)
    )

    sign_flip = edge_attack_summary[edge_attack_summary["attack_mode"] == "sign_flip_topk"]
    targeted = edge_attack_summary[edge_attack_summary["attack_mode"] == "targeted_label"]
    label_flip = edge_attack_summary[edge_attack_summary["attack_mode"] == "label_flip"]
    per_class_collapse_supported = edge_baseline_attack_deltas is not None and not edge_baseline_attack_deltas.empty
    attack_mode_damage_supported = (
        not sign_flip.empty
        and not targeted.empty
        and not label_flip.empty
        and float(sign_flip["l2_to_benign_mean_mean"].mean())
        > float(targeted["l2_to_benign_mean_mean"].mean())
        and float(sign_flip["l2_to_benign_mean_mean"].mean())
        > float(label_flip["l2_to_benign_mean_mean"].mean())
    )

    rows = [
        {
            "claim_id": "RAID-R1",
            "venue": "RAID",
            "claim_state": "main_safe" if median_backbone_supported else "secondary_safe",
            "claim_text": (
                "Coordinate-wise median is the strongest RAID headline defense: it preserves benign IID "
                "performance and consistently beats FedAvg under 30% adversaries across CIC-IDS2017, "
                "Edge-IIoTset, and UNSW-NB15."
            ),
            "blocker": "",
        },
        {
            "claim_id": "RAID-R2",
            "venue": "RAID",
            "claim_state": "main_safe" if krum_dataset_dependent else "secondary_safe",
            "claim_text": (
                "Krum is not a universal winner: it improves resilience on Edge-IIoTset and UNSW-NB15, "
                "but underperforms FedAvg on CIC-IDS2017."
            ),
            "blocker": "",
        },
        {
            "claim_id": "RAID-R3",
            "venue": "RAID",
            "claim_state": "main_safe" if heterogeneity_supported else "secondary_safe",
            "claim_text": "FedAvg heterogeneity response is dataset-dependent across the three core datasets.",
            "blocker": "" if heterogeneity_supported else "Need a supported heterogeneity claim ledger entry.",
        },
        {
            "claim_id": "RAID-R4",
            "venue": "RAID",
            "claim_state": "secondary_safe",
            "claim_text": (
                "Per-class degradation risk is dataset-dependent and should be framed as secondary analysis, "
                "not as a universal phantom-robustness headline."
            ),
            "blocker": "Use UNSW as the positive case and CIC as the counterexample.",
        },
        {
            "claim_id": "RAID-R5",
            "venue": "RAID",
            "claim_state": "forbidden",
            "claim_text": "Bulyan dominates the high-adversary regime across the core datasets.",
            "blocker": "Missing confirmatory 20% and 30% Bulyan cells.",
        },
        {
            "claim_id": "CCS-C1",
            "venue": "CCS",
            "claim_state": "main_safe" if attack_mode_damage_supported else "secondary_safe",
            "claim_text": (
                "On Edge-IIoTset, attack mode materially changes both damage and stealth at 30% adversaries."
            ),
            "blocker": "",
        },
        {
            "claim_id": "CCS-C2",
            "venue": "CCS",
            "claim_state": "main_safe" if attack_mode_damage_supported else "secondary_safe",
            "claim_text": (
                "sign_flip_topk is the highest-magnitude Edge poisoning mode, while label_flip and "
                "targeted_label remain markedly stealthier."
            ),
            "blocker": "",
        },
        {
            "claim_id": "CCS-C3",
            "venue": "CCS",
            "claim_state": (
                "main_safe" if (not missing_edge_baselines and per_class_collapse_supported) else "secondary_safe"
            ),
            "claim_text": (
                "Per-class baseline-vs-attack deltas on Edge reveal class-specific collapse that should accompany macro-F1 in the CCS paper."
                if (not missing_edge_baselines and per_class_collapse_supported)
                else "Per-class baseline-vs-attack collapse on Edge can only be elevated to a headline claim after the exact clean alpha=0.5 baseline family is regenerated."
            ),
            "blocker": (
                ""
                if (not missing_edge_baselines and per_class_collapse_supported)
                else "Exact Edge clean baselines for fedavg/krum/median at alpha=0.5, adv=0, mu=0.0 are missing."
            ),
        },
        {
            "claim_id": "CCS-C4",
            "venue": "CCS",
            "claim_state": "forbidden",
            "claim_text": "The Edge attack-taxonomy conclusions generalize universally across all IDS datasets.",
            "blocker": "CCS scope is Edge-centered, with CIC and UNSW used only as validation checks.",
        },
        {
            "claim_id": "CCS-C5",
            "venue": "CCS",
            "claim_state": "forbidden",
            "claim_text": "Bulyan is the definitive main defense for the CCS paper.",
            "blocker": "Current Bulyan evidence is incomplete and timing out on the cluster.",
        },
    ]

    return pd.DataFrame(rows)


def render_claim_matrix_markdown(claim_matrix: pd.DataFrame) -> str:
    """Render the claim matrix as Markdown."""
    lines = [
        "# Publication Claim Matrix",
        "",
        "| Claim ID | Venue | State | Claim | Blocker |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in claim_matrix.itertuples(index=False):
        lines.append(
            f"| {row.claim_id} | {row.venue} | {row.claim_state} | {row.claim_text} | {row.blocker} |"
        )
    return "\n".join(lines) + "\n"


def render_raid_report(raid_summary: pd.DataFrame, heterogeneity_supported: bool) -> str:
    """Render the RAID evidence summary."""
    lines = [
        "# RAID Readiness Report",
        "",
        "## Core Backbone",
        "",
        "The paper-safe RAID backbone is cross-dataset and should stay centered on FedAvg, Krum, and Median.",
        "",
        "| Dataset | Aggregation | Benign IID macro-F1 | 30% attack macro-F1 | Delta vs FedAvg at 30% | Attack wins vs FedAvg |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in raid_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | {row.aggregation} | {row.benign_macro_f1:.3f} | "
            f"{row.attack30_macro_f1:.3f} | {row.delta_vs_fedavg_attack30:.3f} | {row.attack_wins_vs_fedavg} / 3 |"
        )

    lines.extend(
        [
            "",
            "## Safe Framing",
            "",
            "- Main-safe: Median is the most consistent defense across the three datasets.",
            "- Main-safe: Krum is dataset-dependent and should not be framed as universally better than FedAvg.",
            (
                "- Main-safe: Heterogeneity response is supported and can remain in scope."
                if heterogeneity_supported
                else "- Secondary-safe: Heterogeneity remains useful context but should stay behind the main backbone."
            ),
            "- Secondary-safe: Per-class masking should be positioned as a bounded, dataset-dependent result.",
            "- Forbidden: Any headline that depends on missing Bulyan 20%/30% confirmatory cells.",
            "",
        ]
    )

    return "\n".join(lines)


def render_edge_attack_report(edge_attack_summary: pd.DataFrame, edge_baseline_gaps: pd.DataFrame) -> str:
    """Render the CCS Edge attack-mode report."""
    lines = [
        "# CCS Edge Attack-Mode Report",
        "",
        "## Canonical 30% Edge Attack Summary",
        "",
        "| Aggregation | Attack mode | Runs | Macro-F1 | L2 to benign | Cosine to benign | Pairwise cosine | Update norm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in edge_attack_summary.sort_values(["aggregation", "attack_mode"]).itertuples(index=False):
        lines.append(
            f"| {row.aggregation} | {row.attack_mode} | {row.n_runs} | {row.macro_f1_mean:.3f} | "
            f"{row.l2_to_benign_mean_mean:.3f} | {row.cos_to_benign_mean_mean:.3f} | "
            f"{row.pairwise_cosine_mean_mean:.3f} | {row.update_norm_mean_mean:.3f} |"
        )

    lines.extend(["", "## Readiness Notes", ""])
    lines.append(
        "- Main-safe: attack mode materially changes both measured damage and stealth on Edge-IIoTset."
    )
    lines.append(
        "- Main-safe: sign_flip_topk is the conspicuous high-magnitude mode; label_flip and targeted_label stay much closer to benign update geometry."
    )
    if edge_baseline_gaps.empty:
        lines.append(
            "- Main-safe: exact Edge clean baselines are available for baseline-vs-attack per-class comparisons."
        )
    else:
        lines.append(
            "- Secondary-safe: baseline-vs-attack per-class collapse remains gated because the exact clean Edge alpha=0.5 family is missing."
        )
    lines.append("")
    return "\n".join(lines)


def render_edge_per_class_report(
    edge_per_class_profiles: pd.DataFrame,
    edge_baseline_gaps: pd.DataFrame,
    edge_baseline_attack_deltas: pd.DataFrame,
) -> str:
    """Render the per-class Edge profile report."""
    if not edge_baseline_attack_deltas.empty:
        lines = [
            "# CCS Edge Per-Class Delta Report",
            "",
            "The table below lists the largest baseline-vs-attack per-class collapses from the exact Edge alpha=0.5 baseline family.",
            "",
            "| Aggregation | Attack mode | Largest negative deltas |",
            "| --- | --- | --- |",
        ]
        grouped = edge_baseline_attack_deltas.sort_values(
            ["aggregation", "attack_mode", "delta", "class_name"]
        ).groupby(["aggregation", "attack_mode"], sort=False)
        for (aggregation, attack_mode), group in grouped:
            weakest = ", ".join(
                f"{row.class_name} ({row.delta:.3f})" for row in group.head(5).itertuples(index=False)
            )
            lines.append(f"| {aggregation} | {attack_mode} | {weakest} |")
    else:
        lines = [
            "# CCS Edge Per-Class Profile Report",
            "",
            "The table below lists the weakest recovered classes per aggregation and attack mode from the exact 30% Edge attack family.",
            "",
            "| Aggregation | Attack mode | Lowest-F1 classes |",
            "| --- | --- | --- |",
        ]

        if edge_per_class_profiles.empty:
            lines.append("| n/a | n/a | No recoverable per-class profiles found. |")
        else:
            grouped = edge_per_class_profiles.sort_values(
                ["aggregation", "attack_mode", "per_class_f1_mean", "class_name"]
            ).groupby(["aggregation", "attack_mode"], sort=False)
            for (aggregation, attack_mode), group in grouped:
                weakest = ", ".join(
                    f"{row.class_name} ({row.per_class_f1_mean:.3f})"
                    for row in group.head(5).itertuples(index=False)
                )
                lines.append(f"| {aggregation} | {attack_mode} | {weakest} |")

    lines.extend(["", "## Gating Status", ""])
    if edge_baseline_gaps.empty and not edge_baseline_attack_deltas.empty:
        lines.append(
            "Exact clean Edge baselines are present, so this profile can be promoted into a baseline-vs-attack collapse analysis."
        )
    elif edge_baseline_gaps.empty:
        lines.append(
            "Exact clean Edge baselines are present, but the baseline-vs-attack delta extraction did not recover any rows."
        )
    else:
        lines.append(
            "Do not claim baseline-vs-attack per-class collapse yet. Regenerate the exact clean Edge alpha=0.5 baseline family first."
        )
    lines.append("")
    return "\n".join(lines)


def render_practitioner_walkthrough_report(practitioner_candidates: pd.DataFrame) -> str:
    """Render the practitioner walkthrough candidates."""
    lines = [
        "# Edge Practitioner Walkthrough Report",
        "",
        "These runs satisfy a minimal green-light heuristic: aggregate macro-F1 remains at or above `0.45`, "
        "update geometry stays close to benign (`l2_to_benign_mean <= 3.0`, `cos_to_benign_mean >= 0.98`), "
        "and at least one high-baseline class (`>= 0.5` clean F1) falls to `<= 0.1`.",
        "",
        "| Aggregation | Attack mode | Seed | Macro-F1 | L2 to benign | Cosine to benign | Blind class | Attack F1 | Clean baseline F1 | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    if practitioner_candidates.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | No candidate run met the heuristic. | n/a | n/a | n/a |")
    else:
        for row in practitioner_candidates.head(10).itertuples(index=False):
            lines.append(
                f"| {row.aggregation} | {row.attack_mode} | {row.seed} | {row.macro_f1:.3f} | "
                f"{row.l2_to_benign_mean:.3f} | {row.cos_to_benign_mean:.3f} | {row.class_name} | "
                f"{row.attack_per_class_f1:.3f} | {row.baseline_per_class_f1:.3f} | {row.delta:.3f} |"
            )

    lines.extend(["", "## Interpretation", ""])
    if practitioner_candidates.empty:
        lines.append(
            "The exact Edge corpus does not yet support a concrete practitioner green-light example under the current heuristic."
        )
    else:
        top = practitioner_candidates.iloc[0]
        lines.append(
            "The CCS-safe lesson is an existence claim, not a universality claim: a practitioner can observe "
            f"`{top.macro_f1:.3f}` macro-F1 with benign-looking geometry and still miss {top.class_name} almost completely."
        )
        lines.append(
            f"Top candidate: `{top.run_id}` with {top.class_name} dropping from `{top.baseline_per_class_f1:.3f}` to `{top.attack_per_class_f1:.3f}`."
        )
    lines.append("")
    return "\n".join(lines)


def render_sharper_security_lesson(
    raid_summary: pd.DataFrame,
    edge_attack_summary: pd.DataFrame,
    edge_baseline_attack_deltas: pd.DataFrame,
    practitioner_candidates: pd.DataFrame,
) -> str:
    """Render the cross-track security lesson."""
    lines = [
        "# Sharper Security Lesson",
        "",
        "Robustness in federated IDS is not a single scalar property. The cross-dataset RAID result and the Edge-centered CCS result become coherent only when framed together:",
        "",
        "1. aggregation choice is a first-order security control because FedAvg fails sharply under adversarial participation while Median remains consistently stronger across the three core datasets",
        "2. attack mode changes what 'robust' means, because robust aggregation helps most against conspicuous magnitude attacks and much less against stealthier semantic attacks",
        "3. aggregate macro-F1 is insufficient as a deployment gate, because a run can retain plausible aggregate quality and benign-looking update geometry while a high-baseline class collapses",
        "",
        "## Vital RAID Results",
        "",
    ]

    median_rows = raid_summary[raid_summary["aggregation"] == "median"]
    krum_rows = raid_summary[raid_summary["aggregation"] == "krum"]
    for row in median_rows.itertuples(index=False):
        lines.append(
            f"- {row.dataset_label}: Median benign `{row.benign_macro_f1:.3f}`, 30% attack `{row.attack30_macro_f1:.3f}`, delta vs FedAvg `{row.delta_vs_fedavg_attack30:.3f}`"
        )
    for row in krum_rows.itertuples(index=False):
        lines.append(
            f"- {row.dataset_label}: Krum benign `{row.benign_macro_f1:.3f}`, 30% attack `{row.attack30_macro_f1:.3f}`, delta vs FedAvg `{row.delta_vs_fedavg_attack30:.3f}`"
        )

    lines.extend(["", "## Vital CCS Results", ""])
    for row in edge_attack_summary.sort_values(["aggregation", "attack_mode"]).itertuples(index=False):
        lines.append(
            f"- {row.aggregation} / {row.attack_mode}: macro-F1 `{row.macro_f1_mean:.3f}`, "
            f"`l2_to_benign_mean` `{row.l2_to_benign_mean_mean:.3f}`, "
            f"`cos_to_benign_mean` `{row.cos_to_benign_mean_mean:.3f}`"
        )

    lines.extend(["", "## Largest Edge Per-Class Deltas", ""])
    for aggregation in CORE_AGGREGATIONS:
        subset = edge_baseline_attack_deltas[edge_baseline_attack_deltas["aggregation"] == aggregation]
        if subset.empty:
            continue
        worst = subset.sort_values(["delta", "attack_mode", "class_name"]).head(3)
        summary = ", ".join(
            f"{row.attack_mode}:{row.class_name} ({row.delta:.3f})" for row in worst.itertuples(index=False)
        )
        lines.append(f"- {aggregation}: {summary}")

    lines.extend(["", "## Practitioner Walkthrough Hook", ""])
    if practitioner_candidates.empty:
        lines.append(
            "No exact run currently meets the minimal green-light heuristic, so the CCS story should stay at the dataset-level attack-taxonomy framing."
        )
    else:
        top = practitioner_candidates.iloc[0]
        lines.append(
            f"- existence example: `{top.run_id}` keeps macro-F1 at `{top.macro_f1:.3f}` with "
            f"`l2_to_benign_mean` `{top.l2_to_benign_mean:.3f}` and `cos_to_benign_mean` `{top.cos_to_benign_mean:.3f}`, "
            f"yet {top.class_name} falls from `{top.baseline_per_class_f1:.3f}` to `{top.attack_per_class_f1:.3f}`"
        )
        lines.append(
            "- CCS wording should present this as a minimal proof-of-concept practitioner failure, not as a universal statement about every poisoning run."
        )

    lines.append("")
    return "\n".join(lines)


def render_forbidden_claims(claim_matrix: pd.DataFrame) -> str:
    """Render the forbidden-claims checklist."""
    forbidden = claim_matrix[claim_matrix["claim_state"] == "forbidden"]
    lines = ["# Forbidden Claims", ""]
    for venue, group in forbidden.groupby("venue", sort=False):
        lines.append(f"## {venue}")
        lines.append("")
        for row in group.itertuples(index=False):
            lines.append(f"- {row.claim_text}")
            lines.append(f"  Blocker: {row.blocker}")
        lines.append("")
    return "\n".join(lines)


def build_artifact_manifest(output_dir: Path, source_paths: list[Path]) -> pd.DataFrame:
    """Build a manifest describing the generated publication artifacts."""
    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.iterdir()):
        rows.append(
            {
                "artifact_path": str(path),
                "artifact_name": path.name,
                "source_paths": ";".join(str(source_path) for source_path in source_paths),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate RAID and CCS publication bundles.")
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=Path("reports") / "runs_census" / "coverage_confirmatory.csv",
        help="Path to coverage_confirmatory.csv.",
    )
    parser.add_argument(
        "--claim-ledger-path",
        type=Path,
        default=Path("reports") / "runs_census" / "claim_ledger.csv",
        help="Path to claim_ledger.csv.",
    )
    parser.add_argument(
        "--runs-registry-path",
        type=Path,
        default=Path("reports") / "runs_census" / "runs_registry.csv",
        help="Path to runs_registry.csv.",
    )
    parser.add_argument(
        "--dedup-map-path",
        type=Path,
        default=Path("reports") / "runs_census" / "runs_dedup_map.csv",
        help="Path to runs_dedup_map.csv.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Path to raw run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "publication",
        help="Directory for generated publication artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    coverage = pd.read_csv(args.coverage_path)
    claim_ledger = pd.read_csv(args.claim_ledger_path)
    heterogeneity_supported = bool(
        (
            claim_ledger.loc[claim_ledger["claim_id"] == "C2", "support_level"].fillna("").eq("supported")
        ).any()
    )
    canonical_runs = load_canonical_runs(args.runs_registry_path, args.dedup_map_path)

    raid_summary = build_raid_summary(coverage)
    edge_baseline_gaps = find_missing_edge_clean_baselines(canonical_runs)
    edge_attack_summary = summarize_edge_attack_modes(canonical_runs, args.runs_dir)
    edge_per_class_profiles = build_edge_per_class_profiles(canonical_runs, args.runs_dir)
    edge_baseline_attack_deltas = build_edge_baseline_attack_deltas(canonical_runs, args.runs_dir)
    practitioner_candidates = build_edge_practitioner_walkthrough_candidates(
        canonical_runs, args.runs_dir
    )
    claim_matrix = build_publication_claim_matrix(
        raid_summary=raid_summary,
        edge_attack_summary=edge_attack_summary,
        edge_baseline_gaps=edge_baseline_gaps,
        edge_baseline_attack_deltas=edge_baseline_attack_deltas,
        heterogeneity_supported=heterogeneity_supported,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_specs = {
        "publication_claim_matrix.csv": claim_matrix,
        "raid_backbone_summary.csv": raid_summary,
        "ccs_edge_attack_mode_summary.csv": edge_attack_summary,
        "ccs_edge_baseline_gap.csv": edge_baseline_gaps,
        "ccs_edge_baseline_attack_delta.csv": edge_baseline_attack_deltas,
        "edge_practitioner_walkthrough_candidates.csv": practitioner_candidates,
        "ccs_edge_per_class_profiles.csv": edge_per_class_profiles,
    }

    for filename, frame in output_specs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    markdown_outputs = {
        "publication_claim_matrix.md": render_claim_matrix_markdown(claim_matrix),
        "raid_readiness_report.md": render_raid_report(raid_summary, heterogeneity_supported),
        "ccs_edge_attack_mode_report.md": render_edge_attack_report(
            edge_attack_summary, edge_baseline_gaps
        ),
        "ccs_edge_per_class_report.md": render_edge_per_class_report(
            edge_per_class_profiles, edge_baseline_gaps, edge_baseline_attack_deltas
        ),
        "edge_practitioner_walkthrough_report.md": render_practitioner_walkthrough_report(
            practitioner_candidates
        ),
        "sharper_security_lesson.md": render_sharper_security_lesson(
            raid_summary,
            edge_attack_summary,
            edge_baseline_attack_deltas,
            practitioner_candidates,
        ),
        "forbidden_claims.md": render_forbidden_claims(claim_matrix),
    }

    for filename, contents in markdown_outputs.items():
        (args.output_dir / filename).write_text(contents)

    manifest = build_artifact_manifest(
        args.output_dir,
        [
            args.coverage_path,
            args.claim_ledger_path,
            args.runs_registry_path,
            args.dedup_map_path,
            args.runs_dir,
        ],
    )
    manifest.to_csv(args.output_dir / "publication_artifact_manifest.csv", index=False)

    for path in sorted(args.output_dir.iterdir()):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
