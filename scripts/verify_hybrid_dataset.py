#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_COLS = ["source_dataset", "attack_class", "attack_label_original"]


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    message: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_hybrid_dataset_csv(
    csv_path: Path,
    *,
    chunksize: int = 500_000,
    duplicate_sample_rows: int = 200_000,
    seed: int = 42,
) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    issues: list[AuditIssue] = []
    rng = np.random.default_rng(seed)

    total_rows = 0
    sources: set[str] = set()
    classes: set[int] = set()

    counts_by_source: dict[str, int] = {}
    counts_by_class: dict[int, int] = {}
    counts_by_source_class: dict[str, dict[int, int]] = {}

    feature_cols: list[str] | None = None
    feature_count = 0
    feature_sum: np.ndarray | None = None
    feature_sumsq: np.ndarray | None = None
    feature_min: np.ndarray | None = None
    feature_max: np.ndarray | None = None
    feature_zero_count: np.ndarray | None = None
    nan_count = 0
    inf_count = 0
    all_zero_feature_rows = 0

    dup_hashes: set[int] = set()
    dup_seen = 0
    dup_total_sampled = 0

    reader = pd.read_csv(csv_path, compression="infer", chunksize=chunksize, low_memory=False)
    for chunk in reader:
        total_rows += len(chunk)

        missing_meta = [c for c in METADATA_COLS if c not in chunk.columns]
        if missing_meta:
            raise ValueError(f"Missing required columns: {missing_meta}")

        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in METADATA_COLS]
            feature_count = len(feature_cols)
            feature_sum = np.zeros(feature_count, dtype=np.float64)
            feature_sumsq = np.zeros(feature_count, dtype=np.float64)
            feature_min = np.full(feature_count, np.inf, dtype=np.float64)
            feature_max = np.full(feature_count, -np.inf, dtype=np.float64)
            feature_zero_count = np.zeros(feature_count, dtype=np.int64)

        src_counts = chunk["source_dataset"].astype(str).value_counts()
        for src, cnt in src_counts.items():
            sources.add(src)
            counts_by_source[src] = counts_by_source.get(src, 0) + int(cnt)

        cls_series = pd.to_numeric(chunk["attack_class"], errors="coerce")
        if cls_series.isna().any():
            issues.append(
                AuditIssue(
                    severity="error",
                    message="Non-numeric values found in attack_class",
                )
            )
        chunk_classes = cls_series.fillna(-1).astype(int)
        cls_counts = chunk_classes.value_counts()
        for cls, cnt in cls_counts.items():
            if cls >= 0:
                classes.add(int(cls))
                counts_by_class[int(cls)] = counts_by_class.get(int(cls), 0) + int(cnt)

        grouped = chunk.groupby(["source_dataset", "attack_class"]).size()
        for (src, cls), cnt in grouped.items():
            src = str(src)
            cls_i = int(cls)
            counts_by_source_class.setdefault(src, {})
            counts_by_source_class[src][cls_i] = counts_by_source_class[src].get(cls_i, 0) + int(cnt)

        features = chunk[feature_cols].to_numpy(dtype=np.float64, copy=False)
        nan_count += int(np.isnan(features).sum())
        inf_count += int(np.isinf(features).sum())

        features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        feature_sum += features.sum(axis=0)
        feature_sumsq += np.square(features).sum(axis=0)
        feature_min = np.minimum(feature_min, features.min(axis=0))
        feature_max = np.maximum(feature_max, features.max(axis=0))
        feature_zero_count += (features == 0).sum(axis=0).astype(np.int64)
        all_zero_feature_rows += int((features == 0).all(axis=1).sum())

        if dup_total_sampled < duplicate_sample_rows:
            remaining = duplicate_sample_rows - dup_total_sampled
            take = min(remaining, len(chunk))
            if take > 0:
                sample_idx = rng.choice(len(chunk), size=take, replace=False)
                sample = chunk.iloc[sample_idx]
                sample_hashes = pd.util.hash_pandas_object(sample, index=False).astype(np.uint64).to_numpy()
                for h in sample_hashes.tolist():
                    if int(h) in dup_hashes:
                        dup_seen += 1
                    dup_hashes.add(int(h))
                dup_total_sampled += take

    if feature_cols is None or feature_sum is None or feature_sumsq is None:
        raise ValueError("No rows read from dataset")

    if nan_count > 0:
        issues.append(AuditIssue(severity="error", message=f"Found NaNs in feature matrix: {nan_count}"))
    if inf_count > 0:
        issues.append(AuditIssue(severity="error", message=f"Found infinities in feature matrix: {inf_count}"))

    class_min = min(classes) if classes else None
    class_max = max(classes) if classes else None
    if class_min is None or class_max is None:
        issues.append(AuditIssue(severity="error", message="No attack_class values found"))
    else:
        if class_min < 0 or class_max > 6:
            issues.append(
                AuditIssue(
                    severity="error",
                    message=f"attack_class out of expected range 0..6 (min={class_min}, max={class_max})",
                )
            )

    counts_by_class_str = {str(k): v for k, v in sorted(counts_by_class.items())}
    counts_by_source_class_str = {
        src: {str(k): v for k, v in sorted(inner.items())}
        for src, inner in sorted(counts_by_source_class.items())
    }

    feature_mean = feature_sum / float(total_rows)
    feature_var = np.maximum(feature_sumsq / float(total_rows) - np.square(feature_mean), 0.0)
    feature_std = np.sqrt(feature_var)
    feature_zero_frac = feature_zero_count / float(total_rows)

    feature_stats = {
        name: {
            "min": float(feature_min[i]),
            "max": float(feature_max[i]),
            "mean": float(feature_mean[i]),
            "std": float(feature_std[i]),
            "zero_frac": float(feature_zero_frac[i]),
        }
        for i, name in enumerate(feature_cols)
    }

    duplicate_frac_est = (dup_seen / float(dup_total_sampled)) if dup_total_sampled else 0.0
    all_zero_row_frac = all_zero_feature_rows / float(total_rows) if total_rows else 0.0

    passes_minimum_checks = not any(issue.severity == "error" for issue in issues) and total_rows > 0

    return {
        "generated_at": _utc_now_iso(),
        "input_path": str(csv_path),
        "total_rows": total_rows,
        "n_columns": len(METADATA_COLS) + feature_count,
        "n_features": feature_count,
        "sources": sorted(sources),
        "attack_classes_present": sorted(classes),
        "counts_by_source": {k: counts_by_source[k] for k in sorted(counts_by_source)},
        "counts_by_attack_class": counts_by_class_str,
        "counts_by_source_attack_class": counts_by_source_class_str,
        "all_zero_feature_row_frac": all_zero_row_frac,
        "duplicate_row_frac_estimate": duplicate_frac_est,
        "duplicate_sample_rows": dup_total_sampled,
        "feature_stats": feature_stats,
        "passes_minimum_checks": passes_minimum_checks,
        "issues": [issue.__dict__ for issue in issues],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a hybrid IDS dataset CSV for publication readiness")
    parser.add_argument("--input", type=Path, required=True, help="Path to hybrid dataset CSV(.gz) file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON report (default: <input>.audit.json)",
    )
    parser.add_argument("--chunksize", type=int, default=500_000, help="CSV read chunk size")
    parser.add_argument(
        "--duplicate-sample-rows",
        type=int,
        default=200_000,
        help="Rows to sample for duplicate-rate estimate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    out = args.output or Path(str(args.input) + ".audit.json")
    report = audit_hybrid_dataset_csv(
        args.input,
        chunksize=args.chunksize,
        duplicate_sample_rows=args.duplicate_sample_rows,
        seed=args.seed,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote audit report to {out}")
    print(
        f"passes_minimum_checks={report['passes_minimum_checks']}, rows={report['total_rows']:,}"
    )


if __name__ == "__main__":
    main()
