#!/usr/bin/env python3
"""Ensure full IDS datasets are materialized under data/ for CI runs."""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "unsw_full": {
        "source": ROOT / "datasets" / "real" / "UNSW_NB15_training-set.csv.gz",
        "target": ROOT / "data" / "unsw" / "UNSW_NB15_training-set.csv",
        "required": True,
        "min_bytes": 15_000_000,
    },
    "cic_full": {
        "source": ROOT / "datasets" / "real" / "cic_ids2017_multiclass.csv.gz",
        "target": ROOT / "data" / "cic" / "cic_ids2017_multiclass.csv",
        "required": True,
        "min_bytes": 3_000_000,
    },
    "unsw_sample": {
        "source": ROOT / "datasets" / "real" / "unsw_nb15_sample.csv.gz",
        "target": ROOT / "data" / "unsw" / "unsw_nb15_sample.csv",
        "required": False,
    },
    "cic_sample": {
        "source": ROOT / "datasets" / "real" / "cic_ids2017_sample.csv.gz",
        "target": ROOT / "data" / "cic" / "cic_ids2017_sample.csv",
        "required": False,
    },
}


def extract_dataset(name: str, source: Path, target: Path, required: bool, min_bytes: int | None) -> None:
    if target.exists():
        size = target.stat().st_size
        if min_bytes is not None and size < min_bytes:
            print(f"[{name}] Existing {target.relative_to(ROOT)} ({size} bytes) below expected size, re-extracting")
        else:
            print(f"[{name}] Using cached {target.relative_to(ROOT)} ({size} bytes)")
            return
    if not source.exists():
        message = f"Expected dataset archive missing: {source}"
        if required:
            raise SystemExit(message)
        print(f"[{name}] {message} (optional)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    size = target.stat().st_size
    if min_bytes is not None and size < min_bytes:
        raise SystemExit(
            f"[{name}] Extracted dataset too small: {size} bytes (expected at least {min_bytes}). "
            "Verify that the full dataset archive is present."
        )
    print(f"[{name}] Extracted {source.relative_to(ROOT)} -> {target.relative_to(ROOT)} ({size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize IDS datasets for CI")
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only extract required full datasets (skip samples).",
    )
    args = parser.parse_args()

    for name, paths in DATASETS.items():
        if args.full_only and not paths["required"]:
            continue
        extract_dataset(name, paths["source"], paths["target"], paths["required"], paths.get("min_bytes"))


if __name__ == "__main__":
    main()
