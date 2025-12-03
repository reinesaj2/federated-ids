#!/usr/bin/env python3
"""Ensure full IDS datasets are materialized under data/ for CI runs."""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
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

EDGE_IIOTSET_SAMPLES = (
    "edge_iiotset_quick.csv",
    "edge_iiotset_nightly.csv",
    "edge_iiotset_full.csv",
)


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


def link_processed_edge_iiotset_samples() -> None:
    processed_dir = ROOT / "datasets" / "edge-iiotset" / "processed"
    target_dir = ROOT / "data" / "edge-iiotset"

    if not processed_dir.exists():
        return

    linked = False
    for sample_name in EDGE_IIOTSET_SAMPLES:
        src = processed_dir / sample_name
        if not src.exists():
            continue

        dest = target_dir / sample_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            continue

        try:
            dest.symlink_to(src)
            action = "Linked"
        except OSError:
            shutil.copy2(src, dest)
            action = "Copied"

        print(
            f"[edge-iiotset] {action} {dest.relative_to(ROOT)} -> {src.relative_to(ROOT)}"
        )
        linked = True

    if linked:
        print("[edge-iiotset] Using processed samples from datasets/edge-iiotset/processed/")


def prepare_edge_iiotset_samples() -> None:
    """Generate stratified Edge-IIoTset samples if source dataset exists."""
    source_dataset = (
        ROOT / "datasets" / "edge-iiotset" / "Edge-IIoTset dataset" / "Selected dataset for ML and DL" / "DNN-EdgeIIoT-dataset.csv"
    )

    link_processed_edge_iiotset_samples()

    if not source_dataset.exists():
        print("[edge-iiotset] Source dataset not found, skipping sample generation")
        print(f"             Expected: {source_dataset.relative_to(ROOT)}")
        return

    # Check if samples already exist
    quick_sample = ROOT / "data" / "edge-iiotset" / "edge_iiotset_quick.csv"
    nightly_sample = ROOT / "data" / "edge-iiotset" / "edge_iiotset_nightly.csv"
    full_sample = ROOT / "data" / "edge-iiotset" / "edge_iiotset_full.csv"

    if quick_sample.exists() and nightly_sample.exists() and full_sample.exists():
        print("[edge-iiotset] All samples already exist, skipping generation")
        print(f"             Quick: {quick_sample.stat().st_size:,} bytes")
        print(f"             Nightly: {nightly_sample.stat().st_size:,} bytes")
        print(f"             Full: {full_sample.stat().st_size:,} bytes")
        return

    print("[edge-iiotset] Generating stratified samples (this may take several minutes)...")

    try:
        subprocess.run(
            [
                "python",
                str(ROOT / "scripts" / "prepare_edge_iiotset_samples.py"),
                "--tier",
                "all",
                "--source",
                str(source_dataset),
                "--output-dir",
                str(ROOT / "data" / "edge-iiotset"),
            ],
            check=True,
            cwd=ROOT,
        )
        print("[edge-iiotset] Sample generation complete")
        if quick_sample.exists():
            print(f"             Quick: {quick_sample.stat().st_size:,} bytes")
        if nightly_sample.exists():
            print(f"             Nightly: {nightly_sample.stat().st_size:,} bytes")
        if full_sample.exists():
            print(f"             Full: {full_sample.stat().st_size:,} bytes")
    except subprocess.CalledProcessError as e:
        print(f"[edge-iiotset] ERROR: Sample generation failed: {e}")
        print("             Experiments will fail if Edge-IIoTset samples are required")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize IDS datasets for CI")
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only extract required full datasets (skip samples).",
    )
    args = parser.parse_args()

    # Extract legacy datasets (UNSW, CIC)
    for name, paths in DATASETS.items():
        if args.full_only and not paths["required"]:
            continue
        extract_dataset(name, paths["source"], paths["target"], paths["required"], paths.get("min_bytes"))

    # Generate Edge-IIoTset samples
    print("\n" + "=" * 70)
    prepare_edge_iiotset_samples()
    print("=" * 70)


if __name__ == "__main__":
    main()
