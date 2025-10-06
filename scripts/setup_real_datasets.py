#!/usr/bin/env python3
"""Utility to materialize bundled real IDS dataset samples into data/ for CI."""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLES = {
    "unsw": {
        "source": ROOT / "datasets" / "real" / "unsw_nb15_sample.csv.gz",
        "target": ROOT / "data" / "unsw" / "unsw_nb15_sample.csv",
    },
    "cic": {
        "source": ROOT / "datasets" / "real" / "cic_ids2017_sample.csv.gz",
        "target": ROOT / "data" / "cic" / "cic_ids2017_sample.csv",
    },
    "cic_multiclass": {
        "source": ROOT / "datasets" / "real" / "cic_ids2017_multiclass.csv.gz",
        "target": ROOT / "data" / "cic" / "cic_ids2017_multiclass.csv",
    },
}


def extract_sample(name: str, source: Path, target: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Expected sample archive missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    size = target.stat().st_size
    print(f"[{name}] Extracted {source.relative_to(ROOT)} -> {target.relative_to(ROOT)} ({size} bytes)")


def main() -> None:
    for name, paths in SAMPLES.items():
        extract_sample(name, paths["source"], paths["target"])


if __name__ == "__main__":
    main()
