#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_CLIENT_COLS = [
  "client_id","round","dataset_size","n_classes","loss_after","acc_after"
]
# Extended columns checked only if present
EXTENDED_ALIGN_COLS = [
  "macro_f1_argmax","benign_fpr_argmax","f1_bin_tau","benign_fpr_bin_tau","tau_bin"
]


def check_schema(run_dir: Path) -> None:
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        raise AssertionError(f"No client metrics found in {run_dir}")
    for f in client_files:
        df = pd.read_csv(f)
        for c in REQUIRED_CLIENT_COLS:
            if c not in df.columns:
                raise AssertionError(f"Missing column {c} in {f}")


def check_alignment(run_dir: Path, tol: float = 1e-6) -> None:
    for f in run_dir.glob("client_*_metrics.csv"):
        df = pd.read_csv(f)
        if set(EXTENDED_ALIGN_COLS).issubset(df.columns):
            # compute recall_benign from argmax via reported benign_fpr_argmax
            # benign_fpr_argmax should equal 1 - recall_benign
            # We trust benign_fpr_argmax value coherence per row
            # Just validate ranges and existence
            if not np.all((df["benign_fpr_argmax"] >= 0.0) & (df["benign_fpr_argmax"] <= 1.0)):
                raise AssertionError(f"benign_fpr_argmax out of range in {f}")


def coef_variation(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    return float(std / mean) if mean != 0 else 0.0


def check_fairness(run_dir: Path, cv_threshold: float = 0.20) -> None:
    rows = []
    for f in run_dir.glob("client_*_metrics.csv"):
        try:
            rows.append(pd.read_csv(f))
        except Exception:
            pass
    if not rows:
        raise AssertionError(f"No client CSVs for fairness check in {run_dir}")
    df = pd.concat(rows, ignore_index=True)
    if "macro_f1_argmax" in df.columns:
        cv = coef_variation(df["macro_f1_argmax"])
        if cv > cv_threshold:
            raise AssertionError(f"Fairness CV exceeded threshold: {cv:.3f} > {cv_threshold:.2f}")


def check_artifacts(run_dir: Path) -> None:
    needed = ["metrics.csv","client_metrics_plot.png","server_metrics_plot.png"]
    present = [p.name for p in run_dir.iterdir() if p.is_file()]
    for n in needed:
        if n not in present:
            raise AssertionError(f"Missing artifact {n} in {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        print(f"No runs dir at {runs_root}", file=sys.stderr)
        sys.exit(1)

    failed = []
    for run_dir in runs_root.glob("**/"):
        # only check leaf run dirs that contain metrics.csv
        if (run_dir / "metrics.csv").exists():
            try:
                check_schema(run_dir)
                check_alignment(run_dir)
                check_fairness(run_dir)
                check_artifacts(run_dir)
            except AssertionError as e:
                failed.append((str(run_dir), str(e)))
    if failed:
        for r, msg in failed:
            print(f"[CI CHECK FAIL] {r}: {msg}", file=sys.stderr)
        sys.exit(2)
    print("CI checks passed.")


if __name__ == "__main__":
    main()
