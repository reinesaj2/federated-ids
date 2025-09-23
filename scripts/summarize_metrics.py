#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd


def coef_variation(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    return float(std / mean) if mean != 0 else 0.0


def summarize_clients(run_dir: Path) -> dict:
    rows = []
    for p in sorted(run_dir.glob("client_*_metrics.csv")):
        try:
            df = pd.read_csv(p)
            rows.append(df)
        except Exception:
            pass
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    out = {}
    for key in ["macro_f1_argmax", "benign_fpr_argmax", "f1_bin_tau", "benign_fpr_bin_tau"]:
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce").dropna()
            out[key] = {
                "mean": float(s.mean()) if not s.empty else None,
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "cv": coef_variation(s) if not s.empty else None,
            }
    # Worst/best client macro-F1 (argmax) across clients at last round if available
    try:
        if {"client_id","round","macro_f1_argmax"}.issubset(df.columns):
            last = df.groupby("client_id").apply(lambda d: d.sort_values("round").tail(1)).reset_index(drop=True)
            s = pd.to_numeric(last["macro_f1_argmax"], errors="coerce").dropna()
            if not s.empty:
                out["macro_f1_argmax_last_round"] = {
                    "mean": float(s.mean()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "cv": coef_variation(s),
                }
    except Exception:
        pass

    # Low-FPR snapshot: aggregate the fraction of client-round rows with fpr_after<=0.10
    if "fpr_after" in df.columns:
        sa = pd.to_numeric(df["fpr_after"], errors="coerce").dropna()
        if not sa.empty:
            out["fpr_after"] = {
                "mean": float(sa.mean()),
                "frac_le_0_10": float((sa <= 0.10).mean()),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize FL metrics for fairness and rare-class reporting")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = summarize_clients(run_dir)

    out_path = Path(args.output) if args.output else (run_dir / "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to: {out_path}")


if __name__ == "__main__":
    main()
