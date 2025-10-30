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
    for key in [
        "macro_f1_argmax",
        "benign_fpr_argmax",
        "f1_bin_tau",
        "benign_fpr_bin_tau",
    ]:
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce").dropna()
            out[key] = {
                "mean": float(s.mean()) if not s.empty else None,
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "cv": coef_variation(s) if not s.empty else None,
            }

    # Differential Privacy metrics summary
    dp_fields = ["dp_epsilon", "dp_delta", "dp_sigma", "dp_clip_norm", "dp_sample_rate"]
    dp_present = any(field in df.columns for field in dp_fields)
    if dp_present:
        out["privacy"] = {}
        for field in dp_fields:
            if field in df.columns:
                s = pd.to_numeric(df[field], errors="coerce").dropna()
                if not s.empty:
                    out["privacy"][field] = {
                        "mean": float(s.mean()),
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }
        if "dp_total_steps" in df.columns:
            s_steps = pd.to_numeric(df["dp_total_steps"], errors="coerce").dropna()
            if not s_steps.empty:
                out["privacy"]["dp_total_steps"] = {
                    "mean": float(s_steps.mean()),
                    "min": float(s_steps.min()),
                    "max": float(s_steps.max()),
                }
    # Worst/best client macro-F1 (argmax) across clients at last round if available
    try:
        if {"client_id", "round", "macro_f1_argmax"}.issubset(df.columns):
            last = (
                df.groupby("client_id")
                .apply(lambda d: d.sort_values("round").tail(1))
                .reset_index(drop=True)
            )
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

    # Low-FPR point: extract (τ, FPR, F1) from last round across all clients
    if {"tau_bin", "benign_fpr_bin_tau", "f1_bin_tau", "client_id", "round"}.issubset(
        df.columns
    ):
        try:
            # Get last round for each client
            last_round_df = (
                df.groupby("client_id")
                .apply(lambda d: d.sort_values("round").tail(1))
                .reset_index(drop=True)
            )

            tau_vals = pd.to_numeric(last_round_df["tau_bin"], errors="coerce").dropna()
            fpr_vals = pd.to_numeric(
                last_round_df["benign_fpr_bin_tau"], errors="coerce"
            ).dropna()
            f1_vals = pd.to_numeric(
                last_round_df["f1_bin_tau"], errors="coerce"
            ).dropna()

            if not tau_vals.empty and not fpr_vals.empty and not f1_vals.empty:
                out["low_fpr_point"] = {
                    "tau_mean": float(tau_vals.mean()),
                    "tau_min": float(tau_vals.min()),
                    "tau_max": float(tau_vals.max()),
                    "fpr_mean": float(fpr_vals.mean()),
                    "fpr_min": float(fpr_vals.min()),
                    "fpr_max": float(fpr_vals.max()),
                    "f1_mean": float(f1_vals.mean()),
                    "f1_min": float(f1_vals.min()),
                    "f1_max": float(f1_vals.max()),
                }
        except Exception:
            pass

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize FL metrics for fairness and rare-class reporting"
    )
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
