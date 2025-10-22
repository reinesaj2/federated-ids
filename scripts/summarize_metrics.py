#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict
import pandas as pd


def coef_variation(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    return float(std / mean) if mean != 0 else 0.0


def compute_fairness_metrics(client_metrics_df: pd.DataFrame) -> Dict:
    """
    Compute fairness and disparity metrics across federated clients.

    Analyzes client-level performance to identify disparities, quantify variance,
    and assess fairness across the federation.

    Args:
        client_metrics_df: DataFrame with columns client_id, round, macro_f1_argmax,
                          benign_fpr_argmax (optional)

    Returns:
        Dictionary with fairness metrics:
        - worst_client_macro_f1_argmax: Minimum F1 across clients (final round)
        - best_client_macro_f1_argmax: Maximum F1 across clients (final round)
        - cv_macro_f1_argmax: Coefficient of variation of F1 across clients
        - fraction_clients_fpr_le_0_10: Fraction of clients with FPR <= 0.10
        - rare_class_f1_mean: Mean F1 for rare classes (not implemented yet)
        - rare_class_f1_min: Min F1 for rare classes (not implemented yet)
    """
    if client_metrics_df.empty:
        return {}

    fairness = {}

    if "macro_f1_argmax" in client_metrics_df.columns and "client_id" in client_metrics_df.columns:
        f1_by_client = client_metrics_df.groupby("client_id")["macro_f1_argmax"].last()

        if not f1_by_client.empty:
            fairness["worst_client_macro_f1_argmax"] = float(f1_by_client.min())
            fairness["best_client_macro_f1_argmax"] = float(f1_by_client.max())

            mean_f1 = f1_by_client.mean()
            std_f1 = f1_by_client.std(ddof=0)
            fairness["cv_macro_f1_argmax"] = float(std_f1 / mean_f1) if mean_f1 > 0 else 0.0

    if "benign_fpr_argmax" in client_metrics_df.columns and "client_id" in client_metrics_df.columns:
        fpr_by_client = client_metrics_df.groupby("client_id")["benign_fpr_argmax"].last()

        if not fpr_by_client.empty:
            low_fpr_count = (fpr_by_client <= 0.10).sum()
            fairness["fraction_clients_fpr_le_0_10"] = float(low_fpr_count / len(fpr_by_client))

    return fairness


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
    dp_fields = ["dp_epsilon", "dp_delta", "dp_sigma", "dp_clip_norm"]
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
    # Worst/best client macro-F1 (argmax) across clients at last round if available
    try:
        if {"client_id", "round", "macro_f1_argmax"}.issubset(df.columns):
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

    # Low-FPR point: extract (τ, FPR, F1) from last round across all clients
    if {"tau_bin", "benign_fpr_bin_tau", "f1_bin_tau", "client_id", "round"}.issubset(df.columns):
        try:
            # Get last round for each client
            last_round_df = df.groupby("client_id").apply(lambda d: d.sort_values("round").tail(1)).reset_index(drop=True)

            tau_vals = pd.to_numeric(last_round_df["tau_bin"], errors="coerce").dropna()
            fpr_vals = pd.to_numeric(last_round_df["benign_fpr_bin_tau"], errors="coerce").dropna()
            f1_vals = pd.to_numeric(last_round_df["f1_bin_tau"], errors="coerce").dropna()

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

    fairness_metrics = compute_fairness_metrics(df)
    out.update(fairness_metrics)

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
