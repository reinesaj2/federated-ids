"""
Data Aggregation for Plotting

Pre-processes experiment directories into summary CSV files for efficient
plot generation. Handles parallel loading and progress reporting.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def normalize_dataset_name(raw_name: str) -> str:
    """Normalize dataset name to canonical form."""
    name_lower = raw_name.lower()
    if "cic" in name_lower:
        return "cic"
    if "iiot" in name_lower or "edge" in name_lower:
        return "iiot"
    if "unsw" in name_lower:
        return "unsw"
    return name_lower


def parse_experiment_params(dirname: str) -> dict | None:
    """Parse experiment parameters from directory name."""
    pattern_with_prefix = (
        r"^(?P<dataset_prefix>[\w\-]+)_simple_[a-f0-9]+_"
        r"comp_(?P<aggregator>\w+)_"
        r"alpha(?P<alpha>[\d.]+)_"
        r"adv(?P<adv_pct>\d+)_"
        r"dp(?P<dp>\d+)_pers(?P<pers>\d+)_"
        r"mu(?P<mu>[\d.]+)_"
        r"seed(?P<seed>\d+)"
    )

    pattern_comp = (
        r"^comp_(?P<aggregator>\w+)_"
        r"alpha(?P<alpha>[\d.]+)_"
        r"adv(?P<adv_pct>\d+)_"
        r"dp(?P<dp>\d+)_pers(?P<pers>\d+)_"
        r"mu(?P<mu>[\d.]+)_"
        r"seed(?P<seed>\d+)"
        r"(?:_dataset(?P<dataset>\w+))?"
    )

    pattern_ds_prefix = (
        r"^ds(?P<dataset_prefix>[\w\-]+)_"
        r"comp_(?P<aggregator>\w+)_"
        r"alpha(?P<alpha>[\d.]+)_"
        r"adv(?P<adv_pct>\d+)_"
        r"dp(?P<dp>\d+)_pers(?P<pers>\d+)_"
        r"mu(?P<mu>[\d.]+)_"
        r"seed(?P<seed>\d+)"
    )

    for pattern, dataset_key in [
        (pattern_with_prefix, "dataset_prefix"),
        (pattern_ds_prefix, "dataset_prefix"),
        (pattern_comp, "dataset"),
    ]:
        match = re.match(pattern, dirname)
        if match:
            groups = match.groupdict()
            raw_dataset = groups.get(dataset_key) or groups.get("dataset_prefix") or "iiot"
            return {
                "aggregator": groups["aggregator"],
                "alpha": float(groups["alpha"]),
                "adv_pct": int(groups["adv_pct"]),
                "mu": float(groups["mu"]),
                "seed": int(groups["seed"]),
                "dataset": normalize_dataset_name(raw_dataset),
            }

    return None


def load_run_metrics(run_dir: Path) -> dict | None:
    """Load final metrics from experiment run directory."""
    client_files = list(run_dir.glob("client_*_metrics.csv"))

    if not client_files:
        metrics_file = run_dir / "metrics.csv"
        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)
                if df.empty:
                    return None
                if "macro_f1" in df.columns:
                    return {"macro_f1": df["macro_f1"].iloc[-1]}
            except Exception:
                return None
        return None

    f1_values = []
    for client_file in client_files:
        try:
            df = pd.read_csv(client_file)
            if df.empty:
                continue
            for col in ["macro_f1_after", "macro_f1_argmax", "macro_f1"]:
                if col in df.columns and len(df) > 0:
                    f1_values.append(df[col].iloc[-1])
                    break
        except Exception:
            continue

    if not f1_values:
        return None

    return {"macro_f1": sum(f1_values) / len(f1_values)}


def process_single_run(run_dir: Path) -> dict | None:
    """Process a single run directory and return record."""
    params = parse_experiment_params(run_dir.name)
    if params is None:
        return None

    metrics = load_run_metrics(run_dir)
    if metrics is None:
        return None

    return {**params, **metrics}


def aggregate_runs(
    source_dir: Path | str,
    output_csv: Path | str | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Aggregate all experiment runs into a summary DataFrame.

    Args:
        source_dir: Directory containing experiment runs
        output_csv: Optional path to save the aggregated CSV
        max_workers: Number of parallel workers

    Returns:
        DataFrame with aggregated results
    """
    source_dir = Path(source_dir)
    run_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

    print(f"Found {len(run_dirs)} directories to process...")

    records = []
    processed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_run, d): d for d in run_dirs}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                records.append(result)
                processed += 1
            else:
                skipped += 1

            if (processed + skipped) % 500 == 0:
                print(f"  Processed: {processed}, Skipped: {skipped}")

    print(f"Final: {processed} valid records, {skipped} skipped")

    if not records:
        raise ValueError(f"No valid experiment data found in: {source_dir}")

    df = pd.DataFrame(records)

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return df


def main() -> None:
    """CLI entry point for data aggregation."""
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate experiment runs into CSV")
    parser.add_argument("source", help="Source directory with experiment runs")
    parser.add_argument("-o", "--output", help="Output CSV path", default="results/summary.csv")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Parallel workers")

    args = parser.parse_args()
    aggregate_runs(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
