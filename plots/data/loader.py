"""
Data Loading Utilities

Unified data loading for plotting, supporting both CSV files and
experiment directory structures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Unified data loader for plotting.

    Supports:
    - CSV files with pre-aggregated experiment results
    - Directory structures with per-experiment metrics files
    """

    def __init__(self, source: Path | str) -> None:
        """
        Initialize loader with data source path.

        Args:
            source: Path to CSV file or experiment directory
        """
        self.source = Path(source)

    def load(self) -> pd.DataFrame:
        """
        Load data from configured source.

        Returns:
            DataFrame with standardized column names

        Raises:
            FileNotFoundError: If source path does not exist
            ValueError: If source format is not recognized
        """
        if not self.source.exists():
            raise FileNotFoundError(f"Data source not found: {self.source}")

        if self.source.is_file() and self.source.suffix == ".csv":
            return self._load_csv()
        elif self.source.is_dir():
            return self._load_directory()
        else:
            raise ValueError(f"Unrecognized data source format: {self.source}")

    def _load_csv(self) -> pd.DataFrame:
        """Load and standardize CSV file."""
        df = pd.read_csv(self.source)
        return self._standardize_columns(df)

    def _load_directory(self) -> pd.DataFrame:
        """Load from directory of experiment results."""
        records = []

        for run_dir in self.source.iterdir():
            if not run_dir.is_dir():
                continue

            params = self._parse_experiment_params(run_dir.name)
            if params is None:
                continue

            metrics = self._load_run_metrics(run_dir)
            if metrics is not None:
                records.append({**params, **metrics})

        if not records:
            raise ValueError(f"No valid experiment data found in: {self.source}")

        return pd.DataFrame(records)

    def _parse_experiment_params(self, dirname: str) -> dict | None:
        """
        Parse experiment parameters from directory name.

        Supports multiple naming formats:
        1. {dataset}_simple_{hash}_comp_{agg}_alpha{a}_adv{adv}_dp{dp}_pers{p}_mu{mu}_seed{s}
        2. comp_{aggregator}_alpha{alpha}_adv{adv}_dp{dp}_pers{pers}_mu{mu}_seed{seed}
        3. {aggregator}_alpha{alpha}_adv{adv}_mu{mu}_seed{seed}_{dataset}
        """
        pattern_with_prefix = (
            r"^(?P<dataset_prefix>[\w\-]+)_simple_[a-f0-9]+_"
            r"comp_(?P<aggregator>\w+)_"
            r"alpha(?P<alpha>[\d.]+)_"
            r"adv(?P<adv_pct>\d+)_"
            r"dp\d+_pers\d+_"
            r"mu(?P<mu>[\d.]+)_"
            r"seed(?P<seed>\d+)"
        )

        pattern_comp = (
            r"^comp_(?P<aggregator>\w+)_"
            r"alpha(?P<alpha>[\d.]+)_"
            r"adv(?P<adv_pct>\d+)_"
            r"dp\d+_pers\d+_"
            r"mu(?P<mu>[\d.]+)_"
            r"seed(?P<seed>\d+)"
        )

        pattern_legacy = (
            r"^(?P<aggregator>\w+)_"
            r"alpha(?P<alpha>[\d.]+)_"
            r"adv(?P<adv_pct>\d+)_"
            r"mu(?P<mu>[\d.]+)_"
            r"seed(?P<seed>\d+)_"
            r"(?P<dataset>\w+)$"
        )

        match = re.match(pattern_with_prefix, dirname)
        if match:
            groups = match.groupdict()
            dataset = self._normalize_dataset_name(groups["dataset_prefix"])
            return {
                "aggregator": groups["aggregator"],
                "alpha": float(groups["alpha"]),
                "adv_pct": int(groups["adv_pct"]),
                "mu": float(groups["mu"]),
                "seed": int(groups["seed"]),
                "dataset": dataset,
            }

        match = re.match(pattern_comp, dirname)
        if match:
            groups = match.groupdict()
            return {
                "aggregator": groups["aggregator"],
                "alpha": float(groups["alpha"]),
                "adv_pct": int(groups["adv_pct"]),
                "mu": float(groups["mu"]),
                "seed": int(groups["seed"]),
                "dataset": "iiot",
            }

        match = re.match(pattern_legacy, dirname)
        if match:
            groups = match.groupdict()
            return {
                "aggregator": groups["aggregator"],
                "alpha": float(groups["alpha"]),
                "adv_pct": int(groups["adv_pct"]),
                "mu": float(groups["mu"]),
                "seed": int(groups["seed"]),
                "dataset": self._normalize_dataset_name(groups["dataset"]),
            }

        return None

    def _normalize_dataset_name(self, raw_name: str) -> str:
        """Normalize dataset name to canonical form."""
        name_lower = raw_name.lower()
        if "cic" in name_lower:
            return "cic"
        if "iiot" in name_lower or "edge" in name_lower:
            return "iiot"
        if "unsw" in name_lower:
            return "unsw"
        return name_lower

    def _load_run_metrics(self, run_dir: Path) -> dict | None:
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
                f1_col = None
                for col in ["macro_f1_after", "macro_f1_argmax", "macro_f1"]:
                    if col in df.columns:
                        f1_col = col
                        break
                if f1_col and len(df) > 0:
                    f1_values.append(df[f1_col].iloc[-1])
            except Exception:
                continue

        if not f1_values:
            return None

        return {"macro_f1": sum(f1_values) / len(f1_values)}

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = [
            col.lower().replace(" ", "_").replace("-", "_")
            for col in df.columns
        ]
        return df


def load_chapter4_data(source: Path | str) -> pd.DataFrame:
    """
    Convenience function to load Chapter 4 experiment data.

    Args:
        source: Path to CSV file or experiment directory

    Returns:
        DataFrame with standardized Chapter 4 columns
    """
    loader = DataLoader(source)
    return loader.load()
