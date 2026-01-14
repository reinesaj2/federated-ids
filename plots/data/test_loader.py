"""Tests for data loading utilities."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from plots.data.loader import DataLoader, load_chapter4_data


class TestDataLoader:
    def test_init_with_csv_path(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")
        loader = DataLoader(csv_file)
        assert loader.source == csv_file

    def test_init_with_directory_path(self, tmp_path: Path):
        loader = DataLoader(tmp_path)
        assert loader.source == tmp_path

    def test_load_csv_returns_dataframe(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("dataset,aggregator,macro_f1\niiot,fedavg,0.95\n")
        loader = DataLoader(csv_file)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert "macro_f1" in df.columns

    def test_load_raises_for_nonexistent_path(self):
        loader = DataLoader(Path("/nonexistent/path.csv"))
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_standardizes_column_names(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Dataset,Aggregator,Macro_F1\nIIOT,FedAvg,0.95\n")
        loader = DataLoader(csv_file)
        df = loader.load()
        assert "dataset" in df.columns
        assert "aggregator" in df.columns


class TestLoadChapter4Data:
    def test_load_from_csv(self, tmp_path: Path):
        csv_file = tmp_path / "summary.csv"
        csv_file.write_text(
            "dataset,aggregator,alpha,adv_pct,mu,macro_f1\n"
            "iiot,fedavg,0.1,0,0.0,0.95\n"
        )
        df = load_chapter4_data(csv_file)
        assert len(df) == 1
        assert df.iloc[0]["macro_f1"] == 0.95

    def test_returns_dataframe_with_required_columns(self, tmp_path: Path):
        csv_file = tmp_path / "summary.csv"
        csv_file.write_text(
            "dataset,aggregator,alpha,adv_pct,mu,macro_f1,seed\n"
            "iiot,fedavg,0.1,0,0.0,0.95,42\n"
        )
        df = load_chapter4_data(csv_file)
        required = {"dataset", "aggregator", "alpha", "adv_pct", "mu", "macro_f1"}
        assert required.issubset(set(df.columns))
