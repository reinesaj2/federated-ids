"""Tests for hybrid dataset loading and source-aware partitioning.

Tests cover:
- load_hybrid_dataset: Loading, validation, and error handling
- source_aware_partition: Index coverage, source boundaries, fallback behavior
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    MIN_SAMPLES_PER_CLASS,
    infer_feature_columns,
    load_hybrid_dataset,
    prepare_partitions_from_dataframe,
    source_aware_partition,
)


class TestLoadHybridDataset:
    """Tests for load_hybrid_dataset function."""

    def test_loads_valid_hybrid_csv(self, tmp_path: Path) -> None:
        """Valid hybrid CSV should load with correct columns identified."""
        csv_path = tmp_path / "hybrid.csv"
        df = pd.DataFrame(
            {
                "duration": [1.0, 2.0, 3.0],
                "fwd_packets": [10.0, 20.0, 30.0],
                "source_dataset": ["cic", "unsw", "iiot"],
                "attack_class": [0, 1, 2],
                "attack_label_original": ["BENIGN", "DOS", "PROBE"],
            }
        )
        df.to_csv(csv_path, index=False)

        result_df, label_col, proto_col, source_col = load_hybrid_dataset(csv_path)

        assert label_col == "attack_class"
        assert source_col == "source_dataset"
        assert proto_col is None
        assert len(result_df) == 3
        assert set(result_df["source_dataset"]) == {"cic", "unsw", "iiot"}

    def test_loads_gzipped_csv(self, tmp_path: Path) -> None:
        """Gzipped hybrid CSV should load correctly."""
        csv_path = tmp_path / "hybrid.csv.gz"
        df = pd.DataFrame(
            {
                "duration": [1.0, 2.0],
                "source_dataset": ["cic", "unsw"],
                "attack_class": [0, 1],
                "attack_label_original": ["BENIGN", "DOS"],
            }
        )
        df.to_csv(csv_path, index=False, compression="gzip")

        result_df, label_col, _, source_col = load_hybrid_dataset(csv_path)

        assert len(result_df) == 2
        assert label_col == "attack_class"

    def test_missing_attack_class_raises_value_error(self, tmp_path: Path) -> None:
        """Missing attack_class column should raise ValueError."""
        csv_path = tmp_path / "hybrid.csv"
        df = pd.DataFrame(
            {
                "duration": [1.0],
                "source_dataset": ["cic"],
                "attack_label_original": ["BENIGN"],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="attack_class"):
            load_hybrid_dataset(csv_path)

    def test_missing_source_dataset_raises_value_error(self, tmp_path: Path) -> None:
        """Missing source_dataset column should raise ValueError."""
        csv_path = tmp_path / "hybrid.csv"
        df = pd.DataFrame(
            {
                "duration": [1.0],
                "attack_class": [0],
                "attack_label_original": ["BENIGN"],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="source_dataset"):
            load_hybrid_dataset(csv_path)

    def test_max_samples_limits_rows(self, tmp_path: Path) -> None:
        """max_samples parameter should limit rows loaded."""
        csv_path = tmp_path / "hybrid.csv"
        df = pd.DataFrame(
            {
                "duration": list(range(100)),
                "source_dataset": ["cic"] * 100,
                "attack_class": [0] * 100,
                "attack_label_original": ["BENIGN"] * 100,
            }
        )
        df.to_csv(csv_path, index=False)

        result_df, _, _, _ = load_hybrid_dataset(csv_path, max_samples=10)

        assert len(result_df) == 10

    def test_fills_nan_with_zero(self, tmp_path: Path) -> None:
        """NaN values in feature columns should be filled with 0."""
        csv_path = tmp_path / "hybrid.csv"
        df = pd.DataFrame(
            {
                "duration": [1.0, np.nan, 3.0],
                "source_dataset": ["cic", "unsw", "iiot"],
                "attack_class": [0, 1, 2],
                "attack_label_original": ["BENIGN", "DOS", "PROBE"],
            }
        )
        df.to_csv(csv_path, index=False)

        result_df, _, _, _ = load_hybrid_dataset(csv_path)

        assert result_df["duration"].isna().sum() == 0
        assert result_df["duration"].iloc[1] == 0.0


class TestSourceAwarePartition:
    """Tests for source_aware_partition function."""

    def test_all_indices_covered_exactly_once(self) -> None:
        """Every input index should appear exactly once across all shards."""
        n_samples = 300
        source_labels = ["cic"] * 100 + ["unsw"] * 100 + ["iiot"] * 100
        attack_labels = [i % 7 for i in range(n_samples)]

        shards = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)

        all_indices = []
        for shard in shards:
            all_indices.extend(shard)

        assert sorted(all_indices) == list(range(n_samples))

    def test_no_overlap_between_shards(self) -> None:
        """No index should appear in multiple shards."""
        source_labels = ["cic"] * 100 + ["unsw"] * 100 + ["iiot"] * 100
        attack_labels = [i % 7 for i in range(300)]

        shards = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)

        all_indices = []
        for shard in shards:
            all_indices.extend(shard)

        assert len(all_indices) == len(set(all_indices))

    def test_correct_number_of_clients(self) -> None:
        """Should produce num_sources * clients_per_source shards."""
        source_labels = ["cic"] * 100 + ["unsw"] * 100 + ["iiot"] * 100
        attack_labels = [i % 7 for i in range(300)]

        shards = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)

        assert len(shards) == 9  # 3 sources x 3 clients

    def test_respects_source_boundaries(self) -> None:
        """Clients should only contain indices from their assigned source."""
        source_labels = ["cic"] * 100 + ["unsw"] * 100 + ["iiot"] * 100
        attack_labels = [i % 7 for i in range(300)]

        shards = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)

        source_array = np.array(source_labels)
        unique_sources = sorted(set(source_labels))

        for source_idx, source in enumerate(unique_sources):
            start = source_idx * 3
            end = start + 3
            source_clients = shards[start:end]

            for shard in source_clients:
                sources_in_shard = set(source_array[shard])
                assert sources_in_shard == {source}, f"Client shard for source '{source}' contains indices from: {sources_in_shard}"

    def test_iid_partition_when_alpha_inf(self) -> None:
        """alpha=inf should use IID partitioning within each source."""
        source_labels = ["cic"] * 100 + ["unsw"] * 100
        attack_labels = [i % 7 for i in range(200)]

        shards = source_aware_partition(source_labels, attack_labels, clients_per_source=2, alpha=float("inf"), seed=42)

        assert len(shards) == 4  # 2 sources x 2 clients
        assert sum(len(s) for s in shards) == 200

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical partitions."""
        source_labels = ["cic"] * 100 + ["unsw"] * 100 + ["iiot"] * 100
        attack_labels = [i % 7 for i in range(300)]

        shards1 = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)
        shards2 = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)

        for s1, s2 in zip(shards1, shards2, strict=True):
            assert s1 == s2

    def test_different_seeds_produce_different_partitions(self) -> None:
        """Different seeds should produce different partitions."""
        source_labels = ["cic"] * 100 + ["unsw"] * 100 + ["iiot"] * 100
        attack_labels = [i % 7 for i in range(300)]

        shards1 = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)
        shards2 = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=99)

        any_different = any(s1 != s2 for s1, s2 in zip(shards1, shards2, strict=True))
        assert any_different

    def test_warns_on_fallback_to_iid(self) -> None:
        """Should warn when Dirichlet fails and falls back to IID."""
        source_labels = ["cic"] * 10
        attack_labels = [0] * 5 + [1] * 5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            source_aware_partition(
                source_labels,
                attack_labels,
                clients_per_source=5,
                alpha=0.01,
                seed=42,
                min_samples_per_class=MIN_SAMPLES_PER_CLASS,
            )

            fallback_warnings = [warning for warning in w if "Falling back to IID" in str(warning.message)]
            assert len(fallback_warnings) >= 1

    def test_handles_single_source(self) -> None:
        """Should work with only one source dataset."""
        source_labels = ["cic"] * 100
        attack_labels = [i % 7 for i in range(100)]

        shards = source_aware_partition(source_labels, attack_labels, clients_per_source=3, alpha=0.5, seed=42)

        assert len(shards) == 3
        assert sum(len(s) for s in shards) == 100


def test_prepare_partitions_source_strategy_isolates_sources() -> None:
    seed = 42
    num_clients = 3
    alpha = 0.5
    df = pd.DataFrame(
        {
            "duration": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            "source_id": [0, 0, 1, 1, 2, 2],
            "source_dataset": ["cic", "cic", "unsw", "unsw", "iiot", "iiot"],
            "attack_class": [0, 1, 0, 1, 0, 1],
            "attack_label_original": ["BENIGN", "DOS", "BENIGN", "DOS", "BENIGN", "DOS"],
        }
    )

    _pre, X_parts, _y_parts, num_classes = prepare_partitions_from_dataframe(
        df=df,
        label_col="attack_class",
        partition_strategy="source",
        num_clients=num_clients,
        seed=seed,
        alpha=alpha,
        source_col="source_dataset",
    )

    assert num_classes == 2
    assert len(X_parts) == 3

    numeric_cols, _ = infer_feature_columns(df, "attack_class", drop_cols=[])
    source_idx = numeric_cols.index("source_id")
    shard_values = []
    for shard in X_parts:
        shard_matrix = shard.toarray() if hasattr(shard, "toarray") else shard
        values = np.unique(np.round(shard_matrix[:, source_idx], 6))
        assert values.size == 1
        shard_values.append(values[0])
    assert len(set(shard_values)) == 3
