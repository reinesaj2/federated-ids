"""Unit tests for temporal validation protocol functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    parse_frame_time,
    temporal_sort_indices,
    temporal_train_val_test_split_indices,
    numpy_to_temporal_train_val_test_loaders,
)


class TestParseFrameTime:
    def test_parses_valid_edge_iiotset_format(self):
        series = pd.Series(
            [
                " 2021 19:46:24.393481000 ",
                "2021 18:54:19.817953000",
                "2021 17:28:06.949992000",
            ]
        )
        result = parse_frame_time(series)

        assert len(result) == 3
        assert not result.isna().any()
        assert result.iloc[2] < result.iloc[1] < result.iloc[0]

    def test_returns_nat_for_unparsable_values(self):
        series = pd.Series(
            [
                "2021 19:46:24.393481000",
                "192.168.1.1",
                "invalid_timestamp",
                "",
            ]
        )
        result = parse_frame_time(series)

        assert len(result) == 4
        assert not pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])

    def test_handles_empty_series(self):
        series = pd.Series([], dtype=str)
        result = parse_frame_time(series)

        assert len(result) == 0

    def test_strips_whitespace(self):
        series = pd.Series(["  2021 12:00:00.000000000  "])
        result = parse_frame_time(series)

        assert not pd.isna(result.iloc[0])


class TestTemporalSortIndices:
    def test_sorts_by_timestamp_ascending(self):
        df = pd.DataFrame(
            {
                "frame.time": [
                    "2021 19:00:00.000000000",
                    "2021 17:00:00.000000000",
                    "2021 18:00:00.000000000",
                ],
                "value": [1, 2, 3],
            }
        )
        sort_idx, n_unparsable = temporal_sort_indices(df, "frame.time")

        assert n_unparsable == 0
        assert list(sort_idx) == [1, 2, 0]

    def test_unparsable_rows_placed_at_end(self):
        df = pd.DataFrame(
            {
                "frame.time": [
                    "2021 19:00:00.000000000",
                    "invalid",
                    "2021 17:00:00.000000000",
                ],
                "value": [1, 2, 3],
            }
        )
        sort_idx, n_unparsable = temporal_sort_indices(df, "frame.time")

        assert n_unparsable == 1
        assert list(sort_idx) == [2, 0, 1]

    def test_all_unparsable_returns_original_order(self):
        df = pd.DataFrame(
            {
                "frame.time": ["invalid1", "invalid2", "invalid3"],
                "value": [1, 2, 3],
            }
        )
        sort_idx, n_unparsable = temporal_sort_indices(df, "frame.time")

        assert n_unparsable == 3
        assert list(sort_idx) == [0, 1, 2]

    def test_missing_column_returns_original_order(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        sort_idx, n_unparsable = temporal_sort_indices(df, "frame.time")

        assert n_unparsable == 3
        assert list(sort_idx) == [0, 1, 2]

    def test_preserves_relative_order_of_unparsable(self):
        df = pd.DataFrame(
            {
                "frame.time": [
                    "2021 19:00:00.000000000",
                    "invalid_a",
                    "2021 17:00:00.000000000",
                    "invalid_b",
                ],
                "value": [1, 2, 3, 4],
            }
        )
        sort_idx, n_unparsable = temporal_sort_indices(df, "frame.time")

        assert n_unparsable == 2
        assert list(sort_idx) == [2, 0, 1, 3]


class TestTemporalTrainValTestSplitIndices:
    def test_default_70_15_15_split(self):
        n_samples = 100
        train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples)

        assert len(train_idx) == 70
        assert len(val_idx) == 15
        assert len(test_idx) == 15

        assert list(train_idx) == list(range(0, 70))
        assert list(val_idx) == list(range(70, 85))
        assert list(test_idx) == list(range(85, 100))

    def test_custom_split_fractions(self):
        n_samples = 100
        train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples, train_frac=0.6, val_frac=0.2, test_frac=0.2)

        assert len(train_idx) == 60
        assert len(val_idx) == 20
        assert len(test_idx) == 20

    def test_rejects_invalid_fractions(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            temporal_train_val_test_split_indices(100, 0.5, 0.3, 0.1)

    def test_no_overlap_between_splits(self):
        n_samples = 1000
        train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples)

        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        assert len(all_idx) == n_samples

        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_train_is_earliest_test_is_latest(self):
        n_samples = 100
        train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples)

        assert max(train_idx) < min(val_idx)
        assert max(val_idx) < min(test_idx)


class TestNumpyToTemporalTrainValTestLoaders:
    def test_creates_three_loaders(self):
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, size=n_samples).astype(np.int64)

        train_loader, val_loader, test_loader = numpy_to_temporal_train_val_test_loaders(X, y, batch_size=32)

        assert len(train_loader.dataset) == 70
        assert len(val_loader.dataset) == 15
        assert len(test_loader.dataset) == 15

    def test_applies_sort_indices(self):
        n_samples = 10
        X = np.arange(n_samples).reshape(-1, 1).astype(np.float32)
        y = np.arange(n_samples).astype(np.int64)
        sort_indices = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)

        train_loader, val_loader, test_loader = numpy_to_temporal_train_val_test_loaders(X, y, batch_size=32, sort_indices=sort_indices)

        train_X = train_loader.dataset.tensors[0].numpy()
        test_X = test_loader.dataset.tensors[0].numpy()

        assert train_X[0, 0] == 9.0
        assert test_X[-1, 0] == 0.0

    def test_rejects_invalid_splits(self):
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=100).astype(np.int64)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            numpy_to_temporal_train_val_test_loaders(X, y, batch_size=32, splits=(0.5, 0.2, 0.1))


class TestTemporalValidationProtocolInvariants:
    """Test invariants required by the validation protocol."""

    def test_train_val_test_are_disjoint(self):
        n_samples = 1000
        train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples)

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert train_set.isdisjoint(val_set)
        assert val_set.isdisjoint(test_set)
        assert train_set.isdisjoint(test_set)

    def test_train_val_test_cover_all_samples(self):
        n_samples = 1000
        train_idx, val_idx, test_idx = temporal_train_val_test_split_indices(n_samples)

        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == n_samples
        assert set(all_idx) == set(range(n_samples))

    def test_temporal_ordering_preserved_after_sort(self):
        timestamps = [
            "2021 10:00:00.000000000",
            "2021 12:00:00.000000000",
            "2021 11:00:00.000000000",
            "2021 09:00:00.000000000",
        ]
        df = pd.DataFrame({"frame.time": timestamps, "value": range(4)})

        sort_idx, _ = temporal_sort_indices(df, "frame.time")
        sorted_times = [timestamps[i] for i in sort_idx]

        parsed = parse_frame_time(pd.Series(sorted_times))
        for i in range(len(parsed) - 1):
            assert parsed.iloc[i] <= parsed.iloc[i + 1]
