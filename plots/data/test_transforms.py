import numpy as np
import pandas as pd
import pytest

from plots.data.transforms import aggregate_by_seed, compute_confidence_interval, prepare_ci_matrix


def test_compute_confidence_interval_constant_values():
    values = np.array([0.5, 0.5, 0.5])
    assert compute_confidence_interval(values) == (0.5, 0.5, 0.5)


def test_compute_confidence_interval_empty_raises():
    values = np.array([])
    with pytest.raises(ValueError):
        compute_confidence_interval(values)


def test_aggregate_by_seed_returns_expected_summary():
    data = pd.DataFrame(
        {
            "aggregator": ["fedavg", "fedavg", "krum", "krum"],
            "seed": [1, 2, 1, 2],
            "macro_f1": [0.7, 0.7, 0.5, 0.5],
        }
    )
    result = aggregate_by_seed(data, ["aggregator"])
    expected = [
        {"aggregator": "fedavg", "mean": 0.7, "ci_lower": 0.7, "ci_upper": 0.7},
        {"aggregator": "krum", "mean": 0.5, "ci_lower": 0.5, "ci_upper": 0.5},
    ]
    assert result.to_dict("records") == expected


def test_prepare_ci_matrix_returns_expected_arrays():
    data = pd.DataFrame(
        {
            "round": [1, 2, 1, 2],
            "seed": [1, 1, 2, 2],
            "macro_f1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    rounds, matrix = prepare_ci_matrix(data, "macro_f1")
    assert (rounds.tolist(), matrix.tolist()) == ([1, 2], [[0.1, 0.2], [0.3, 0.4]])


def test_prepare_ci_matrix_returns_none_without_seed():
    data = pd.DataFrame({"round": [1, 2], "macro_f1": [0.1, 0.2]})
    assert prepare_ci_matrix(data, "macro_f1") is None
