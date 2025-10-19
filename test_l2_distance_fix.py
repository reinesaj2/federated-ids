from __future__ import annotations

import numpy as np

from server_metrics import calculate_robustness_metrics
from robust_aggregation import aggregate_weights, AggregationMethod


def test_l2_distance_is_not_zero_for_median_aggregation():
    """
    Replicates the L2=0 bug scenario and provides a test case for the fix.

    The bug occurs when the benign_mean reference is calculated using the
    same aggregation as the method being tested (e.g., Median). With two
    clients, the median is the average, making the reference identical to
    the aggregated result.
    """
    # 1. Define two distinct client updates
    client_updates = [
        [np.array([1.0, 2.0])],  # Client 1
        [np.array([3.0, 4.0])],  # Client 2
    ]

    # 2. Simulate the OLD, buggy logic where benign_mean is the same as aggregated
    # For 2 clients, median is the same as the average
    buggy_benign_mean = aggregate_weights(client_updates, AggregationMethod.MEDIAN)
    aggregated_result = aggregate_weights(client_updates, AggregationMethod.MEDIAN)

    # 3. Calculate metrics using the buggy reference
    # This should yield L2 distance of 0.0
    metrics_before_fix = calculate_robustness_metrics(client_updates, buggy_benign_mean, aggregated_result)
    assert np.isclose(metrics_before_fix["l2_to_benign_mean"], 0.0)
    assert np.isclose(metrics_before_fix["cos_to_benign_mean"], 1.0)

    # 4. With 2 clients, Median equals FedAvg, so L2 would still be 0.
    # The test needs a more complex scenario with 3+ clients to demonstrate
    # that FedAvg and Median produce different results.

    client_updates_3 = [
        [np.array([1.0, 2.0])],  # Client 1
        [np.array([3.0, 4.0])],  # Client 2
        [np.array([8.0, 9.0])],  # Client 3 (outlier)
    ]

    # Median will pick the middle elements: [3.0, 4.0]
    aggregated_median_3 = aggregate_weights(client_updates_3, AggregationMethod.MEDIAN)
    assert np.allclose(aggregated_median_3[0], [3.0, 4.0])

    # FedAvg will be the average: [(1+3+8)/3, (2+4+9)/3] = [4.0, 5.0]
    reference_fedavg_3 = aggregate_weights(client_updates_3, AggregationMethod.FED_AVG)
    assert np.allclose(reference_fedavg_3[0], [4.0, 5.0])

    final_metrics = calculate_robustness_metrics(client_updates_3, reference_fedavg_3, aggregated_median_3)

    # The L2 distance between [3, 4] and [4, 5] should be sqrt((4-3)^2 + (5-4)^2) = sqrt(2)
    expected_l2_dist = np.sqrt(2.0)
    assert np.isclose(final_metrics["l2_to_benign_mean"], expected_l2_dist)
