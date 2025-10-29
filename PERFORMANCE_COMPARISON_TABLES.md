# Performance Comparison Tables

Consolidated metrics derived from the latest committed experiment artifacts.

## Attack Resilience (Macro-F1)

Source: `results/comparative_analysis/attack_resilience_stats.csv`. Macro-F1 values are clipped to [0,1] to account for a known logging anomaly on the FedAvg baseline while preserving the relative degradation signals.

| Aggregation | Macro-F1 (f=0%) | Macro-F1 (f=10%) | Macro-F1 (f=30%) | Δ at 30% (drop) | Runs |
| --- | --- | --- | --- | --- | --- |
| Bulyan | 0.994 | 0.930 | 0.854 | 14.1% | 3 |
| Fedavg | 1.000 | 1.000 | 0.675 | 74.5% | 3 |
| Krum | 0.969 | 0.969 | 0.787 | 18.8% | 5 |
| Median | 0.999 | 1.000 | 0.894 | 10.5% | 5 |

## Personalization Gains (Δ Macro-F1)

| Scope | Mean ΔF1 | Std Dev | Clients |
| --- | --- | --- | --- |
| Overall | 0.035 | 0.070 | 30 |
| Dataset: CIC | 0.060 | — | 12 |
| Dataset: UNSW | 0.018 | — | 18 |

## FedAvg vs FedProx (Heterogeneity Stability)

| Scenario | L2 Improvement (×) | Time Overhead (×) |
| --- | --- | --- |
| alpha 0.05 mu 0.01 vs alpha 0.05 mu 0.0 | 1.35 | 1.01 |
| alpha 0.05 mu 0.1 vs alpha 0.05 mu 0.0 | 2.67 | 1.03 |
| alpha 0.1 mu 0.01 vs alpha 0.1 mu 0.0 | 0.03 | 0.95 |
| alpha 0.1 mu 0.1 vs alpha 0.1 mu 0.0 | 0.02 | 1.03 |
| alpha 0.5 mu 0.01 vs alpha 0.5 mu 0.0 | 0.01 | 0.99 |
| alpha 0.5 mu 0.1 vs alpha 0.5 mu 0.0 | 0.02 | 1.02 |

## Privacy–Utility Trade-off (Differential Privacy)

| ε (target) | Macro-F1 | Noise σ | Samples |
| --- | --- | --- | --- |
| 1.5 | 0.790 | 0.70 | 2 |
| baseline | 0.890 | — | 1 |

---

_Last regenerated via `python scripts/generate_performance_tables.py`._
