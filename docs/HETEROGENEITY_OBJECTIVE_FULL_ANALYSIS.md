Heterogeneity Objective: Full Analysis (Non-IID Impact and FedProx Mitigation)

Objective
Assess how non-IID client data affects federated IDS performance and whether FedProx mitigates heterogeneity. This directly addresses Objective 2 in `deliverable1/FL.txt`: quantify performance under varying alpha and evaluate FedProx across datasets.

Data Sources and Selection
- CIC-IDS2017 and UNSW-NB15 runs: `federated-ids/runs`
  - Filters: adversary_fraction = 0, dp_enabled = false, personalization_epochs = 0, fedprox_mu = 0 for heterogeneity sweeps.
  - Metrics: macro-F1 computed as the mean of client macro-F1 at the final round (from `client_*_metrics.csv`).
- Edge-IIoTset (full) runs: `cluster-experiments/cluster-runs`, summarized in `federated-ids/full_iiot_all_results.csv`.
  - Filters: adv_percent = 0, mu = 0 for heterogeneity sweeps.
  - Metrics: macro-F1 from aggregated summary file.
- Alpha values: {0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf}.
- Statistical tests: Welch t-tests; effect size via Cohen’s d. Monotonic trend captured with Spearman rho using ordinal alpha ranks.

Results: Heterogeneity Sweep (FedAvg, adv=0)
Macro-F1 (mean ± 95% CI) across alpha values.

CIC-IDS2017
| Alpha | Macro-F1 (mean ± 95% CI) | n |
|---|---|---|
| 0.02 | 0.355 ± 0.027 | 25 |
| 0.05 | 0.425 ± 0.017 | 26 |
| 0.1 | 0.466 ± 0.017 | 25 |
| 0.2 | 0.502 ± 0.016 | 25 |
| 0.5 | 0.497 ± 0.011 | 25 |
| 1.0 | 0.503 ± 0.011 | 31 |
| inf | 0.523 ± 0.010 | 25 |

UNSW-NB15
| Alpha | Macro-F1 (mean ± 95% CI) | n |
|---|---|---|
| 0.02 | 0.607 ± 0.065 | 38 |
| 0.05 | 0.577 ± 0.066 | 37 |
| 0.1 | 0.589 ± 0.072 | 32 |
| 0.2 | 0.562 ± 0.063 | 36 |
| 0.5 | 0.551 ± 0.066 | 35 |
| 1.0 | 0.497 ± 0.045 | 50 |
| inf | 0.515 ± 0.072 | 34 |

Edge-IIoTset (Full)
| Alpha | Macro-F1 (mean ± 95% CI) | n |
|---|---|---|
| 0.02 | 0.385 ± 0.035 | 18 |
| 0.05 | 0.473 ± 0.039 | 18 |
| 0.1 | 0.592 ± 0.024 | 15 |
| 0.2 | 0.644 ± 0.012 | 15 |
| 0.5 | 0.664 ± 0.009 | 15 |
| 1.0 | 0.387 ± 0.150 | 23 |
| inf | 0.713 ± 0.005 | 15 |

Summary of Heterogeneity Sensitivity
| Dataset | Spearman rho (p) | Low alpha vs IID | Low alpha vs inf | Range (max-min) | Low vs IID p (d) | Low vs inf p (d) | Best alpha |
|---|---|---|---|---|---|---|---|
| cic | 0.721 (p=1.64e-30) | -29.5% | -32.3% | 0.169 | p=9.35e-12, d=-3.03 | p=3.97e-13, d=-3.38 | inf |
| unsw | -0.310 (p=3.14e-07) | 22.2% | 17.9% | 0.110 | p=6.38e-03, d=0.62 | p=5.82e-02, d=0.46 | 0.02 |
| iiot | 0.654 (p=7.02e-16) | -0.3% | -45.9% | 0.328 | p=9.88e-01, d=-0.00 | p=2.08e-13, d=-6.24 | inf |

Interpretation
- CIC-IDS2017 shows a strong, monotonic improvement as alpha increases. Non-IID partitions (alpha=0.02) reduce macro-F1 by ~30% relative to IID, with large effect sizes (d≈-3.0).
- UNSW-NB15 exhibits the opposite trend: lower alpha improves macro-F1. This suggests that for this dataset and client partitioning, local specialization can outweigh global aggregation losses, likely due to class imbalance structure and client sampling.
- Edge-IIoTset demonstrates the largest heterogeneity sensitivity (range 0.328). Alpha=1.0 performs poorly relative to alpha=inf, while alpha=0.02 is nearly identical to IID. This indicates that alpha=1.0 is not necessarily the best proxy for IID behavior in this dataset.

Results: FedProx Mitigation (alpha=0.1, adv=0)
Best FedProx mu per dataset vs FedAvg baseline.

| Dataset | FedAvg (alpha=0.1) | Best FedProx mu | Best FedProx | Delta vs FedAvg | p (d) |
|---|---|---|---|---|---|
| cic | 0.466 ± 0.017 (n=25) | 0.002 | 0.514 ± 0.046 (n=5) | +10.4% | p=0.038, d=1.17 |
| unsw | 0.589 ± 0.072 (n=32) | 0.005 | 0.490 ± 0.064 (n=9) | -16.9% | p=0.033, d=-0.55 |
| iiot | 0.592 ± 0.024 (n=15) | 0.5 | 0.721 ± 0.142 (n=3) | +21.8% | p=0.047, d=2.88 |

Stability note: Edge-IIoTset FedProx runs at alpha=0.1 show a 59.3% zero macro-F1 rate (n=27), indicating instability for several mu settings. The best mu result is promising but based on a small sample.

Implications for Objective 2
- Heterogeneity impact is dataset-dependent. CIC aligns with expected non-IID degradation, while UNSW and Edge-IIoTset reveal non-monotonic behavior, emphasizing the need to analyze heterogeneity in context rather than assuming a universal trend.
- FedProx provides meaningful gains on CIC and Edge-IIoTset but is harmful on UNSW for alpha=0.1. This indicates FedProx tuning is necessary and not universally beneficial.
- The heterogeneity objective is satisfied by quantifying alpha effects, identifying statistically significant changes, and testing FedProx mitigation with clear mixed outcomes.

Limitations and Follow-Up
- Alpha=1.0 is not always the empirical peak; alpha=inf often yields higher macro-F1, indicating that alpha choice can interact with dataset and client sampling in non-trivial ways.
- Edge-IIoTset FedProx instability warrants a targeted sweep to identify stable mu regimes or alternative regularization strategies.
- UNSW’s inverted alpha trend suggests evaluating stratified or label-balanced splits to isolate heterogeneity from class imbalance effects.
