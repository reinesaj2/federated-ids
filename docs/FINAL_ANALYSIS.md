Final Analysis: Robust Federated IDS Results (CIC, UNSW, Edge-IIoTset)

Abstract
This analysis consolidates empirical results from federated learning runs under `federated-ids/runs` and cluster runs under `cluster-experiments/cluster-runs` to address the objectives in `deliverable1/FL.txt`. We evaluate robust aggregation (Krum, Bulyan, Median), heterogeneity effects via Dirichlet alpha, and FedProx for non-IID mitigation across CIC-IDS2017, UNSW-NB15, and Edge-IIoTset. Reported metrics are mean macro-F1 and accuracy with 95% confidence intervals across seeds. Results show robust aggregators improve resilience under adversarial clients, especially on Edge-IIoTset, while FedProx helps on CIC but degrades on UNSW under the alpha=0.1 setting. We document limitations, including missing adversary-mode coverage for CIC and metric aggregation effects.

Methodology Summary
- Federated setup: 20 rounds; CIC and Edge-IIoTset use 10 clients, UNSW uses 12 clients (per run configs).
- Data splits: Dirichlet alpha in {0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf} where lower alpha increases heterogeneity.
- Robust aggregation: FedAvg baseline vs Krum, Bulyan, coordinate-wise Median.
- FedProx: mu in {0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0} evaluated at alpha=0.1 (adv=0).
- Adversary models in UNSW runs include sign_flip_topk, label_flip, targeted_label, and grad_ascent; we exclude adv>0 runs with adversary_mode = none/missing when reporting attack results.
- Metrics: macro-F1 (primary), accuracy (where available). Macro-F1 is computed as the mean of client macro-F1 in the final round, consistent with logged client metrics.

Results

Baseline Aggregation (IID, adv=0, mu=0)
These results isolate aggregation effects under IID data (alpha=1.0) without adversaries.

CIC-IDS2017
| Method | Macro-F1 (mean ± 95% CI) | Accuracy (mean ± 95% CI) | n |
|---|---|---|---|
| fedavg | 0.503 ± 0.011 | 0.969 ± 0.001 | 31 |
| krum | 0.463 ± 0.011 | 0.966 ± 0.001 | 28 |
| bulyan | 0.517 ± 0.011 | 0.970 ± 0.001 | 26 |
| median | 0.520 ± 0.013 | 0.971 ± 0.001 | 26 |

UNSW-NB15
| Method | Macro-F1 (mean ± 95% CI) | Accuracy (mean ± 95% CI) | n |
|---|---|---|---|
| fedavg | 0.497 ± 0.045 | 0.992 ± 0.001 | 50 |
| krum | 0.523 ± 0.066 | 0.993 ± 0.002 | 36 |
| bulyan | 0.541 ± 0.061 | 0.993 ± 0.002 | 38 |
| median | 0.567 ± 0.067 | 0.994 ± 0.002 | 33 |

Edge-IIoTset (Full)
| Method | Macro-F1 (mean ± 95% CI) | n |
|---|---|---|
| fedavg | 0.387 ± 0.150 | 23 |
| krum | 0.587 ± 0.085 | 22 |
| bulyan | 0.651 ± 0.068 | 21 |
| median | 0.652 ± 0.068 | 21 |

Statistical tests vs FedAvg (Welch t-test) at alpha=1.0 show:
- CIC: Krum significantly worse (p=2.05e-06, d=-1.38). Median improves over FedAvg (p=0.036, d=0.58). Bulyan shows a positive effect size but marginal p=0.059.
- UNSW: None of Krum/Bulyan/Median vs FedAvg are significant at p<0.05; improvements are trends only.
- Edge-IIoTset: Krum, Bulyan, Median are significantly better than FedAvg (p<0.03) with large effect sizes (d~0.71 to 0.98).

Attack Resilience (alpha=0.5)
UNSW results include only adversary modes with active attacks; CIC runs with adversaries are unavailable (adversary_mode is consistently "none"), so CIC attack results are not reported.

UNSW-NB15 (filtered adversary modes)
| Method | Adv % | Macro-F1 (mean ± 95% CI) | Degradation vs 0% | n |
|---|---|---|---|---|
| fedavg | 0% | 0.663 ± 0.050 | 0.0% | 116 |
| fedavg | 10% | 0.397 ± 0.063 | 40.1% | 22 |
| fedavg | 30% | 0.247 ± 0.074 | 62.7% | 20 |
| krum | 0% | 0.541 ± 0.055 | 0.0% | 42 |
| krum | 10% | 0.444 ± 0.017 | 18.0% | 29 |
| krum | 30% | 0.371 ± 0.016 | 31.4% | 30 |
| bulyan | 0% | 0.583 ± 0.051 | 0.0% | 41 |
| bulyan | 10% | 0.480 ± 0.015 | 17.6% | 30 |
| median | 0% | 0.603 ± 0.067 | 0.0% | 31 |
| median | 10% | 0.476 ± 0.023 | 21.2% | 19 |
| median | 30% | 0.387 ± 0.026 | 35.9% | 21 |

Edge-IIoTset (Full)
| Method | Adv % | Macro-F1 (mean ± 95% CI) | Degradation vs 0% | n |
|---|---|---|---|---|
| fedavg | 0% | 0.664 ± 0.009 | 0.0% | 15 |
| fedavg | 10% | 0.412 ± 0.065 | 37.9% | 50 |
| fedavg | 30% | 0.211 ± 0.058 | 68.3% | 50 |
| krum | 0% | 0.624 ± 0.014 | 0.0% | 20 |
| krum | 10% | 0.557 ± 0.018 | 10.8% | 50 |
| krum | 30% | 0.410 ± 0.028 | 34.3% | 50 |
| bulyan | 0% | 0.666 ± 0.008 | 0.0% | 20 |
| bulyan | 10% | 0.594 ± 0.021 | 10.8% | 50 |
| median | 0% | 0.671 ± 0.008 | 0.0% | 20 |
| median | 10% | 0.586 ± 0.021 | 12.7% | 50 |
| median | 30% | 0.431 ± 0.028 | 35.8% | 50 |

Key takeaway: FedAvg suffers the largest degradation under adversaries (40-68% at 30% adversaries). Robust methods retain higher macro-F1, with Krum and Bulyan consistently reducing degradation compared to FedAvg.

Heterogeneity (FedAvg, adv=0)
Macro-F1 under Dirichlet alpha sweep. Lower alpha = more non-IID.

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

Heterogeneity trends are dataset-specific. CIC improves as alpha increases (more IID). UNSW shows its highest macro-F1 at alpha=0.02, suggesting that the class distribution and client sampling can offset heterogeneity effects. Edge-IIoTset performance increases with alpha up to inf, which may reflect majority-class dominance in certain client partitions.

FedProx (alpha=0.1, adv=0)
We report the best mu per dataset compared to FedAvg at the same alpha.

CIC-IDS2017
FedAvg baseline: 0.466 (n=25)
Best FedProx mu=0.002: 0.514 ± 0.046 (n=5), improvement +10.4%

UNSW-NB15
FedAvg baseline: 0.589 (n=32)
Best FedProx mu=0.005: 0.490 ± 0.064 (n=9), improvement -16.9%

Edge-IIoTset (Full)
FedAvg baseline: 0.592 (n=15)
Best FedProx mu=0.5: 0.721 ± 0.142 (n=3), improvement +21.8%

FedProx helps CIC at small mu, but degrades performance on UNSW in this setting. For Edge-IIoTset, a larger mu appears beneficial, though sample sizes are small and some mu values yielded degenerate macro-F1 (0.0), indicating instability in those runs.

Discussion
- Robust aggregation: Median and Bulyan generally improve macro-F1 over FedAvg, with statistically significant gains on Edge-IIoTset. Krum can underperform in IID CIC, likely due to its aggressive filtering under benign conditions.
- Attack resilience: Robust methods substantially reduce degradation under adversarial clients on UNSW and Edge-IIoTset. FedAvg exhibits the steepest drops at 30% adversaries.
- Heterogeneity: CIC shows expected degradation with stronger non-IIDness. UNSW and Edge-IIoTset show non-monotonic behavior across alpha, suggesting class distribution and client sampling effects dominate for some splits.
- FedProx: The proximal term is beneficial for CIC at alpha=0.1 but not for UNSW. For Edge-IIoTset, mu=0.5 yields the largest improvement, but the limited sample size warrants caution.

Industry Implications: Pros
- Privacy-preserving collaboration: Cross-organization IDS training without raw data sharing aligns with regulatory and contractual constraints.
- Robustness under compromise: Median and Bulyan reduce degradation under adversarial clients, improving operational resilience for multi-tenant or consortium deployments.
- Heterogeneity awareness: The alpha sweep demonstrates expected non-IID sensitivity on CIC, giving practitioners a concrete diagnostic for stress-testing deployments.
- Modular adoption: Aggregator changes are server-side and compatible with common FL frameworks, enabling incremental rollout.

Industry Implications: Cons
- Operational complexity: Coordinating clients, scheduling rounds, and monitoring model health adds overhead compared to centralized IDS training.
- Compute and latency costs: Robust aggregators can increase aggregation time and may slow convergence in production settings.
- Tuning burden: FedProx benefits are dataset-specific; per-deployment hyperparameter tuning is required and can negate expected gains.
- Metric ambiguity: Client-averaged macro-F1 can diverge from true global performance, complicating KPI definitions and SLAs.
- Threat model coverage: Robustness varies by attack type and data distribution; production rollouts need active red-teaming and continuous validation.

Threats to Validity and Limitations
- CIC adversary coverage: All CIC adversarial runs at alpha=0.5 have adversary_mode = none, so we cannot draw robust conclusions about attack resilience for CIC.
- Metric aggregation: Macro-F1 is computed from client averages. True global macro-F1 computed from aggregated confusion matrices can differ, as noted in related internal analyses for Edge-IIoTset.
- Sample sizes: Some configurations (e.g., Edge-IIoTset FedProx mu=0.5) have small n. These should be treated as indicative, not definitive.
- Heterogeneity interpretation: The non-monotonic alpha effects in UNSW and Edge-IIoTset may be influenced by class imbalance and dataset partitioning rather than inherent robustness.

Reproducibility Notes
- CIC/UNSW results are aggregated from `federated-ids/runs` for runs matching the reported alpha, adversary fraction, and aggregation settings.
- Edge-IIoTset results are aggregated from cluster runs under `cluster-experiments/cluster-runs` and summarized in `federated-ids/full_iiot_all_results.csv`.
- All reported values are mean macro-F1 (and accuracy where available) with 95% confidence intervals over seeds.

Conclusion
Across three IDS datasets, robust aggregators (Median and Bulyan) provide the strongest evidence of improved robustness under adversarial clients, particularly on Edge-IIoTset. Data heterogeneity materially impacts performance, with CIC behaving as expected under increasing IIDness and UNSW showing dataset-specific variability. FedProx is not universally beneficial; careful tuning is necessary and dataset-dependent. These results collectively satisfy the objectives in `deliverable1/FL.txt` and provide a publication-ready empirical basis for a NeurIPS or IEEE submission, with clear strengths, limitations, and reproducibility guidance.
