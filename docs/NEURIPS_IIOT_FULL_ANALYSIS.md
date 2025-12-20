# NeurIPS Success Criteria Analysis: Full Edge-IIoTset Experiments

**Date:** 2025-12-19
**Dataset:** Edge-IIoTset (dsedge-iiotset-full)
**Total Experiments:** 1,740 valid, complete runs (98.3% completion rate)
**Status:** Partial readiness - critical gaps identified

---

## Executive Summary

This document provides a self-skeptical assessment of the full Edge-IIoTset experimental corpus against NeurIPS publication success criteria. The analysis identifies both significant strengths and critical gaps that must be addressed before submission.

**Key Findings:**

1. Robust aggregation (Krum) delivers **669% improvement** over FedAvg under 30% Byzantine attack
2. **Zero experiments** exist for the "Combined Robustness + Heterogeneity" ablation
3. Only one attack type (grad_ascent) tested - insufficient for generalizability claims
4. FedProx domain-specific failure mode is a **novel, publication-ready finding**

---

## Criterion 2: Experimental Rigor

### 2.1 Baseline Coverage

| Requirement | Status | Evidence        |
| ----------- | ------ | --------------- |
| FedAvg      | PASS   | 567 experiments |
| FedProx     | PASS   | 355 experiments |
| Krum        | PASS   | 563 experiments |
| Bulyan      | PASS   | 266 experiments |
| Median      | PASS   | 165 experiments |

**Verdict:** PASS - All required baselines present

### 2.2 Attack Model Coverage

| Attack Type                      | Status   | Count                       |
| -------------------------------- | -------- | --------------------------- |
| grad_ascent (gradient sign-flip) | PASS     | All adversarial experiments |
| label_flipping                   | **FAIL** | 0                           |
| gaussian_noise                   | **FAIL** | 0                           |
| model_replacement                | **FAIL** | 0                           |

**Verdict:** FAIL - Only 1 of 4 required attack types

### 2.3 Adversarial Fraction Coverage

| Fraction | Status | Count |
| -------- | ------ | ----- |
| 0%       | PASS   | 846   |
| 10%      | PASS   | 465   |
| 20%      | PASS   | 305   |
| 30%      | PASS   | 300   |

**Verdict:** PASS - 0-30% adversary coverage complete (40% deferred)

### 2.4 Non-IID Partition Coverage

| Alpha | Status | Interpretation                 |
| ----- | ------ | ------------------------------ |
| 0.02  | PASS   | Extreme heterogeneity          |
| 0.05  | PASS   | Severe heterogeneity           |
| 0.10  | PASS   | High heterogeneity             |
| 0.20  | PASS   | Moderate heterogeneity         |
| 0.50  | PASS   | Low heterogeneity              |
| 1.00  | PASS   | Near-IID                       |
| inf   | PASS   | IID (equivalent to alpha=10.0) |

**Verdict:** PASS - Excellent alpha coverage

### 2.5 Statistical Significance

| Metric                   | Value |
| ------------------------ | ----- |
| Configs with >= 5 seeds  | 91    |
| Configs with >= 10 seeds | 77    |
| Configs with >= 20 seeds | 70    |
| Mean seeds per config    | ~13   |

**Verdict:** PASS - Exceeds minimum requirement of 5 seeds

---

## Criterion 4: Ablation Studies

### 4.1 Aggregator Alone (Robust Agg under Non-IID, no FedProx)

| Aggregator | Alpha | Mean F1 | Count |
| ---------- | ----- | ------- | ----- |
| Bulyan     | 0.02  | 0.389   | 40    |
| Bulyan     | 1.00  | 0.656   | 40    |
| Krum       | 0.02  | 0.246   | 80    |
| Krum       | 1.00  | 0.560   | 80    |
| Median     | 0.02  | 0.289   | 37    |
| Median     | 1.00  | 0.654   | 40    |

**Verdict:** PASS - 977 experiments, comprehensive coverage

### 4.2 FedProx Alone (under Byzantine attack)

| Experiments          | Count |
| -------------------- | ----- |
| FedProx with adv > 0 | **0** |

**Verdict:** FAIL - No FedProx experiments under Byzantine attack

### 4.3 Combined (Robust Agg + FedProx)

| Experiments            | Count |
| ---------------------- | ----- |
| Robust Agg with mu > 0 | **0** |

**Verdict:** FAIL - Zero combined experiments

---

## Criterion 6: Key Claims Verification

### Claim A: Robust aggregators degrade under severe non-IID (alpha < 0.5)

| Condition                            | Mean F1   |
| ------------------------------------ | --------- |
| Robust Agg at IID (alpha >= 1.0)     | 0.686     |
| Robust Agg at Non-IID (alpha <= 0.1) | 0.415     |
| **Degradation**                      | **39.4%** |

**Verdict:** VERIFIABLE - Strong evidence supports this claim

### Claim B: FedProx alone fails under >20% Byzantine

| Condition          | Mean F1 |
| ------------------ | ------- |
| FedProx at 0% adv  | 0.597   |
| FedProx at 20% adv | NO DATA |
| FedProx at 30% adv | NO DATA |

**Verdict:** NOT VERIFIABLE - Missing data

### Claim C: Combined approach maintains performance under stress

**Verdict:** NOT VERIFIABLE - Zero combined experiments

### Claim D: Attack-type-specific insights

**Verdict:** NOT VERIFIABLE - Only one attack type

### Claim E: Computational overhead acceptable

| Aggregator | Mean Time (ms) | Std (ms) |
| ---------- | -------------- | -------- |
| FedAvg     | 2.3            | 0.5      |
| Krum       | 16.3           | 1.6      |
| Bulyan     | 49.6           | 5.1      |
| Median     | 45.9           | 10.4     |

**Verdict:** VERIFIABLE - Overhead is reasonable (7-25x FedAvg)

---

## Publication-Ready Strengths

### Strength 1: Robust Aggregation Effectiveness

```
FedAvg under 30% Byzantine (alpha=1.0): 0.061 +/- 0.052 F1
Krum under 30% Byzantine (alpha=1.0):   0.471 +/- 0.044 F1
IMPROVEMENT: 669%
```

### Strength 2: Non-IID Degradation Quantified

| Alpha | FedAvg F1 | Interpretation |
| ----- | --------- | -------------- |
| inf   | 0.713     | IID baseline   |
| 1.0   | 0.684     | -4%            |
| 0.5   | 0.664     | -7%            |
| 0.2   | 0.644     | -10%           |
| 0.1   | 0.592     | -17%           |
| 0.05  | 0.473     | -34%           |
| 0.02  | 0.385     | -46%           |

### Strength 3: Novel FedProx Finding

From `docs/FEDPROX_NOVELTY_ANALYSIS.md`:

- Li et al. (2020): FedProx improves accuracy on balanced tasks
- Our finding: FedProx DEGRADES macro-F1 on imbalanced IDS
- Root cause: Proximal term constrains minority class specialization
- This is a **first-of-its-kind domain-specific insight**

---

## Critical Gaps Summary

| Gap                    | Impact                        | Required Action                   |
| ---------------------- | ----------------------------- | --------------------------------- |
| Single attack type     | Cannot claim generalizability | Add label_flipping, noise attacks |
| No FedProx + Byzantine | Cannot test Claim B           | Run FedProx at adv={10,20,30}%    |
| No Combined ablation   | Cannot prove synergy          | Run Krum+FedProx, Bulyan+FedProx  |

---

## Recommended Publication Paths

### Option A: Robust Aggregation for Federated IDS (80% Ready)

**Focus:** Krum/Bulyan/Median effectiveness against Byzantine attacks in IDS domain

**Novel contribution:** First comprehensive evaluation of Byzantine-tolerant FL for network intrusion detection with 15-class Edge-IIoTset

**Remaining work:**

- Add 2+ additional attack types

### Option B: FedProx Failure Mode Discovery (90% Ready)

**Focus:** Domain-specific failure of FedProx on imbalanced security tasks

**Novel contribution:** First to demonstrate FedProx degrades macro-F1 despite reducing model drift, contradicting Li et al. (2020) in imbalanced domains

**Remaining work:**

- Minor additional experiments for robustness
- Theoretical analysis of why proximal term hurts minority classes

### Option C: Combined Robustness + Heterogeneity (0% Ready)

**Focus:** Synergy between robust aggregation and proximal regularization

**Current status:** ZERO experiments exist. Requires complete new experimental campaign.

---

## Appendix: Experiment Configuration Space

```
Aggregators: {fedavg, fedprox, krum, bulyan, median}
Alpha (Non-IID): {0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf}
Adversary Fraction: {0%, 10%, 20%, 30%}
FedProx Mu: {0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0}
Seeds: 42-61 (up to 20 per config)
Clients: 10
Rounds: 20
Dataset: Edge-IIoTset Full (15 classes, ~1.5M samples)
```

---

## References

- `docs/FEDPROX_NOVELTY_ANALYSIS.md` - FedProx domain-specific findings
- `docs/OBJECTIVE_2_INVESTIGATION_SUMMARY.md` - FedProx implementation verification
- `docs/CLUSTER_RUNS_ANALYSIS.md` - Raw experiment coverage statistics
- `iiot_comprehensive_plots/` - Generated visualizations
