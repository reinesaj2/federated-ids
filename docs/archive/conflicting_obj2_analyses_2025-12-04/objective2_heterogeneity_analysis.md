# Objective 2: Heterogeneity Analysis for IIoT

**Date:** December 2, 2025  
**Dataset:** Edge-IIoTset  
**Total Experiments Analyzed:** 616 runs

---

## Executive Summary

Objective 2 (Heterogeneity/Non-IID data handling) shows a **robust null result** across all tested dimensions for the IIoT dataset. This finding is consistent across all three datasets (UNSW, CIC, IIoT) and represents a scientifically valuable contribution.

| Dimension         | Result    | p-value | Cohen's d | Verdict    |
| ----------------- | --------- | ------- | --------- | ---------- |
| Final Accuracy    | NO EFFECT | 0.94    | 0.006     | NULL       |
| Convergence Speed | NO EFFECT | 0.51    | -0.19     | NULL       |
| Early Performance | MINIMAL   | -       | -         | NULL       |
| FedProx Benefit   | +2.5% max | -       | -         | NEGLIGIBLE |

---

## Current Experimental Coverage

### Heterogeneity-Relevant Runs (adv=0, dp=0, pers=0)

| Alpha | Mu=0.0 | Mu=0.01 | Mu=0.05 | Mu=0.1 | Status          |
| ----- | ------ | ------- | ------- | ------ | --------------- |
| 0.02  | 228    | 48      | 48      | 48     | COMPLETE        |
| 0.05  | 179    | 30      | 30      | 30     | COMPLETE        |
| 0.1   | 164    | 24      | 24      | 24     | LOW             |
| 0.2   | 156    | 30      | 30      | 30     | COMPLETE        |
| 0.5   | 763    | 48      | 18      | 18     | LOW             |
| 1.0   | 453    | 0       | 0       | 0      | MISSING FedProx |
| inf   | 17     | 0       | 0       | 0      | Baseline only   |

---

## Statistical Analysis Results

### 1. Effect of Alpha (IID vs Non-IID) at mu=0

```
IID (alpha=1.0):     F1 = 0.7062 +/- 0.059 (n=453)
Non-IID (alpha=0.02): F1 = 0.7059 +/- 0.033 (n=228)

Difference: 0.0003 (essentially ZERO)
p-value: 0.9434 (NOT significant)
Cohen's d: 0.006 (NEGLIGIBLE)
```

**Interpretation:** There is NO detectable difference in final model accuracy between IID and extreme non-IID data distributions.

### 2. Convergence Speed Analysis

| Alpha | Final F1 | Rounds to 90% | Rounds to 95% | Rounds to 99% |
| ----- | -------- | ------------- | ------------- | ------------- |
| 0.02  | 0.7114   | 5.2           | 9.8           | 14.3          |
| 0.05  | 0.7020   | 4.7           | 9.6           | 13.7          |
| 0.1   | 0.6995   | 5.0           | 10.2          | 15.2          |
| 0.5   | 0.6732   | 4.2           | 7.5           | 13.5          |
| 1.0   | 0.7144   | 5.3           | 10.4          | 14.8          |

```
Statistical Test (Rounds to 95% F1):
  IID:     10.4 +/- 3.5 rounds (n=72)
  Non-IID: 9.8 +/- 2.5 rounds (n=20)

  Difference: -0.6 rounds (Non-IID slightly FASTER)
  p-value: 0.5092 (NOT significant)
  Cohen's d: -0.185 (NEGLIGIBLE)
```

**Interpretation:** Heterogeneity does NOT slow convergence. If anything, extreme non-IID converges slightly faster (not significant).

### 3. FedProx Effect at Extreme Non-IID (alpha=0.02)

| FedProx mu   | F1     | Improvement vs FedAvg | p-value |
| ------------ | ------ | --------------------- | ------- |
| 0.0 (FedAvg) | 0.7059 | baseline              | -       |
| 0.01         | 0.7023 | -0.36%                | 0.49    |
| 0.05         | 0.7066 | +0.07%                | 0.89    |
| 0.1          | 0.7033 | -0.26%                | 0.61    |

**Interpretation:** FedProx provides NO benefit at extreme non-IID. Some mu values actually hurt performance.

### 4. Best FedProx Improvements (Any Alpha)

| Alpha | Mu   | FedAvg F1 | FedProx F1 | Improvement | N   |
| ----- | ---- | --------- | ---------- | ----------- | --- |
| 0.5   | 0.05 | 0.6681    | 0.6935     | +2.54%      | 18  |
| 0.5   | 0.1  | 0.6681    | 0.6880     | +2.00%      | 18  |
| 0.5   | 0.01 | 0.6681    | 0.6774     | +0.93%      | 48  |

**Interpretation:** Best case is +2.54% improvement at alpha=0.5, mu=0.05. However:

- Sample size is small (n=18)
- Effect is not consistent across configs
- Most configs show no improvement or regression

---

## Theoretical Explanation

### Why IDS Data is Heterogeneity-Resilient

1. **Universal Attack Signatures:** Network attacks have consistent patterns regardless of deployment location. A DDoS attack looks the same whether observed in Factory A or Factory B.

2. **Anomaly-Based Detection:** IDS fundamentally looks for deviations from normal behavior. The "normal" baseline may vary, but attack patterns are consistent.

3. **Feature Engineering:** Network flow features (packet sizes, timing, protocol usage) are standardized and don't exhibit the semantic drift seen in image/text data.

4. **Class Distribution vs Feature Distribution:** Non-IID in FL-IDS primarily affects which attack types each client sees, not the feature distributions of those attacks.

---

## Paths to Publishability

### PATH 1: Embrace the Null Result (RECOMMENDED)

**Publication Angle:** "On the Natural Robustness of Federated IDS to Non-IID Data"

**Key Arguments:**

1. Consistent null result across 3 diverse IDS datasets
2. Theoretical basis: attack signatures are universal
3. Practical implication: simpler FL systems work for IDS
4. Community value: prevents wasted research effort

**Required Work:**

- Cross-dataset comparison analysis (data exists)
- Statistical power analysis
- Theoretical explanation writeup

**Target Venues:**

- FL-ICML Workshop, NeurIPS FL Workshop
- ACSAC, RAID (short papers)
- Negative results tracks

**Effort:** LOW (analysis only)
**Risk:** LOW (guaranteed outcome)

### PATH 2: Pursue Extreme Heterogeneity

**Hypothesis:** Current alpha range (0.02-1.0) may not be extreme enough.

**New Experiments Required:**

#### Priority 1: Fill Grid Gaps (38 runs)

```
alpha=0.1, mu=[0.01, 0.05, 0.1], seeds=[42-46]  # 15 runs
alpha=0.5, mu=[0.05, 0.1], seeds=[43-46]        # 8 runs
alpha=1.0, mu=[0.01, 0.05, 0.1], seeds=[42-46]  # 15 runs
```

#### Priority 2: Extreme Alpha (40 runs)

```
alpha=0.005, mu=[0.0, 0.01, 0.05, 0.1], seeds=[42-46]  # 20 runs
alpha=0.01, mu=[0.0, 0.01, 0.05, 0.1], seeds=[42-46]   # 20 runs
```

**Total New Experiments:** 78 runs

**Effort:** HIGH (~40-80 GPU hours)
**Risk:** HIGH (likely to remain null)

---

## Recommended Action

**STRONG RECOMMENDATION: PATH 1**

The null result IS the publishable finding for Objective 2. Running additional experiments has high risk of the same outcome with significant compute cost.

### If PATH 2 is Chosen

1. Run Priority 2 (extreme alpha) FIRST - 40 runs
2. Analyze results immediately
3. If still null, STOP and accept the finding
4. DO NOT proceed to "more clients" experiments

---

## Experiment Commands (PATH 2)

```bash
# Priority 2: Extreme Heterogeneity Experiments
for alpha in 0.005 0.01; do
  for mu in 0.0 0.01 0.05 0.1; do
    for seed in 42 43 44 45 46; do
      python server.py \
        --dataset edge-iiotset-nightly \
        --alpha $alpha \
        --fedprox_mu $mu \
        --seed $seed \
        --num_rounds 20 \
        --num_clients 5 \
        --adversary_fraction 0.0 \
        --dp_enabled 0 \
        --personalization_epochs 0
    done
  done
done
```

---

## Cross-Dataset Comparison

| Dataset  | IID F1 | Non-IID F1 | p-value | Cohen's d |
| -------- | ------ | ---------- | ------- | --------- |
| **UNSW** | 0.9998 | 0.9998     | 0.83    | 0.07      |
| **CIC**  | 0.7243 | 0.7455     | 0.57    | -0.24     |
| **IIoT** | 0.7062 | 0.7059     | 0.94    | 0.006     |

**All three datasets show null results for heterogeneity effect.**

This cross-dataset consistency strengthens the case for PATH 1 (null result publication).

---

## References

1. Li, T., et al. (2020). Federated optimization in heterogeneous networks. MLSys. (FedProx paper)
2. Zhao, Y., et al. (2018). Federated learning with non-IID data. arXiv:1806.00582.
3. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. ICML.
