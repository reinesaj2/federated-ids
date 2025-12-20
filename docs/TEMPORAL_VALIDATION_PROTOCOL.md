# Temporal Validation Protocol for FedProx Hyperparameter Selection

**Created:** 2025-12-17  
**Status:** Pre-registered Protocol  
**Purpose:** Define a rigorous, publishable methodology for comparing FedProx vs FedAvg under non-IID conditions.

---

## 1. Problem Statement

FedProx introduces a proximal regularization term controlled by hyperparameter `mu`. Prior experiments showed:

- No single fixed `mu` (0.002-0.2) consistently beats FedAvg across all heterogeneity levels (alpha)
- Oracle selection of `mu` (best per alpha/seed after the fact) yields 31/35 wins for FedProx
- This oracle approach is not publishable without a pre-registered hyperparameter selection protocol

**Goal:** Establish a validation protocol that allows honest mu selection without data leakage.

---

## 2. Validation Strategy: Temporal Split

### 2.1 Rationale

Temporal splitting is chosen over held-out clients because:

1. **Realistic deployment scenario:** IDS models must generalize to future attacks, not just unseen clients
2. **Stability:** All 10 clients contribute to train/val/test, reducing variance from client sampling
3. **Domain appropriateness:** Network intrusion data is inherently temporal; attack patterns evolve
4. **Literature alignment:** Standard practice in IDS benchmarking (CIC-IDS2017, UNSW-NB15 papers)

### 2.2 Split Definition

For each client's local data, ordered by `frame.time` (or equivalent temporal field):

| Split      | Proportion     | Purpose                       |
| ---------- | -------------- | ----------------------------- |
| Train      | 70% (earliest) | Client local training         |
| Validation | 15% (middle)   | Hyperparameter selection (mu) |
| Test       | 15% (latest)   | Final metric reporting        |

```
Timeline per client:
[========== Train (70%) ==========][=== Val (15%) ===][=== Test (15%) ===]
       ^                                   ^                    ^
  Oldest samples              Used for mu selection    Final reporting only
```

### 2.3 Implementation Requirements

1. **Temporal ordering:** Use `frame.time` column to sort samples before splitting
2. **Timestamp parsing:** Parse `frame.time` as datetime using format `%Y %H:%M:%S.%f` (e.g., `2021 19:46:24.393481000`). For unparsable rows (~5-6% of dataset), fall back to original file order (row index) to preserve relative ordering.
3. **Feature exclusion:** Drop `frame.time` from feature matrix after ordering (no temporal leakage)
4. **Per-client splitting:** Each client applies the 70/15/15 split independently to its local partition
5. **Global metrics:** Computed as sample-weighted mean across clients (see Section 4.3)

### 2.4 Final Model Evaluation

After the last federated round completes:

1. Server performs one additional **eval-only round** (no training, no aggregation)
2. Each client evaluates the final global model on its local validation and test splits
3. Clients return predictions and sample counts (not gradients)
4. Server computes global metrics via sample-weighted aggregation

This ensures metrics reflect the final aggregated model, not the one-round-behind state.

---

## 3. Seed Separation

### 3.1 Tuning Phase

| Parameter   | Value                                 |
| ----------- | ------------------------------------- |
| Seeds       | {42, 43, 44}                          |
| Metric used | `macro_f1` on **validation** set      |
| Purpose     | Select optimal `mu` per `alpha` level |

### 3.2 Evaluation Phase

| Parameter   | Value                                          |
| ----------- | ---------------------------------------------- |
| Seeds       | {45, 46, 47, 48, 49}                           |
| Metric used | `macro_f1` on **test** set                     |
| Purpose     | Report final results with confidence intervals |

### 3.3 Seed Isolation Guarantee

- Tuning phase results (seeds 42-44) are used **only** for mu selection
- Evaluation phase results (seeds 45-49) are used **only** for final reporting
- No cross-contamination: mu is locked before evaluation seeds are run

---

## 4. Metrics

### 4.1 Primary Metric (Selection and Reporting)

**`macro_f1`** on global test set (aggregated across all clients)

Rationale:

- Handles class imbalance (common in IDS datasets)
- Treats all attack classes equally regardless of frequency
- Standard metric in multi-class IDS literature

### 4.2 Secondary Metrics (Reporting Only)

| Metric               | Definition                             | Rationale                           |
| -------------------- | -------------------------------------- | ----------------------------------- |
| Client-mean macro_f1 | Unweighted mean of per-client macro_f1 | Fairness across participants        |
| Benign FPR           | False positive rate on benign class    | Operational cost of false alarms    |
| Per-class recall     | Recall for each attack type            | Identifies which attacks are missed |
| Accuracy             | Overall correctness                    | Baseline comparison                 |

### 4.3 Global Metric Computation (Sample-Weighted)

**Definition:** Global macro_f1 is computed as a **sample-weighted mean** across clients.

```
global_macro_f1 = sum(n_i * macro_f1_i) / sum(n_i)

where:
  n_i = number of samples in client i's test set
  macro_f1_i = macro F1 score on client i's test set
```

**Rationale:**

- Reflects true global performance across the entire distributed dataset
- Larger clients (more data) contribute proportionally more
- Matches deployment scenario where total prediction quality matters

**Alternative (reported as secondary):** Client-mean macro_f1 (unweighted) for fairness analysis.

---

## 5. Hyperparameter Selection Protocol

### 5.1 Mu Selection Rule

For each heterogeneity level `alpha`:

1. Run FedProx with candidate mu values: **{0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0}**
2. For each (alpha, mu) pair, compute sample-weighted mean `macro_f1` on validation set across tuning seeds {42, 43, 44}
3. Select `mu*[alpha]` = argmax over mu of mean validation macro_f1

```
mu*[alpha] = argmax_{mu} ( mean_{seed in {42,43,44}} global_macro_f1_val(alpha, mu, seed) )
```

**Mu grid rationale:**

- {0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2}: Matches existing 315 runs for comparability
- {0.5, 1.0}: Added to empirically test whether higher regularization helps under extreme non-IID

### 5.2 Baseline Comparison

FedAvg (mu=0) is run with the same seeds and data splits. No hyperparameter selection needed.

### 5.3 Final Evaluation

After mu selection is locked:

1. Run FedProx with `mu*[alpha]` on evaluation seeds {45, 46, 47, 48, 49}
2. Run FedAvg on evaluation seeds {45, 46, 47, 48, 49}
3. Report mean and 95% CI for `macro_f1` on **test** set
4. Perform paired t-test or Wilcoxon signed-rank test for significance

---

## 6. Experimental Matrix

### 6.1 Alpha Grid (Heterogeneity Levels)

Matches existing 315 runs for comparability:

| Alpha | Interpretation                                      |
| ----- | --------------------------------------------------- |
| 0.02  | Extreme non-IID (highly skewed label distributions) |
| 0.05  | Very high non-IID                                   |
| 0.1   | High non-IID                                        |
| 0.2   | Moderate-high non-IID                               |
| 0.5   | Moderate non-IID                                    |
| 1.0   | Mild non-IID                                        |
| inf   | IID (uniform label distribution)                    |

**Total alpha values:** 7

### 6.2 Mu Grid (FedProx Regularization)

| Mu                                             | Source                             |
| ---------------------------------------------- | ---------------------------------- |
| 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2 | Existing sweep (8 values)          |
| 0.5, 1.0                                       | New high-mu exploration (2 values) |

**Total mu values:** 10

### 6.3 Tuning Phase Jobs

| Alpha | Mu Values | Seeds | Total Configs |
| ----- | --------- | ----- | ------------- |
| 0.02  | 10 values | 3     | 30            |
| 0.05  | 10 values | 3     | 30            |
| 0.1   | 10 values | 3     | 30            |
| 0.2   | 10 values | 3     | 30            |
| 0.5   | 10 values | 3     | 30            |
| 1.0   | 10 values | 3     | 30            |
| inf   | 10 values | 3     | 30            |

**Total tuning jobs (FedProx):** 210 (7 alphas x 10 mu x 3 seeds)  
**Total tuning jobs (FedAvg):** 21 (7 alphas x 3 seeds)  
**Grand total tuning phase:** 231 jobs

### 6.4 Evaluation Phase Jobs

| Alpha | Mu         | Seeds | Total Configs |
| ----- | ---------- | ----- | ------------- |
| 0.02  | mu\*[0.02] | 5     | 5             |
| 0.05  | mu\*[0.05] | 5     | 5             |
| 0.1   | mu\*[0.1]  | 5     | 5             |
| 0.2   | mu\*[0.2]  | 5     | 5             |
| 0.5   | mu\*[0.5]  | 5     | 5             |
| 1.0   | mu\*[1.0]  | 5     | 5             |
| inf   | mu\*[inf]  | 5     | 5             |

**Total evaluation jobs (FedProx):** 35 (7 alphas x 5 seeds)  
**Total evaluation jobs (FedAvg):** 35 (7 alphas x 5 seeds)  
**Grand total evaluation phase:** 70 jobs

### 6.5 Compute Budget Summary

| Phase      | Jobs    | Est. Time/Job | Total Time (1 node) | Total Time (17 nodes) |
| ---------- | ------- | ------------- | ------------------- | --------------------- |
| Tuning     | 231     | ~7 min        | ~27 hours           | ~1.6 hours            |
| Evaluation | 70      | ~7 min        | ~8 hours            | ~0.5 hours            |
| **Total**  | **301** | -             | **~35 hours**       | **~2.1 hours**        |

---

## 7. Reporting Requirements

### 7.1 Main Results Table

```
| Alpha | FedAvg macro_f1 (95% CI) | FedProx macro_f1 (95% CI) | mu* | Delta | p-value | Cohen's d |
|-------|--------------------------|---------------------------|-----|-------|---------|-----------|
| 0.02  | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
| 0.05  | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
| 0.1   | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
| 0.2   | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
| 0.5   | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
| 1.0   | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
| inf   | 0.XXX (0.XXX, 0.XXX)    | 0.XXX (0.XXX, 0.XXX)     | X.X | +X.XX | 0.XXX   | X.XX      |
```

### 7.2 Required Disclosures

1. **Tuning budget:** 231 configurations explored during mu selection (210 FedProx + 21 FedAvg)
2. **Mu grid:** {0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0}
3. **Alpha grid:** {0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf}
4. **Selection metric:** Sample-weighted validation macro_f1 (not test)
5. **Evaluation metric:** Sample-weighted test macro_f1 (never seen during tuning)
6. **Seed separation:** Tuning seeds {42, 43, 44} vs Evaluation seeds {45, 46, 47, 48, 49}
7. **Temporal split:** 70/15/15 train/val/test per client, ordered by `frame.time`
8. **Timestamp format:** `%Y %H:%M:%S.%f` (e.g., `2021 19:46:24.393481000`)
9. **Timestamp fallback:** Unparsable rows (~5-6%) retain original file order position

### 7.3 Negative Result Handling

If FedProx does not significantly outperform FedAvg:

- Report the negative result with confidence intervals
- Discuss sensitivity of FedProx to mu selection
- Note that even with validation-based tuning, fixed mu may not generalize
- This is a valid contribution to the literature

---

## 8. Implementation Checklist

### 8.1 Data Pipeline

- [ ] Add `parse_frame_time()` function with datetime format `%Y %H:%M:%S.%f`
- [ ] Add `temporal_sort_indices()` function with NaT fallback to row index
- [ ] Modify `load_edge_iiotset()` to preserve `frame.time` for ordering
- [ ] Implement temporal 70/15/15 split (positional, not random)
- [ ] Ensure `frame.time` is dropped from features after ordering

### 8.2 Federated Training Loop

- [ ] Add validation set evaluation during training (for early stopping / monitoring)
- [ ] Implement eval-only round after final aggregation
- [ ] Create separate metric logging for val vs test splits
- [ ] Implement sample-weighted global metric aggregation

### 8.3 Experiment Scripts

- [ ] Update Slurm scripts for tuning phase (seeds 42-44, mu grid with 0.5/1.0)
- [ ] Update Slurm scripts for evaluation phase (seeds 45-49)
- [ ] Add alpha=inf handling (IID baseline)

### 8.4 Analysis Scripts

- [ ] Create mu selection script (argmax over validation macro_f1)
- [ ] Create final reporting script with 95% CIs
- [ ] Implement paired t-test and Cohen's d calculation
- [ ] Generate results table in publication format

---

## 9. Appendix: Statistical Tests

### 9.1 Confidence Intervals

95% CI for mean macro_f1 over 5 evaluation seeds:

```
CI = mean +/- t_{0.975, df=4} * (std / sqrt(5))
```

where t\_{0.975, 4} = 2.776

### 9.2 Significance Testing

**Paired t-test** (if normality holds) or **Wilcoxon signed-rank test** (non-parametric):

- Null hypothesis: FedProx and FedAvg have equal mean macro_f1
- Alternative: Two-sided (FedProx differs from FedAvg)
- Significance level: alpha = 0.05
- Report actual p-values, not just significance

### 9.3 Effect Size

Report Cohen's d for practical significance:

```
d = (mean_FedProx - mean_FedAvg) / pooled_std
```

| d         | Interpretation |
| --------- | -------------- |
| < 0.2     | Negligible     |
| 0.2 - 0.5 | Small          |
| 0.5 - 0.8 | Medium         |
| > 0.8     | Large          |

---

## 10. References

1. Li et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys.
2. Sharafaldin et al. (2018). "Toward Generating a New Intrusion Detection Dataset." ICISSP.
3. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks." AISTATS.
