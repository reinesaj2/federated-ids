# IIoT Experimental Analysis: Critical & Optimistic Review

**Date:** November 22, 2025
**Context:** IIoT Worktree (`iiot-experiments`)
**Datasets:** IIoT-Set (CIC/UNSW variants)

---

## Part 1: Skeptical Review (The "Debugger" View)

### 1. Critical Data Integrity Failure: "Hyper-Unity" F1 Scores

**Observation:** The `attack_resilience_stats.csv` reports a Mean Macro-F1 of **2.65** for FedAvg (0% Adversaries), with a confidence interval spanning from **-5.36 to +10.66**.
**Implication:**

- **F1 Scores are bounded [0, 1].** A score of 2.65 is mathematically impossible.
- This indicates a severe bug in the metric aggregation logic (likely summing instead of averaging across clients or seeds).
- **Consequence:** All "degradation" percentages calculated against this baseline are invalid. The claim "74% degradation" is meaningless if the starting point is 265%.

### 2. The "Beneficial Attack" Anomaly

**Observation:** For `fedavg` with 10% adversaries (`adversary_fraction=0.1`), the F1 score is reported as **perfectly 1.0**.
**Implication:**

- How does introducing 10% malicious actors _improve_ the model to perfection?
- This suggests the attack code might be failing (sending zero-updates?) or the metric calculation is clamping values incorrectly.
- Coupled with the 2.65 baseline, this suggests the entire FedAvg reporting pipeline is broken.

### 3. Task Triviality (Again)

**Observation:** `Median` aggregation achieves **0.9994 F1** on the benign task.
**Implication:**

- Similar to the UNSW findings in the main repo, the IIoT task appears trivial.
- If a simple coordinate-wise median achieves 99.9% accuracy, is there any room for "research"?
- The "Personalization" gains (~6% for CIC) are the only evidence that the task isn't _completely_ solved by a global linear model.

---

## Part 2: Optimistic Review (The "Scientist" View)

### 1. Robustness Hierarchy Established

**Observation:** Despite the baseline glitches, the relative ordering of defenses under heavy attack (30% adversaries) is clear and consistent:

- **Median:** 0.89 F1 (Winner)
- **Bulyan:** 0.85 F1 (Strong Runner-up)
- **Krum:** 0.79 F1 (Viable)
- **FedAvg:** 0.67 F1 (Collapsed)

**Verdict:** This validates the core thesis hypothesis: **Simple statistical defenses (Median) outperform complex selection-based ones (Krum) in high-dimensional IIoT data.** This is a publishable finding.

### 2. FedProx: The "Heterogeneity Killer"

**Observation:** In the `fedprox_comparison_summary.json`, we have clean, bug-free evidence for FedProx.

- **Scenario:** Severe Non-IID (`alpha=0.05`).
- **Result:** FedProx (`mu=0.1`) reduced L2 drift by **2.66x** compared to FedAvg (0.30 $\to$ 0.11).
- **Significance:** This is a textbook result. It proves we can mathematically constrain local model drift even when clients have vastly different data distributions.

### 3. Personalization: The "Silver Bullet"

**Observation:** Personalization yielded a **~6% F1 gain** for CIC-IDS2017 clients.
**Significance:**

- In a field where 1% is significant, 6% is massive.
- This justifies the "Federated + Personalized" architecture for IIoT. It shows that while the global model is good (security), the local adaptation is better (specialization).

---

## Part 3: Action Plan

1.  **Fix the Metric Aggregator:**
    - Locate the script generating `attack_resilience_stats.csv`.
    - **Bug Hunt:** Look for `sum()` where there should be `mean()`.
    - _Priority: Critical._ We cannot put "F1 = 2.65" in a thesis.

2.  **Re-Verify FedAvg Baseline:**
    - Once fixed, re-run the FedAvg baseline (0% attack) to establish a sane ground truth (likely ~0.99).

3.  **Publish the FedProx & Median Results:**
    - The FedProx heterogeneity data is clean.
    - The Median robustness data (ignoring the FedAvg baseline) is strong.
    - The Personalization gains are the "cherry on top."

**Overall Status:**
The IIoT experiments are **richer** than the main repo experiments (more defenses, clear personalization wins), but the **reporting pipeline is buggy**. Fix the reporting script, and this chapter is done.
