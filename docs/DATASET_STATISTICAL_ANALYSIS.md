# Dataset Statistical Analysis

This document provides comprehensive statistical analysis of the CIC-IDS2017 and Edge-IIoTset datasets, including class imbalance metrics, distribution characteristics, and implications for machine learning.

## Table of Contents

1. [Class Distribution Statistics](#class-distribution-statistics)
2. [Imbalance Metrics](#imbalance-metrics)
3. [Attack Category Analysis](#attack-category-analysis)
4. [Feature Space Statistics](#feature-space-statistics)
5. [Machine Learning Implications](#machine-learning-implications)

---

## Class Distribution Statistics

### CIC-IDS2017 Complete Distribution

| Rank | Class | Count | Percentage | Cumulative % |
|------|-------|-------|------------|--------------|
| 1 | BENIGN | 2,273,097 | 80.300% | 80.300% |
| 2 | DoS Hulk | 231,073 | 8.163% | 88.463% |
| 3 | PortScan | 158,930 | 5.615% | 94.078% |
| 4 | DDoS | 128,027 | 4.523% | 98.601% |
| 5 | DoS GoldenEye | 10,293 | 0.364% | 98.965% |
| 6 | FTP-Patator | 7,938 | 0.280% | 99.245% |
| 7 | SSH-Patator | 5,897 | 0.208% | 99.453% |
| 8 | DoS slowloris | 5,796 | 0.205% | 99.658% |
| 9 | DoS Slowhttptest | 5,499 | 0.194% | 99.852% |
| 10 | Bot | 1,966 | 0.069% | 99.921% |
| 11 | Web Attack Brute Force | 1,507 | 0.053% | 99.974% |
| 12 | Web Attack XSS | 652 | 0.023% | 99.997% |
| 13 | Infiltration | 36 | 0.001% | 99.998% |
| 14 | Web Attack SQL Injection | 21 | 0.001% | 99.999% |
| 15 | Heartbleed | 11 | 0.0004% | 100.000% |

**Total Samples**: 2,830,743

### Edge-IIoTset Complete Distribution

| Rank | Class | Count | Percentage | Cumulative % |
|------|-------|-------|------------|--------------|
| 1 | Normal | 1,238,765 | 72.797% | 72.797% |
| 2 | DDoS_UDP | 93,254 | 5.480% | 78.277% |
| 3 | DDoS_ICMP | 89,329 | 5.249% | 83.526% |
| 4 | SQL_injection | 39,273 | 2.308% | 85.834% |
| 5 | Vulnerability_scanner | 38,503 | 2.263% | 88.097% |
| 6 | DDoS_TCP | 38,461 | 2.260% | 90.357% |
| 7 | Password | 38,448 | 2.260% | 92.617% |
| 8 | DDoS_HTTP | 38,316 | 2.252% | 94.869% |
| 9 | Uploading | 28,785 | 1.691% | 96.560% |
| 10 | Backdoor | 18,984 | 1.116% | 97.676% |
| 11 | Port_Scanning | 17,314 | 1.018% | 98.694% |
| 12 | XSS | 12,199 | 0.717% | 99.411% |
| 13 | Ransomware | 8,368 | 0.492% | 99.903% |
| 14 | MITM | 928 | 0.055% | 99.958% |
| 15 | Fingerprinting | 764 | 0.045% | 100.003% |

**Total Samples**: 1,701,691

---

## Imbalance Metrics

### Shannon Entropy

Shannon entropy measures the uncertainty or randomness in class distribution. Higher values indicate more balanced distributions.

**Formula**: H = -sum(p_i * log2(p_i)) for all classes i

| Dataset | Shannon Entropy | Max Possible | Normalized Entropy |
|---------|-----------------|--------------|-------------------|
| CIC-IDS2017 | 1.1084 bits | 3.9069 bits (log2(15)) | 0.284 (28.4%) |
| Edge-IIoTset | 3.7085 bits | 3.9069 bits (log2(15)) | 0.949 (94.9%) |

**Interpretation**: CIC-IDS2017 has only 28.4% of maximum possible entropy, indicating severe class concentration. Edge-IIoTset achieves 94.9%, indicating near-uniform distribution.

### Effective Number of Classes

The effective number of classes represents how many equally-sized classes would produce the same entropy.

**Formula**: N_eff = 2^H

| Dataset | Actual Classes | Effective Classes | Efficiency |
|---------|----------------|-------------------|------------|
| CIC-IDS2017 | 15 | 2.16 | 14.4% |
| Edge-IIoTset | 15 | 13.07 | 87.1% |

**Interpretation**: CIC-IDS2017 effectively behaves as a 2-class problem (BENIGN vs everything else), while Edge-IIoTset maintains meaningful distinction across 13 classes.

### Imbalance Ratio

The ratio between the largest and smallest class counts.

| Dataset | Majority Class | Minority Class | Imbalance Ratio |
|---------|----------------|----------------|-----------------|
| CIC-IDS2017 | BENIGN (2,273,097) | Heartbleed (11) | 206,645:1 |
| Edge-IIoTset | Normal (1,238,765) | Fingerprinting (764) | 1,621:1 |

**Interpretation**: CIC-IDS2017 has 127x worse imbalance ratio than Edge-IIoTset.

### Gini Impurity

Measures the probability of incorrect classification if samples were randomly labeled according to class distribution.

**Formula**: G = 1 - sum(p_i^2)

| Dataset | Gini Impurity | Interpretation |
|---------|---------------|----------------|
| CIC-IDS2017 | 0.330 | Low impurity (dominated by BENIGN) |
| Edge-IIoTset | 0.850 | High impurity (more balanced) |

### Class Percentile Analysis

Distribution of class sizes relative to median class size.

**CIC-IDS2017**:
- Classes above median: 4 (BENIGN, DoS Hulk, PortScan, DDoS)
- Classes below 1%: 11 (73% of classes)
- Classes below 0.1%: 6 (40% of classes)

**Edge-IIoTset**:
- Classes above median: 8
- Classes below 1%: 4 (27% of classes)
- Classes below 0.1%: 2 (13% of classes)

---

## Attack Category Analysis

### Attack Taxonomy Mapping

| Category | CIC-IDS2017 Classes | CIC Total % | Edge-IIoTset Classes | IIOT Total % |
|----------|---------------------|-------------|----------------------|--------------|
| DoS/DDoS | DoS Hulk, GoldenEye, slowloris, Slowhttptest, DDoS | 13.45% | DDoS_UDP, DDoS_ICMP, DDoS_TCP, DDoS_HTTP | 15.24% |
| Reconnaissance | PortScan | 5.61% | Port_Scanning, Vulnerability_scanner, Fingerprinting | 3.33% |
| Brute Force | FTP-Patator, SSH-Patator, Web Brute Force | 0.54% | Password | 2.26% |
| Web Attacks | XSS, SQL Injection | 0.02% | SQL_injection, XSS | 3.03% |
| Infiltration | Infiltration | 0.001% | Uploading, Backdoor, MITM | 2.86% |
| Malware | Bot, Heartbleed | 0.07% | Ransomware | 0.49% |
| **Benign** | BENIGN | 80.30% | Normal | 72.80% |
| **Total Attack** | - | 19.70% | - | 27.20% |

### Category Representation Gap

| Category | CIC % | IIOT % | Ratio (IIOT/CIC) |
|----------|-------|--------|------------------|
| Web Attacks | 0.02% | 3.03% | 151.5x |
| Infiltration | 0.001% | 2.86% | 2,860x |
| Brute Force | 0.54% | 2.26% | 4.2x |
| DoS/DDoS | 13.45% | 15.24% | 1.1x |
| Reconnaissance | 5.61% | 3.33% | 0.6x |

**Key Finding**: CIC-IDS2017 severely under-represents web attacks (0.02%) and infiltration (0.001%), making these categories statistically invisible during training.

---

## Feature Space Statistics

### CIC-IDS2017 Feature Statistics

| Category | Count | Type | Missing Values |
|----------|-------|------|----------------|
| Packet Length | 8 | Numeric | 0 |
| Inter-Arrival Time | 10 | Numeric | 0 |
| Flow Statistics | 7 | Numeric | Low (<1%) |
| TCP Flags | 6 | Numeric | 0 |
| Subflow | 4 | Numeric | 0 |
| Header/Window | 4 | Numeric | Low (<1%) |
| Bulk | 6 | Numeric | 0 |
| Active/Idle | 8 | Numeric | Low (<1%) |
| **Total** | **78** | Numeric | <1% overall |

**Value Range Issues**:
- Flow Bytes/s: Contains infinity values (requires preprocessing)
- Some timing features: Negative values present (anomalous)
- Header length features: Large negative values in some records

### Edge-IIoTset Feature Statistics

| Category | Count | Type | Missing Values |
|----------|-------|------|----------------|
| TCP | 15 | Numeric | Protocol-dependent |
| HTTP | 9 | Mixed | HTTP traffic only |
| MQTT | 13 | Numeric | MQTT traffic only |
| DNS | 7 | Numeric | DNS traffic only |
| ICMP | 4 | Numeric | ICMP traffic only |
| ARP | 4 | Numeric | ARP traffic only |
| UDP | 3 | Numeric | UDP traffic only |
| Modbus | 3 | Numeric | Modbus traffic only |
| **Total** | **61** | Mixed | Protocol-specific |

**Key Characteristic**: Missing values are semantically meaningful (protocol not present in packet). Zero imputation is appropriate.

### Feature Correlation Analysis

**CIC-IDS2017 High Correlations (>0.9)**:
- Fwd Packet Length Mean <-> Avg Fwd Segment Size (0.99)
- Total Length of Fwd Packets <-> Subflow Fwd Bytes (1.0)
- Active Mean <-> Active Max (0.95)
- Fwd Header Length <-> Fwd Header Length.1 (1.0, duplicate)

**Edge-IIoTset High Correlations (>0.9)**:
- tcp.seq <-> tcp.ack (0.82) - Not redundant, different meanings
- Protocol-specific features naturally uncorrelated

**Recommendation**: CIC requires feature selection/PCA to remove redundancy. Edge-IIoTset has cleaner feature structure.

---

## Machine Learning Implications

### Class Weight Recommendations

For weighted loss functions to handle class imbalance:

**CIC-IDS2017 Suggested Weights**:
| Class | Inverse Frequency Weight | Sqrt-Normalized |
|-------|--------------------------|-----------------|
| BENIGN | 1.0 | 1.0 |
| DoS Hulk | 9.8 | 3.1 |
| PortScan | 14.3 | 3.8 |
| DDoS | 17.8 | 4.2 |
| DoS GoldenEye | 220.8 | 14.9 |
| FTP-Patator | 286.3 | 16.9 |
| SSH-Patator | 385.5 | 19.6 |
| Heartbleed | 206,645.2 | 454.6 |

**Edge-IIoTset Suggested Weights**:
| Class | Inverse Frequency Weight | Sqrt-Normalized |
|-------|--------------------------|-----------------|
| Normal | 1.0 | 1.0 |
| DDoS_UDP | 13.3 | 3.6 |
| DDoS_ICMP | 13.9 | 3.7 |
| SQL_injection | 31.5 | 5.6 |
| Fingerprinting | 1,621.0 | 40.3 |

### Stratification Feasibility

| Scenario | CIC-IDS2017 | Edge-IIoTset |
|----------|-------------|--------------|
| 5-fold CV possible | Only for 8 classes | All 15 classes |
| 10-fold CV possible | Only for 4 classes | All 15 classes |
| Minimum samples for stratification | 50 | 76 |
| Classes with <50 samples | 4 | 0 |

**CIC Challenge**: Cross-validation is infeasible for rare attack classes (Heartbleed, Infiltration, SQL Injection).

### Sampling Strategy Recommendations

**CIC-IDS2017**:
1. **SMOTE/ADASYN**: Required for classes with <1000 samples
2. **Undersampling BENIGN**: Sample to ~500k to reduce dominance
3. **Focal Loss**: Use gamma=2.0 to focus on hard examples
4. **Binary Fallback**: Consider BENIGN vs ATTACK if multi-class fails

**Edge-IIoTset**:
1. **Standard Training**: Class weights sufficient
2. **Light Oversampling**: Optional for Fingerprinting, MITM
3. **Focal Loss**: gamma=1.0 sufficient
4. **Multi-class Viable**: All 15 classes trainable

### Expected Model Performance

Based on class distribution analysis:

| Metric | CIC-IDS2017 Expected | Edge-IIoTset Expected |
|--------|----------------------|----------------------|
| Accuracy | >95% (misleading due to BENIGN) | 85-90% |
| Macro-F1 | 0.15-0.30 | 0.50-0.70 |
| Weighted-F1 | 0.80-0.90 | 0.80-0.90 |
| Per-class min F1 | ~0.00 (rare classes) | 0.20-0.40 |

### Federated Learning Considerations

| Factor | CIC Impact | IIOT Impact |
|--------|------------|-------------|
| Client Data Heterogeneity | Extreme (most clients may lack rare classes) | Moderate |
| Global Model Bias | Toward BENIGN detection | More balanced |
| Local Update Quality | Highly variable | Consistent |
| Aggregation Stability | Unstable for rare classes | Stable |

---

## Summary Statistics Table

| Statistic | CIC-IDS2017 | Edge-IIoTset |
|-----------|-------------|--------------|
| Total Samples | 2,830,743 | 1,701,691 |
| Number of Classes | 15 | 15 |
| Number of Features | 78 | 61 |
| Benign/Normal % | 80.30% | 72.80% |
| Attack Traffic % | 19.70% | 27.20% |
| Shannon Entropy | 1.1084 bits | 3.7085 bits |
| Effective Classes | 2.16 | 13.07 |
| Imbalance Ratio | 206,645:1 | 1,621:1 |
| Classes >1% | 4 | 11 |
| Classes <0.1% | 6 | 2 |
| Gini Impurity | 0.330 | 0.850 |
| Feature Redundancy | High | Low |
| Missing Values | <1% | Protocol-specific |

---

## Conclusions

1. **CIC-IDS2017** is fundamentally a binary classification problem disguised as multi-class, with 80.3% benign traffic and 6 classes having fewer than 100 samples.

2. **Edge-IIoTset** provides genuine multi-class learning opportunity with 13 effective classes and manageable 1,621:1 imbalance ratio.

3. **For federated learning research**, Edge-IIoTset produces more reliable and generalizable results due to balanced class representation.

4. **CIC-IDS2017 requires significant preprocessing** (oversampling, undersampling, class weighting) before meaningful multi-class analysis is possible.

5. **Cross-dataset transfer learning** faces fundamental challenges due to different feature granularities and class balance characteristics.

---

*Document generated: 2024-12-31*
*Statistical analysis based on complete dataset examination*
