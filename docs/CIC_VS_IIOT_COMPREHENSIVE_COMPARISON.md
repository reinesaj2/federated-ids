# CIC-IDS2017 vs Edge-IIoTset: Comprehensive Dataset Comparison

This document provides a complete analysis of the two primary intrusion detection datasets used in this federated learning research: CIC-IDS2017 and Edge-IIoTset.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset Overview](#dataset-overview)
3. [Class Distribution Analysis](#class-distribution-analysis)
4. [Feature Space Comparison](#feature-space-comparison)
5. [Top 5 Attack Classes Analysis](#top-5-attack-classes-analysis)
6. [Federated Learning Performance](#federated-learning-performance)
7. [Statistical Significance](#statistical-significance)
8. [Recommendations](#recommendations)

---

## Executive Summary

Based on analysis of 5,221 full experimental runs (2,747 IIOT, 2,474 CIC), the Edge-IIoTset dataset demonstrates significantly better federated learning performance than CIC-IDS2017:

| Metric | CIC-IDS2017 | Edge-IIoTset | Difference |
|--------|-------------|--------------|------------|
| Mean Macro-F1 | 0.1774 | 0.4317 | +0.254 (2.4x) |
| Best F1 (benign baseline) | 0.2526 | 0.6189 | +0.366 |
| Effective Classes | 2.16 / 15 | 13.07 / 15 | 6x more usable |
| Imbalance Ratio | 206,645:1 | 24:1 | 8,610x better |

The performance gap is statistically significant (p < 0.001, Cohen's d = -3.33).

---

## Dataset Overview

### CIC-IDS2017

| Property | Value |
|----------|-------|
| Total Samples | 2,830,743 |
| Total Features | 78 |
| Number of Classes | 15 |
| Majority Class | BENIGN (80.30%) |
| Feature Granularity | Flow-level (aggregated statistics) |
| Source | Canadian Institute for Cybersecurity, 2017 |

### Edge-IIoTset

| Property | Value |
|----------|-------|
| Total Samples | 1,701,691 |
| Total Features | 61 |
| Number of Classes | 15 |
| Majority Class | Normal (72.80%) |
| Feature Granularity | Packet-level (protocol fields) |
| Source | Ferrag et al., IEEE Access 2022 |

---

## Class Distribution Analysis

### CIC-IDS2017 Full Class Distribution

| Class | Count | Percentage | Category |
|-------|-------|------------|----------|
| BENIGN | 2,273,097 | 80.30% | Normal |
| DoS Hulk | 231,073 | 8.16% | DoS |
| PortScan | 158,930 | 5.61% | Reconnaissance |
| DDoS | 128,027 | 4.52% | DDoS |
| DoS GoldenEye | 10,293 | 0.36% | DoS |
| FTP-Patator | 7,938 | 0.28% | Brute Force |
| SSH-Patator | 5,897 | 0.21% | Brute Force |
| DoS slowloris | 5,796 | 0.20% | DoS |
| DoS Slowhttptest | 5,499 | 0.19% | DoS |
| Bot | 1,966 | 0.07% | Malware |
| Web Attack Brute Force | 1,507 | 0.05% | Web Attack |
| Web Attack XSS | 652 | 0.02% | Web Attack |
| Infiltration | 36 | 0.00% | Infiltration |
| Web Attack SQL Injection | 21 | 0.00% | Web Attack |
| Heartbleed | 11 | 0.00% | Vulnerability |

### Edge-IIoTset Full Class Distribution

| Class | Count | Percentage | Category |
|-------|-------|------------|----------|
| Normal | 1,238,765 | 72.80% | Normal |
| DDoS_UDP | 93,254 | 5.48% | DDoS |
| DDoS_ICMP | 89,329 | 5.25% | DDoS |
| SQL_injection | 39,273 | 2.31% | Web Attack |
| Vulnerability_scanner | 38,503 | 2.26% | Reconnaissance |
| DDoS_TCP | 38,461 | 2.26% | DDoS |
| Password | 38,448 | 2.26% | Brute Force |
| DDoS_HTTP | 38,316 | 2.25% | DDoS |
| Uploading | 28,785 | 1.69% | Infiltration |
| Backdoor | 18,984 | 1.12% | Malware |
| Port_Scanning | 17,314 | 1.02% | Reconnaissance |
| XSS | 12,199 | 0.72% | Web Attack |
| Ransomware | 8,368 | 0.49% | Malware |
| MITM | 928 | 0.05% | Network Attack |
| Fingerprinting | 764 | 0.04% | Reconnaissance |

### Class Imbalance Metrics

| Metric | CIC-IDS2017 | Edge-IIoTset | Interpretation |
|--------|-------------|--------------|----------------|
| Shannon Entropy (bits) | 1.1084 | 3.7085 | Higher = more balanced |
| Imbalance Ratio (max/min) | 206,645:1 | 24:1 | Lower = more balanced |
| Effective Classes | 2.16 | 13.07 | Closer to actual = better |
| Gini Impurity | 0.33 | 0.85 | Higher = more balanced |

---

## Feature Space Comparison

### CIC-IDS2017 Feature Categories

The CIC-IDS2017 dataset uses flow-level aggregated statistics:

| Category | Features | Examples |
|----------|----------|----------|
| Flow Statistics | 7 | Flow Duration, Flow Bytes/s, Flow Packets/s |
| Packet Length | 8 | Fwd Packet Length Max/Min/Mean/Std, Bwd Packet Length Max/Min/Mean/Std |
| TCP Flags | 6 | FIN Flag Count, SYN Flag Count, RST/PSH/ACK/URG Flag Count |
| Inter-Arrival Time | 10 | Flow IAT Mean/Std/Max/Min, Fwd IAT Total/Mean/Std/Max/Min |
| Subflow Metrics | 4 | Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets/Bytes |
| Header/Window | 4 | Init_Win_bytes_forward/backward, Fwd/Bwd Header Length |
| Bulk Statistics | 6 | Fwd/Bwd Avg Bytes/Packets/Bulk Rate |
| Active/Idle | 8 | Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min |

### Edge-IIoTset Feature Categories

The Edge-IIoTset dataset uses packet-level protocol-specific fields:

| Category | Features | Examples |
|----------|----------|----------|
| TCP | 15 | tcp.flags, tcp.dstport, tcp.srcport, tcp.ack, tcp.seq, tcp.len, tcp.checksum |
| HTTP | 9 | http.request.method, http.request.uri.query, http.response, http.content_length |
| MQTT | 13 | mqtt.hdrflags, mqtt.topic_len, mqtt.len, mqtt.msgtype, mqtt.qos |
| DNS | 7 | dns.qry.name, dns.qry.type, dns.flags.response |
| ARP | 4 | arp.opcode, arp.src.proto_ipv4, arp.hw.size |
| ICMP | 4 | icmp.checksum, icmp.seq_le, icmp.type |
| Modbus | 3 | modbus.trans_id, modbus.unit_id, modbus.len |
| UDP | 3 | udp.port, udp.stream, udp.time_delta |

### Feature Type Overlap

| Feature Type | CIC-IDS2017 | Edge-IIoTset | Semantic Match |
|--------------|-------------|--------------|----------------|
| Packet Statistics | Yes (8 features) | Yes (4 features) | High |
| TCP Flags/State | Yes (6 features) | Yes (6 features) | High |
| Timing/IAT | Yes (10 features) | Partial (1 feature) | Medium |
| Protocol-Specific (IoT) | No | Yes (23+ features) | None |
| Header Info | Yes (4 features) | Partial (2 features) | Medium |
| Application Layer | No | Yes (HTTP, MQTT, DNS) | None |

---

## Top 5 Attack Classes Analysis

### CIC-IDS2017 Top 5 Attacks

#### 1. DoS Hulk (231,073 samples, 8.16%)

| Feature | Importance | Attack Mean | BENIGN Mean | Ratio |
|---------|------------|-------------|-------------|-------|
| Bwd Packets/s | 0.087 | 424.5 | 6,393.5 | 0.07x |
| Max Packet Length | 0.082 | 4,005.0 | 494.3 | 8.1x |
| Packet Length Std | 0.070 | 1,217.2 | 147.8 | 8.2x |
| Avg Bwd Segment Size | 0.065 | 1,281.4 | 160.8 | 8.0x |
| Average Packet Size | 0.051 | 640.0 | 124.1 | 5.2x |

**Signature**: Very large packets with high variance, lower backward packet rate.

#### 2. PortScan (158,930 samples, 5.61%)

| Feature | Importance | Attack Mean | BENIGN Mean | Ratio |
|---------|------------|-------------|-------------|-------|
| Bwd Packets/s | 0.093 | 31,313.5 | 6,393.5 | 4.9x |
| Fwd Packet Length Mean | 0.081 | 1.0 | 66.4 | 0.02x |
| Subflow Fwd Bytes | 0.073 | 1.1 | 635.5 | 0.002x |
| Average Packet Size | 0.059 | 4.6 | 124.1 | 0.04x |
| Total Length of Fwd Packets | 0.057 | 1.1 | 635.5 | 0.002x |

**Signature**: Minimal forward payload (probe packets), very high backward packet rate.

#### 3. DDoS (128,027 samples, 4.52%)

| Feature | Importance | Attack Mean | BENIGN Mean | Ratio |
|---------|------------|-------------|-------------|-------|
| Fwd Packet Length Mean | 0.088 | 7.4 | 66.4 | 0.11x |
| Avg Fwd Segment Size | 0.081 | 7.4 | 66.4 | 0.11x |
| Fwd Packet Length Max | 0.062 | 14.9 | 230.4 | 0.06x |
| Subflow Fwd Packets | 0.060 | 4.5 | 10.7 | 0.42x |
| Fwd Header Length | 0.052 | 97.1 | -32,404 | N/A |

**Signature**: Extremely uniform small packets with very low variance (automated flooding).

#### 4. DoS GoldenEye (10,293 samples, 0.36%)

| Feature | Importance | Attack Mean | BENIGN Mean | Ratio |
|---------|------------|-------------|-------------|-------|
| Bwd Packet Length Max | 0.086 | 4,205.5 | 396.5 | 10.6x |
| Avg Bwd Segment Size | 0.069 | 1,256.7 | 160.8 | 7.8x |
| Max Packet Length | 0.068 | 4,219.9 | 494.3 | 8.5x |
| Bwd Packet Length Std | 0.067 | 1,951.9 | 123.1 | 15.9x |
| Packet Length Std | 0.056 | 1,271.8 | 147.8 | 8.6x |

**Signature**: Very large backward packets with high variance (HTTP slow-read pattern).

#### 5. FTP-Patator (7,938 samples, 0.28%)

| Feature | Importance | Attack Mean | BENIGN Mean | Ratio |
|---------|------------|-------------|-------------|-------|
| Destination Port | 0.191 | 21.0 | 9,419.5 | Fixed |
| Fwd Packet Length Std | 0.068 | 9.7 | 74.6 | 0.13x |
| Fwd Packet Length Mean | 0.065 | 9.4 | 66.4 | 0.14x |
| Avg Fwd Segment Size | 0.065 | 9.4 | 66.4 | 0.14x |
| Max Packet Length | 0.063 | 24.0 | 494.3 | 0.05x |

**Signature**: Exclusively port 21, small uniform packets (brute-force login attempts).

### Edge-IIoTset Top 5 Attacks

#### 1. DDoS_UDP (93,254 samples, 5.48%)

| Feature | Importance | Attack Mean | Normal Mean | Key Pattern |
|---------|------------|-------------|-------------|-------------|
| arp.src.proto_ipv4 | 0.242 | 0.00 | 1.00 | Absent |
| tcp.payload | 0.161 | 0.00 | 1,369.06 | Absent |
| tcp.flags | 0.114 | 0.00 | 16.25 | Absent |
| tcp.options | 0.111 | 0.00 | 1.93 | Absent |
| tcp.dstport | 0.104 | 0.00 | 31,611.15 | Absent |

**Signature**: Complete absence of TCP/ARP fields (UDP-only traffic).

#### 2. DDoS_ICMP (89,329 samples, 5.25%)

| Feature | Importance | Attack Mean | Normal Mean | Key Pattern |
|---------|------------|-------------|-------------|-------------|
| icmp.checksum | 0.211 | 32,860.33 | 0.00 | Present |
| tcp.payload | 0.162 | 0.00 | 1,316.49 | Absent |
| tcp.options | 0.141 | 0.00 | 1.91 | Absent |
| icmp.seq_le | 0.130 | 32,767.98 | 0.00 | Present |
| tcp.flags | 0.110 | 0.00 | 16.26 | Absent |

**Signature**: High ICMP fields with absence of TCP features.

#### 3. SQL_injection (39,273 samples, 2.31%)

| Feature | Importance | Attack Mean | Normal Mean | Key Pattern |
|---------|------------|-------------|-------------|-------------|
| http.request.uri.query | 0.207 | 119.68 | 2,992.00 | Shorter |
| http.request.version | 0.202 | 0.09 | 2.00 | Lower |
| tcp.srcport | 0.175 | 1,094.54 | 6,862.04 | Lower |
| http.request.method | 0.165 | 0.09 | 2.00 | Lower |
| tcp.options | 0.099 | 17,173.82 | 3,870.79 | Higher |

**Signature**: Short URI query strings, low source ports, elevated tcp.options.

#### 4. Vulnerability_scanner (38,503 samples, 2.26%)

| Feature | Importance | Attack Mean | Normal Mean | Key Pattern |
|---------|------------|-------------|-------------|-------------|
| http.referer | 0.210 | 0.00 | 2.00 | Absent |
| http.request.uri.query | 0.144 | 100.04 | 1,622.00 | Shorter |
| http.request.method | 0.143 | 0.58 | 8.00 | Lower |
| http.request.version | 0.126 | 0.58 | 12.00 | Lower |
| tcp.srcport | 0.089 | 216.15 | 3,577.57 | Lower |

**Signature**: Missing HTTP referer, low source ports, short URI queries.

#### 5. DDoS_TCP (38,461 samples, 2.26%)

| Feature | Importance | Attack Mean | Normal Mean | Key Pattern |
|---------|------------|-------------|-------------|-------------|
| tcp.options | 0.241 | 0.00 | 1.54 | Absent |
| tcp.srcport | 0.195 | 5,337.22 | 19,433.33 | Lower |
| tcp.ack | 0.148 | 627,244,221.59 | 5,244,558.90 | 120x Higher |
| tcp.dstport | 0.081 | 13,643.59 | 31,586.79 | Lower |
| tcp.flags | 0.072 | 9.48 | 16.26 | Lower |

**Signature**: Extremely high tcp.ack values (SYN flood pattern), absent tcp.options.

---

## Federated Learning Performance

### Overall Performance by Dataset

| Metric | IIOT | CIC |
|--------|------|-----|
| Mean F1 (all runs) | 0.4317 | 0.1774 |
| Std F1 | 0.2031 | 0.0681 |
| Best F1 | 0.8389 | 0.8268 |
| Total Runs Analyzed | 2,747 | 2,474 |

### Performance by Aggregation Method (Benign, adv=0%)

| Method | IIOT F1 | CIC F1 | Gap |
|--------|---------|--------|-----|
| Bulyan | 0.6015 | 0.2118 | +0.39 |
| Median | 0.5976 | 0.2029 | +0.39 |
| FedAvg | 0.5829 | 0.2053 | +0.38 |
| Krum | 0.5070 | 0.2024 | +0.30 |

### Performance by Adversarial Fraction

| Adversary % | IIOT F1 | CIC F1 | IIOT Degradation | CIC Degradation |
|-------------|---------|--------|------------------|-----------------|
| 0% | 0.573 | 0.202 | baseline | baseline |
| 10% | 0.463 | 0.181 | -19.2% | -10.4% |
| 20% | 0.365 | 0.157 | -36.3% | -22.3% |
| 30% | 0.292 | 0.140 | -49.0% | -30.7% |

### Performance by Heterogeneity (Alpha)

| Alpha | IIOT F1 | CIC F1 | IIOT Trend | CIC Trend |
|-------|---------|--------|------------|-----------|
| 0.02 (extreme) | 0.390 | 0.248 | Low | High |
| 0.05 | 0.467 | 0.218 | - | - |
| 0.10 | 0.534 | 0.176 | - | - |
| 0.20 | 0.619 | 0.169 | - | - |
| 0.50 | 0.649 | 0.177 | - | - |
| 1.00 (IID) | 0.663 | 0.195 | High | Low |

### Per-Class F1 Performance

#### IIOT Top Performing Attack Classes

| Class | Mean F1 | Std | Best Aggregation |
|-------|---------|-----|------------------|
| DDoS_UDP | 0.895 | 0.05 | FedAvg |
| Vulnerability_scanner | 0.837 | 0.08 | FedAvg |
| DDoS_ICMP | 0.812 | 0.07 | Bulyan |
| SQL_injection | 0.701 | 0.12 | Median |
| DDoS_TCP | 0.685 | 0.11 | FedAvg |

#### CIC Top Performing Attack Classes

| Class | Mean F1 | Std | Best Aggregation |
|-------|---------|-----|------------------|
| DoS Hulk | 0.494 | 0.15 | Bulyan |
| PortScan | 0.458 | 0.12 | Bulyan |
| DDoS | 0.301 | 0.18 | Median |
| DoS GoldenEye | 0.287 | 0.14 | Bulyan |
| FTP-Patator | 0.253 | 0.11 | Krum |

---

## Statistical Significance

### Baseline FedAvg Comparison (Benign Conditions)

| Statistic | Value |
|-----------|-------|
| CIC Mean F1 | 0.2526 |
| IIOT Mean F1 | 0.6189 |
| t-statistic | -31.8421 |
| p-value | < 0.000001 |
| Cohen's d | -3.3286 |
| Significance | Highly Significant |

### Effect Size Interpretation

| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |
| **3.33** | **Very large effect** |

The observed effect size of 3.33 indicates that the performance difference between IIOT and CIC is not only statistically significant but also practically meaningful, with IIOT outperforming CIC by over 3 standard deviations.

---

## Recommendations

### For Multi-Class Attack Detection Research

1. **Prefer Edge-IIoTset** for multi-class experiments without heavy preprocessing
2. CIC-IDS2017 requires aggressive class balancing (SMOTE, focal loss, class weighting)
3. CIC effectively degrades to binary classification due to 80% BENIGN dominance

### For Federated Learning Experiments

1. IIOT provides more reliable convergence with standard FL algorithms
2. CIC requires specialized techniques to handle extreme class imbalance
3. Heterogeneity (alpha) has stronger effects on IIOT than CIC

### For Feature Engineering

1. Protocol-specific fields (IIOT) provide better attack separation than flow statistics (CIC)
2. Single features like tcp.dstport can achieve 20% importance in IIOT
3. CIC requires combinations of features for comparable discrimination

### For Cross-Dataset Generalization

1. Direct transfer learning between datasets is challenging due to different feature granularities
2. Attack category mapping provides semantic alignment but not feature alignment
3. Hybrid approaches may require feature harmonization preprocessing

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP 2018.

2. Ferrag, M. A., et al. (2022). Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning. IEEE Access.

---

*Document generated: 2024-12-31*
*Analysis based on 5,221 full experimental runs*
