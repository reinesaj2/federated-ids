# Attack Class Feature Analysis

This document provides detailed feature importance analysis for the top attack classes in both CIC-IDS2017 and Edge-IIoTset datasets, focusing on discriminative features that distinguish attacks from benign traffic.

## Table of Contents

1. [Methodology](#methodology)
2. [CIC-IDS2017 Attack Signatures](#cic-ids2017-attack-signatures)
3. [Edge-IIoTset Attack Signatures](#edge-iiotset-attack-signatures)
4. [Cross-Dataset Feature Comparison](#cross-dataset-feature-comparison)
5. [Attack Detection Strategies](#attack-detection-strategies)

---

## Methodology

### Feature Importance Computation

Feature importance was computed using RandomForest classifiers with the following configuration:

```
n_estimators: 50
max_depth: 10
random_state: 42
Sample size: 50,000 samples per analysis
Binary classification: Attack class vs Benign/Normal
```

For each attack class, a binary classifier was trained to distinguish that specific attack from benign traffic, and feature importances were extracted from the trained model.

### Metrics Reported

- **Importance**: RandomForest feature importance score (sum to 1.0)
- **Attack Mean**: Mean feature value for attack samples
- **Attack Std**: Standard deviation of feature for attack samples
- **Benign Mean**: Mean feature value for benign/normal samples
- **Ratio**: Attack Mean / Benign Mean (indicates direction of difference)

---

## CIC-IDS2017 Attack Signatures

### Feature Categories Used

CIC-IDS2017 uses flow-level aggregated statistics organized into these categories:

| Category | Feature Count | Examples |
|----------|---------------|----------|
| Packet Length | 8 | Fwd/Bwd Packet Length Max, Min, Mean, Std |
| Inter-Arrival Time | 10 | Flow/Fwd/Bwd IAT Mean, Std, Max, Min, Total |
| Flow Statistics | 7 | Duration, Bytes/s, Packets/s |
| TCP Flags | 6 | FIN, SYN, RST, PSH, ACK, URG counts |
| Subflow | 4 | Subflow Fwd/Bwd Packets, Bytes |
| Header/Window | 4 | Init_Win_bytes, Header Length |

---

### DoS Hulk Attack

**Sample Count**: 231,073 (8.16% of dataset)

**Attack Behavior**: HTTP POST flood attack that creates high-bandwidth connections with large packet payloads.

| Rank | Feature | Importance | Attack Mean (Std) | Benign Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | Bwd Packets/s | 0.087 | 424.5 (12,149) | 6,393.5 (37,971) | Lower backward rate |
| 2 | Max Packet Length | 0.082 | 4,005.0 (3,130) | 494.3 (1,074) | 8x larger packets |
| 3 | Packet Length Std | 0.070 | 1,217.2 (925) | 147.8 (307) | High size variance |
| 4 | Avg Bwd Segment Size | 0.065 | 1,281.4 (929) | 160.8 (284) | Large backward segments |
| 5 | Average Packet Size | 0.051 | 640.0 (466) | 124.1 (197) | 5x larger average |

**Detection Rule**: Flag flows where Max Packet Length > 2000 AND Packet Length Std > 500 AND Bwd Packets/s < 1000.

---

### PortScan Attack

**Sample Count**: 158,930 (5.61% of dataset)

**Attack Behavior**: Network reconnaissance using minimal probe packets to discover open ports.

| Rank | Feature | Importance | Attack Mean (Std) | Benign Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | Bwd Packets/s | 0.093 | 31,313.5 (63,949) | 6,393.5 (37,971) | 5x higher backward rate |
| 2 | Fwd Packet Length Mean | 0.081 | 1.0 (1.3) | 66.4 (204) | Near-zero forward payload |
| 3 | Subflow Fwd Bytes | 0.073 | 1.1 (5.7) | 635.5 (10,618) | Minimal forward data |
| 4 | Average Packet Size | 0.059 | 4.6 (28) | 124.1 (197) | Tiny packets |
| 5 | Total Length of Fwd Packets | 0.057 | 1.1 (5.7) | 635.5 (10,634) | Minimal payload |

**Detection Rule**: Flag flows where Fwd Packet Length Mean < 5 AND Bwd Packets/s > 10000 AND Average Packet Size < 20.

---

### DDoS Attack

**Sample Count**: 128,027 (4.52% of dataset)

**Attack Behavior**: Distributed denial of service with automated, uniform packet flooding.

| Rank | Feature | Importance | Attack Mean (Std) | Benign Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | Fwd Packet Length Mean | 0.088 | 7.4 (1.2) | 66.4 (204) | Very small, uniform |
| 2 | Avg Fwd Segment Size | 0.081 | 7.4 (1.2) | 66.4 (204) | Consistent segment size |
| 3 | Fwd Packet Length Max | 0.062 | 14.9 (6.7) | 230.4 (791) | Small max packet |
| 4 | Subflow Fwd Packets | 0.060 | 4.5 (1.9) | 10.7 (837) | Consistent count |
| 5 | Fwd Header Length | 0.052 | 97.1 (38) | -32,404 (23M) | Stable headers |

**Detection Rule**: Flag flows where Fwd Packet Length Std < 5 AND Fwd Packet Length Mean < 15 (low variance indicates automation).

---

### DoS GoldenEye Attack

**Sample Count**: 10,293 (0.36% of dataset)

**Attack Behavior**: HTTP-based slow-read attack that keeps connections open with large response data.

| Rank | Feature | Importance | Attack Mean (Std) | Benign Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | Bwd Packet Length Max | 0.086 | 4,205.5 (3,451) | 396.5 (813) | 10x larger backward packets |
| 2 | Avg Bwd Segment Size | 0.069 | 1,256.7 (977) | 160.8 (284) | Large backward segments |
| 3 | Max Packet Length | 0.068 | 4,219.9 (3,434) | 494.3 (1,074) | Very large max |
| 4 | Bwd Packet Length Std | 0.067 | 1,951.9 (1,598) | 123.1 (275) | High backward variance |
| 5 | Packet Length Std | 0.056 | 1,271.8 (974) | 147.8 (307) | High overall variance |

**Detection Rule**: Flag flows where Bwd Packet Length Max > 3000 AND Bwd Packet Length Std > 1000.

---

### FTP-Patator Attack

**Sample Count**: 7,938 (0.28% of dataset)

**Attack Behavior**: Brute-force password attack against FTP servers on port 21.

| Rank | Feature | Importance | Attack Mean (Std) | Benign Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | Destination Port | 0.191 | 21.0 (0.7) | 9,419.5 (19,755) | Fixed port 21 |
| 2 | Fwd Packet Length Std | 0.068 | 9.7 (0.8) | 74.6 (311) | Very uniform |
| 3 | Fwd Packet Length Mean | 0.065 | 9.4 (2.5) | 66.4 (204) | Small, consistent |
| 4 | Avg Fwd Segment Size | 0.065 | 9.4 (2.5) | 66.4 (204) | Small segments |
| 5 | Max Packet Length | 0.063 | 24.0 (10) | 494.3 (1,074) | Very small max |

**Detection Rule**: Destination Port = 21 is sufficient (0.191 importance alone).

---

## Edge-IIoTset Attack Signatures

### Feature Categories Used

Edge-IIoTset uses packet-level protocol-specific fields:

| Category | Feature Count | Protocol Coverage |
|----------|---------------|-------------------|
| TCP | 15 | Connection state, flags, ports, checksum |
| HTTP | 9 | Request method, URI, response, content |
| MQTT | 13 | IoT messaging protocol fields |
| DNS | 7 | Query and response fields |
| ICMP | 4 | Network control protocol |
| ARP | 4 | Address resolution fields |
| UDP | 3 | Datagram fields |
| Modbus | 3 | Industrial control protocol |

---

### DDoS_UDP Attack

**Sample Count**: 93,254 (5.48% of dataset)

**Attack Behavior**: UDP-based volumetric attack that floods targets with UDP packets.

| Rank | Feature | Importance | Attack Mean (Std) | Normal Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | arp.src.proto_ipv4 | 0.242 | 0.00 (0.00) | 1.00 (0.10) | ARP absent |
| 2 | tcp.payload | 0.161 | 0.00 (0.00) | 1,369.06 (4,287.62) | TCP absent |
| 3 | tcp.flags | 0.114 | 0.00 (0.00) | 16.25 (6.65) | TCP absent |
| 4 | tcp.options | 0.111 | 0.00 (0.00) | 1.93 (14.84) | TCP absent |
| 5 | tcp.dstport | 0.104 | 0.00 (0.00) | 31,611.15 (27,834.35) | TCP absent |

**Detection Rule**: tcp.payload = 0 AND tcp.flags = 0 AND high packet volume (pure UDP traffic).

---

### DDoS_ICMP Attack

**Sample Count**: 89,329 (5.25% of dataset)

**Attack Behavior**: ICMP flood attack (ping flood) to exhaust target resources.

| Rank | Feature | Importance | Attack Mean (Std) | Normal Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | icmp.checksum | 0.211 | 32,860.33 (18,967.35) | 0.00 (0.00) | ICMP present |
| 2 | tcp.payload | 0.162 | 0.00 (0.00) | 1,316.49 (4,117.73) | TCP absent |
| 3 | tcp.options | 0.141 | 0.00 (0.00) | 1.91 (14.35) | TCP absent |
| 4 | icmp.seq_le | 0.130 | 32,767.98 (18,663.15) | 0.00 (0.00) | ICMP present |
| 5 | tcp.flags | 0.110 | 0.00 (0.00) | 16.26 (6.65) | TCP absent |

**Detection Rule**: icmp.checksum > 0 AND icmp.seq_le > 0 AND tcp.* = 0.

---

### SQL_injection Attack

**Sample Count**: 39,273 (2.31% of dataset)

**Attack Behavior**: Web application attack injecting SQL commands via HTTP requests.

| Rank | Feature | Importance | Attack Mean (Std) | Normal Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | http.request.uri.query | 0.207 | 119.68 (467.85) | 2,992.00 (0.00) | Shorter queries |
| 2 | http.request.version | 0.202 | 0.09 (0.28) | 2.00 (0.00) | Lower version |
| 3 | tcp.srcport | 0.175 | 1,094.54 (1,381.31) | 6,862.04 (3,901.72) | Lower source port |
| 4 | http.request.method | 0.165 | 0.09 (0.28) | 2.00 (0.00) | Different method |
| 5 | tcp.options | 0.099 | 17,173.82 (10,512.75) | 3,870.79 (11,096.75) | Higher options |

**Detection Rule**: http.request.uri.query < 500 AND tcp.srcport < 2000 AND tcp.options > 10000.

---

### Vulnerability_scanner Attack

**Sample Count**: 38,503 (2.26% of dataset)

**Attack Behavior**: Automated scanning to discover vulnerabilities in web applications.

| Rank | Feature | Importance | Attack Mean (Std) | Normal Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | http.referer | 0.210 | 0.00 (0.06) | 2.00 (0.00) | No referer header |
| 2 | http.request.uri.query | 0.144 | 100.04 (280.20) | 1,622.00 (0.00) | Short queries |
| 3 | http.request.method | 0.143 | 0.58 (0.50) | 8.00 (0.00) | Unusual method |
| 4 | http.request.version | 0.126 | 0.58 (0.52) | 12.00 (0.00) | Lower version |
| 5 | tcp.srcport | 0.089 | 216.15 (294.61) | 3,577.57 (3,592.76) | Very low source port |

**Detection Rule**: http.referer = 0 AND http.request.uri.query < 200 AND tcp.srcport < 500.

---

### DDoS_TCP Attack

**Sample Count**: 38,461 (2.26% of dataset)

**Attack Behavior**: TCP-based flood attack, typically SYN flood exhausting connection tables.

| Rank | Feature | Importance | Attack Mean (Std) | Normal Mean (Std) | Interpretation |
|------|---------|------------|-------------------|-------------------|----------------|
| 1 | tcp.options | 0.241 | 0.00 (0.00) | 1.54 (6.16) | No TCP options |
| 2 | tcp.srcport | 0.195 | 5,337.22 (6,167.47) | 19,433.33 (4,637.34) | Lower source port |
| 3 | tcp.ack | 0.148 | 627,244,221.59 (709M) | 5,244,558.90 (26.6M) | 120x higher ACK |
| 4 | tcp.dstport | 0.081 | 13,643.59 (19,928.40) | 31,586.79 (27,823.42) | Lower dest port |
| 5 | tcp.flags | 0.072 | 9.48 (8.87) | 16.26 (6.63) | Different flag pattern |

**Detection Rule**: tcp.ack > 100,000,000 AND tcp.options = 0 (SYN flood signature).

---

## Cross-Dataset Feature Comparison

### Attack Category Mapping

| Attack Category | CIC-IDS2017 Classes | Edge-IIoTset Classes | Feature Overlap |
|-----------------|---------------------|----------------------|-----------------|
| DoS/DDoS | DoS Hulk, DDoS, GoldenEye | DDoS_UDP, DDoS_ICMP, DDoS_TCP | Medium |
| Reconnaissance | PortScan | Vulnerability_scanner, Port_Scanning | Low |
| Brute Force | FTP-Patator, SSH-Patator | Password | Medium |
| Web Attacks | XSS, SQL Injection | SQL_injection, XSS | High |

### Feature Type Comparison

| Feature Type | CIC Best For | IIOT Best For |
|--------------|--------------|---------------|
| Packet Length Stats | DoS Hulk, GoldenEye | - |
| Packet Rate | PortScan | - |
| Packet Uniformity | DDoS | - |
| Destination Port | FTP-Patator | DDoS_TCP |
| Protocol Presence/Absence | - | DDoS_UDP, DDoS_ICMP |
| HTTP Fields | - | SQL_injection, Vuln_scanner |
| TCP State | - | DDoS_TCP |

### Top Discriminative Features by Dataset

**CIC-IDS2017 Global Top 10**:
1. Bwd Packet Length Max (0.108)
2. Avg Bwd Segment Size (0.066)
3. Fwd Packet Length Max (0.061)
4. Init_Win_bytes_forward (0.049)
5. act_data_pkt_fwd (0.044)
6. Fwd Packet Length Mean (0.043)
7. Avg Fwd Segment Size (0.043)
8. Packet Length Variance (0.039)
9. Total Length of Bwd Packets (0.036)
10. Fwd IAT Std (0.034)

**Edge-IIoTset Global Top 10**:
1. tcp.dstport (0.199)
2. udp.stream (0.115)
3. tcp.flags (0.112)
4. tcp.ack (0.111)
5. tcp.seq (0.084)
6. icmp.checksum (0.084)
7. tcp.checksum (0.063)
8. tcp.len (0.054)
9. icmp.seq_le (0.034)
10. http.content_length (0.032)

---

## Attack Detection Strategies

### CIC-IDS2017 Recommended Approach

1. **Primary Features**: Packet length statistics (Bwd/Fwd Packet Length Max, Mean, Std)
2. **Secondary Features**: Inter-arrival time statistics (IAT Mean, Std)
3. **Port-Based Rules**: Destination Port for service-specific attacks (FTP, SSH)
4. **Uniformity Detection**: Low variance indicates automated attacks (DDoS)

### Edge-IIoTset Recommended Approach

1. **Protocol Separation**: Use protocol field presence/absence as primary discriminator
2. **TCP State Analysis**: tcp.ack, tcp.flags, tcp.options for TCP-based attacks
3. **HTTP Inspection**: URI query length, referer, method for web attacks
4. **ICMP/UDP Detection**: Protocol-specific fields (icmp.checksum, udp.stream)

### Cross-Dataset Generalization Challenges

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| Feature mismatch | Flow-level vs packet-level | Feature harmonization layer |
| Protocol coverage | CIC lacks IoT protocols | Train separate models |
| Class imbalance | CIC extreme imbalance | Stratified sampling, class weights |
| Attack semantics | Different attack implementations | Category-level transfer |

---

## Summary

### Key Findings

1. **Single-feature discrimination**: FTP-Patator (CIC) and tcp.dstport (IIOT) achieve high importance (>0.19) individually
2. **Protocol separation**: IIOT attacks are separable by protocol presence/absence patterns
3. **Statistical signatures**: CIC attacks require combinations of flow statistics
4. **Uniformity as indicator**: Low variance strongly indicates automated attacks in both datasets
5. **Port-based detection**: Effective for service-specific attacks (FTP, common IoT ports)

### Recommendations for IDS Implementation

1. Use protocol-level features when available (higher discrimination power)
2. Implement hierarchical detection: protocol type, then protocol-specific rules
3. Monitor packet uniformity metrics for automated attack detection
4. Combine statistical anomaly detection with signature-based rules
5. Maintain separate models for different protocol families

---

*Document generated: 2024-12-31*
*Feature importance computed using RandomForest classifiers*
