# Hybrid IDS Dataset

This document describes the intelligent fusion of three benchmark intrusion detection datasets into a unified "super dataset" for federated learning research.

## Overview

The hybrid dataset combines:
- **CIC-IDS2017**: Flow-level network traffic features
- **UNSW-NB15**: Connection-level features
- **Edge-IIoTset**: Packet-level IoT/IIoT protocol features

Each source dataset captures network intrusions at different granularities and from different network environments, providing complementary coverage for robust IDS model training.

## Dataset Sources

### CIC-IDS2017
- **Source**: Canadian Institute for Cybersecurity
- **Reference**: Sharafaldin et al., ICISSP 2018
- **Features**: 78 flow-based features (packet lengths, IATs, TCP flags)
- **Attack Types**: DoS, DDoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration

### UNSW-NB15
- **Source**: University of New South Wales
- **Reference**: Moustafa & Slay, MilCIS 2015
- **Features**: 49 connection-based features (duration, bytes, TTL, jitter)
- **Attack Types**: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

### Edge-IIoTset
- **Source**: IEEE DataPort
- **Reference**: Ferrag et al., IEEE Access 2022
- **Features**: 63 packet-level IoT protocol features (MQTT, Modbus, TCP, UDP, DNS, HTTP)
- **Attack Types**: DDoS variants, SQL Injection, XSS, Ransomware, MITM, Backdoor, Password attacks

## Unified Attack Taxonomy

The hybrid dataset harmonizes attack labels into a 7-class taxonomy:

| Class | Name       | Description                                    | Source Mappings                           |
|-------|------------|------------------------------------------------|-------------------------------------------|
| 0     | BENIGN     | Normal/legitimate traffic                      | Normal, Benign (all datasets)             |
| 1     | DOS        | Denial of Service attacks                      | DoS*, DDoS*, DDOS_ICMP/UDP/TCP/HTTP       |
| 2     | PROBE      | Reconnaissance and scanning                    | PortScan, Reconnaissance, Fingerprinting  |
| 3     | EXPLOIT    | Injection and exploitation attacks             | SQL_Injection, XSS, Exploits, Web Attacks |
| 4     | BRUTEFORCE | Password and credential attacks                | FTP-Patator, SSH-Patator, Password        |
| 5     | MALWARE    | Malicious software and backdoors               | Backdoor, Shellcode, Worms, Ransomware    |
| 6     | OTHER      | Generic and uncategorized attacks              | Generic, Fuzzers, MITM, Analysis          |

## Feature Alignment Strategy

Since each dataset has fundamentally different feature spaces, the hybrid dataset uses **semantic feature extraction**:

### Extracted Feature Groups

1. **Timing/Duration Features**
   - CIC: Flow Duration, IAT Mean/Std
   - UNSW: dur, sinpkt, dinpkt
   - IIoT: udp.time_delta

2. **Packet Count Features**
   - CIC: Total Fwd/Bwd Packets
   - UNSW: spkts, dpkts
   - IIoT: (packet-level, no aggregation)

3. **Byte Count Features**
   - CIC: Total Length Fwd/Bwd Packets
   - UNSW: sbytes, dbytes
   - IIoT: tcp.len

4. **TCP State Features**
   - CIC: FIN/SYN/RST/ACK Flag Counts
   - UNSW: tcprtt, synack, state
   - IIoT: tcp.flags.*, tcp.connection.*

5. **Protocol-Specific Features**
   - IIoT-only: mqtt.*, mbtcp.*, dns.qry.*

## Usage

### Generate Hybrid Dataset

```bash
python scripts/create_hybrid_dataset.py \
    --data-dir data \
    --output data/hybrid/hybrid_ids_dataset.csv \
    --iiot-variant nightly \
    --balance stratified \
    --seed 42
```

### Command Line Options

| Option           | Default                              | Description                              |
|------------------|--------------------------------------|------------------------------------------|
| `--data-dir`     | `data`                               | Root directory with dataset folders      |
| `--output`       | `data/hybrid/hybrid_ids_dataset.csv` | Output CSV path                          |
| `--iiot-variant` | `nightly`                            | IIoT variant: nightly, full, 500k        |
| `--max-samples`  | None                                 | Limit samples per source dataset         |
| `--balance`      | `none`                               | Balance strategy: none, undersample, stratified |
| `--no-normalize` | False                                | Skip StandardScaler normalization        |
| `--seed`         | 42                                   | Random seed for reproducibility          |

### Balance Strategies

- **none**: Concatenate all datasets without balancing
- **undersample**: Undersample majority classes to match minority class count
- **stratified**: Balance samples across source datasets (equal contribution)

## Output Format

The hybrid dataset CSV contains:

| Column                 | Type    | Description                              |
|------------------------|---------|------------------------------------------|
| `duration`             | float   | Connection/flow duration                 |
| `fwd_packets`          | float   | Forward packet count                     |
| `bwd_packets`          | float   | Backward packet count                    |
| ... (50+ features)     | float   | Semantic features from each source       |
| `source_dataset`       | string  | Origin dataset: cic, unsw, iiot          |
| `attack_class`         | int     | Unified class index (0-6)                |
| `attack_label_original`| string  | Original attack label from source        |

## Integration with Federated Learning

The hybrid dataset is designed for federated learning experiments:

1. **Heterogeneity Simulation**: Use `source_dataset` column to partition data by origin, simulating real-world data heterogeneity across organizations

2. **Cross-Domain Evaluation**: Train on one/two datasets, test on held-out dataset to measure generalization

3. **Attack Coverage**: Unified taxonomy enables consistent evaluation across diverse attack scenarios

### Example: Dirichlet Partitioning with Source Awareness

```python
from data_preprocessing import dirichlet_partition

# Load hybrid dataset
df = pd.read_csv("data/hybrid/hybrid_ids_dataset.csv")

# Partition by attack class with alpha=0.5 (moderate heterogeneity)
labels = df["attack_class"].values
client_indices = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=42)
```

## Experimental Considerations

### Limitations

1. **Feature Misalignment**: Not all features map semantically across datasets; some are zero-filled
2. **Temporal Disjoint**: Datasets collected at different times with different traffic patterns
3. **Label Noise**: Fuzzy matching may introduce minor label inconsistencies

### Best Practices

1. Report results separately by source dataset in addition to aggregate metrics
2. Use stratified balancing for fair cross-dataset comparison
3. Document which IIoT variant (nightly/full/500k) was used for reproducibility

## References

1. Sharafaldin, I., Lashkari, A.H., and Ghorbani, A.A. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." ICISSP.

2. Moustafa, N., and Slay, J. (2015). "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems." MilCIS.

3. Ferrag, M.A., et al. (2022). "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning." IEEE Access.
