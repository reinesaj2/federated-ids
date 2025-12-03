# Edge-IIoTset Dataset

## Overview

Edge-IIoTset is a comprehensive realistic cyber security dataset of IoT and IIoT applications designed for both centralized and federated learning. This dataset contains 2,219,201 samples with 14 distinct attack types captured from IoT/IIoT devices in a seven-layer architecture.

## Dataset Information

**Authors**: Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke

**Publication**: IEEE Access, 2022

**DOI**: 10.36227/techrxiv.18857336.v1

**License**: Free use for academic research. For commercial use, contact the lead author.

**Contact**: mohamed.amine.ferrag@gmail.com

## Citation

```bibtex
@article{ferrag2022edgeiiot,
  title={Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning},
  author={Ferrag, Mohamed Amine and Friha, Othmane and Hamouda, Djallel and Maglaras, Leandros and Janicke, Helge},
  journal={IEEE Access},
  year={2022},
  doi={10.36227/techrxiv.18857336.v1}
}
```

## Dataset Structure

```
Edge-IIoTset dataset/
├── Attack traffic/           # 13 attack type CSV and PCAP files
├── Normal traffic/           # 10 IoT sensor CSV and PCAP files
└── Selected dataset for ML and DL/
    ├── DNN-EdgeIIoT-dataset.csv   # 1.1GB - Full dataset for deep learning
    └── ML-EdgeIIoT-dataset.csv    # 78MB - Subset for traditional ML
```

## Dataset Characteristics

- **Total Samples**: 2,219,201
- **Features**: 61 network flow features
- **Attack Types**: 14 distinct categories
- **Normal Traffic**: 1,615,643 samples (73%)
- **Attack Traffic**: 603,558 samples (27%)

### Classification Labels

#### Binary Classification (Attack_label)
- `0`: Normal
- `1`: Attack

#### Multi-class Classification (Attack_type)
1. BENIGN (Normal)
2. DDoS_UDP
3. DDoS_ICMP
4. DDoS_TCP
5. DDoS_HTTP
6. SQL_injection
7. Password
8. Vulnerability_scanner
9. Uploading
10. Backdoor
11. Port_Scanning
12. XSS
13. Ransomware
14. MITM
15. Fingerprinting

## Attack Distribution

| Attack Type | Samples | Percentage |
|-------------|---------|------------|
| BENIGN | 1,615,643 | 72.8% |
| DDoS_UDP | 121,568 | 5.5% |
| DDoS_ICMP | 116,436 | 5.2% |
| SQL_injection | 51,203 | 2.3% |
| Password | 50,153 | 2.3% |
| Vulnerability_scanner | 50,110 | 2.3% |
| DDoS_TCP | 50,062 | 2.3% |
| DDoS_HTTP | 49,911 | 2.2% |
| Uploading | 37,634 | 1.7% |
| Backdoor | 24,862 | 1.1% |
| Port_Scanning | 22,564 | 1.0% |
| XSS | 15,915 | 0.7% |
| Ransomware | 10,925 | 0.5% |
| MITM | 1,214 | 0.1% |
| Fingerprinting | 1,001 | 0.0% |

## IoT Devices

The dataset includes traffic from 10 types of IoT sensors:

1. Distance (Ultrasonic sensor)
2. Flame Sensor
3. Heart Rate sensor
4. IR (Infrared) Receiver
5. Modbus sensor
6. pH Sensor (PH-4502C)
7. Soil Moisture Sensor v1.2
8. Sound Detection Sensor (LM393)
9. Temperature and Humidity (DHT11)
10. Water Level sensor

## Features

The dataset contains 61 network flow features extracted from:

- Frame attributes
- IP layer (source/destination hosts)
- ARP protocol
- ICMP protocol
- HTTP protocol
- TCP protocol
- UDP protocol
- DNS protocol
- MQTT protocol
- Modbus TCP protocol

Key feature categories:
- Protocol-specific headers and flags
- Packet sizes and counts
- Timing information
- Connection states

## Network Architecture

The testbed is organized into seven layers:

1. **Cloud Computing Layer**
2. **Network Functions Virtualization Layer**
3. **Blockchain Network Layer**
4. **Fog Computing Layer**
5. **Software-Defined Networking Layer**
6. **Edge Computing Layer**
7. **IoT and IIoT Perception Layer**

Technologies used: ThingsBoard IoT platform, OPNFV, Hyperledger Sawtooth, ONOS SDN controller, Mosquitto MQTT brokers

## Usage in Federated Learning

This dataset is explicitly designed for federated learning scenarios, making it ideal for:

- **Realistic non-IID partitioning**: Different IoT devices naturally have heterogeneous data distributions
- **Byzantine robustness testing**: Multiple attack types stress-test robust aggregation methods
- **Privacy evaluation**: Large-scale differential privacy validation
- **Personalization**: Per-device model adaptation

## Preprocessing Recommendations

From the dataset authors (Readme.txt):

### Columns to Drop (for privacy/leakage prevention)
```python
drop_columns = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
]
```

### Preprocessing Steps
1. Drop null rows and duplicate rows
2. Shuffle data
3. Encode categorical features with dummy encoding
4. Standardize numerical features

## Downloads

- **Kaggle**: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- **IEEE DataPort**: https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications

## Integration with This Project

See `docs/edge_iiotset_integration.md` for detailed integration instructions.

Quick start:
```bash
# Generate stratified samples
python scripts/prepare_edge_iiotset_samples.py --tier all

# Run experiment
python scripts/comparative_analysis.py \
    --dataset edge-iiotset-quick \
    --preset comp_fedavg_alpha1.0_seed42
```

## References

1. Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning", IEEE Access, 2022
2. Dataset on Kaggle
3. Dataset on IEEE DataPort
4. Official documentation: Edge_IIoTset__DatasetFL.pdf

## Support

For questions about the dataset:
- **Lead Author**: Dr. Mohamed Amine Ferrag
- **Email**: mohamed.amine.ferrag@gmail.com or ferrag.mohamedamine@univ-guelma.dz
- **Google Scholar**: https://scholar.google.fr/citations?user=IkPeqxMAAAAJ
- **ORCID**: 0000-0002-0632-3172

## Last Updated

18 March 2022 (Original dataset)
12 January 2025 (This integration)
