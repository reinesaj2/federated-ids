"""Hybrid IDS Dataset Creator: Intelligent fusion of CIC-IDS2017, UNSW-NB15, Edge-IIoTset.

This script creates a unified benchmark dataset by:
1. Extracting semantically aligned features from all three datasets
2. Harmonizing attack labels into a unified 7-class taxonomy
3. Balancing class distributions across dataset sources
4. Preserving source provenance for analysis

Feature Alignment Strategy:
    Each dataset has fundamentally different feature spaces (flow/connection/packet level).
    We extract common semantic properties: timing, size, protocol, TCP state.
    Dataset-specific features are preserved with source-aware encoding.

Attack Taxonomy (7 classes):
    0: BENIGN      - Normal/benign traffic
    1: DOS         - DoS/DDoS attacks (volumetric, protocol, application layer)
    2: PROBE       - Reconnaissance, port scanning, fingerprinting
    3: EXPLOIT     - SQL injection, XSS, exploits, web attacks
    4: BRUTEFORCE  - Password attacks, credential stuffing
    5: MALWARE     - Backdoors, ransomware, worms, shellcode
    6: OTHER       - Generic attacks, fuzzing, analysis, MITM

References:
    CIC-IDS2017: Sharafaldin et al., ICISSP 2018
    UNSW-NB15: Moustafa & Slay, MilCIS 2015
    Edge-IIoTset: Ferrag et al., IEEE Access 2022
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


UNIFIED_ATTACK_TAXONOMY: dict[str, int] = {
    "BENIGN": 0,
    "NORMAL": 0,
    "DOS": 1,
    "DDOS": 1,
    "DOS HULK": 1,
    "DOS GOLDENEYE": 1,
    "DOS SLOWLORIS": 1,
    "DOS SLOWHTTPTEST": 1,
    "DDOS_ICMP": 1,
    "DDOS_UDP": 1,
    "DDOS_TCP": 1,
    "DDOS_HTTP": 1,
    "PORTSCAN": 2,
    "RECONNAISSANCE": 2,
    "FINGERPRINTING": 2,
    "VULNERABILITY_SCANNER": 2,
    "PORT_SCANNING": 2,
    "ANALYSIS": 2,
    "EXPLOITS": 3,
    "SQL_INJECTION": 3,
    "XSS": 3,
    "WEB ATTACK - XSS": 3,
    "WEB ATTACK - SQL INJECTION": 3,
    "FTP-PATATOR": 4,
    "SSH-PATATOR": 4,
    "PASSWORD": 4,
    "BRUTEFORCE": 4,
    "WEB ATTACK - BRUTE FORCE": 4,
    "WEB ATTACK � BRUTE FORCE": 4,
    "WEB ATTACK � XSS": 3,
    "WEB ATTACK � SQL INJECTION": 3,
    "INFILTRATION": 3,
    "HEARTBLEED": 3,
    "BACKDOOR": 5,
    "BACKDOORS": 5,
    "SHELLCODE": 5,
    "WORMS": 5,
    "RANSOMWARE": 5,
    "UPLOADING": 5,
    "BOT": 5,
    "GENERIC": 6,
    "FUZZERS": 6,
    "MITM": 6,
}

UNIFIED_CLASS_NAMES: list[str] = [
    "BENIGN",
    "DOS",
    "PROBE",
    "EXPLOIT",
    "BRUTEFORCE",
    "MALWARE",
    "OTHER",
]

_WARNED_UNKNOWN_LABELS: set[str] = set()


@dataclass
class DatasetConfig:
    """Configuration for a source dataset."""

    name: str
    csv_path: Path | list[Path]
    label_col: str
    feature_mapping: dict[str, str]
    drop_cols: list[str]


def get_cic_config(data_dir: Path, use_full: bool = False) -> DatasetConfig:
    """CIC-IDS2017 dataset configuration.

    Args:
        data_dir: Root data directory
        use_full: If True, load all raw CIC CSV files (~2.8M rows)
    """
    if use_full:
        raw_dir = data_dir / "cic" / "raw" / "MachineLearningCVE"
        csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
        fallback = data_dir / "cic" / "cic_ids2017_multiclass.csv"
        csv_path: Path | list[Path] = csv_files if csv_files else fallback
    else:
        csv_path = data_dir / "cic" / "cic_ids2017_multiclass.csv"

    return DatasetConfig(
        name="cic",
        csv_path=csv_path,
        label_col="Label",
        feature_mapping={
            "duration": "Flow Duration",
            "fwd_packets": "Total Fwd Packets",
            "bwd_packets": "Total Backward Packets",
            "fwd_bytes": "Total Length of Fwd Packets",
            "bwd_bytes": "Total Length of Bwd Packets",
            "fwd_pkt_len_mean": "Fwd Packet Length Mean",
            "fwd_pkt_len_std": "Fwd Packet Length Std",
            "bwd_pkt_len_mean": "Bwd Packet Length Mean",
            "bwd_pkt_len_std": "Bwd Packet Length Std",
            "flow_bytes_per_sec": "Flow Bytes/s",
            "flow_pkts_per_sec": "Flow Packets/s",
            "flow_iat_mean": "Flow IAT Mean",
            "flow_iat_std": "Flow IAT Std",
            "fwd_iat_mean": "Fwd IAT Mean",
            "bwd_iat_mean": "Bwd IAT Mean",
            "fin_flag": "FIN Flag Count",
            "syn_flag": "SYN Flag Count",
            "rst_flag": "RST Flag Count",
            "psh_flag": "PSH Flag Count",
            "ack_flag": "ACK Flag Count",
            "pkt_len_mean": "Packet Length Mean",
            "pkt_len_std": "Packet Length Std",
            "pkt_len_min": "Min Packet Length",
            "pkt_len_max": "Max Packet Length",
            "init_win_fwd": "Init_Win_bytes_forward",
            "init_win_bwd": "Init_Win_bytes_backward",
            "active_mean": "Active Mean",
            "idle_mean": "Idle Mean",
        },
        drop_cols=["Flow ID", "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port"],
    )


def get_unsw_config(data_dir: Path) -> DatasetConfig:
    """UNSW-NB15 dataset configuration."""
    return DatasetConfig(
        name="unsw",
        csv_path=data_dir / "unsw" / "UNSW_NB15_training-set.csv",
        label_col="attack_cat",
        feature_mapping={
            "duration": "dur",
            "fwd_packets": "spkts",
            "bwd_packets": "dpkts",
            "fwd_bytes": "sbytes",
            "bwd_bytes": "dbytes",
            "rate": "rate",
            "src_ttl": "sttl",
            "dst_ttl": "dttl",
            "src_load": "sload",
            "dst_load": "dload",
            "src_loss": "sloss",
            "dst_loss": "dloss",
            "src_interpkt": "sinpkt",
            "dst_interpkt": "dinpkt",
            "src_jitter": "sjit",
            "dst_jitter": "djit",
            "tcp_rtt": "tcprtt",
            "synack_time": "synack",
            "ackdat_time": "ackdat",
            "src_mean_pkt": "smean",
            "dst_mean_pkt": "dmean",
            "ct_srv_src": "ct_srv_src",
            "ct_state_ttl": "ct_state_ttl",
            "ct_dst_ltm": "ct_dst_ltm",
            "ct_src_dport_ltm": "ct_src_dport_ltm",
            "is_ftp_login": "is_ftp_login",
            "ct_ftp_cmd": "ct_ftp_cmd",
            "ct_flw_http_mthd": "ct_flw_http_mthd",
        },
        drop_cols=["id", "proto", "service", "state"],
    )


def get_iiot_config(data_dir: Path, variant: str = "nightly") -> DatasetConfig:
    """Edge-IIoTset dataset configuration."""
    csv_variants = {
        "nightly": "edge_iiotset_nightly.csv",
        "full": "edge_iiotset_full.csv",
        "500k": "edge_iiotset_500k_curated.csv",
    }
    return DatasetConfig(
        name="iiot",
        csv_path=data_dir / "edge-iiotset" / csv_variants.get(variant, csv_variants["nightly"]),
        label_col="Attack_type",
        feature_mapping={
            "tcp_ack": "tcp.ack",
            "tcp_checksum": "tcp.checksum",
            "tcp_fin": "tcp.connection.fin",
            "tcp_rst": "tcp.connection.rst",
            "tcp_syn": "tcp.connection.syn",
            "tcp_synack": "tcp.connection.synack",
            "tcp_dstport": "tcp.dstport",
            "tcp_flags": "tcp.flags",
            "tcp_flags_ack": "tcp.flags.ack",
            "tcp_len": "tcp.len",
            "tcp_seq": "tcp.seq",
            "udp_port": "udp.port",
            "udp_stream": "udp.stream",
            "udp_time_delta": "udp.time_delta",
            "dns_qry_len": "dns.qry.name.len",
            "dns_qry_type": "dns.qry.type",
            "http_content_len": "http.content_length",
            "mqtt_len": "mqtt.len",
            "mqtt_msgtype": "mqtt.msgtype",
            "mqtt_topic_len": "mqtt.topic_len",
            "mbtcp_len": "mbtcp.len",
            "mbtcp_trans_id": "mbtcp.trans_id",
            "icmp_checksum": "icmp.checksum",
            "icmp_seq": "icmp.seq_le",
            "arp_opcode": "arp.opcode",
        },
        drop_cols=[
            "frame.time",
            "ip.src_host",
            "ip.dst_host",
            "tcp.payload",
            "tcp.options",
            "tcp.srcport",
            "http.request.full_uri",
            "http.file_data",
            "mqtt.topic",
            "mqtt.msg",
            "mqtt.protoname",
            "dns.qry.name",
        ],
    )


def harmonize_label(label: str) -> int:
    """Map attack label to unified taxonomy class index.

    Args:
        label: Original attack label string from any dataset

    Returns:
        Integer class index (0-6) in unified taxonomy
    """
    normalized = str(label).strip().upper()

    if normalized in UNIFIED_ATTACK_TAXONOMY:
        return UNIFIED_ATTACK_TAXONOMY[normalized]

    if "DOS" in normalized or "DDOS" in normalized:
        return 1
    if "SCAN" in normalized or "RECON" in normalized or "PROBE" in normalized:
        return 2
    if "INJECT" in normalized or "XSS" in normalized or "EXPLOIT" in normalized:
        return 3
    if "BRUTE" in normalized or "PATATOR" in normalized or "PASSWORD" in normalized:
        return 4
    if "BACKDOOR" in normalized or "MALWARE" in normalized or "RANSOM" in normalized:
        return 5
    if "WORM" in normalized or "SHELL" in normalized:
        return 5

    if normalized not in _WARNED_UNKNOWN_LABELS:
        _WARNED_UNKNOWN_LABELS.add(normalized)
        logger.warning(f"Unknown attack label '{label}' mapped to OTHER (class 6)")
    return 6


def load_and_extract_features(
    config: DatasetConfig, max_samples: int | None = None
) -> pd.DataFrame:
    """Load dataset and extract semantically aligned features.

    Args:
        config: Dataset configuration with paths and feature mappings
        max_samples: Optional limit on number of rows to load

    Returns:
        DataFrame with extracted features, source identifier, and harmonized labels
    """
    csv_paths = config.csv_path if isinstance(config.csv_path, list) else [config.csv_path]
    logger.info(f"Loading {config.name} from {len(csv_paths)} file(s)")

    extracted_parts: list[pd.DataFrame] = []
    samples_remaining = max_samples
    for csv_path in csv_paths:
        if not csv_path.exists():
            logger.warning(f"  Skipping missing file: {csv_path}")
            continue

        nrows = samples_remaining if samples_remaining else None

        header = pd.read_csv(csv_path, low_memory=False, nrows=0)
        normalized_to_actual: dict[str, str] = {
            str(col).strip(): str(col) for col in header.columns
        }
        needed_normalized = {config.label_col.strip()} | {
            str(col).strip() for col in config.feature_mapping.values()
        }

        missing_label = config.label_col.strip() not in normalized_to_actual
        if missing_label:
            label_like = [c for c in normalized_to_actual if "label" in c.lower()]
            raise ValueError(
                f"Label column '{config.label_col}' not found in {config.name} header. "
                f"Label-like columns: {label_like}"
            )

        usecols = [
            normalized_to_actual[col]
            for col in needed_normalized
            if col in normalized_to_actual
        ]

        raw_df = pd.read_csv(csv_path, low_memory=False, nrows=nrows, usecols=usecols)
        raw_df.columns = [
            col.strip() if isinstance(col, str) else col for col in raw_df.columns
        ]

        extracted: dict[str, pd.Series] = {}
        for semantic_name, actual_col in config.feature_mapping.items():
            actual_col_normalized = str(actual_col).strip()
            if actual_col_normalized in raw_df.columns:
                extracted[semantic_name] = pd.to_numeric(
                    raw_df[actual_col_normalized], errors="coerce"
                )
            else:
                extracted[semantic_name] = pd.Series(np.nan, index=raw_df.index)

        result = pd.DataFrame(extracted)
        result["source_dataset"] = config.name

        label_col = config.label_col.strip()
        original_labels = raw_df[label_col].astype(str).str.strip()
        result["attack_class"] = original_labels.apply(harmonize_label)
        result["attack_label_original"] = original_labels

        metadata_cols = ["source_dataset", "attack_class", "attack_label_original"]
        feature_cols = [c for c in result.columns if c not in metadata_cols]
        result[feature_cols] = result[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        result[feature_cols] = result[feature_cols].astype(np.float32, copy=False)

        extracted_parts.append(result)

        logger.info(f"  Loaded {len(raw_df)} samples from {csv_path.name}")
        if samples_remaining:
            samples_remaining -= len(raw_df)
            if samples_remaining <= 0:
                break

    if not extracted_parts:
        raise FileNotFoundError(f"No valid CSV files found for {config.name}")

    combined = pd.concat(extracted_parts, ignore_index=True)
    metadata_cols = ["source_dataset", "attack_class", "attack_label_original"]
    feature_cols = [c for c in combined.columns if c not in metadata_cols]
    logger.info(f"  Extracted {len(feature_cols)} features, {len(combined)} samples")
    return combined


def downsample_hybrid_dataset(
    df: pd.DataFrame,
    total_rows: int,
    *,
    seed: int,
    min_rows_per_source: int | None = None,
) -> pd.DataFrame:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")

    if min_rows_per_source is None:
        if total_rows >= len(df):
            return df
        return df.sample(n=total_rows, random_state=seed).reset_index(drop=True)

    if min_rows_per_source <= 0:
        raise ValueError("min_rows_per_source must be positive when provided")

    source_counts = df["source_dataset"].value_counts()
    sources = list(source_counts.index)
    required = min_rows_per_source * len(sources)
    if required > total_rows:
        raise ValueError(
            f"min_rows_per_source={min_rows_per_source} across {len(sources)} sources "
            f"requires at least {required} total_rows"
        )

    if (source_counts < min_rows_per_source).any():
        too_small = source_counts[source_counts < min_rows_per_source].to_dict()
        raise ValueError(f"Insufficient rows for sources: {too_small}")

    if total_rows >= len(df):
        return df

    reserved_parts = []
    for idx, source in enumerate(sources):
        reserved_parts.append(
            df[df["source_dataset"] == source].sample(
                n=min_rows_per_source, random_state=seed + idx
            )
        )
    reserved = pd.concat(reserved_parts, ignore_index=False)

    remaining_needed = total_rows - len(reserved)
    remaining_pool = df.drop(index=reserved.index)
    remaining = remaining_pool.sample(n=remaining_needed, random_state=seed)

    sampled = pd.concat([reserved, remaining], ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def dedupe_hybrid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def create_hybrid_dataset(
    data_dir: Path,
    output_path: Path,
    iiot_variant: str = "nightly",
    use_full_cic: bool = False,
    max_samples_per_dataset: int | None = None,
    total_rows: int | None = None,
    min_rows_per_source: int | None = None,
    dedupe: bool = False,
    balance_strategy: Literal["none", "undersample", "stratified"] = "none",
    normalize: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Create unified hybrid dataset from all three sources.

    Args:
        data_dir: Root directory containing dataset folders
        output_path: Path to save the hybrid dataset CSV
        iiot_variant: Edge-IIoTset variant ("nightly", "full", "500k")
        use_full_cic: If True, load all CIC raw CSV files (~2.8M rows)
        max_samples_per_dataset: Limit samples per source dataset
        total_rows: If set, downsample final hybrid dataset to this many rows
        min_rows_per_source: When downsampling, guarantee at least this many rows per source
        dedupe: Whether to drop exact-duplicate rows after feature extraction
        balance_strategy: How to balance class distributions
        normalize: Whether to apply StandardScaler normalization
        seed: Random seed for reproducibility

    Returns:
        Combined DataFrame with unified features and labels
    """
    configs = [
        get_cic_config(data_dir, use_full=use_full_cic),
        get_unsw_config(data_dir),
        get_iiot_config(data_dir, iiot_variant),
    ]

    datasets = []
    for config in configs:
        try:
            df = load_and_extract_features(config, max_samples_per_dataset)
            datasets.append(df)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {config.name}: {e}")
        except Exception as e:
            logger.error(f"Error loading {config.name}: {e}")
            raise

    if not datasets:
        raise ValueError("No datasets loaded successfully")

    logger.info("Combining datasets...")
    combined = pd.concat(datasets, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} samples")

    metadata_cols = ["source_dataset", "attack_class", "attack_label_original"]
    feature_cols = [c for c in combined.columns if c not in metadata_cols]
    combined[feature_cols] = combined[feature_cols].fillna(0)

    if dedupe:
        logger.info("Dropping exact duplicate rows...")
        before = len(combined)
        combined = dedupe_hybrid_dataset(combined)
        logger.info(f"After dedupe: {len(combined)} samples (dropped {before - len(combined)})")

    if balance_strategy == "undersample":
        logger.info("Applying undersampling to balance classes...")
        min_class_count = combined["attack_class"].value_counts().min()
        balanced_dfs = []
        for class_idx in range(len(UNIFIED_CLASS_NAMES)):
            class_df = combined[combined["attack_class"] == class_idx]
            if len(class_df) > min_class_count:
                class_df = class_df.sample(n=min_class_count, random_state=seed)
            balanced_dfs.append(class_df)
        combined = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"After undersampling: {len(combined)} samples")

    elif balance_strategy == "stratified":
        logger.info("Applying stratified sampling to balance datasets...")
        min_dataset_count = combined["source_dataset"].value_counts().min()
        balanced_dfs = []
        for source in combined["source_dataset"].unique():
            source_df = combined[combined["source_dataset"] == source]
            if len(source_df) > min_dataset_count:
                source_df = source_df.sample(n=min_dataset_count, random_state=seed)
            balanced_dfs.append(source_df)
        combined = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"After stratified sampling: {len(combined)} samples")

    if normalize:
        logger.info("Normalizing features with StandardScaler...")
        scaler = StandardScaler(copy=False)
        combined[feature_cols] = scaler.fit_transform(combined[feature_cols])

    if total_rows is not None:
        logger.info(f"Downsampling to total_rows={total_rows}...")
        combined = downsample_hybrid_dataset(
            combined, total_rows, seed=seed, min_rows_per_source=min_rows_per_source
        )
    else:
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved hybrid dataset to {output_path}")

    logger.info("\n=== Hybrid Dataset Summary ===")
    logger.info(f"Total samples: {len(combined)}")
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info("\nSamples by source:")
    for source, count in combined["source_dataset"].value_counts().items():
        pct = 100 * count / len(combined)
        logger.info(f"  {source}: {count} ({pct:.1f}%)")
    logger.info("\nSamples by attack class:")
    for class_idx, class_name in enumerate(UNIFIED_CLASS_NAMES):
        count = (combined["attack_class"] == class_idx).sum()
        pct = 100 * count / len(combined)
        logger.info(f"  {class_idx} ({class_name}): {count} ({pct:.1f}%)")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create hybrid IDS dataset from CIC, UNSW, and Edge-IIoTset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing dataset folders (default: data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hybrid/hybrid_ids_dataset.csv"),
        help="Output path for hybrid dataset CSV",
    )
    parser.add_argument(
        "--iiot-variant",
        choices=["nightly", "full", "500k"],
        default="nightly",
        help="Edge-IIoTset variant to use (default: nightly)",
    )
    parser.add_argument(
        "--use-full-cic",
        action="store_true",
        help="Load all CIC raw CSV files (~2.8M rows) instead of sample",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per source dataset (default: unlimited)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact duplicate rows after feature extraction",
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=None,
        help="Downsample final hybrid dataset to this many rows (default: unlimited)",
    )
    parser.add_argument(
        "--min-rows-per-source",
        type=int,
        default=None,
        help="When using --total-rows, guarantee at least this many rows per source dataset",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "undersample", "stratified"],
        default="none",
        help="Class balancing strategy (default: none)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip feature normalization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    create_hybrid_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        iiot_variant=args.iiot_variant,
        use_full_cic=args.use_full_cic,
        max_samples_per_dataset=args.max_samples,
        total_rows=args.total_rows,
        min_rows_per_source=args.min_rows_per_source,
        dedupe=args.dedupe,
        balance_strategy=args.balance,
        normalize=not args.no_normalize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
