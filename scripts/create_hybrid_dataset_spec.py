"""Tests for create_hybrid_dataset module.

Tests the hybrid dataset fusion functionality including:
- Label harmonization across attack taxonomies
- Feature extraction from each dataset type
- Dataset combination and balancing strategies
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.create_hybrid_dataset import (
    UNIFIED_ATTACK_TAXONOMY,
    UNIFIED_CLASS_NAMES,
    DatasetConfig,
    dedupe_hybrid_dataset,
    downsample_hybrid_dataset,
    get_cic_config,
    get_iiot_config,
    get_unsw_config,
    harmonize_label,
    load_and_extract_features,
)


class TestHarmonizeLabel:
    """Tests for harmonize_label function."""

    @pytest.mark.parametrize(
        "label,expected_class",
        [
            ("BENIGN", 0),
            ("Normal", 0),
            ("benign", 0),
            ("NORMAL", 0),
            ("DOS", 1),
            ("DDoS", 1),
            ("DOS HULK", 1),
            ("DDOS_ICMP", 1),
            ("DDOS_TCP", 1),
            ("DoS GoldenEye", 1),
            ("PORTSCAN", 2),
            ("PortScan", 2),
            ("Reconnaissance", 2),
            ("FINGERPRINTING", 2),
            ("SQL_INJECTION", 3),
            ("XSS", 3),
            ("Exploits", 3),
            ("Infiltration", 3),
            ("Heartbleed", 3),
            ("WEB ATTACK - SQL INJECTION", 3),
            ("WEB ATTACK - XSS", 3),
            ("FTP-Patator", 4),
            ("SSH-Patator", 4),
            ("PASSWORD", 4),
            ("WEB ATTACK - BRUTE FORCE", 4),
            ("Web Attack � Brute Force", 4),
            ("BACKDOOR", 5),
            ("Backdoors", 5),
            ("RANSOMWARE", 5),
            ("Shellcode", 5),
            ("Worms", 5),
            ("Bot", 5),
            ("GENERIC", 6),
            ("Fuzzers", 6),
            ("MITM", 6),
            ("Web Attack � XSS", 3),
            ("Web Attack � Sql Injection", 3),
        ],
    )
    def test_known_labels_map_correctly(self, label: str, expected_class: int) -> None:
        """Known attack labels should map to correct unified class."""
        assert harmonize_label(label) == expected_class

    def test_unknown_label_maps_to_other(self) -> None:
        """Unknown labels should map to OTHER (class 6)."""
        assert harmonize_label("UNKNOWN_ATTACK_TYPE") == 6

    def test_label_normalization_case_insensitive(self) -> None:
        """Label matching should be case-insensitive."""
        assert harmonize_label("benign") == harmonize_label("BENIGN")
        assert harmonize_label("dos hulk") == harmonize_label("DOS HULK")

    def test_label_normalization_strips_whitespace(self) -> None:
        """Labels with whitespace should be normalized."""
        assert harmonize_label("  BENIGN  ") == 0
        assert harmonize_label("\tDOS\n") == 1

    def test_fuzzy_matching_dos_variants(self) -> None:
        """DoS-like labels should match via fuzzy matching."""
        assert harmonize_label("DOS_SOMETHING_NEW") == 1
        assert harmonize_label("DDOS_CUSTOM") == 1

    def test_fuzzy_matching_scan_variants(self) -> None:
        """Scan-like labels should match via fuzzy matching."""
        assert harmonize_label("NETWORK_SCAN") == 2
        assert harmonize_label("RECON_ATTACK") == 2


class TestUnifiedTaxonomy:
    """Tests for unified attack taxonomy constants."""

    def test_taxonomy_has_seven_classes(self) -> None:
        """Unified taxonomy should have exactly 7 classes."""
        assert len(UNIFIED_CLASS_NAMES) == 7

    def test_class_names_indexed_correctly(self) -> None:
        """Class names should be in correct order."""
        assert UNIFIED_CLASS_NAMES[0] == "BENIGN"
        assert UNIFIED_CLASS_NAMES[1] == "DOS"
        assert UNIFIED_CLASS_NAMES[2] == "PROBE"
        assert UNIFIED_CLASS_NAMES[3] == "EXPLOIT"
        assert UNIFIED_CLASS_NAMES[4] == "BRUTEFORCE"
        assert UNIFIED_CLASS_NAMES[5] == "MALWARE"
        assert UNIFIED_CLASS_NAMES[6] == "OTHER"

    def test_all_taxonomy_values_in_range(self) -> None:
        """All taxonomy values should be valid class indices."""
        for class_idx in UNIFIED_ATTACK_TAXONOMY.values():
            assert 0 <= class_idx <= 6


class TestDatasetConfigs:
    """Tests for dataset configuration functions."""

    def test_cic_config_has_required_fields(self) -> None:
        """CIC config should have all required fields."""
        config = get_cic_config(Path("data"))
        assert config.name == "cic"
        assert config.label_col == "Label"
        assert len(config.feature_mapping) > 0
        assert "duration" in config.feature_mapping

    def test_unsw_config_has_required_fields(self) -> None:
        """UNSW config should have all required fields."""
        config = get_unsw_config(Path("data"))
        assert config.name == "unsw"
        assert config.label_col == "attack_cat"
        assert len(config.feature_mapping) > 0
        assert "duration" in config.feature_mapping

    def test_iiot_config_has_required_fields(self) -> None:
        """IIoT config should have all required fields."""
        config = get_iiot_config(Path("data"))
        assert config.name == "iiot"
        assert config.label_col == "Attack_type"
        assert len(config.feature_mapping) > 0

    def test_iiot_config_variants(self) -> None:
        """IIoT config should support different variants."""
        nightly = get_iiot_config(Path("data"), "nightly")
        full = get_iiot_config(Path("data"), "full")
        assert "nightly" in str(nightly.csv_path)
        assert "full" in str(full.csv_path)


class TestLoadAndExtractFeatures:
    """Tests for load_and_extract_features function."""

    def test_missing_dataset_raises_file_not_found(self) -> None:
        """Missing dataset file should raise FileNotFoundError."""
        config = DatasetConfig(
            name="test",
            csv_path=Path("/nonexistent/path.csv"),
            label_col="label",
            feature_mapping={"feat": "col"},
            drop_cols=[],
        )
        with pytest.raises(FileNotFoundError):
            load_and_extract_features(config)

    def test_extracted_features_include_metadata(self, tmp_path: Path) -> None:
        """Extracted DataFrame should include source and label columns."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [4.0, 5.0, 6.0],
                "label": ["BENIGN", "DOS", "XSS"],
            }
        )
        df.to_csv(csv_path, index=False)

        config = DatasetConfig(
            name="test_ds",
            csv_path=csv_path,
            label_col="label",
            feature_mapping={"feat1": "col1", "feat2": "col2"},
            drop_cols=[],
        )
        result = load_and_extract_features(config)

        assert "source_dataset" in result.columns
        assert "attack_class" in result.columns
        assert "attack_label_original" in result.columns
        assert result["source_dataset"].iloc[0] == "test_ds"

    def test_labels_are_harmonized(self, tmp_path: Path) -> None:
        """Attack labels should be mapped to unified taxonomy."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],
                "label": ["BENIGN", "DOS", "XSS"],
            }
        )
        df.to_csv(csv_path, index=False)

        config = DatasetConfig(
            name="test",
            csv_path=csv_path,
            label_col="label",
            feature_mapping={"feat1": "col1"},
            drop_cols=[],
        )
        result = load_and_extract_features(config)

        assert result["attack_class"].tolist() == [0, 1, 3]

    def test_missing_features_filled_with_zero(self, tmp_path: Path) -> None:
        """Missing feature columns should be filled with zeros."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "col1": [1.0, 2.0],
                "label": ["BENIGN", "DOS"],
            }
        )
        df.to_csv(csv_path, index=False)

        config = DatasetConfig(
            name="test",
            csv_path=csv_path,
            label_col="label",
            feature_mapping={"feat1": "col1", "feat_missing": "nonexistent_col"},
            drop_cols=[],
        )
        result = load_and_extract_features(config)

        assert "feat_missing" in result.columns
        assert result["feat_missing"].isna().sum() == 0 or (result["feat_missing"] == 0).all()

    def test_max_samples_limits_rows(self, tmp_path: Path) -> None:
        """max_samples parameter should limit number of rows loaded."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "col1": range(100),
                "label": ["BENIGN"] * 100,
            }
        )
        df.to_csv(csv_path, index=False)

        config = DatasetConfig(
            name="test",
            csv_path=csv_path,
            label_col="label",
            feature_mapping={"feat1": "col1"},
            drop_cols=[],
        )
        result = load_and_extract_features(config, max_samples=10)

        assert len(result) == 10


class TestDownsampleHybridDataset:
    """Tests for downsample_hybrid_dataset helper."""

    def test_downsample_to_exact_total_rows(self) -> None:
        total_rows = 4
        seed = 123
        df = pd.DataFrame(
            {
                "duration": list(range(10)),
                "source_dataset": ["cic"] * 5 + ["unsw"] * 5,
                "attack_class": [0] * 10,
                "attack_label_original": ["BENIGN"] * 10,
            }
        )

        result = downsample_hybrid_dataset(df, total_rows, seed=seed)

        assert len(result) == total_rows

    def test_min_rows_per_source_is_respected(self) -> None:
        total_rows = 6
        min_rows_per_source = 2
        seed = 42
        df = pd.DataFrame(
            {
                "duration": list(range(12)),
                "source_dataset": ["cic"] * 4 + ["unsw"] * 4 + ["iiot"] * 4,
                "attack_class": [0] * 12,
                "attack_label_original": ["BENIGN"] * 12,
            }
        )

        result = downsample_hybrid_dataset(
            df,
            total_rows,
            seed=seed,
            min_rows_per_source=min_rows_per_source,
        )

        counts = result["source_dataset"].value_counts().to_dict()
        assert len(result) == total_rows
        assert counts["cic"] >= min_rows_per_source
        assert counts["unsw"] >= min_rows_per_source
        assert counts["iiot"] >= min_rows_per_source

    def test_min_rows_per_source_requires_sufficient_total_rows(self) -> None:
        total_rows = 5
        min_rows_per_source = 2
        seed = 42
        df = pd.DataFrame(
            {
                "duration": list(range(9)),
                "source_dataset": ["cic"] * 3 + ["unsw"] * 3 + ["iiot"] * 3,
                "attack_class": [0] * 9,
                "attack_label_original": ["BENIGN"] * 9,
            }
        )

        with pytest.raises(ValueError):
            downsample_hybrid_dataset(
                df,
                total_rows,
                seed=seed,
                min_rows_per_source=min_rows_per_source,
            )


class TestDedupeHybridDataset:
    def test_dedupe_removes_exact_duplicates(self) -> None:
        df = pd.DataFrame(
            {
                "duration": [1.0, 1.0, 2.0],
                "source_dataset": ["cic", "cic", "cic"],
                "attack_class": [0, 0, 1],
                "attack_label_original": ["BENIGN", "BENIGN", "DOS"],
            }
        )

        result = dedupe_hybrid_dataset(df)

        assert len(result) == 2

    def test_min_rows_per_source_requires_sufficient_rows_per_source(self) -> None:
        total_rows = 5
        min_rows_per_source = 3
        seed = 42
        df = pd.DataFrame(
            {
                "duration": list(range(5)),
                "source_dataset": ["cic"] * 3 + ["unsw"] * 1 + ["iiot"] * 1,
                "attack_class": [0] * 5,
                "attack_label_original": ["BENIGN"] * 5,
            }
        )

        with pytest.raises(ValueError):
            downsample_hybrid_dataset(
                df,
                total_rows,
                seed=seed,
                min_rows_per_source=min_rows_per_source,
            )
