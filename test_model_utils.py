"""
Tests for model persistence utilities.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from client import SimpleNet
from model_utils import (
    get_model_metadata,
    list_model_artifacts,
    load_model,
    save_client_model,
    save_final_model,
    save_global_model,
    save_model,
)


class TestModelUtils:
    """Test model persistence utilities."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return SimpleNet(num_features=10, num_classes=2)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load_model_basic(self, simple_model, temp_dir):
        """Test basic model save and load functionality."""
        model_path = temp_dir / "test_model.pth"

        # Save model
        save_model(simple_model, model_path)

        # Verify file exists
        assert model_path.exists()

        # Load model
        loaded_model = load_model(model_path, SimpleNet, num_features=10, num_classes=2)

        # Verify models have same state
        original_state = simple_model.state_dict()
        loaded_state = loaded_model.state_dict()

        assert set(original_state.keys()) == set(loaded_state.keys())
        for key in original_state.keys():
            torch.testing.assert_close(original_state[key], loaded_state[key])

    def test_save_model_with_metadata(self, simple_model, temp_dir):
        """Test saving model with metadata."""
        model_path = temp_dir / "model_with_metadata.pth"
        test_metadata = {"round": 5, "client_id": 1, "accuracy": 0.95}

        save_model(simple_model, model_path, metadata=test_metadata)

        # Verify metadata can be retrieved
        retrieved_metadata = get_model_metadata(model_path)
        assert retrieved_metadata == test_metadata

    def test_save_client_model_naming_convention(self, simple_model, temp_dir):
        """Test client model naming follows expected convention."""
        client_id = 3
        round_num = 7

        saved_path = save_client_model(simple_model, temp_dir, client_id, round_num)

        expected_filename = f"client_{client_id}_model_round_{round_num}.pth"
        assert saved_path.name == expected_filename
        assert saved_path.exists()

        # Verify metadata
        metadata = get_model_metadata(saved_path)
        assert metadata["client_id"] == client_id
        assert metadata["round"] == round_num
        assert metadata["model_type"] == "client"

    def test_save_global_model_naming_convention(self, simple_model, temp_dir):
        """Test global model naming follows expected convention."""
        round_num = 10

        saved_path = save_global_model(simple_model, temp_dir, round_num)

        expected_filename = f"global_model_round_{round_num}.pth"
        assert saved_path.name == expected_filename
        assert saved_path.exists()

        # Verify metadata
        metadata = get_model_metadata(saved_path)
        assert metadata["round"] == round_num
        assert metadata["model_type"] == "global"

    def test_save_final_model_naming_convention(self, simple_model, temp_dir):
        """Test final model naming follows expected convention."""
        saved_path = save_final_model(simple_model, temp_dir)

        expected_filename = "final_global_model.pth"
        assert saved_path.name == expected_filename
        assert saved_path.exists()

        # Verify metadata
        metadata = get_model_metadata(saved_path)
        assert metadata["model_type"] == "final_global"

    def test_model_directory_creation(self, simple_model):
        """Test that model saving creates directories as needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "deep" / "model.pth"

            # Directory doesn't exist yet
            assert not nested_path.parent.exists()

            save_model(simple_model, nested_path)

            # Directory and file should now exist
            assert nested_path.parent.exists()
            assert nested_path.exists()

    def test_load_model_file_not_found(self, temp_dir):
        """Test load_model raises appropriate error for missing file."""
        nonexistent_path = temp_dir / "does_not_exist.pth"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_model(nonexistent_path, SimpleNet, num_features=10, num_classes=2)

    def test_list_model_artifacts_empty_directory(self, temp_dir):
        """Test listing artifacts in empty directory."""
        artifacts = list_model_artifacts(temp_dir)

        expected_structure = {
            "client_models": [],
            "global_models": [],
            "final_model": [],
        }
        assert artifacts == expected_structure

    def test_list_model_artifacts_mixed_files(self, simple_model, temp_dir):
        """Test listing artifacts with mixed model files."""
        # Create various model files
        save_client_model(simple_model, temp_dir, 0, 1)
        save_client_model(simple_model, temp_dir, 1, 1)
        save_client_model(simple_model, temp_dir, 0, 2)

        save_global_model(simple_model, temp_dir, 1)
        save_global_model(simple_model, temp_dir, 2)

        save_final_model(simple_model, temp_dir)

        # Create non-model file to verify it's ignored
        (temp_dir / "not_a_model.txt").write_text("ignore me")

        artifacts = list_model_artifacts(temp_dir)

        # Verify client models
        assert len(artifacts["client_models"]) == 3
        client_names = [p.name for p in artifacts["client_models"]]
        assert "client_0_model_round_1.pth" in client_names
        assert "client_1_model_round_1.pth" in client_names
        assert "client_0_model_round_2.pth" in client_names

        # Verify global models
        assert len(artifacts["global_models"]) == 2
        global_names = [p.name for p in artifacts["global_models"]]
        assert "global_model_round_1.pth" in global_names
        assert "global_model_round_2.pth" in global_names

        # Verify final model
        assert len(artifacts["final_model"]) == 1
        assert artifacts["final_model"][0].name == "final_global_model.pth"

    def test_get_model_metadata_no_metadata(self, simple_model, temp_dir):
        """Test getting metadata from model without metadata."""
        model_path = temp_dir / "no_metadata.pth"
        save_model(simple_model, model_path)  # No metadata provided

        metadata = get_model_metadata(model_path)
        assert metadata is None

    def test_get_model_metadata_nonexistent_file(self, temp_dir):
        """Test getting metadata from nonexistent file."""
        nonexistent_path = temp_dir / "missing.pth"

        metadata = get_model_metadata(nonexistent_path)
        assert metadata is None

    def test_load_model_preserves_architecture(self, temp_dir):
        """Test that loaded model preserves original architecture."""
        original_model = SimpleNet(num_features=15, num_classes=3)
        model_path = temp_dir / "architecture_test.pth"

        save_model(original_model, model_path)
        loaded_model = load_model(model_path, SimpleNet, num_features=15, num_classes=3)

        # Verify architecture matches
        assert isinstance(loaded_model, SimpleNet)

        # Check that forward pass works with expected input shape
        test_input = torch.randn(1, 15)
        output = loaded_model(test_input)
        assert output.shape == (1, 3)

    def test_model_roundtrip_preserves_predictions(self, temp_dir):
        """Test that save/load roundtrip preserves model predictions."""
        model = SimpleNet(num_features=5, num_classes=2)
        model_path = temp_dir / "roundtrip_test.pth"

        # Generate test input
        test_input = torch.randn(3, 5)

        # Get original predictions
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)

        # Save and load model
        save_model(model, model_path)
        loaded_model = load_model(model_path, SimpleNet, num_features=5, num_classes=2)

        # Get loaded model predictions
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(test_input)

        # Predictions should be identical
        torch.testing.assert_close(original_output, loaded_output)

    def test_save_models_with_custom_metadata(self, simple_model, temp_dir):
        """Test saving different model types with custom metadata."""
        base_metadata = {"experiment": "test_run", "seed": 42}

        # Test client model with custom metadata
        client_path = save_client_model(
            simple_model, temp_dir, 0, 1, metadata=base_metadata
        )
        client_meta = get_model_metadata(client_path)
        assert client_meta["experiment"] == "test_run"
        assert client_meta["client_id"] == 0
        assert client_meta["model_type"] == "client"

        # Test global model with custom metadata
        global_path = save_global_model(
            simple_model, temp_dir, 1, metadata=base_metadata
        )
        global_meta = get_model_metadata(global_path)
        assert global_meta["experiment"] == "test_run"
        assert global_meta["round"] == 1
        assert global_meta["model_type"] == "global"

        # Test final model with custom metadata
        final_path = save_final_model(simple_model, temp_dir, metadata=base_metadata)
        final_meta = get_model_metadata(final_path)
        assert final_meta["experiment"] == "test_run"
        assert final_meta["model_type"] == "final_global"
