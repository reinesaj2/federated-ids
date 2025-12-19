import unittest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make sure to be able to import from client
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from client import TorchClient, SimpleNet, create_model  # noqa: E402
from models.per_dataset_encoder import PerDatasetEncoderNet  # noqa: E402


class TestTorchClient(unittest.TestCase):
    def setUp(self):
        """Set up a mock client and its dependencies."""
        self.mock_model = SimpleNet(num_features=10, num_classes=2)
        # Create dummy data loaders
        dummy_X = torch.randn(10, 10)
        dummy_y = torch.randint(0, 2, (10,))
        dummy_dataset = TensorDataset(dummy_X, dummy_y)
        self.mock_train_loader = DataLoader(dummy_dataset, batch_size=4)
        self.mock_test_loader = DataLoader(dummy_dataset, batch_size=4)
        self.mock_device = torch.device("cpu")
        self.mock_metrics_logger = MagicMock()
        self.mock_fit_timer = MagicMock()
        self.data_stats = {"dataset_size": 10, "n_classes": 2}

    def _create_client(self, runtime_config):
        """Helper to create a TorchClient instance."""
        return TorchClient(
            model=self.mock_model,
            train_loader=self.mock_train_loader,
            test_loader=self.mock_test_loader,
            device=self.mock_device,
            metrics_logger=self.mock_metrics_logger,
            fit_timer=self.mock_fit_timer,
            data_stats=self.data_stats,
            runtime_config=runtime_config,
        )

    @patch("client.DPAccountant")
    def test_dp_accountant_step(self, MockDPAccountant):
        """Verify DPAccountant is stepped when DP is enabled."""
        mock_instance = MockDPAccountant.return_value
        mock_instance.get_epsilon.return_value = 1.0
        mock_instance.delta = 1e-5

        runtime_config = {
            "dp_enabled": True,
            "dp_clip": 1.0,
            "dp_noise_multiplier": 1.0,
        }
        client = self._create_client(runtime_config)

        # Mock initial parameters that match the model's state_dict structure
        initial_params = [p.detach().cpu().numpy() for p in self.mock_model.parameters()]

        # Call fit
        client.fit(initial_params, {})

        # Assertions
        mock_instance.step.assert_called_once_with(noise_multiplier=1.0, sample_rate=1.0)
        self.mock_metrics_logger.log_round_metrics.assert_called()
        # Check that dp_epsilon was passed to the logger
        call_args = self.mock_metrics_logger.log_round_metrics.call_args[1]
        self.assertAlmostEqual(call_args["dp_epsilon"], 1.0)
        self.assertAlmostEqual(call_args["dp_delta"], 1e-5)

    @patch("client.get_logger")
    def test_secure_aggregation_log(self, mock_get_logger):
        """Verify a log is created when secure aggregation is enabled."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        runtime_config = {"secure_aggregation": True}
        client = self._create_client(runtime_config)

        initial_params = [p.detach().cpu().numpy() for p in self.mock_model.parameters()]
        client.fit(initial_params, {})

        mock_logger.info.assert_called_with(
            "secure_aggregation_enabled",
            extra={"client_id": self.mock_metrics_logger.client_id, "round": 1},
        )

    def test_fit_metrics_payload_excludes_none_values(self):
        dp_enabled = False
        runtime_config = {"dp_enabled": dp_enabled}
        empty_config = {}
        client = self._create_client(runtime_config)

        initial_params = [p.detach().cpu().numpy() for p in self.mock_model.parameters()]
        _weights, _num_examples, metrics = client.fit(initial_params, empty_config)

        none_keys = [key for key, value in metrics.items() if value is None]

        self.assertEqual(none_keys, [])


class TestCreateModel(unittest.TestCase):
    def test_create_model_simple_for_synthetic(self):
        model, metadata = create_model(dataset_name="synthetic", num_features=64, num_classes=5, model_arch="auto", encoder_latent_dim=0)
        assert isinstance(model, SimpleNet)
        assert metadata["model_arch"] == "simple"

    def test_create_model_simple_explicit(self):
        model, metadata = create_model(dataset_name="cic", num_features=220, num_classes=12, model_arch="simple", encoder_latent_dim=0)
        assert isinstance(model, SimpleNet)
        assert metadata["model_arch"] == "simple"

    def test_create_model_encoder_for_unsw_auto(self):
        model, metadata = create_model(dataset_name="unsw", num_features=196, num_classes=10, model_arch="auto", encoder_latent_dim=0)
        assert isinstance(model, PerDatasetEncoderNet)
        assert metadata["model_arch"] == "per_dataset_encoder"
        assert metadata["latent_dim"] == 192

    def test_create_model_encoder_for_cic_auto(self):
        model, metadata = create_model(dataset_name="cic", num_features=220, num_classes=12, model_arch="auto", encoder_latent_dim=0)
        assert isinstance(model, PerDatasetEncoderNet)
        assert metadata["model_arch"] == "per_dataset_encoder"
        assert metadata["latent_dim"] == 256

    def test_create_model_encoder_explicit(self):
        model, metadata = create_model(
            dataset_name="synthetic",
            num_features=64,
            num_classes=5,
            model_arch="per_dataset_encoder",
            encoder_latent_dim=0,
        )
        assert isinstance(model, PerDatasetEncoderNet)
        assert metadata["model_arch"] == "per_dataset_encoder"

    def test_create_model_encoder_with_latent_override(self):
        override_latent = 512
        model, metadata = create_model(
            dataset_name="unsw",
            num_features=196,
            num_classes=10,
            model_arch="per_dataset_encoder",
            encoder_latent_dim=override_latent,
        )
        assert isinstance(model, PerDatasetEncoderNet)
        assert metadata["latent_dim"] == override_latent

    def test_create_model_forward_pass_simple(self):
        model, _ = create_model(dataset_name="synthetic", num_features=64, num_classes=5, model_arch="simple", encoder_latent_dim=0)
        x = torch.randn(32, 64)
        logits = model(x)
        assert logits.shape == (32, 5)

    def test_create_model_forward_pass_encoder(self):
        model, _ = create_model(
            dataset_name="unsw", num_features=196, num_classes=10, model_arch="per_dataset_encoder", encoder_latent_dim=0
        )
        x = torch.randn(32, 196)
        logits = model(x)
        assert logits.shape == (32, 10)


if __name__ == "__main__":
    unittest.main()
