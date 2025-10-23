import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make sure to be able to import from client
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from client import TorchClient, SimpleNet


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
        mock_instance.step.assert_called_once_with(
            noise_multiplier=1.0, sample_rate=1.0
        )
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


if __name__ == "__main__":
    unittest.main()
