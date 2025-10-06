"""Tests for multi-class IDS model support."""

import json
import numpy as np
import pandas as pd
import torch
from client import SimpleNet
from data_preprocessing import prepare_partitions_from_dataframe


class TestSimpleNetArchitecture:
    """Test SimpleNet model architecture with various num_classes."""

    def test_simplenet_with_2_classes(self):
        """Test SimpleNet with binary classification (default)."""
        num_features = 20
        num_classes = 2

        model = SimpleNet(num_features=num_features, num_classes=num_classes)
        last_layer = list(model.modules())[-1]

        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == num_classes

    def test_simplenet_with_8_classes(self):
        """Test SimpleNet with 8 classes for multi-class attack detection."""
        num_features = 20
        num_classes = 8
        batch_size = 4

        model = SimpleNet(num_features=num_features, num_classes=num_classes)
        last_layer = list(model.modules())[-1]

        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == num_classes

        x = torch.randn(batch_size, num_features)
        output = model(x)
        assert output.shape == (batch_size, num_classes)

    def test_simplenet_with_variable_classes(self):
        """Test SimpleNet with various num_classes values."""
        num_features = 20
        test_num_classes = [2, 3, 5, 8, 10, 15]

        for num_classes in test_num_classes:
            model = SimpleNet(num_features=num_features, num_classes=num_classes)
            last_layer = list(model.modules())[-1]
            assert last_layer.out_features == num_classes


class TestMulticlassDataPreprocessing:
    """Test data preprocessing with multi-class labels."""

    def test_multiclass_data_preprocessing(self):
        """Test that prepare_partitions_from_dataframe handles multi-class data."""
        n_samples = 1000
        n_features = 10
        n_classes = 8
        num_clients = 3
        partition_strategy = "dirichlet"
        alpha = 0.1
        seed = 42

        data = {f"feat_{i}": np.random.randn(n_samples) for i in range(n_features)}
        data["label"] = np.random.randint(0, n_classes, size=n_samples)
        df = pd.DataFrame(data)

        pre, X_parts, y_parts, num_classes_global = prepare_partitions_from_dataframe(
            df=df,
            label_col="label",
            partition_strategy=partition_strategy,
            num_clients=num_clients,
            seed=seed,
            alpha=alpha,
        )

        assert num_classes_global == n_classes

        dims = [Xp.shape[1] for Xp in X_parts]
        assert len(set(dims)) == 1

        assert sum(len(y) for y in y_parts) == len(df)

        for y_part in y_parts:
            assert y_part.min() >= 0
            assert y_part.max() < n_classes


class TestPerClassMetrics:
    """Test per-class metric JSON serialization."""

    def test_per_class_metrics_json_format(self):
        """Test that per-class metrics are stored as JSON with correct format."""
        num_classes = 8
        sample_f1s = [0.92, 0.88, 0.95, 0.83, 0.90, 0.87, 0.91, 0.89]
        sample_precisions = [0.93, 0.89, 0.96, 0.84, 0.91, 0.88, 0.92, 0.90]
        sample_recalls = [0.91, 0.87, 0.94, 0.82, 0.89, 0.86, 0.90, 0.88]

        f1_json = json.dumps({str(i): f for i, f in enumerate(sample_f1s)})
        precision_json = json.dumps(
            {str(i): p for i, p in enumerate(sample_precisions)}
        )
        recall_json = json.dumps({str(i): r for i, r in enumerate(sample_recalls)})

        f1_parsed = json.loads(f1_json)
        precision_parsed = json.loads(precision_json)
        recall_parsed = json.loads(recall_json)

        assert len(f1_parsed) == num_classes
        assert len(precision_parsed) == num_classes
        assert len(recall_parsed) == num_classes

        for i in range(num_classes):
            assert f1_parsed[str(i)] == sample_f1s[i]
            assert precision_parsed[str(i)] == sample_precisions[i]
            assert recall_parsed[str(i)] == sample_recalls[i]
