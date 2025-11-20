import numpy as np
import pandas as pd

from data_preprocessing import DEFAULT_DROP_COLS, fit_preprocessor_train_only_and_transform_all


def test_train_only_fit_preserves_feature_dim():
    # Build a small dataframe with identifiers and categorical/numeric mix
    df = pd.DataFrame(
        {
            'Flow ID': [f'id{i}' for i in range(200)],
            'Timestamp': [f't{i}' for i in range(200)],
            'Src IP': [f'10.0.0.{i%5}' for i in range(200)],
            'num1': np.linspace(0, 1, 200),
            'cat1': np.array(['A', 'B', 'C', 'D'])[np.random.randint(0, 4, 200)],
            'label': np.random.randint(0, 2, 200),
        }
    )
    pre, X_all, y_all = fit_preprocessor_train_only_and_transform_all(df, 'label', drop_cols=DEFAULT_DROP_COLS, seed=123)
    # No identifiers columns should remain; feature dim must be > 0
    assert X_all.shape[0] == df.shape[0]
    assert X_all.shape[1] > 0
