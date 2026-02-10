"""Tests for training data loading and feature resolution."""

import numpy as np
import pandas as pd
import pytest

from src.training.train_resnet import load_training_data


def test_load_training_data_sorts_numeric_feature_suffix(tmp_path):
    csv_path = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "ph": [5.2, 6.1],
            "band_10": [10.0, 11.0],
            "band_2": [2.0, 3.0],
            "band_1": [1.0, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    X, y, targets = load_training_data(
        csv_path=str(csv_path),
        target_cols=["ph"],
        feature_prefix="band_",
        n_features=3,
    )

    assert targets == ["ph"]
    assert X.shape == (2, 3)
    assert y.shape == (2, 1)
    np.testing.assert_array_equal(X[0], np.array([1.0, 2.0, 10.0]))


def test_load_training_data_falls_back_when_prefix_missing(tmp_path):
    csv_path = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "ph": [5.2, 6.1],
            "band_0": [0.1, 0.2],
            "band_1": [0.3, np.nan],
        }
    )
    df.to_csv(csv_path, index=False)

    X, y, targets = load_training_data(
        csv_path=str(csv_path),
        target_cols=["ph"],
        feature_prefix="ae_",
        n_features=2,
    )

    # One row is removed because of NaN in feature columns.
    assert X.shape == (1, 2)
    assert y.shape == (1, 1)
    assert targets == ["ph"]


def test_load_training_data_enforces_feature_count(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "ph": [5.2],
            "band_0": [0.1],
            "band_1": [0.3],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Expected 64 feature columns"):
        load_training_data(
            csv_path=str(csv_path),
            target_cols=["ph"],
            feature_prefix="band_",
            n_features=64,
        )
