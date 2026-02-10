"""Tests for baseline feature-column resolution."""

import pandas as pd

from src.training.train_baselines import _resolve_feature_columns


def test_resolve_feature_columns_band_order():
    df = pd.DataFrame(
        {
            "band_10": [1.0],
            "band_2": [2.0],
            "band_1": [3.0],
            "ph": [5.5],
        }
    )

    cols = _resolve_feature_columns(df, feature_prefix="band_")
    assert cols == ["band_1", "band_2", "band_10"]


def test_resolve_feature_columns_fallback_from_auto():
    df = pd.DataFrame({"band_0": [1.0], "band_1": [2.0], "ph": [5.5]})
    cols = _resolve_feature_columns(df, feature_prefix="auto")
    assert cols == ["band_0", "band_1"]
