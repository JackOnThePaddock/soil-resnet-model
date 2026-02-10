"""Tests for feature fusion helpers."""

import pandas as pd

from src.features.fusion import detect_alphaearth_feature_columns


def test_detect_alphaearth_feature_columns_band():
    df = pd.DataFrame({"band_10": [1], "band_2": [1], "band_1": [1]})
    cols = detect_alphaearth_feature_columns(df)
    assert cols == ["band_1", "band_2", "band_10"]


def test_detect_alphaearth_feature_columns_a_prefix():
    df = pd.DataFrame({"A10": [1], "A02": [1], "A01": [1]})
    cols = detect_alphaearth_feature_columns(df)
    assert cols == ["A01", "A02", "A10"]
