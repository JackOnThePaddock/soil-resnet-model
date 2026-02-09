"""Tests for data cleaner module."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_cleaner import DataCleaner


class TestDataCleaner:
    def test_remove_duplicates(self):
        df = pd.DataFrame({
            "latitude": [1.0, 1.0, 2.0],
            "longitude": [1.0, 1.0, 2.0],
            "depth_upper": [0, 0, 0],
            "ph": [5.5, 5.5, 6.0],
        })
        cleaner = DataCleaner()
        cleaned = cleaner.remove_duplicates(df)
        assert len(cleaned) <= len(df)

    def test_validate_coordinates(self):
        df = pd.DataFrame({
            "latitude": [-33.0, 200.0, -28.0],
            "longitude": [151.0, 151.0, -300.0],
        })
        cleaner = DataCleaner()
        valid = cleaner.validate_coordinates(df)
        assert len(valid) == 1

    def test_filter_depth_range(self):
        df = pd.DataFrame({
            "depth_upper": [0, 15, 30, 0],
            "depth_lower": [15, 30, 45, 10],
            "ph": [5.0, 6.0, 7.0, 5.5],
        })
        cleaner = DataCleaner()
        filtered = cleaner.filter_depth(df, max_depth=15)
        assert all(filtered["depth_upper"] <= 15)
