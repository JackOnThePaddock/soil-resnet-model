"""Tests for data cleaner module."""

import numpy as np
import pandas as pd
import pytest

from src.data.data_cleaner import (
    clean_soil_data,
    validate_value_ranges,
    standardize_columns,
    convert_mg_kg_to_cmol_kg,
    convert_ph_water_to_cacl2,
    calculate_esp,
    merge_property_data,
)


class TestUnitConversions:
    def test_mg_kg_to_cmol_kg_calcium(self):
        result = convert_mg_kg_to_cmol_kg(400.8, "Ca")
        assert abs(result - 1.0) < 0.01  # 400.8 mg/kg Ca = ~1.0 cmol/kg

    def test_mg_kg_to_cmol_kg_unknown_element(self):
        result = convert_mg_kg_to_cmol_kg(100.0, "Fe")
        assert result is None

    def test_mg_kg_to_cmol_kg_nan(self):
        result = convert_mg_kg_to_cmol_kg(float("nan"), "Ca")
        assert result is None

    def test_ph_water_to_cacl2(self):
        result = convert_ph_water_to_cacl2(7.0)
        assert abs(result - 6.3) < 0.01

    def test_ph_water_to_cacl2_nan(self):
        result = convert_ph_water_to_cacl2(float("nan"))
        assert result is None

    def test_calculate_esp(self):
        result = calculate_esp(2.0, 10.0)
        assert abs(result - 20.0) < 0.01

    def test_calculate_esp_zero_cec(self):
        result = calculate_esp(2.0, 0.0)
        assert result is None


class TestValidateValueRanges:
    def test_outliers_set_to_nan(self):
        df = pd.DataFrame({
            "ph_cacl2": [5.0, 1.0, 9.5, 6.0],  # 1.0 and 9.5 are outliers
            "cec_cmol_kg": [10.0, 250.0, 5.0, 20.0],  # 250 is outlier
        })
        cleaned = validate_value_ranges(df)
        assert pd.isna(cleaned.loc[1, "ph_cacl2"])
        assert pd.isna(cleaned.loc[2, "ph_cacl2"])
        assert pd.isna(cleaned.loc[1, "cec_cmol_kg"])
        assert cleaned.loc[0, "ph_cacl2"] == 5.0

    def test_valid_values_unchanged(self):
        df = pd.DataFrame({"ph_cacl2": [4.5, 5.5, 7.0]})
        cleaned = validate_value_ranges(df)
        pd.testing.assert_frame_equal(df, cleaned)


class TestStandardizeColumns:
    def test_renames_columns(self):
        df = pd.DataFrame({"Latitude": [1.0], "Longitude": [2.0], "pH_CaCl2": [5.5]})
        result = standardize_columns(df)
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        assert "ph_cacl2" in result.columns


class TestCleanSoilData:
    def test_deduplication(self):
        df = pd.DataFrame({
            "site_id": ["A", "A", "B"],
            "observation_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "upper_depth_cm": [0, 0, 0],
            "lower_depth_cm": [15, 15, 15],
            "ph_cacl2": [5.5, 5.5, 6.0],
        })
        cleaned = clean_soil_data(df)
        assert len(cleaned) == 2

    def test_ph_conversion_when_missing(self):
        df = pd.DataFrame({"ph_water": [7.0, 7.5]})
        cleaned = clean_soil_data(df)
        assert "ph_cacl2" in cleaned.columns
        assert abs(cleaned["ph_cacl2"].iloc[0] - 6.3) < 0.01

    def test_esp_calculation(self):
        df = pd.DataFrame({
            "na_cmol_kg": [1.0, 2.0],
            "cec_cmol_kg": [10.0, 20.0],
        })
        cleaned = clean_soil_data(df)
        assert "esp_percent" in cleaned.columns
        assert abs(cleaned["esp_percent"].iloc[0] - 10.0) < 0.01


class TestMergePropertyData:
    def test_merge_two_dataframes(self):
        df1 = pd.DataFrame({"site_id": ["A", "B"], "ph_cacl2": [5.0, 6.0]})
        df2 = pd.DataFrame({"site_id": ["A", "B"], "cec_cmol_kg": [10.0, 20.0]})
        merged = merge_property_data([df1, df2])
        assert "ph_cacl2" in merged.columns
        assert "cec_cmol_kg" in merged.columns
        assert len(merged) == 2

    def test_empty_list(self):
        result = merge_property_data([])
        assert result.empty
