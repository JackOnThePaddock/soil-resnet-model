"""Data cleaning and transformation for soil data."""

from typing import List, Optional

import numpy as np
import pandas as pd

ATOMIC_WEIGHTS = {"Ca": 40.08, "Mg": 24.31, "Na": 22.99, "K": 39.10}


def convert_mg_kg_to_cmol_kg(value: float, element: str) -> Optional[float]:
    """Convert mg/kg to cmol(+)/kg for exchangeable cations."""
    if pd.isna(value) or element not in ATOMIC_WEIGHTS:
        return None
    return value / (ATOMIC_WEIGHTS[element] * 10)


def convert_ph_water_to_cacl2(ph_water: float) -> Optional[float]:
    """Approximate conversion: pH(CaCl2) = pH(Water) - 0.7."""
    if pd.isna(ph_water):
        return None
    return ph_water - 0.7


def calculate_esp(na_cmol_kg: float, cec_cmol_kg: float) -> Optional[float]:
    """Calculate Exchangeable Sodium Percentage: ESP = (Na / CEC) * 100."""
    if pd.isna(na_cmol_kg) or pd.isna(cec_cmol_kg) or cec_cmol_kg == 0:
        return None
    return (na_cmol_kg / cec_cmol_kg) * 100


def validate_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Set outlier soil property values to NaN."""
    ranges = {
        "ph_cacl2": (2.0, 8.5), "cec_cmol_kg": (0, 200), "soc_percent": (0, 60),
        "esp_percent": (0, 25), "ca_cmol_kg": (0, 150), "mg_cmol_kg": (0, 100),
        "na_cmol_kg": (0, 50),
    }
    df_clean = df.copy()
    for col, (lo, hi) in ranges.items():
        if col in df_clean.columns:
            outliers = (df_clean[col] < lo) | (df_clean[col] > hi)
            if outliers.sum() > 0:
                print(f"Warning: {outliers.sum()} outliers in {col}, set to NaN")
                df_clean.loc[outliers, col] = np.nan
    return df_clean


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to match database schema."""
    mapping = {
        "Longitude": "longitude", "Latitude": "latitude", "X": "longitude", "Y": "latitude",
        "SiteID": "site_id", "LocationID": "site_id", "DataSet": "dataset_source",
        "UpperDepth": "upper_depth_cm", "LowerDepth": "lower_depth_cm",
        "ObservationDate": "observation_date", "SampleDate": "observation_date",
        "pH_CaCl2": "ph_cacl2", "pH_H2O": "ph_water",
        "CEC": "cec_cmol_kg", "OC": "soc_percent", "SOC": "soc_percent",
        "ExchCa": "ca_cmol_kg", "ExchMg": "mg_cmol_kg", "ExchNa": "na_cmol_kg",
        "ESP": "esp_percent",
    }
    return df.rename(columns=mapping)


def clean_soil_data(df: pd.DataFrame, min_date: str = "2017-01-01") -> pd.DataFrame:
    """Main cleaning pipeline for soil data."""
    df_clean = standardize_columns(df)

    # pH conversion
    if "ph_water" in df_clean.columns and "ph_cacl2" not in df_clean.columns:
        df_clean["ph_cacl2"] = df_clean["ph_water"].apply(convert_ph_water_to_cacl2)

    # ESP calculation
    if "esp_percent" not in df_clean.columns:
        if "na_cmol_kg" in df_clean.columns and "cec_cmol_kg" in df_clean.columns:
            df_clean["esp_percent"] = df_clean.apply(
                lambda r: calculate_esp(r["na_cmol_kg"], r["cec_cmol_kg"]), axis=1
            )

    df_clean = validate_value_ranges(df_clean)

    # Deduplicate
    dedup_cols = [c for c in ["site_id", "observation_date", "upper_depth_cm", "lower_depth_cm"] if c in df_clean.columns]
    if dedup_cols:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=dedup_cols, keep="first")
        removed = before - len(df_clean)
        if removed:
            print(f"Removed {removed} duplicates")

    return df_clean


def merge_property_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge data from multiple property group queries."""
    if not dataframes:
        return pd.DataFrame()
    merged = dataframes[0].copy()
    for df in dataframes[1:]:
        if df.empty:
            continue
        merge_cols = [c for c in ["site_id", "latitude", "longitude", "observation_date", "upper_depth_cm", "lower_depth_cm"]
                      if c in merged.columns and c in df.columns]
        if merge_cols:
            merged = pd.merge(merged, df, on=merge_cols, how="outer", suffixes=("", "_dup"))
            merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
        else:
            merged = pd.concat([merged, df], ignore_index=True)
    return merged
