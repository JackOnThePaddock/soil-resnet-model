"""
Data Cleaning and Transformation for Soil Data

Handles unit conversions, ESP calculation, and data validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

from config import (
    ATOMIC_WEIGHTS,
    DEFAULT_UPPER_DEPTH,
    DEFAULT_LOWER_DEPTH,
    MIN_OBSERVATION_DATE
)


def convert_mg_kg_to_cmol_kg(value: float, element: str) -> Optional[float]:
    """
    Convert mg/kg to cmol(+)/kg for exchangeable cations.

    Formula: cmol(+)/kg = mg/kg / (atomic_weight * 10)

    Args:
        value: Value in mg/kg
        element: Element symbol (Ca, Mg, Na, K)

    Returns:
        Value in cmol(+)/kg or None if conversion fails
    """
    if pd.isna(value) or value is None:
        return None

    if element not in ATOMIC_WEIGHTS:
        raise ValueError(f"Unknown element: {element}")

    return value / (ATOMIC_WEIGHTS[element] * 10)


def convert_ph_water_to_cacl2(ph_water: float) -> Optional[float]:
    """
    Convert pH(Water) to pH(CaCl2).

    Approximate conversion: pH(CaCl2) = pH(Water) - 0.7
    Note: This is an approximation and may vary by soil type.

    Args:
        ph_water: pH measured in water

    Returns:
        Estimated pH in CaCl2
    """
    if pd.isna(ph_water) or ph_water is None:
        return None

    return ph_water - 0.7


def calculate_esp(na_cmol_kg: float, cec_cmol_kg: float) -> Optional[float]:
    """
    Calculate Exchangeable Sodium Percentage (ESP).

    Formula: ESP = (Na / CEC) * 100

    Args:
        na_cmol_kg: Exchangeable sodium in cmol(+)/kg
        cec_cmol_kg: Cation exchange capacity in cmol(+)/kg

    Returns:
        ESP as percentage
    """
    if pd.isna(na_cmol_kg) or pd.isna(cec_cmol_kg):
        return None

    if cec_cmol_kg == 0:
        return None

    return (na_cmol_kg / cec_cmol_kg) * 100


def validate_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate soil property values are within expected ranges.
    Sets outliers to NaN with a warning.

    Args:
        df: DataFrame with soil data

    Returns:
        DataFrame with validated/cleaned values
    """
    expected_ranges = {
        'ph_cacl2': (2.0, 8.5),       # Capped at 8.5 per training spec
        'cec_cmol_kg': (0, 200),
        'soc_percent': (0, 60),
        'esp_percent': (0, 25),        # Capped at 25% per training spec
        'ca_cmol_kg': (0, 150),
        'mg_cmol_kg': (0, 100),
        'na_cmol_kg': (0, 50)
    }

    df_clean = df.copy()

    for col, (min_val, max_val) in expected_ranges.items():
        if col in df_clean.columns:
            outliers = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                print(f"Warning: {outlier_count} outliers found in {col} "
                      f"(outside {min_val}-{max_val}). Setting to NaN.")
                df_clean.loc[outliers, col] = np.nan

    return df_clean


def filter_by_depth(
    df: pd.DataFrame,
    upper_depth: int = DEFAULT_UPPER_DEPTH,
    lower_depth: int = DEFAULT_LOWER_DEPTH
) -> pd.DataFrame:
    """
    Filter records by depth range.

    Args:
        df: DataFrame with depth columns
        upper_depth: Minimum upper depth (cm)
        lower_depth: Maximum lower depth (cm)

    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()

    # Check for depth columns
    if 'upper_depth' in df_filtered.columns and 'lower_depth' in df_filtered.columns:
        mask = (
            (df_filtered['upper_depth'] >= upper_depth) &
            (df_filtered['lower_depth'] <= lower_depth)
        )
        df_filtered = df_filtered[mask]

    elif 'UpperDepth' in df_filtered.columns and 'LowerDepth' in df_filtered.columns:
        mask = (
            (df_filtered['UpperDepth'] >= upper_depth) &
            (df_filtered['LowerDepth'] <= lower_depth)
        )
        df_filtered = df_filtered[mask]

    return df_filtered


def filter_by_date(
    df: pd.DataFrame,
    min_date: str = MIN_OBSERVATION_DATE
) -> pd.DataFrame:
    """
    Filter records by observation date.

    Args:
        df: DataFrame with date column
        min_date: Minimum date string (YYYY-MM-DD)

    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    min_datetime = pd.to_datetime(min_date)

    # Find the date column
    date_cols = ['observation_date', 'ObservationDate', 'date', 'Date', 'SampleDate']
    date_col = None

    for col in date_cols:
        if col in df_filtered.columns:
            date_col = col
            break

    if date_col:
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
        df_filtered = df_filtered[df_filtered[date_col] >= min_datetime]

    return df_filtered


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to match database schema.

    Args:
        df: DataFrame with raw API column names

    Returns:
        DataFrame with standardized column names
    """
    column_mapping = {
        # Location
        'Longitude': 'longitude',
        'Latitude': 'latitude',
        'longitude': 'longitude',
        'latitude': 'latitude',
        'X': 'longitude',
        'Y': 'latitude',

        # Site info
        'SiteID': 'site_id',
        'site_id': 'site_id',
        'LocationID': 'site_id',
        'DataSet': 'dataset_source',
        'Provider': 'dataset_source',

        # Depth
        'UpperDepth': 'upper_depth_cm',
        'LowerDepth': 'lower_depth_cm',
        'upper_depth': 'upper_depth_cm',
        'lower_depth': 'lower_depth_cm',

        # Date
        'ObservationDate': 'observation_date',
        'SampleDate': 'observation_date',
        'Date': 'observation_date',

        # Properties - these will vary by API response
        'pH_CaCl2': 'ph_cacl2',
        'pH_H2O': 'ph_water',
        'CEC': 'cec_cmol_kg',
        'ECEC': 'cec_cmol_kg',
        'OC': 'soc_percent',
        'SOC': 'soc_percent',
        'OrganicCarbon': 'soc_percent',
        'ExchCa': 'ca_cmol_kg',
        'ExchMg': 'mg_cmol_kg',
        'ExchNa': 'na_cmol_kg',
        'ESP': 'esp_percent',

        # Methods
        'Method': 'method_cations',
        'MethodCode': 'method_cations'
    }

    df_std = df.copy()
    df_std = df_std.rename(columns=column_mapping)

    return df_std


def clean_soil_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning pipeline for soil data.

    Args:
        df: Raw DataFrame from API

    Returns:
        Cleaned and standardized DataFrame
    """
    # Step 1: Standardize column names
    df_clean = standardize_columns(df)

    # Step 2: Filter by depth
    df_clean = filter_by_depth(df_clean)

    # Step 3: Filter by date
    df_clean = filter_by_date(df_clean)

    # Step 4: Convert pH(Water) to pH(CaCl2) if needed
    if 'ph_water' in df_clean.columns and 'ph_cacl2' not in df_clean.columns:
        df_clean['ph_cacl2'] = df_clean['ph_water'].apply(convert_ph_water_to_cacl2)
        print("Converted pH(Water) to pH(CaCl2)")

    # Step 5: Calculate ESP if not present
    if 'esp_percent' not in df_clean.columns:
        if 'na_cmol_kg' in df_clean.columns and 'cec_cmol_kg' in df_clean.columns:
            df_clean['esp_percent'] = df_clean.apply(
                lambda row: calculate_esp(row['na_cmol_kg'], row['cec_cmol_kg']),
                axis=1
            )
            print("Calculated ESP from Na and CEC")

    # Step 6: Validate value ranges
    df_clean = validate_value_ranges(df_clean)

    # Step 7: Remove duplicates
    dedup_cols = ['site_id', 'observation_date', 'upper_depth_cm', 'lower_depth_cm']
    existing_cols = [c for c in dedup_cols if c in df_clean.columns]

    if existing_cols:
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=existing_cols, keep='first')
        removed = initial_count - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} duplicate records")

    return df_clean


def merge_property_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge data from multiple property group queries.

    Args:
        dataframes: List of DataFrames from different property queries

    Returns:
        Merged DataFrame
    """
    if not dataframes:
        return pd.DataFrame()

    # Start with first DataFrame
    merged = dataframes[0].copy()

    # Merge remaining DataFrames
    for df in dataframes[1:]:
        if df.empty:
            continue

        # Find common columns for merging (location + site + date)
        merge_cols = ['site_id', 'latitude', 'longitude', 'observation_date',
                      'upper_depth_cm', 'lower_depth_cm']
        available_merge_cols = [c for c in merge_cols if c in merged.columns and c in df.columns]

        if available_merge_cols:
            merged = pd.merge(
                merged,
                df,
                on=available_merge_cols,
                how='outer',
                suffixes=('', '_dup')
            )

            # Remove duplicate columns
            merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]
        else:
            # If no common columns, just concatenate
            merged = pd.concat([merged, df], ignore_index=True)

    return merged


if __name__ == "__main__":
    # Test conversions
    print("Testing unit conversions:")
    print(f"  Ca: 800 mg/kg = {convert_mg_kg_to_cmol_kg(800, 'Ca'):.2f} cmol(+)/kg")
    print(f"  Mg: 200 mg/kg = {convert_mg_kg_to_cmol_kg(200, 'Mg'):.2f} cmol(+)/kg")
    print(f"  Na: 100 mg/kg = {convert_mg_kg_to_cmol_kg(100, 'Na'):.2f} cmol(+)/kg")
    print(f"  pH(Water) 6.5 = pH(CaCl2) {convert_ph_water_to_cacl2(6.5):.1f}")
    print(f"  ESP: Na=2, CEC=20 = {calculate_esp(2, 20):.1f}%")
