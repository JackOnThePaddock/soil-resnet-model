"""
ANSIS JSON-LD Parser

Parses the nested ANSIS soil data format and extracts soil chemistry observations.
"""

import json
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import DEFAULT_UPPER_DEPTH, DEFAULT_LOWER_DEPTH, MIN_OBSERVATION_DATE


def parse_wkt_point(wkt_string: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse WKT POINT geometry to extract coordinates.

    Args:
        wkt_string: WKT string like "SRID=4283;POINT(lon lat)"

    Returns:
        Tuple of (longitude, latitude) or (None, None)
    """
    if not wkt_string:
        return None, None

    # Extract coordinates from POINT(lon lat)
    match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', wkt_string)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lon, lat

    return None, None


def parse_depth_value(depth_obj: Dict) -> Optional[float]:
    """
    Parse depth value and convert to centimeters.

    Args:
        depth_obj: Depth object with 'result' containing 'value' and 'unit'

    Returns:
        Depth in centimeters
    """
    if not depth_obj or 'result' not in depth_obj:
        return None

    result = depth_obj['result']
    value = result.get('value')
    unit = result.get('unit', '')

    if value is None:
        return None

    # Convert to cm based on unit
    if 'M' in unit.upper() and 'CM' not in unit.upper():
        # Meters to centimeters
        return float(value) * 100
    elif 'CM' in unit.upper():
        return float(value)
    else:
        # Assume meters if no unit specified
        return float(value) * 100


def extract_property_value(
    prop_data: Any,
    preferred_methods: List[str] = None
) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract property value, preferring specific methods.

    Args:
        prop_data: Property data (can be list or dict)
        preferred_methods: List of preferred method codes (e.g., ['4A1', '15A1'])

    Returns:
        Tuple of (value, method_code)
    """
    if prop_data is None:
        return None, None

    # Handle list of measurements
    if isinstance(prop_data, list):
        if not prop_data:
            return None, None

        # If we have preferred methods, try to find a match
        if preferred_methods:
            for item in prop_data:
                method = item.get('usedProcedure', '')
                method_code = method.replace('scm:', '').upper()
                if method_code in [m.upper() for m in preferred_methods]:
                    result = item.get('result', {})
                    return result.get('value'), method_code

        # Otherwise return the first value
        result = prop_data[0].get('result', {})
        method = prop_data[0].get('usedProcedure', '').replace('scm:', '')
        return result.get('value'), method

    # Handle single measurement
    elif isinstance(prop_data, dict):
        result = prop_data.get('result', {})
        method = prop_data.get('usedProcedure', '').replace('scm:', '')
        return result.get('value'), method

    return None, None


def parse_ansis_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a single ANSIS JSON file and extract soil observations.

    Args:
        file_path: Path to ANSIS JSON file

    Returns:
        List of observation dictionaries
    """
    observations = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'data' not in data:
        return observations

    for site in data['data']:
        # Skip non-soil sites
        if site.get('type') != 'ansis:SoilSite':
            continue

        site_id = site.get('id', '')

        # Extract coordinates
        geometry = site.get('geometry', {})
        geom_result = geometry.get('result', '')
        lon, lat = parse_wkt_point(geom_result)

        if lon is None or lat is None:
            continue

        # Get scoped identifier if available
        scoped_ids = site.get('scopedIdentifier', [])
        external_id = None
        authority = None
        if scoped_ids:
            external_id = scoped_ids[0].get('value')
            authority = scoped_ids[0].get('authority')

        # Process site visits
        for visit in site.get('siteVisit', []):
            visit_date = visit.get('startedAtTime') or visit.get('endedAtTime')

            # Parse date
            if visit_date:
                try:
                    visit_date = datetime.fromisoformat(visit_date.replace('Z', '+00:00'))
                except:
                    visit_date = None

            # Process soil profiles
            for profile in visit.get('soilProfile', []):

                # Process soil layers
                for layer in profile.get('soilLayer', []):
                    # Get depth
                    upper_depth = parse_depth_value(layer.get('depthUpper'))
                    lower_depth = parse_depth_value(layer.get('depthLower'))

                    if upper_depth is None or lower_depth is None:
                        continue

                    # Extract soil properties
                    obs = {
                        'site_id': site_id,
                        'external_id': external_id,
                        'authority': authority,
                        'latitude': lat,
                        'longitude': lon,
                        'observation_date': visit_date.date() if visit_date else None,
                        'upper_depth_cm': upper_depth,
                        'lower_depth_cm': lower_depth,
                        'dataset_source': 'ANSIS',
                    }

                    # pH - prefer CaCl2 (4A1) over water
                    ph_data = layer.get('ph')
                    ph_val, ph_method = extract_property_value(ph_data, ['4A1', '4B1', '4B2'])
                    if ph_val is not None:
                        obs['ph_cacl2'] = ph_val
                        obs['method_ph'] = ph_method

                    # Exchangeable Calcium
                    ca_data = layer.get('exchangeableCalcium')
                    ca_val, ca_method = extract_property_value(ca_data, ['15A1', '15C1', '15D1'])
                    if ca_val is not None:
                        obs['ca_cmol_kg'] = ca_val
                        obs['method_cations'] = ca_method

                    # Exchangeable Magnesium
                    mg_data = layer.get('exchangeableMagnesium')
                    mg_val, _ = extract_property_value(mg_data, ['15A1', '15C1', '15D1'])
                    if mg_val is not None:
                        obs['mg_cmol_kg'] = mg_val

                    # Exchangeable Sodium
                    na_data = layer.get('exchangeableSodium')
                    na_val, _ = extract_property_value(na_data, ['15A1', '15C1', '15D1'])
                    if na_val is not None:
                        obs['na_cmol_kg'] = na_val

                    # Exchangeable Potassium (for CEC calculation)
                    k_data = layer.get('exchangeablePotassium')
                    k_val, _ = extract_property_value(k_data, ['15A1', '15C1', '15D1'])
                    if k_val is not None:
                        obs['k_cmol_kg'] = k_val

                    # Total Organic Carbon
                    toc_data = layer.get('totalOrganicCarbon')
                    toc_val, toc_method = extract_property_value(toc_data, ['6B2', '6B2b', '6A1'])
                    if toc_val is not None:
                        obs['soc_percent'] = toc_val
                        obs['method_soc'] = toc_method

                    # CEC - direct measurement
                    cec_data = layer.get('cationExchangeCapacity') or layer.get('effectiveCEC')
                    cec_val, cec_method = extract_property_value(cec_data, ['15J1', '15E1'])
                    if cec_val is not None:
                        obs['cec_cmol_kg'] = cec_val
                        obs['method_cec'] = cec_method

                    # Electrical Conductivity (bonus)
                    ec_data = layer.get('electricalConductivity')
                    ec_val, _ = extract_property_value(ec_data)
                    if ec_val is not None:
                        obs['ec_ds_m'] = ec_val

                    # Only add if we have at least some chemistry data
                    has_chemistry = any([
                        obs.get('ph_cacl2'),
                        obs.get('ca_cmol_kg'),
                        obs.get('soc_percent'),
                        obs.get('cec_cmol_kg')
                    ])

                    if has_chemistry:
                        observations.append(obs)

    return observations


def parse_ansis_directory(
    directory: str,
    upper_depth_max: float = DEFAULT_LOWER_DEPTH,
    min_date: str = MIN_OBSERVATION_DATE
) -> pd.DataFrame:
    """
    Parse all ANSIS JSON files in a directory.

    Args:
        directory: Path to directory containing ANSIS JSON files
        upper_depth_max: Maximum upper depth to include (cm)
        min_date: Minimum observation date (YYYY-MM-DD)

    Returns:
        DataFrame with all observations
    """
    directory = Path(directory)
    all_observations = []

    files = list(directory.iterdir())
    print(f"Found {len(files)} files to process")

    for i, file_path in enumerate(files):
        if file_path.is_file():
            try:
                observations = parse_ansis_file(str(file_path))
                all_observations.extend(observations)

                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(files)} files, {len(all_observations)} observations")

            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse {file_path.name}: {e}")
            except Exception as e:
                print(f"  Warning: Error processing {file_path.name}: {e}")

    print(f"Total raw observations: {len(all_observations)}")

    if not all_observations:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_observations)

    # Filter by depth (0-15cm by default)
    if 'upper_depth_cm' in df.columns:
        df = df[df['upper_depth_cm'] <= upper_depth_max]
        print(f"After depth filter (0-{upper_depth_max}cm): {len(df)} observations")

    # Filter by date
    if 'observation_date' in df.columns and min_date:
        min_datetime = pd.to_datetime(min_date).date()
        df['observation_date'] = pd.to_datetime(df['observation_date']).dt.date
        df = df[df['observation_date'] >= min_datetime]
        print(f"After date filter (>={min_date}): {len(df)} observations")

    # Calculate CEC if not present but have all cations
    if 'cec_cmol_kg' not in df.columns or df['cec_cmol_kg'].isna().all():
        cation_cols = ['ca_cmol_kg', 'mg_cmol_kg', 'na_cmol_kg', 'k_cmol_kg']
        available_cols = [c for c in cation_cols if c in df.columns]
        if len(available_cols) >= 3:
            df['cec_cmol_kg'] = df[available_cols].sum(axis=1, skipna=True)
            df['method_cec'] = 'calculated'
            print("Calculated CEC from exchangeable cations")

    # Calculate ESP
    if 'na_cmol_kg' in df.columns and 'cec_cmol_kg' in df.columns:
        df['esp_percent'] = (df['na_cmol_kg'] / df['cec_cmol_kg']) * 100
        df.loc[df['cec_cmol_kg'] == 0, 'esp_percent'] = np.nan
        print("Calculated ESP from Na and CEC")

    return df


def process_ansis_zip(
    zip_path: str,
    extract_dir: str = 'ansis_data',
    db_path: str = 'soil_data.db'
) -> Dict[str, Any]:
    """
    Process a downloaded ANSIS ZIP file.

    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract files
        db_path: Path to SQLite database

    Returns:
        Summary dict
    """
    import zipfile
    from db_handler import init_database, insert_soil_data, get_data_summary

    print("=" * 60)
    print("Processing ANSIS Data Export")
    print("=" * 60)

    # Extract ZIP if needed
    if not os.path.exists(extract_dir):
        print(f"\nExtracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    # Initialize database
    print("\nInitializing database...")
    init_database(db_path)

    # Parse all files
    print("\nParsing ANSIS files...")
    df = parse_ansis_directory(extract_dir)

    if df.empty:
        print("ERROR: No data extracted")
        return {'error': 'No data extracted'}

    print(f"\nFinal dataset: {len(df)} observations")
    print(f"Columns: {list(df.columns)}")

    # Insert into database
    print("\nInserting into database...")
    records_inserted = insert_soil_data(df, db_path)
    print(f"Records inserted: {records_inserted}")

    # Get summary
    summary = get_data_summary(db_path)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total records: {summary['total_records']}")

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Process provided directory or ZIP
        path = sys.argv[1]
        if path.endswith('.zip'):
            process_ansis_zip(path)
        else:
            df = parse_ansis_directory(path)
            print(f"\nDataset shape: {df.shape}")
            print(df.head())
    else:
        print("Usage: python ansis_parser.py <directory_or_zip>")
        print("\nExample:")
        print("  python ansis_parser.py ansis_data/")
        print("  python ansis_parser.py download.zip")
