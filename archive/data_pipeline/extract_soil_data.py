"""
Soil Data Extraction and Consolidation Script
Extracts soil data from multiple sources and formats per year (2017+)
Output format: id,lat,lon,band_0,...,band_63,ph,cec,esp,soc,ca,mg,na
"""

import pandas as pd
import struct
from pathlib import Path
import os

# Paths
BASE_DIR = Path(r"C:\Users\jackc\Downloads\Soil Data")
ALPHAEARTH_DIR = BASE_DIR / "National Soil Data Standardised" / "by_year_cleaned_top10cm_metrics_alphaearth"
SHAPEFILE_DIR = BASE_DIR / "SOIL MODEL Training DATA" / "Soil Tests"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Band column mapping (A00-A63 -> band_0-band_63)
BAND_COLS_OLD = [f"A{i:02d}" for i in range(64)]
BAND_COLS_NEW = [f"band_{i}" for i in range(64)]
BAND_RENAME = dict(zip(BAND_COLS_OLD, BAND_COLS_NEW))

def read_dbf(dbf_path):
    """Read a DBF file and return as list of dicts."""
    records = []
    with open(dbf_path, 'rb') as f:
        # Read header
        f.read(4)  # version + date
        numrec = struct.unpack('<I', f.read(4))[0]
        headerlen = struct.unpack('<H', f.read(2))[0]
        recordlen = struct.unpack('<H', f.read(2))[0]
        f.read(20)  # reserved

        # Read field descriptors
        fields = []
        field_sizes = []
        while True:
            field_data = f.read(32)
            if field_data[0] == 0x0D:
                break
            name = field_data[:11].split(b'\x00')[0].decode('ascii')
            size = field_data[16]
            fields.append(name)
            field_sizes.append(size)

        # Skip to first record
        f.seek(headerlen)

        # Read records
        for _ in range(numrec):
            rec = {}
            f.read(1)  # deletion flag
            for name, size in zip(fields, field_sizes):
                val = f.read(size).decode('latin-1').strip()
                try:
                    val = float(val) if val and '.' in val else (int(val) if val else None)
                except ValueError:
                    pass
                rec[name] = val
            records.append(rec)

    return pd.DataFrame(records)

def read_shapefile_coordinates(shp_path):
    """Read coordinates from SHP file (point geometry)."""
    coords = []
    with open(shp_path, 'rb') as f:
        # Skip header (100 bytes)
        f.seek(100)
        while True:
            try:
                # Record header
                rec_num = struct.unpack('>I', f.read(4))[0]
                content_len = struct.unpack('>I', f.read(4))[0]
                # Shape type
                shape_type = struct.unpack('<I', f.read(4))[0]
                if shape_type == 1:  # Point
                    x = struct.unpack('<d', f.read(8))[0]
                    y = struct.unpack('<d', f.read(8))[0]
                    coords.append((y, x))  # lat, lon
                else:
                    # Skip other geometries
                    f.read(content_len * 2 - 4)
            except struct.error:
                break
    return coords

def load_shapefile_data():
    """Load Ca, Mg, Na data from shapefiles."""
    all_data = []

    dbf_files = list(SHAPEFILE_DIR.glob("*.dbf"))
    for dbf_path in dbf_files:
        shp_path = dbf_path.with_suffix('.shp')

        if not shp_path.exists():
            continue

        # Read DBF attributes
        df = read_dbf(dbf_path)

        # Read coordinates
        coords = read_shapefile_coordinates(shp_path)

        if len(coords) == len(df):
            df['lat'] = [c[0] for c in coords]
            df['lon'] = [c[1] for c in coords]
            all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def process_year(year, shapefile_df):
    """Process and merge data for a specific year."""

    # File paths for this year
    ph_file = ALPHAEARTH_DIR / f"top10cm_ph_{year}_alphaearth_{year}.csv"
    cec_file = ALPHAEARTH_DIR / f"top10cm_cec_cmolkg_{year}_alphaearth_{year}.csv"
    esp_file = ALPHAEARTH_DIR / f"top10cm_esp_pct_{year}_alphaearth_{year}.csv"

    dfs = []

    # Load pH data (primary source for bands)
    if ph_file.exists():
        df_ph = pd.read_csv(ph_file)
        df_ph = df_ph.rename(columns=BAND_RENAME)
        df_ph = df_ph.rename(columns={'ph': 'ph'})
        dfs.append(('ph', df_ph))
        print(f"  Loaded pH: {len(df_ph)} records")

    # Load CEC data
    if cec_file.exists():
        df_cec = pd.read_csv(cec_file)
        df_cec = df_cec.rename(columns=BAND_RENAME)
        df_cec = df_cec.rename(columns={'cec_cmolkg': 'cec'})
        dfs.append(('cec', df_cec))
        print(f"  Loaded CEC: {len(df_cec)} records")

    # Load ESP data
    if esp_file.exists():
        df_esp = pd.read_csv(esp_file)
        df_esp = df_esp.rename(columns=BAND_RENAME)
        df_esp = df_esp.rename(columns={'esp_pct': 'esp'})
        dfs.append(('esp', df_esp))
        print(f"  Loaded ESP: {len(df_esp)} records")

    if not dfs:
        print(f"  No data files found for {year}")
        return None

    # Start with the first dataset
    _, merged = dfs[0]

    # Merge additional datasets
    for metric, df in dfs[1:]:
        # Keep only the metric column and merge keys
        cols_to_add = ['lat', 'lon', 'date', metric]
        df_subset = df[cols_to_add].copy()

        # Merge on lat, lon, date
        merged = pd.merge(
            merged,
            df_subset,
            on=['lat', 'lon', 'date'],
            how='outer',
            suffixes=('', '_dup')
        )

    # Handle duplicates - group by lat/lon and average numeric values
    numeric_cols = BAND_COLS_NEW + ['ph', 'cec', 'esp']
    existing_numeric = [c for c in numeric_cols if c in merged.columns]

    # Group by location and take mean for duplicates
    merged = merged.groupby(['lat', 'lon'], as_index=False).agg({
        **{c: 'first' for c in merged.columns if c not in existing_numeric + ['lat', 'lon']},
        **{c: 'mean' for c in existing_numeric if c in merged.columns}
    })

    # Add empty columns for missing metrics
    for col in ['ph', 'cec', 'esp', 'soc', 'ca', 'mg', 'na']:
        if col not in merged.columns:
            merged[col] = None

    # Try to match with shapefile data for Ca, Mg, Na
    if not shapefile_df.empty and 'lat' in shapefile_df.columns:
        # Round coordinates for matching
        merged['lat_round'] = merged['lat'].round(4)
        merged['lon_round'] = merged['lon'].round(4)

        sf_copy = shapefile_df.copy()
        sf_copy['lat_round'] = sf_copy['lat'].round(4)
        sf_copy['lon_round'] = sf_copy['lon'].round(4)

        # Match and update Ca, Mg, Na
        for idx, row in merged.iterrows():
            matches = sf_copy[
                (sf_copy['lat_round'] == row['lat_round']) &
                (sf_copy['lon_round'] == row['lon_round'])
            ]
            if len(matches) > 0:
                match = matches.iloc[0]
                if 'Ca' in match and pd.notna(match['Ca']):
                    merged.at[idx, 'ca'] = match['Ca']
                if 'Mg' in match and pd.notna(match['Mg']):
                    merged.at[idx, 'mg'] = match['Mg']
                if 'Na' in match and pd.notna(match['Na']):
                    merged.at[idx, 'na'] = match['Na']

        merged = merged.drop(columns=['lat_round', 'lon_round'], errors='ignore')

    # Generate ID
    merged['id'] = [f"{year}_{i:05d}" for i in range(len(merged))]

    # Ensure all band columns exist
    for col in BAND_COLS_NEW:
        if col not in merged.columns:
            merged[col] = None

    # Select and order final columns
    final_cols = ['id', 'lat', 'lon'] + BAND_COLS_NEW + ['ph', 'cec', 'esp', 'soc', 'ca', 'mg', 'na']

    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in merged.columns]
    merged = merged[final_cols]

    return merged

def main():
    print("=" * 60)
    print("Soil Data Extraction and Consolidation")
    print("=" * 60)

    # Load shapefile data (for Ca, Mg)
    print("\nLoading shapefile data (Ca, Mg, Na)...")
    shapefile_df = load_shapefile_data()
    print(f"Loaded {len(shapefile_df)} shapefile records")

    # Process each year
    years = range(2017, 2025)

    for year in years:
        print(f"\nProcessing year {year}...")
        df = process_year(year, shapefile_df)

        if df is not None and len(df) > 0:
            output_path = OUTPUT_DIR / f"soil_data_{year}.csv"
            df.to_csv(output_path, index=False)
            print(f"  Saved {len(df)} records to {output_path.name}")
        else:
            print(f"  No data for {year}")

    print("\n" + "=" * 60)
    print("Done! Output files in:", OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()
