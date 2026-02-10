"""
Fast targeted ESPADE extraction - only processes files with 2017+ data
"""
import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(r"C:\Users\jackc\Downloads\Soil Data")
ESPADE_DIR = BASE_DIR / "espade_soil_data"
PROFILE_DIR = ESPADE_DIR / "profile_data"
SAMPLE_DIR = ESPADE_DIR / "sample_data"
OUTPUT_DIR = BASE_DIR / "output"

PROFILE_COLS = ['SoilProfileID', 'Latitude', 'Longitude', 'SoilProfileDate']
SAMPLE_COLS = [
    'SoilProfileID', 'BoundUpper', 'BoundLower', 'SampleDate',
    'N4A1', 'N4B1',  # Lab pH
    'N15A1_CA', 'N15A1_MG', 'N15A1_NA', 'N15A1_ECEC',
    'N15B1_CA', 'N15B1_MG', 'N15B1_NA', 'N15B1_CEC',
    'N15C1_CA', 'N15C1_MG', 'N15C1_NA', 'N15C1_CEC',
]

MAX_DEPTH_CM = 15
MAX_PH = 8.5
MAX_ESP = 25.0

def log(msg):
    print(msg, flush=True)

def extract_year(date_str):
    if pd.isna(date_str) or not date_str:
        return None
    matches = re.findall(r'(\d{4})', str(date_str))
    for m in reversed(matches):
        year = int(m)
        if 2000 <= year <= 2030:
            return year
    return None

def safe_float(val):
    try:
        if pd.isna(val) or val in ('', 'NA', 'na'):
            return None
        return float(val)
    except:
        return None

def first_valid(*values):
    for v in values:
        f = safe_float(v)
        if f is not None:
            return f
    return None

log("=" * 60)
log("ESPADE TARGETED EXTRACTION")
log(f"Filters: depth <= {MAX_DEPTH_CM}cm, pH <= {MAX_PH}, ESP <= {MAX_ESP}%")
log("=" * 60)

# Step 1: Load profile coordinates and dates
log("\n[1/4] Loading profile data...")
profile_coords = {}
profile_dates = {}
profile_years = {}

profile_files = sorted(PROFILE_DIR.glob("profile_data_part_*.xlsx"))
for i, pf in enumerate(profile_files):
    try:
        df = pd.read_excel(pf, usecols=lambda x: x in PROFILE_COLS, engine='openpyxl')
        for _, row in df.iterrows():
            pid = row.get('SoilProfileID')
            lat = row.get('Latitude')
            lon = row.get('Longitude')
            if pd.notna(pid) and pd.notna(lat) and pd.notna(lon):
                try:
                    pid_int = int(float(pid))
                    profile_coords[pid_int] = (float(lat), float(lon))
                    date = row.get('SoilProfileDate')
                    if pd.notna(date):
                        profile_dates[pid_int] = str(date)
                        year = extract_year(str(date))
                        if year:
                            profile_years[pid_int] = year
                except:
                    pass
    except Exception as e:
        log(f"  Error {pf.name}: {e}")

    if (i+1) % 10 == 0:
        log(f"  {i+1}/{len(profile_files)} files")

log(f"  Total: {len(profile_coords)} profiles")
profiles_2017_plus = {pid for pid, year in profile_years.items() if year >= 2017}
log(f"  Profiles with 2017+ dates: {len(profiles_2017_plus)}")

# Step 2: Pre-screen sample files for 2017+ data
log("\n[2/4] Pre-screening sample files for 2017+ data...")
files_with_recent_data = []
sample_files = sorted(SAMPLE_DIR.glob("sample_data*.xlsx"))

for i, sf in enumerate(sample_files):
    try:
        # Just read SampleDate column to check for recent data
        df = pd.read_excel(sf, usecols=['SampleDate', 'SoilProfileID'], nrows=1000, engine='openpyxl')
        has_recent = False

        # Check sample dates
        for date in df['SampleDate'].dropna().head(100):
            year = extract_year(str(date))
            if year and year >= 2017:
                has_recent = True
                break

        # Also check if any profiles are from 2017+
        if not has_recent:
            for pid in df['SoilProfileID'].dropna().head(100):
                try:
                    pid_int = int(float(pid))
                    if pid_int in profiles_2017_plus:
                        has_recent = True
                        break
                except:
                    pass

        if has_recent:
            files_with_recent_data.append(sf)

    except Exception as e:
        pass  # Skip files that can't be read

    if (i+1) % 50 == 0:
        log(f"  Scanned {i+1}/{len(sample_files)} files, {len(files_with_recent_data)} have 2017+ data")

log(f"  Files with 2017+ data: {len(files_with_recent_data)} / {len(sample_files)}")

if not files_with_recent_data:
    log("\n  No files found with 2017+ data in ESPADE sample_data!")
    log("  The ESPADE dataset is mostly historical (pre-2017).")
    exit()

# Step 3: Process only files with recent data
log(f"\n[3/4] Processing {len(files_with_recent_data)} files with 2017+ data...")
all_records = []
skipped_depth = 0
skipped_ph = 0
skipped_esp = 0
skipped_no_data = 0

for i, sf in enumerate(files_with_recent_data):
    try:
        df = pd.read_excel(sf, engine='openpyxl')
        available_cols = [c for c in SAMPLE_COLS if c in df.columns]
        if not available_cols:
            continue
        df = df[available_cols]

        for _, row in df.iterrows():
            pid = row.get('SoilProfileID')
            try:
                pid = int(float(pid))
            except:
                continue

            if pid not in profile_coords:
                continue

            lat, lon = profile_coords[pid]
            if not (-45 <= lat <= -10 and 112 <= lon <= 155):
                continue

            # Depth filter
            upper = safe_float(row.get('BoundUpper'))
            lower = safe_float(row.get('BoundLower'))
            if upper is None or lower is None:
                continue
            lower_cm = lower * 100 if lower < 1 else lower
            if upper != 0 or lower_cm > MAX_DEPTH_CM:
                skipped_depth += 1
                continue

            # Year filter
            date_str = row.get('SampleDate') if pd.notna(row.get('SampleDate')) else profile_dates.get(pid)
            year = extract_year(date_str)
            if not year or year < 2017:
                continue

            # Get lab values
            ph = first_valid(row.get('N4A1'), row.get('N4B1'))
            ca = first_valid(row.get('N15A1_CA'), row.get('N15B1_CA'), row.get('N15C1_CA'))
            mg = first_valid(row.get('N15A1_MG'), row.get('N15B1_MG'), row.get('N15C1_MG'))
            na = first_valid(row.get('N15A1_NA'), row.get('N15B1_NA'), row.get('N15C1_NA'))
            cec = first_valid(row.get('N15A1_ECEC'), row.get('N15B1_CEC'), row.get('N15C1_CEC'))

            if ph is None and ca is None and cec is None:
                skipped_no_data += 1
                continue

            if ph is not None and ph > MAX_PH:
                skipped_ph += 1
                continue

            esp = None
            if na is not None and cec is not None and cec > 0:
                esp = (na / cec) * 100

            if esp is not None and esp > MAX_ESP:
                skipped_esp += 1
                continue

            all_records.append({
                'year': year, 'lat': lat, 'lon': lon,
                'ph': ph, 'cec': cec, 'esp': esp, 'soc': None,
                'ca': ca, 'mg': mg, 'na': na
            })
    except Exception as e:
        log(f"  Error {sf.name}: {e}")

    if (i+1) % 5 == 0 or i == len(files_with_recent_data)-1:
        log(f"  {i+1}/{len(files_with_recent_data)} files, {len(all_records)} valid records")

log(f"\n  Valid: {len(all_records)}")
log(f"  Skipped - depth > {MAX_DEPTH_CM}cm: {skipped_depth}")
log(f"  Skipped - pH > {MAX_PH}: {skipped_ph}")
log(f"  Skipped - ESP > {MAX_ESP}%: {skipped_esp}")
log(f"  Skipped - no lab data: {skipped_no_data}")

# Step 4: Merge with existing
log("\n[4/4] Merging with existing data...")

from collections import Counter
year_counts = Counter(r['year'] for r in all_records)
log("Records by year:")
for y in sorted(year_counts.keys()):
    log(f"  {y}: {year_counts[y]}")

for year in range(2017, 2025):
    output_file = OUTPUT_DIR / f"soil_data_{year}.csv"
    existing_df = pd.read_csv(output_file)
    existing_count = len(existing_df)

    year_records = [r for r in all_records if r['year'] == year]

    if year_records:
        new_df = pd.DataFrame(year_records)
        new_df['id'] = [f"ESPADE_LAB_{year}_{i}" for i in range(len(new_df))]
        new_df = new_df[['id', 'lat', 'lon', 'ph', 'cec', 'esp', 'soc', 'ca', 'mg', 'na']]

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined['lat_r'] = combined['lat'].round(4)
        combined['lon_r'] = combined['lon'].round(4)
        combined = combined.drop_duplicates(subset=['lat_r', 'lon_r'], keep='first')
        combined = combined.drop(columns=['lat_r', 'lon_r'])

        # Apply filters to ALL data
        if 'ph' in combined.columns:
            combined = combined[(combined['ph'].isna()) | (combined['ph'] <= MAX_PH)]
        if 'esp' in combined.columns:
            combined = combined[(combined['esp'].isna()) | (combined['esp'] <= MAX_ESP)]

        combined.to_csv(output_file, index=False)
        final_count = len(combined)
        log(f"  {year}: {final_count} total records")
    else:
        log(f"  {year}: no new records")

log("\nDone!")
