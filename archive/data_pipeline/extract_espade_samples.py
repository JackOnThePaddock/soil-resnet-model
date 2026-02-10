"""
Extract full ESPADE sample data including Ca, Mg, SOC from lab analysis
"""
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import re

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
BASE_DIR = Path(r"C:\Users\jackc\Downloads\Soil Data")
ESPADE_DIR = BASE_DIR / "espade_soil_data"
PROFILE_DIR = ESPADE_DIR / "profile_data"
SAMPLE_DIR = ESPADE_DIR / "sample_data"
OUTPUT_DIR = BASE_DIR / "output"

# Columns we need from sample_data
NEEDED_COLS = [
    'SoilProfileID', 'SoilProfileLayerID', 'BoundUpper', 'BoundLower', 'SampleDate',
    'N4A1', 'N4B1', 'N4B2',
    'N6B1', 'N6B2', 'N6B3', 'N6B4',
    'N15A1_CA', 'N15A1_MG', 'N15A1_NA', 'N15A1_ECEC',
    'N15B1_CA', 'N15B1_MG', 'N15B1_NA', 'N15B1_CEC',
    'N15C1_CA', 'N15C1_MG', 'N15C1_NA', 'N15C1_CEC',
    'N15D1_CA', 'N15D1_MG', 'N15D1_NA', 'N15D1_CEC',
]

def load_shared_strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    s_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    return [t.text if t.text is not None else '' for t in s_root.iter(NS + 't')]

def cell_value(cell, shared):
    t = cell.attrib.get('t')
    v = cell.find(NS + 'v')
    if v is None:
        return ''
    val = v.text or ''
    if t == 's':
        try:
            return shared[int(val)]
        except:
            return ''
    return val

def col_letter(ref):
    m = re.match(r'([A-Z]+)', ref)
    return m.group(1) if m else ''

def read_xlsx_data(xlsx_path, needed_cols):
    records = []
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            shared = load_shared_strings(z)
            with z.open('xl/worksheets/sheet1.xml') as sheet:
                context = ET.iterparse(sheet, events=('end',))
                headers_by_letter = {}
                for event, elem in context:
                    if elem.tag == NS + 'row':
                        for c in elem.findall(NS + 'c'):
                            ref = c.attrib.get('r', '')
                            letter = col_letter(ref)
                            headers_by_letter[letter] = cell_value(c, shared)
                        elem.clear()
                        break

            needed_letters = {k for k, v in headers_by_letter.items() if v in needed_cols}

            with z.open('xl/worksheets/sheet1.xml') as sheet:
                context = ET.iterparse(sheet, events=('end',))
                first = True
                for event, elem in context:
                    if elem.tag != NS + 'row':
                        continue
                    if first:
                        first = False
                        elem.clear()
                        continue
                    row = {}
                    for c in elem.findall(NS + 'c'):
                        ref = c.attrib.get('r', '')
                        letter = col_letter(ref)
                        if letter in needed_letters:
                            header = headers_by_letter.get(letter, '')
                            if header in needed_cols:
                                row[header] = cell_value(c, shared)
                    if row.get('SoilProfileID'):
                        records.append(row)
                    elem.clear()
    except Exception as e:
        print(f"Error reading {xlsx_path.name}: {e}")
    return records

def extract_year(date_str):
    if not date_str:
        return None
    matches = re.findall(r'(\d{4})', str(date_str))
    for m in reversed(matches):
        year = int(m)
        if 2000 <= year <= 2030:
            return year
    return None

def to_float(val):
    try:
        return float(val) if val not in (None, '', 'NA', 'na') else None
    except:
        return None

def main():
    print("=" * 70)
    print("EXTRACTING FULL ESPADE SAMPLE DATA")
    print("=" * 70)

    # Step 1: Load profile coordinates
    print("\nStep 1: Loading profile coordinates...")
    profile_coords = {}
    profile_dates = {}
    profile_files = sorted(PROFILE_DIR.glob("profile_data_part_*.xlsx"))

    for i, pf in enumerate(profile_files):
        rows = read_xlsx_data(pf, ['SoilProfileID', 'Latitude', 'Longitude', 'SoilProfileDate'])
        for row in rows:
            pid = row.get('SoilProfileID', '')
            lat = row.get('Latitude', '')
            lon = row.get('Longitude', '')
            date = row.get('SoilProfileDate', '')
            if pid and lat and lon:
                try:
                    pid_int = int(float(pid))
                    profile_coords[pid_int] = (float(lat), float(lon))
                    if date:
                        profile_dates[pid_int] = date
                except:
                    pass
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(profile_files)} profile files...")

    print(f"  Found {len(profile_coords)} profiles with coordinates")

    # Step 2: Extract sample data
    print("\nStep 2: Extracting sample data...")
    all_samples = []
    sample_files = sorted(SAMPLE_DIR.glob("sample_data_*.xlsx"))

    for i, sf in enumerate(sample_files):
        records = read_xlsx_data(sf, NEEDED_COLS)
        all_samples.extend(records)
        if (i+1) % 50 == 0 or i == len(sample_files)-1:
            print(f"  Processed {i+1}/{len(sample_files)} files ({len(all_samples)} records)")

    print(f"  Total sample records: {len(all_samples)}")

    # Step 3: Process and filter
    print("\nStep 3: Processing top 10cm samples...")

    processed = []
    for rec in all_samples:
        pid = rec.get('SoilProfileID')
        try:
            pid = int(float(pid))
        except:
            continue

        if pid not in profile_coords:
            continue

        lat, lon = profile_coords[pid]
        if not (-45 <= lat <= -10 and 112 <= lon <= 155):
            continue

        upper = to_float(rec.get('BoundUpper'))
        lower = to_float(rec.get('BoundLower'))
        if upper is None or lower is None:
            continue
        if not (upper == 0 and lower <= 0.15):
            continue

        date_str = rec.get('SampleDate') or profile_dates.get(pid, '')
        year = extract_year(date_str)
        if not year or year < 2017:
            continue

        ph = to_float(rec.get('N4A1')) or to_float(rec.get('N4B1')) or to_float(rec.get('N4B2'))
        soc = to_float(rec.get('N6B1')) or to_float(rec.get('N6B2')) or to_float(rec.get('N6B3')) or to_float(rec.get('N6B4'))
        ca = to_float(rec.get('N15A1_CA')) or to_float(rec.get('N15B1_CA')) or to_float(rec.get('N15C1_CA')) or to_float(rec.get('N15D1_CA'))
        mg = to_float(rec.get('N15A1_MG')) or to_float(rec.get('N15B1_MG')) or to_float(rec.get('N15C1_MG')) or to_float(rec.get('N15D1_MG'))
        na = to_float(rec.get('N15A1_NA')) or to_float(rec.get('N15B1_NA')) or to_float(rec.get('N15C1_NA')) or to_float(rec.get('N15D1_NA'))
        cec = to_float(rec.get('N15A1_ECEC')) or to_float(rec.get('N15B1_CEC')) or to_float(rec.get('N15C1_CEC')) or to_float(rec.get('N15D1_CEC'))

        if ph is None and soc is None and ca is None and cec is None:
            continue

        processed.append({
            'year': year, 'lat': lat, 'lon': lon,
            'ph': ph, 'cec': cec, 'esp': None, 'soc': soc, 'ca': ca, 'mg': mg, 'na': na,
        })

    print(f"  Valid records: {len(processed)}")

    # Step 4: Merge with existing data
    print("\nStep 4: Merging with existing data...")

    for year in range(2017, 2025):
        output_file = OUTPUT_DIR / f"soil_data_{year}.csv"
        existing_df = pd.read_csv(output_file)
        existing_count = len(existing_df)

        year_records = [r for r in processed if r['year'] == year]

        if year_records:
            new_df = pd.DataFrame(year_records)
            new_df['id'] = [f"ESPADE_LAB_{year}_{i}" for i in range(len(new_df))]
            new_df = new_df[['id', 'lat', 'lon', 'ph', 'cec', 'esp', 'soc', 'ca', 'mg', 'na']]

            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined['lat_r'] = combined['lat'].round(4)
            combined['lon_r'] = combined['lon'].round(4)
            combined = combined.drop_duplicates(subset=['lat_r', 'lon_r'], keep='first')
            combined = combined.drop(columns=['lat_r', 'lon_r'])

            combined.to_csv(output_file, index=False)
            added = len(combined) - existing_count
            print(f"  {year}: +{added} records (total: {len(combined)})")
        else:
            print(f"  {year}: No new records")

    print("\nDone!")

if __name__ == "__main__":
    main()
