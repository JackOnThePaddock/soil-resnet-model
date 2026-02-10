import csv, re, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path('espade_soil_data')
PROFILE_DIR = BASE_DIR / 'profile_data'
LAYER_DIR = BASE_DIR / 'layer_data'
OUT_DIR = BASE_DIR / 'ph_by_year'
OUT_DIR.mkdir(exist_ok=True)

YEARS = set(range(2017, 2025))
TOP_CM = 10.0

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def load_shared_strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    s_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    return [t.text if t.text is not None else '' for t in s_root.iter(NS + 't')]


def cell_value(cell, shared):
    t = cell.attrib.get('t')
    if t == 'inlineStr':
        is_node = cell.find(NS + 'is')
        if is_node is None:
            return ''
        return ''.join(t_el.text or '' for t_el in is_node.iter(NS + 't'))
    v = cell.find(NS + 'v')
    if v is None:
        return ''
    val = v.text or ''
    if t == 's':
        try:
            return shared[int(val)]
        except Exception:
            return ''
    return val


def col_letter(cell_ref):
    m = re.match(r'([A-Z]+)', cell_ref)
    return m.group(1) if m else ''


def to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def extract_year(date_str):
    if not date_str:
        return None
    # Look for 4-digit year
    matches = re.findall(r'(\d{4})', str(date_str))
    if not matches:
        return None
    # Use the last 4-digit token (often the year)
    for token in reversed(matches):
        year = int(token)
        if 1900 <= year <= 2100:
            return year
    return None


def get_header_maps(z):
    shared = load_shared_strings(z)
    with z.open('xl/worksheets/sheet1.xml') as sheet:
        context = ET.iterparse(sheet, events=('end',))
        for event, elem in context:
            if elem.tag != NS + 'row':
                continue
            headers_by_letter = {}
            for c in elem.findall(NS + 'c'):
                ref = c.attrib.get('r', '')
                letter = col_letter(ref)
                headers_by_letter[letter] = cell_value(c, shared)
            elem.clear()
            return headers_by_letter, {v: k for k, v in headers_by_letter.items() if v}
    return {}, {}


# 1) Build SoilProfileID -> year map using SoilProfileDate
profile_files = sorted(PROFILE_DIR.glob('profile_data_part_*.xlsx'))
if not profile_files:
    raise SystemExit('No profile_data_part_*.xlsx files found.')

# Map SoilProfileID -> (year, date_str)
date_by_profile = {}

for path in profile_files:
    with zipfile.ZipFile(path) as z:
        headers_by_letter, header_to_letter = get_header_maps(z)
        id_letter = header_to_letter.get('SoilProfileID')
        date_letter = header_to_letter.get('SoilProfileDate')
        if not id_letter or not date_letter:
            continue

        needed_letters = {id_letter, date_letter}
        shared = load_shared_strings(z)

        with z.open('xl/worksheets/sheet1.xml') as sheet:
            context = ET.iterparse(sheet, events=('end',))
            for event, elem in context:
                if elem.tag != NS + 'row':
                    continue
                # skip header row
                if elem.attrib.get('r') == '1':
                    elem.clear()
                    continue

                vals = {}
                for c in elem.findall(NS + 'c'):
                    ref = c.attrib.get('r', '')
                    letter = col_letter(ref)
                    if letter not in needed_letters:
                        continue
                    header = headers_by_letter.get(letter, '')
                    if not header:
                        continue
                    vals[header] = cell_value(c, shared)

                spid = vals.get('SoilProfileID')
                if spid in (None, ''):
                    elem.clear()
                    continue
                try:
                    spid_int = int(float(spid))
                except Exception:
                    elem.clear()
                    continue

                date_str = vals.get('SoilProfileDate')
                year = extract_year(date_str)
                if year in YEARS:
                    date_by_profile[spid_int] = (year, date_str)

                elem.clear()

print(f"Profiles with SoilProfileDate in 2017-2024: {len(date_by_profile)}")

# 2) Prepare output writers per year
writers = {}
files = {}

output_headers = [
    'SoilProfileID',
    'SoilProfileLayerID',
    'SoilProfileDate',
    'Year',
    'BoundUpper',
    'BoundLower',
    'ChemicalTestsFieldPH',
]

for year in sorted(YEARS):
    out_path = OUT_DIR / f"ph_top10cm_{year}.csv"
    f = out_path.open('w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(output_headers)
    files[year] = f
    writers[year] = w

# 3) Process layer data, filter top 10 cm and pH, join to year
layer_files = sorted(LAYER_DIR.glob('layer_data_part_*.xlsx'))
if not layer_files:
    raise SystemExit('No layer_data_part_*.xlsx files found.')

for path in layer_files:
    with zipfile.ZipFile(path) as z:
        headers_by_letter, header_to_letter = get_header_maps(z)
        id_letter = header_to_letter.get('SoilProfileID')
        layer_id_letter = header_to_letter.get('SoilProfileLayerID')
        upper_letter = header_to_letter.get('BoundUpper')
        lower_letter = header_to_letter.get('BoundLower')
        ph_letter = header_to_letter.get('ChemicalTestsFieldPH')

        if not (id_letter and upper_letter and lower_letter and ph_letter):
            continue

        needed_letters = {id_letter, upper_letter, lower_letter, ph_letter}
        if layer_id_letter:
            needed_letters.add(layer_id_letter)

        shared = load_shared_strings(z)

        with z.open('xl/worksheets/sheet1.xml') as sheet:
            context = ET.iterparse(sheet, events=('end',))
            for event, elem in context:
                if elem.tag != NS + 'row':
                    continue
                if elem.attrib.get('r') == '1':
                    elem.clear()
                    continue

                vals = {}
                for c in elem.findall(NS + 'c'):
                    ref = c.attrib.get('r', '')
                    letter = col_letter(ref)
                    if letter not in needed_letters:
                        continue
                    header = headers_by_letter.get(letter, '')
                    if not header:
                        continue
                    vals[header] = cell_value(c, shared)

                spid = vals.get('SoilProfileID')
                if spid in (None, ''):
                    elem.clear()
                    continue
                try:
                    spid_int = int(float(spid))
                except Exception:
                    elem.clear()
                    continue

                date_info = date_by_profile.get(spid_int)
                if date_info is None:
                    elem.clear()
                    continue
                year, date_str = date_info

                upper = to_float(vals.get('BoundUpper'))
                lower = to_float(vals.get('BoundLower'))
                if upper is None or lower is None:
                    elem.clear()
                    continue
                if not (0 <= upper < TOP_CM and 0 < lower <= TOP_CM and upper < lower):
                    elem.clear()
                    continue

                ph = vals.get('ChemicalTestsFieldPH')
                if ph in (None, ''):
                    elem.clear()
                    continue

                row = [
                    spid_int,
                    vals.get('SoilProfileLayerID', ''),
                    date_str or '',
                    year,
                    upper,
                    lower,
                    ph,
                ]

                # Write the actual date string if we want it; optional
                # Here we leave blank to reduce join cost; year is already set.

                writers[year].writerow(row)
                elem.clear()

# Close files
for f in files.values():
    f.close()

print('Done. Outputs in', OUT_DIR)
