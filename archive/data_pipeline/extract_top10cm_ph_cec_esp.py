import csv, glob, re, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path('espade_soil_data/layer_data')
OUT_PATH = Path('espade_soil_data/top10cm_ph_cec_esp.csv')
TOP_CM = 10.0

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

BASE_COLS = [
    'SoilProfileID',
    'SoilProfileLayerID',
    'SurveyNumber',
    'SurveyTitle',
    'StationNumber',
    'SoilProfileNumber',
    'LayerNumber',
    'BoundUpper',
    'BoundLower',
]

PH_PAT = re.compile(r'\bph\b|ph$', re.IGNORECASE)
CEC_PAT = re.compile(r'\bcec\b|cation', re.IGNORECASE)
ESP_PAT = re.compile(r'\besp\b|exchangeablesodium', re.IGNORECASE)


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


xlsx_files = sorted(DATA_DIR.glob('layer_data_part_*.xlsx'))
if not xlsx_files:
    raise SystemExit('No layer_data_part_*.xlsx files found.')

# Determine header mapping from the first file
with zipfile.ZipFile(xlsx_files[0]) as z:
    shared = load_shared_strings(z)
    sheet = z.open('xl/worksheets/sheet1.xml')
    context = ET.iterparse(sheet, events=('end',))
    header_row = None
    for event, elem in context:
        if elem.tag == NS + 'row':
            header_row = elem
            break
    if header_row is None:
        raise SystemExit('Failed to read header row.')

    headers_by_letter = {}
    for c in header_row.findall(NS + 'c'):
        ref = c.attrib.get('r', '')
        letter = col_letter(ref)
        headers_by_letter[letter] = cell_value(c, shared)

# Find columns for pH/CEC/ESP
ph_cols = [h for h in headers_by_letter.values() if PH_PAT.search(h or '')]
cec_cols = [h for h in headers_by_letter.values() if CEC_PAT.search(h or '')]
esp_cols = [h for h in headers_by_letter.values() if ESP_PAT.search(h or '')]

# Build output columns
extra_cols = []
for col in ph_cols + cec_cols + esp_cols:
    if col and col not in extra_cols:
        extra_cols.append(col)

output_cols = BASE_COLS + extra_cols

# Map header -> letter for quick lookup
header_to_letter = {v: k for k, v in headers_by_letter.items() if v}

# Needed letters (for performance)
needed_letters = set()
for col in output_cols:
    letter = header_to_letter.get(col)
    if letter:
        needed_letters.add(letter)

# Ensure BoundUpper/BoundLower letters for filtering
bound_upper_letter = header_to_letter.get('BoundUpper')
bound_lower_letter = header_to_letter.get('BoundLower')
if bound_upper_letter:
    needed_letters.add(bound_upper_letter)
if bound_lower_letter:
    needed_letters.add(bound_lower_letter)

print('Detected pH columns:', ph_cols)
print('Detected CEC columns:', cec_cols)
print('Detected ESP columns:', esp_cols)
print('Writing to', OUT_PATH)

with OUT_PATH.open('w', newline='', encoding='utf-8') as fw:
    writer = csv.writer(fw)
    writer.writerow(output_cols)

    row_count = 0
    kept_count = 0

    for path in xlsx_files:
        with zipfile.ZipFile(path) as z:
            shared = load_shared_strings(z)
            with z.open('xl/worksheets/sheet1.xml') as sheet:
                context = ET.iterparse(sheet, events=('end',))
                for event, elem in context:
                    if elem.tag != NS + 'row':
                        continue

                    # Skip header row (row 1)
                    row_num = elem.attrib.get('r')
                    if row_num == '1':
                        elem.clear()
                        continue

                    row_count += 1
                    values_by_header = {}
                    # Only parse needed letters
                    for c in elem.findall(NS + 'c'):
                        ref = c.attrib.get('r', '')
                        letter = col_letter(ref)
                        if letter not in needed_letters:
                            continue
                        header = headers_by_letter.get(letter, '')
                        if not header:
                            continue
                        values_by_header[header] = cell_value(c, shared)

                    # Filter top 10 cm (fully within 0-10 cm)
                    upper = to_float(values_by_header.get('BoundUpper'))
                    lower = to_float(values_by_header.get('BoundLower'))
                    if upper is None or lower is None:
                        elem.clear()
                        continue
                    if not (0 <= upper < TOP_CM and 0 < lower <= TOP_CM):
                        elem.clear()
                        continue

                    # Keep rows that have at least one pH/CEC/ESP value
                    has_value = False
                    for col in extra_cols:
                        v = values_by_header.get(col)
                        if v not in (None, ''):
                            has_value = True
                            break
                    if not has_value:
                        elem.clear()
                        continue

                    row = [values_by_header.get(col, '') for col in output_cols]
                    writer.writerow(row)
                    kept_count += 1

                    elem.clear()

    print(f'Processed {row_count} rows; kept {kept_count} rows')
