import zipfile, xml.etree.ElementTree as ET, re, time
from pathlib import Path

DATA_DIR = Path('C:/Users/jackc/Downloads/espade_soil_data/sample_data')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

patterns = {
    'ph': re.compile(r'\bph\b|_ph|ph_', re.IGNORECASE),
    'esp': re.compile(r'\besp\b|_esp|esp_|exchangeable\s*sodium|sodium\s*percent', re.IGNORECASE),
    'cec': re.compile(r'\bcec\b|ecec', re.IGNORECASE),
    'date': re.compile(r'date', re.IGNORECASE),
    'depth': re.compile(r'bound|depth|upper|lower', re.IGNORECASE),
}

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
        except Exception:
            return ''
    return val

def get_headers(path):
    with zipfile.ZipFile(path) as z:
        shared = load_shared_strings(z)
        sheet_xml = z.read('xl/worksheets/sheet1.xml')
        s_root = ET.fromstring(sheet_xml)
        header_row = s_root.find('.//' + NS + 'row')
        return [cell_value(c, shared) for c in header_row.findall(NS + 'c')]

files = sorted(DATA_DIR.glob('sample_data_200_part_*.xlsx'))
if not files:
    files = sorted(DATA_DIR.glob('sample_data_part_*.xlsx'))

total = len(files)
print('files', total)

all_headers = set()
match = {k: set() for k in patterns}

start = time.time()
for i, path in enumerate(files, 1):
    headers = get_headers(path)
    for h in headers:
        all_headers.add(h)
        for k, pat in patterns.items():
            if pat.search(h or ''):
                match[k].add(h)

    if i % 10 == 0 or i == total:
        elapsed = time.time() - start
        pct = (i / total) * 100 if total else 100
        print(f"scanned {i}/{total} ({pct:.1f}%) in {elapsed:.1f}s")

for k in match:
    print('\n', k, 'matches', len(match[k]))
    for h in sorted(match[k])[:200]:
        print(' ', h)

print('\nTotal headers', len(all_headers))
