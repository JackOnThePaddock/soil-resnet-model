import csv, os, collections, math
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output"
files = {"CEC":"CEC.csv","ESP":"ESP.csv","pH":"pH.csv"}

for label, fname in files.items():
    path = os.path.join(out_dir, fname)
    pair_counts = collections.Counter()
    missing = 0
    total = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            lat = row.get('lat')
            lon = row.get('lon')
            if lat in (None, '') or lon in (None, ''):
                missing += 1
                continue
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                missing += 1
                continue
            pair_counts[(lat_f, lon_f)] += 1

    dup_pairs = sum(1 for c in pair_counts.values() if c > 1)
    dup_rows = sum(c-1 for c in pair_counts.values() if c > 1)
    print(f"{label}:")
    print(f"  total rows: {total}")
    print(f"  rows with missing/invalid lat/lon: {missing}")
    print(f"  unique lat/lon pairs: {len(pair_counts)}")
    print(f"  lat/lon pairs with duplicates (>1 row): {dup_pairs}")
    print(f"  duplicate rows beyond one per lat/lon: {dup_rows}")
