import csv, os, collections
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output"
files = {"CEC":"CEC.csv","ESP":"ESP.csv","pH":"pH.csv"}

for label, fname in files.items():
    path = os.path.join(out_dir, fname)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    counts = collections.Counter()
    for row in rows:
        lat = row.get('lat')
        lon = row.get('lon')
        counts[(lat, lon)] += 1

    filtered = [r for r in rows if counts[(r.get('lat'), r.get('lon'))] == 1]

    out_path = os.path.join(out_dir, f"{label}_unique_latlon.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"{label}: kept {len(filtered)} of {len(rows)} rows -> {out_path}")
