import csv, os, collections
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output"
files = {"CEC":"CEC.csv","ESP":"ESP.csv","pH":"pH.csv"}

for label, fname in files.items():
    path = os.path.join(out_dir, fname)
    site_counts = collections.Counter()
    total_rows = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            site_id = row.get('site_id') or ''
            site_counts[site_id] += 1
    # duplicates beyond 1 per site
    dup_rows = sum(c - 1 for c in site_counts.values() if c > 1)
    print(f"{label}:")
    print(f"  total rows: {total_rows}")
    print(f"  unique sites: {len(site_counts)}")
    print(f"  duplicate rows beyond 1 per site: {dup_rows}")
    print(f"  % of rows that are duplicates: {dup_rows/total_rows*100:.1f}%")
