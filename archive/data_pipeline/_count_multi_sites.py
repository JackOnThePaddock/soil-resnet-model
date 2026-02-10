import csv, os, collections
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output"
files = {"CEC":"CEC.csv","ESP":"ESP.csv","pH":"pH.csv"}

for label, fname in files.items():
    path = os.path.join(out_dir, fname)
    site_counts = collections.Counter()
    site_year_counts = collections.Counter()
    total_rows = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            site_id = row.get('site_id') or ''
            year = row.get('year') or ''
            site_counts[site_id] += 1
            site_year_counts[(site_id, year)] += 1
    multi_sites = sum(1 for c in site_counts.values() if c > 1)
    multi_site_years = sum(1 for c in site_year_counts.values() if c > 1)
    print(f"{label}: total rows={total_rows}")
    print(f"  sites with >1 value (any year): {multi_sites} of {len(site_counts)}")
    print(f"  site-years with >1 value: {multi_site_years} of {len(site_year_counts)}")
