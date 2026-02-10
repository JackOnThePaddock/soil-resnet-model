import csv, os, statistics, collections
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output"
files = {"CEC":"CEC.csv","ESP":"ESP.csv","pH":"pH.csv"}

for label, fname in files.items():
    path = os.path.join(out_dir, fname)
    values_by_site = collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            site_id = row.get('site_id') or ''
            try:
                val = float(row.get('value'))
            except Exception:
                continue
            values_by_site[site_id].append(val)

    # only sites with duplicates
    dup_sites = {k:v for k,v in values_by_site.items() if len(v) > 1}
    if not dup_sites:
        print(f"{label}: no duplicate sites")
        continue

    mins = []
    maxs = []
    ranges = []
    for vals in dup_sites.values():
        mn = min(vals)
        mx = max(vals)
        mins.append(mn)
        maxs.append(mx)
        ranges.append(mx - mn)

    overall_min = min(mins)
    overall_max = max(maxs)
    min_range = min(ranges)
    max_range = max(ranges)

    print(f"{label}:")
    print(f"  duplicate sites: {len(dup_sites)}")
    print(f"  values (min..max) across duplicate sites: {overall_min} .. {overall_max}")
    print(f"  per-site range (min..max): {min_range} .. {max_range}")
