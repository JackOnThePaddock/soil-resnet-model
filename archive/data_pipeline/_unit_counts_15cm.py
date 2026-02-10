import csv, os, collections
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output_15cm"
files = {"CEC":"CEC.csv","ESP":"ESP.csv","pH":"pH.csv"}

for label, fname in files.items():
    path = os.path.join(out_dir, fname)
    unit_counts = collections.Counter()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit_counts[row.get('unit') or ''] += 1
    print(label)
    for unit, count in unit_counts.most_common():
        print(f"  {unit!r}: {count}")
