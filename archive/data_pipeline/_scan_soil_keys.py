import json, os, re
root = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_extracted_1"
keys = set()
cec_keys = set()
esp_keys = set()
ph_keys = set()

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            if 'cation' in k.lower() and 'exchange' in k.lower():
                cec_keys.add(k)
            if 'exchangeable' in k.lower() and 'sodium' in k.lower() and 'percent' in k.lower():
                esp_keys.add(k)
            if k.lower() in ('ph', 'pH'):
                ph_keys.add(k)
            walk(v)
    elif isinstance(obj, list):
        for item in obj:
            walk(item)

for fname in os.listdir(root):
    path = os.path.join(root, fname)
    if not os.path.isfile(path):
        continue
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('Failed', path, e)
        continue
    walk(data)

print('CEC-related keys:')
for k in sorted(cec_keys):
    print(' ', k)
print('ESP keys:')
for k in sorted(esp_keys):
    print(' ', k)
print('pH keys:')
for k in sorted(ph_keys):
    print(' ', k)

extra = sorted([k for k in keys if re.search(r'cation|CEC|cec|exchangeableSodiumPercentage', k)])
print('Keys with cation/CEC/ESP:')
for k in extra:
    print(' ', k)
