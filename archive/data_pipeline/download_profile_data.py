import json, math, time, urllib.request, http.cookiejar
from pathlib import Path

# Downloads eSPADE soil profile data in 1000-profile chunks.
# Resume-safe: skips already-downloaded parts.

chunk_size = 1000
max_retries = 3
request_timeout = 60

base_dir = Path('espade_soil_data')
profile_dir = base_dir / 'profile_data'
profile_dir.mkdir(exist_ok=True)

ids_path = base_dir / 'profile_ids.json'
failed_path = base_dir / 'failed_profile_parts.json'

if not ids_path.exists():
    raise SystemExit('profile_ids.json not found. Run layer download first.')

ids = json.loads(ids_path.read_text(encoding='utf-8'))
print(f"Loaded {len(ids)} profile IDs")

post_url = 'https://espade.environment.nsw.gov.au/soilprofilecollection/GetSoilProfileDataIds'
profile_url = 'https://espade.environment.nsw.gov.au/soilprofilecollection/GetSoilProfileDataByCode'

total_parts = math.ceil(len(ids) / chunk_size)
failed_parts = []

for part in range(total_parts):
    start = part * chunk_size
    end = min(start + chunk_size, len(ids))
    chunk = ids[start:end]
    part_label = f"{part+1:03d}"
    profile_file = profile_dir / f"profile_data_part_{part_label}.xlsx"

    if profile_file.exists():
        print(f"Part {part_label}: already downloaded")
        continue

    ok = False
    for attempt in range(1, max_retries + 1):
        try:
            cj = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

            req = urllib.request.Request(
                post_url,
                data=json.dumps({"ids": chunk}).encode('utf-8'),
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )
            with opener.open(req, timeout=request_timeout) as f:
                resp = f.read().decode('utf-8', errors='ignore')
                if 'success' not in resp.lower():
                    raise RuntimeError(f"Unexpected response: {resp[:200]}")

            with opener.open(profile_url, timeout=request_timeout) as f:
                data = f.read()
            profile_file.write_bytes(data)

            print(f"Part {part_label}: saved ({profile_file.stat().st_size} bytes profile)")
            ok = True
            break
        except Exception as e:
            print(f"Part {part_label}: attempt {attempt} failed: {e}")
            time.sleep(2)

    if not ok:
        failed_parts.append(part_label)

    time.sleep(1)

if failed_parts:
    failed_path.write_text(json.dumps(failed_parts), encoding='utf-8')
    print(f"Failed parts: {failed_parts} (saved to {failed_path})")
else:
    if failed_path.exists():
        failed_path.unlink()
    print("All parts downloaded successfully")
