import json, math, time, urllib.request, http.cookiejar
from pathlib import Path

# Downloads eSPADE soil layer data in 1000-profile chunks (public).
# Attempts sample-data export for CEC/ESP if the endpoint is accessible.
# Resume-safe: skips already-downloaded parts.

boundary = {"SWLatitude": -38.5, "SWLongitude": 140.0, "NELatitude": -27.0, "NELongitude": 154.0, "ColourSchemeID": ""}
layer_chunk_size = 1000
sample_chunk_size = 200
max_retries = 3
request_timeout = 60

download_layers = True
download_samples = True

base_dir = Path(__file__).resolve().parent
layer_dir = base_dir / 'layer_data'
sample_dir = base_dir / 'sample_data'
cookie_file = base_dir / 'espade_cookies.txt'
base_dir.mkdir(exist_ok=True)
layer_dir.mkdir(exist_ok=True)
sample_dir.mkdir(exist_ok=True)

ids_path = base_dir / 'profile_ids.json'
failed_path = base_dir / 'failed_parts.json'
failed_sample_path = base_dir / 'failed_sample_parts.json'

# Load or fetch profile IDs
if ids_path.exists():
    ids = json.loads(ids_path.read_text(encoding='utf-8'))
    print(f"Loaded {len(ids)} profile IDs")
else:
    url = 'https://espade.environment.nsw.gov.au/api/soilprofile/getwithinboundary'
    req = urllib.request.Request(
        url,
        data=json.dumps(boundary).encode('utf-8'),
        headers={'Content-Type': 'application/json; charset=utf-8'}
    )
    with urllib.request.urlopen(req, timeout=request_timeout) as f:
        profiles = json.load(f)
    ids = sorted({p['ProfileId'] for p in profiles if 'ProfileId' in p})
    ids_path.write_text(json.dumps(ids), encoding='utf-8')
    print(f"Fetched and saved {len(ids)} profile IDs")

post_url = 'https://espade.environment.nsw.gov.au/soilprofilecollection/GetSoilProfileDataIds'
layer_url = 'https://espade.environment.nsw.gov.au/soilprofilecollection/GetLayerDataByCode'
sample_url = 'https://espade.environment.nsw.gov.au/soilprofilecollection/GetSoilProfileSampleData'

failed_parts = []
failed_sample_parts = []

def build_opener():
    cj = http.cookiejar.CookieJar()
    # Load browser cookies if provided (for power-user sample export)
    if cookie_file.exists():
        try:
            cj = http.cookiejar.MozillaCookieJar(str(cookie_file))
            cj.load(ignore_discard=True, ignore_expires=True)
        except Exception as e:
            print(f"Warning: failed to load cookies from {cookie_file}: {e}")
            cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def post_ids(opener, chunk):
    req = urllib.request.Request(
        post_url,
        data=json.dumps({"ids": chunk}).encode('utf-8'),
        headers={'Content-Type': 'application/json; charset=utf-8'}
    )
    with opener.open(req, timeout=request_timeout) as f:
        resp = f.read().decode('utf-8', errors='ignore')
        if 'success' not in resp.lower():
            raise RuntimeError(f"Unexpected response: {resp[:200]}")

if download_layers:
    total_parts = math.ceil(len(ids) / layer_chunk_size)
    for part in range(total_parts):
        start = part * layer_chunk_size
        end = min(start + layer_chunk_size, len(ids))
        chunk = ids[start:end]
        part_label = f"{part+1:03d}"
        layer_file = layer_dir / f"layer_data_part_{part_label}.xlsx"
        if layer_file.exists():
            print(f"Layer part {part_label}: already downloaded")
            continue

        ok = False
        for attempt in range(1, max_retries + 1):
            try:
                opener = build_opener()
                post_ids(opener, chunk)
                with opener.open(layer_url, timeout=request_timeout) as f:
                    data = f.read()
                layer_file.write_bytes(data)
                print(f"Layer part {part_label}: saved")
                ok = True
                break
            except Exception as e:
                print(f"Layer part {part_label}: attempt {attempt} failed: {e}")
                time.sleep(2)

        if not ok:
            failed_parts.append(part_label)

        time.sleep(1)

if download_samples:
    total_parts = math.ceil(len(ids) / sample_chunk_size)
    for part in range(total_parts):
        start = part * sample_chunk_size
        end = min(start + sample_chunk_size, len(ids))
        chunk = ids[start:end]
        part_label = f"{part+1:03d}"
        sample_file = sample_dir / f"sample_data_{sample_chunk_size}_part_{part_label}.xlsx"
        if sample_file.exists():
            print(f"Sample part {part_label}: already downloaded")
            continue

        ok = False
        last_sample_error = None
        for attempt in range(1, max_retries + 1):
            try:
                opener = build_opener()
                post_ids(opener, chunk)
                with opener.open(sample_url, timeout=request_timeout) as f:
                    data = f.read()
                sample_file.write_bytes(data)
                print(f"Sample part {part_label}: saved")
                ok = True
                break
            except Exception as e:
                last_sample_error = e
                print(f"Sample part {part_label}: attempt {attempt} failed: {e}")
                time.sleep(2)

        if not ok:
            failed_sample_parts.append(part_label)
            print(f"Sample part {part_label}: download failed: {last_sample_error}")

        time.sleep(1)

if failed_parts:
    failed_path.write_text(json.dumps(failed_parts), encoding='utf-8')
    print(f"Failed parts: {failed_parts} (saved to {failed_path})")
else:
    if failed_path.exists():
        failed_path.unlink()
    print("All parts downloaded successfully")

if failed_sample_parts:
    failed_sample_path.write_text(json.dumps(failed_sample_parts), encoding='utf-8')
    print(f"Failed sample parts: {failed_sample_parts} (saved to {failed_sample_path})")
else:
    if failed_sample_path.exists():
        failed_sample_path.unlink()
