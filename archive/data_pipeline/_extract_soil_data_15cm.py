import json, os, re, csv

root = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_extracted_1"
out_dir = r"C:\Users\jackc\Downloads\d2b08e8a-87e8-421b-9153-503208cbbfbf_output_15cm"
os.makedirs(out_dir, exist_ok=True)

CEC_KEYS = ["cationExchangeCapacity", "effectiveCationExchangeCapacity"]
ESP_KEY = "exchangeableSodiumPercentage"
PH_KEY = "ph"

columns = [
    "year",
    "date",
    "site_id",
    "site_identifier",
    "site_identifier_authority",
    "project_id",
    "lat",
    "lon",
    "depth_upper_m",
    "depth_lower_m",
    "layer_type",
    "property_type",
    "value",
    "value_raw",
    "unit",
    "source_file",
]

cec_rows = []
esp_rows = []
ph_rows = []

point_re = re.compile(r"POINT\(([-0-9.]+) ([-0-9.]+)\)")

def parse_point(s: str):
    if not s:
        return None, None
    m = point_re.search(s)
    if not m:
        return None, None
    lon, lat = m.group(1), m.group(2)
    try:
        return float(lat), float(lon)
    except Exception:
        return None, None


def first_year(date_str: str):
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except Exception:
        return None


def safe_get_depth(layer: dict, key: str):
    v = layer.get(key)
    if not isinstance(v, dict):
        return None
    res = v.get("result")
    if not isinstance(res, dict):
        return None
    val = res.get("value")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def normalize_scoped_identifier(si):
    if not si:
        return None, None
    values = []
    authorities = []
    if isinstance(si, dict):
        si = [si]
    if isinstance(si, list):
        for item in si:
            if not isinstance(item, dict):
                continue
            if item.get("value") is not None:
                values.append(str(item.get("value")))
            if item.get("authority") is not None:
                authorities.append(str(item.get("authority")))
    return (";".join(values) if values else None, ";".join(authorities) if authorities else None)


def normalize_project(proj):
    if proj is None:
        return None
    if isinstance(proj, list):
        return ";".join(str(x) for x in proj if x is not None)
    return str(proj)


def add_row(target_list, *, year, date_str, site_id, site_identifier, site_identifier_authority,
            project_id, lat, lon, depth_upper_m, depth_lower_m, layer_type, property_type,
            value, value_raw, unit, source_file):
    target_list.append({
        "year": year,
        "date": date_str,
        "site_id": site_id,
        "site_identifier": site_identifier,
        "site_identifier_authority": site_identifier_authority,
        "project_id": project_id,
        "lat": lat,
        "lon": lon,
        "depth_upper_m": depth_upper_m,
        "depth_lower_m": depth_lower_m,
        "layer_type": layer_type,
        "property_type": property_type,
        "value": value,
        "value_raw": value_raw,
        "unit": unit,
        "source_file": source_file,
    })

for fname in os.listdir(root):
    path = os.path.join(root, fname)
    if not os.path.isfile(path):
        continue
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        continue

    data = payload.get("data", [])
    if not isinstance(data, list):
        continue

    for site in data:
        if not isinstance(site, dict):
            continue
        site_id = site.get("id")
        lat, lon = parse_point((site.get("geometry") or {}).get("result"))
        site_identifier, site_identifier_authority = normalize_scoped_identifier(site.get("scopedIdentifier"))

        site_visits = site.get("siteVisit", [])
        if isinstance(site_visits, dict):
            site_visits = [site_visits]
        if not isinstance(site_visits, list):
            continue

        for visit in site_visits:
            if not isinstance(visit, dict):
                continue
            date_str = visit.get("endedAtTime") or visit.get("startedAtTime")
            year = first_year(date_str)
            if year is None or year < 2017:
                continue

            project_id = normalize_project(visit.get("relatedProject") or site.get("relatedProject"))

            profiles = visit.get("soilProfile", [])
            if isinstance(profiles, dict):
                profiles = [profiles]
            if not isinstance(profiles, list):
                continue

            for profile in profiles:
                if not isinstance(profile, dict):
                    continue
                layers = profile.get("soilLayer", [])
                if isinstance(layers, dict):
                    layers = [layers]
                if not isinstance(layers, list):
                    continue

                for layer in layers:
                    if not isinstance(layer, dict):
                        continue
                    depth_upper_m = safe_get_depth(layer, "depthUpper")
                    depth_lower_m = safe_get_depth(layer, "depthLower")
                    if depth_upper_m is None or depth_lower_m is None:
                        continue
                    # top 15 cm overlap: include layers starting above 0.15 m
                    if not (depth_upper_m < 0.15 and depth_lower_m > 0):
                        continue

                    layer_type = layer.get("type")

                    for cec_key in CEC_KEYS:
                        if cec_key in layer and isinstance(layer[cec_key], list):
                            for entry in layer[cec_key]:
                                if not isinstance(entry, dict):
                                    continue
                                res = entry.get("result") or {}
                                value_raw = res.get("value")
                                unit = res.get("unit")
                                if value_raw is None:
                                    continue
                                try:
                                    value_raw_f = float(value_raw)
                                except Exception:
                                    continue
                                add_row(
                                    cec_rows,
                                    year=year,
                                    date_str=date_str,
                                    site_id=site_id,
                                    site_identifier=site_identifier,
                                    site_identifier_authority=site_identifier_authority,
                                    project_id=project_id,
                                    lat=lat,
                                    lon=lon,
                                    depth_upper_m=depth_upper_m,
                                    depth_lower_m=depth_lower_m,
                                    layer_type=layer_type,
                                    property_type=cec_key,
                                    value=value_raw_f,
                                    value_raw=value_raw_f,
                                    unit=unit,
                                    source_file=fname,
                                )

                    if ESP_KEY in layer and isinstance(layer[ESP_KEY], list):
                        for entry in layer[ESP_KEY]:
                            if not isinstance(entry, dict):
                                continue
                            res = entry.get("result") or {}
                            value_raw = res.get("value")
                            unit = res.get("unit")
                            if value_raw is None:
                                continue
                            try:
                                value_raw_f = float(value_raw)
                            except Exception:
                                continue
                            value_capped = min(value_raw_f, 25.0)
                            add_row(
                                esp_rows,
                                year=year,
                                date_str=date_str,
                                site_id=site_id,
                                site_identifier=site_identifier,
                                site_identifier_authority=site_identifier_authority,
                                project_id=project_id,
                                lat=lat,
                                lon=lon,
                                depth_upper_m=depth_upper_m,
                                depth_lower_m=depth_lower_m,
                                layer_type=layer_type,
                                property_type=ESP_KEY,
                                value=value_capped,
                                value_raw=value_raw_f,
                                unit=unit,
                                source_file=fname,
                            )

                    if PH_KEY in layer and isinstance(layer[PH_KEY], list):
                        for entry in layer[PH_KEY]:
                            if not isinstance(entry, dict):
                                continue
                            res = entry.get("result") or {}
                            value_raw = res.get("value")
                            unit = res.get("unit")
                            if value_raw is None:
                                continue
                            try:
                                value_raw_f = float(value_raw)
                            except Exception:
                                continue
                            value_capped = min(value_raw_f, 8.5)
                            add_row(
                                ph_rows,
                                year=year,
                                date_str=date_str,
                                site_id=site_id,
                                site_identifier=site_identifier,
                                site_identifier_authority=site_identifier_authority,
                                project_id=project_id,
                                lat=lat,
                                lon=lon,
                                depth_upper_m=depth_upper_m,
                                depth_lower_m=depth_lower_m,
                                layer_type=layer_type,
                                property_type=PH_KEY,
                                value=value_capped,
                                value_raw=value_raw_f,
                                unit=unit,
                                source_file=fname,
                            )


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

write_csv(os.path.join(out_dir, "CEC.csv"), cec_rows)
write_csv(os.path.join(out_dir, "ESP.csv"), esp_rows)
write_csv(os.path.join(out_dir, "pH.csv"), ph_rows)

print("Rows written:")
print("  CEC:", len(cec_rows))
print("  ESP:", len(esp_rows))
print("  pH:", len(ph_rows))
print("Output dir:", out_dir)
