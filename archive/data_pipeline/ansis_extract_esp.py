import csv
import json
import re
from pathlib import Path
import zipfile
from datetime import datetime, timezone

ZIP_PATH = Path(r"C:\Users\jackc\Downloads\c73e85c5-73c5-435b-99d4-c7d02015db4d.zip")
OUT_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data\external_sources")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "ansis_exchangeable_sodium_pct_raw.csv"


def parse_point_wkt(wkt):
    if not wkt:
        return None, None
    m = re.search(r"POINT\(([^ ]+) ([^\)]+)\)", wkt)
    if not m:
        return None, None
    lon, lat = float(m.group(1)), float(m.group(2))
    return lat, lon


def parse_date(dt_str):
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def first_value_with_unit(layer, key):
    vals = layer.get(key, [])
    if not isinstance(vals, list) or len(vals) == 0:
        return None, None
    for v in vals:
        result = v.get("result", {})
        val = result.get("value")
        if val is not None:
            return val, result.get("unit")
    return None, None


def depth_to_meters(layer, key):
    d = layer.get(key)
    if not isinstance(d, dict):
        return None, None
    result = d.get("result", {})
    val = result.get("value")
    unit = result.get("unit")
    if val is None:
        return None, unit
    if unit in ("unit:M", "unit:METRE", "unit:meter", "unit:METERS", "unit:metre"):
        return val, unit
    if unit in ("unit:CM", "unit:CENTIMETRE", "unit:centimetre"):
        return val / 100.0, unit
    if unit in ("unit:MM", "unit:MILLIMETRE", "unit:millimetre"):
        return val / 1000.0, unit
    return val, unit


def extract_from_doc(doc):
    rows = []
    for site in doc.get("data", []):
        if site.get("type") != "ansis:SoilSite":
            continue
        lat, lon = parse_point_wkt(site.get("geometry", {}).get("result"))
        site_id = site.get("id") or site.get("scopedIdentifier", {}).get("result")
        for sv in site.get("siteVisit", []):
            date = parse_date(sv.get("startedAtTime") or sv.get("endedAtTime"))
            for prof in sv.get("soilProfile", []):
                for layer in prof.get("soilLayer", []):
                    esp, esp_unit = first_value_with_unit(layer, "exchangeableSodiumPercentage")
                    if esp is None:
                        continue
                    depth_upper_m, depth_upper_unit = depth_to_meters(layer, "depthUpper")
                    depth_lower_m, depth_lower_unit = depth_to_meters(layer, "depthLower")
                    rows.append({
                        "site_id": site_id,
                        "lat": lat,
                        "lon": lon,
                        "date": date.date().isoformat() if date else None,
                        "depth_upper_m": depth_upper_m,
                        "depth_lower_m": depth_lower_m,
                        "depth_upper_unit": depth_upper_unit,
                        "depth_lower_unit": depth_lower_unit,
                        "esp_pct": esp,
                        "esp_pct_unit": esp_unit,
                    })
    return rows


def main():
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as zp:
        for name in zp.namelist():
            doc = json.loads(zp.read(name))
            rows.extend(extract_from_doc(doc))

    fieldnames = [
        "site_id",
        "lat",
        "lon",
        "date",
        "depth_upper_m",
        "depth_lower_m",
        "depth_upper_unit",
        "depth_lower_unit",
        "esp_pct",
        "esp_pct_unit",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
