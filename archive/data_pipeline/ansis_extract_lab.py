import csv
import json
import re
from pathlib import Path
import zipfile
from datetime import datetime, timezone

ZIP_PATH = Path(r"C:\Users\jackc\Downloads\adcc94b0-66f9-4a77-9281-dc9455d712df.zip")
OUT_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data\external_sources")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "ansis_soil_cores_last10yrs.csv"

CUTOFF = datetime(2017, 1, 1, tzinfo=timezone.utc)


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
    # take first numeric value
    for v in vals:
        result = v.get("result", {})
        val = result.get("value")
        if val is not None:
            return val, result.get("unit")
    return None, None


def mean_values_with_unit(layer, key):
    vals = layer.get(key, [])
    if not isinstance(vals, list) or len(vals) == 0:
        return None, None
    out = []
    unit = None
    for v in vals:
        result = v.get("result", {})
        val = result.get("value")
        if val is not None:
            out.append(val)
            unit = unit or result.get("unit")
    if not out:
        return None, None
    return sum(out) / len(out), unit


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
            if date is None or date < CUTOFF:
                continue
            for prof in sv.get("soilProfile", []):
                for layer in prof.get("soilLayer", []):
                    ph, ph_unit = mean_values_with_unit(layer, "ph")
                    cec, cec_unit = first_value_with_unit(layer, "cationExchangeCapacity")
                    esp, esp_unit = first_value_with_unit(layer, "exchangeableSodiumPercentage")
                    na, na_unit = first_value_with_unit(layer, "exchangeableSodium")
                    depth_upper_m, depth_upper_unit = depth_to_meters(layer, "depthUpper")
                    depth_lower_m, depth_lower_unit = depth_to_meters(layer, "depthLower")
                    if any(v is not None for v in [ph, cec, esp, na]):
                        rows.append({
                            "site_id": site_id,
                            "lat": lat,
                            "lon": lon,
                            "date": date.date().isoformat(),
                            "depth_upper_m": depth_upper_m,
                            "depth_lower_m": depth_lower_m,
                            "depth_upper_unit": depth_upper_unit,
                            "depth_lower_unit": depth_lower_unit,
                            "pH": ph,
                            "pH_unit": ph_unit,
                            "CEC": cec,
                            "CEC_unit": cec_unit,
                            "ESP": esp,
                            "ESP_unit": esp_unit,
                            "Na_cmol": na,
                            "Na_cmol_unit": na_unit,
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
        "pH",
        "pH_unit",
        "CEC",
        "CEC_unit",
        "ESP",
        "ESP_unit",
        "Na_cmol",
        "Na_cmol_unit",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
