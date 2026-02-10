import json
import re
import time
from pathlib import Path

import ee
import pandas as pd
import requests


BASE_DIR = Path(r"C:\Users\jackc\Documents\National Soil Data Standardised")
IN_DIR = BASE_DIR / "by_year_cleaned_top10cm_metrics"
OUT_DIR = BASE_DIR / "by_year_cleaned_top10cm_metrics_alphaearth"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_feature(row, prop_cols):
    lon = row["lon"]
    lat = row["lat"]
    props = {}
    for c in prop_cols:
        v = row.get(c)
        if pd.isna(v):
            continue
        props[c] = v
    return ee.Feature(ee.Geometry.Point([lon, lat]), props)


def parse_years(df, filename):
    if "year" in df.columns:
        years = sorted(df["year"].dropna().astype(int).unique().tolist())
        if years:
            return years
    match = re.search(r"(19|20)\\d{2}", filename)
    if match:
        return [int(match.group(0))]
    return []


def sample_year(df_year, prop_cols, year, out_csv):
    df_year = df_year[df_year["lat"].notna() & df_year["lon"].notna()].copy()
    if df_year.empty:
        return False

    features = [build_feature(row, prop_cols) for _, row in df_year.iterrows()]
    fc = ee.FeatureCollection(features)

    emb = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .filterBounds(fc)
    )
    if emb.size().getInfo() == 0:
        return False

    img = emb.mosaic()
    band_names = img.bandNames().getInfo()
    selectors = prop_cols + band_names

    sampled = img.sampleRegions(
        collection=fc,
        properties=prop_cols,
        scale=10,
        geometries=False,
        tileScale=2,
    )

    request = {"table": sampled, "format": "CSV", "selectors": ",".join(selectors)}
    for attempt in range(3):
        try:
            download_id = ee.data.getTableDownloadId(request)
            url = ee.data.makeTableDownloadUrl(download_id)
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            out_csv.write_bytes(resp.content)
            return band_names, len(df_year)
        except Exception as exc:
            if attempt == 2:
                print(f"Download failed for {out_csv}: {exc}")
                return False
            time.sleep(5)


def main():
    ee.Initialize()

    csv_files = sorted(IN_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {IN_DIR}")
        return

    for path in csv_files:
        df = pd.read_csv(path)
        if "lat" not in df.columns or "lon" not in df.columns:
            print(f"Skipping {path.name}: missing lat/lon")
            continue

        years = parse_years(df, path.name)
        if not years:
            print(f"Skipping {path.name}: could not determine year")
            continue

        prop_cols = df.columns.tolist()
        out_meta = []

        for year in years:
            df_year = df.copy()
            if "year" in df_year.columns:
                df_year = df_year[df_year["year"].astype(int) == int(year)].copy()
            if df_year.empty:
                continue

            out_csv = OUT_DIR / f"{path.stem}_alphaearth_{year}.csv"
            result = sample_year(df_year, prop_cols, int(year), out_csv)
            if not result:
                print(f"{path.name} year {year}: no image or failed")
                continue

            band_names, count = result
            out_meta.append(
                {
                    "input_file": path.name,
                    "year": int(year),
                    "count_points": int(count),
                    "band_names": band_names,
                    "output_csv": out_csv.name,
                }
            )
            print(f"{path.name} year {year}: {count} points -> {out_csv}")

        if out_meta:
            meta_path = OUT_DIR / f"{path.stem}_alphaearth_meta.json"
            meta_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
            print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
