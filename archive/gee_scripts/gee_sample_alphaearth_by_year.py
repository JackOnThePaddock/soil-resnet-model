import ee
import pandas as pd
from pathlib import Path
import requests
import json
import time

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
OUT_DIR = BASE_DIR / "external_sources" / "by_year"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "all": {
        "input": BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm.csv",
        "props": ["site_id", "paddock", "sample_id", "lat", "lon", "date", "depth_upper_m", "depth_lower_m", "ph", "cec_cmolkg", "esp_pct", "na_cmolkg"],
    },
    "na": {
        "input": BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_top10cm.csv",
        "props": ["site_id", "lat", "lon", "date", "depth_upper_m", "depth_lower_m", "na_cmolkg"],
    },
    "esp": {
        "input": BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_pct_top10cm.csv",
        "props": ["site_id", "lat", "lon", "date", "depth_upper_m", "depth_lower_m", "esp_pct"],
    },
}


def build_feature(row, prop_cols, year):
    lon = row["lon"]
    lat = row["lat"]
    props = {"year": year}
    for c in prop_cols:
        if c not in row:
            continue
        v = row.get(c)
        if pd.isna(v):
            continue
        props[c] = v
    return ee.Feature(ee.Geometry.Point([lon, lat]), props)


def sample_year(df_year, prop_cols, year, out_csv):
    df_year = df_year[df_year["lat"].notna() & df_year["lon"].notna()].copy()
    if df_year.empty:
        return False

    features = [build_feature(row, prop_cols, year) for _, row in df_year.iterrows()]
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
    selectors = ["year"] + [c for c in prop_cols if c in df_year.columns] + band_names

    sampled = img.sampleRegions(
        collection=fc,
        properties=[c for c in prop_cols if c in df_year.columns] + ["year"],
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
            return True
        except Exception as exc:
            if attempt == 2:
                print(f"Download failed for {year}: {exc}")
                return False
            time.sleep(5)


def run_dataset(name, cfg):
    df = pd.read_csv(cfg["input"])
    if "date" not in df.columns:
        return None
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_parsed"].notna()].copy()
    df["year"] = df["date_parsed"].dt.year
    df = df[df["year"] >= 2017].copy()

    years = sorted(df["year"].unique().tolist())
    out_files = []
    for yr in years:
        df_year = df[df["year"] == yr].copy()
        if df_year.empty:
            continue
        out_csv = OUT_DIR / f"{name}_alphaearth_{yr}.csv"
        ok = sample_year(df_year, cfg["props"], int(yr), out_csv)
        if ok:
            out_files.append(out_csv)
            print(f"{name} year {yr}: {len(df_year)} points -> {out_csv}")
        else:
            print(f"{name} year {yr}: no image")

    # combine
    if out_files:
        combined = pd.concat([pd.read_csv(p) for p in out_files], ignore_index=True)
        combined_out = BASE_DIR / "external_sources" / f"{name}_alphaearth_by_year.csv"
        combined.to_csv(combined_out, index=False)
        print(f"Combined -> {combined_out} ({len(combined)} rows)")


def main():
    ee.Initialize()
    for name, cfg in DATASETS.items():
        run_dataset(name, cfg)


if __name__ == "__main__":
    main()
