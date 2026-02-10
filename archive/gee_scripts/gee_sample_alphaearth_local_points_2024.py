import ee
import pandas as pd
from pathlib import Path
import requests
import json

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "outputs" / "training" / "training_points_1x1.csv"
OUT_CSV = BASE_DIR / "outputs" / "training" / "training_points_alphaearth_2024.csv"
OUT_META = BASE_DIR / "outputs" / "training" / "training_points_alphaearth_2024_meta.json"

TARGET_YEAR = 2024


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


def main():
    ee.Initialize()

    df = pd.read_csv(IN_CSV)
    # normalize target names
    rename_targets = {
        "Na_cmol": "na_cmolkg",
        "ESP": "esp_pct",
        "pH": "ph",
        "CEC": "cec_cmolkg",
    }
    df = df.rename(columns={k: v for k, v in rename_targets.items() if k in df.columns})

    prop_cols = [
        "paddock",
        "sample_id",
        "lat",
        "lon",
        "ph",
        "cec_cmolkg",
        "esp_pct",
        "na_cmolkg",
    ]
    for c in prop_cols:
        if c not in df.columns:
            df[c] = None

    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    features = [build_feature(row, prop_cols) for _, row in df.iterrows()]
    fc = ee.FeatureCollection(features)

    emb = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filter(ee.Filter.calendarRange(TARGET_YEAR, TARGET_YEAR, "year"))
        .filterBounds(fc)
    )
    if emb.size().getInfo() == 0:
        raise RuntimeError(f"No AlphaEarth annual imagery for {TARGET_YEAR}")

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

    request = {
        "table": sampled,
        "format": "CSV",
        "selectors": ",".join(selectors),
    }
    download_id = ee.data.getTableDownloadId(request)
    url = ee.data.makeTableDownloadUrl(download_id)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    OUT_CSV.write_bytes(resp.content)

    meta = {
        "year": TARGET_YEAR,
        "count_points": len(df),
        "band_names": band_names,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
