import ee
import pandas as pd
from pathlib import Path
import requests
import json

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm.csv"
OUT_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
OUT_META = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median_meta.json"


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
    # keep rows with valid coordinates
    df = df[df["lat"].notna() & df["lon"].notna()].copy()

    prop_cols = [
        "site_id",
        "date",
        "depth_upper_m",
        "depth_lower_m",
        "ph",
        "cec_cmolkg",
        "esp_pct",
        "na_cmolkg",
    ]
    # ensure required columns exist
    for c in prop_cols:
        if c not in df.columns:
            df[c] = None

    features = [build_feature(row, prop_cols) for _, row in df.iterrows()]
    fc = ee.FeatureCollection(features)

    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterBounds(fc)
    emb = emb.map(lambda img: img.set("year", ee.Date(img.get("system:time_start")).get("year")))
    years = ee.List(emb.aggregate_array("year")).distinct().sort()
    last5 = years.slice(-5)
    emb5 = emb.filter(ee.Filter.inList("year", last5))
    median = emb5.median()

    band_names = median.bandNames()

    sampled = median.sampleRegions(
        collection=fc,
        properties=prop_cols,
        scale=10,
        geometries=False,
        tileScale=2,
    )

    # download to local CSV
    band_list = band_names.getInfo()
    selectors = prop_cols + band_list
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
        "years": last5.getInfo(),
        "count_points": len(df),
        "band_names": band_list,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
