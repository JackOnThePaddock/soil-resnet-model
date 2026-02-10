import ee
import pandas as pd
from pathlib import Path
import requests
import json
import math

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm.csv"
OUT_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_physics_5yr.csv"
OUT_META = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_physics_5yr_meta.json"


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
    for c in prop_cols:
        if c not in df.columns:
            df[c] = None

    features = [build_feature(row, prop_cols) for _, row in df.iterrows()]
    fc = ee.FeatureCollection(features)

    # AlphaEarth 5-year median (latest 5 years available in collection)
    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterBounds(fc)
    emb = emb.map(lambda img: img.set("year", ee.Date(img.get("system:time_start")).get("year")))
    years = ee.List(emb.aggregate_array("year")).distinct().sort()
    last5 = years.slice(-5)
    emb5 = emb.filter(ee.Filter.inList("year", last5))
    emb_median = emb5.median()

    # Terrain + TWI from MERIT/Hydro
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    elv = merit.select("elv")
    upa = merit.select("upa")
    slope_deg = ee.Terrain.slope(elv).rename("slope_deg")
    slope_rad = slope_deg.multiply(math.pi / 180.0)
    tan_slope = slope_rad.tan().max(ee.Image.constant(1e-4))
    twi = upa.add(1).divide(tan_slope).log().rename("twi")

    # Climate: Prescott Index from ERA5-Land monthly aggregates (2020-2024)
    clim = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    clim = clim.filterDate("2020-01-01", "2025-01-01")
    precip = clim.select("total_precipitation_sum").sum().rename("precip_m")
    pet = clim.select("potential_evaporation_sum").sum().multiply(-1).rename("pet_m")
    prescott = precip.divide(pet.add(1e-6)).rename("prescott")

    img = emb_median.addBands([twi, slope_deg, prescott])

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
        "years": last5.getInfo(),
        "count_points": len(df),
        "band_names": band_names,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
