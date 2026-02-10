"""
Sample additional covariates from GEE for ESP modeling.
- Sentinel-2 indices (annual median): NDVI, NDMI, NDSI, BSI
- Seasonal (wet/dry) indices: NDVI, NDSI (Jan-Mar vs Jul-Sep)
- Sentinel-1 SAR: VV_dB, VH_dB, VV_minus_VH (annual median)
- Topography: elevation, slope (SRTM)
- Climate: precipitation sum, PET sum, aridity (TerraClimate)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import ee
except Exception as e:
    print("ERROR: earthengine-api (ee) is not available:", e)
    sys.exit(1)

BASE = Path(r"C:\Users\jackc\Downloads\Soil Data\Soil Tests")
IN_PATH = BASE / "soil_esp_base_2017_2024.csv"
OUT_PATH = BASE / "soil_esp_features_2017_2024.csv"
YEAR_OUT_DIR = BASE / "gee_features_years"

PROJECT = "agenticagro"
CHUNK_SIZE = 250
FORCE_REDO = True


def mask_s2_sr(img):
    qa = img.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return img.updateMask(mask).divide(10000)


def add_s2_indices(img):
    b2 = img.select('B2')  # Blue
    b3 = img.select('B3')  # Green
    b4 = img.select('B4')  # Red
    b8 = img.select('B8')  # NIR
    b11 = img.select('B11')  # SWIR1

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('ndvi')
    ndmi = b8.subtract(b11).divide(b8.add(b11)).rename('ndmi')
    ndsi = b3.subtract(b11).divide(b3.add(b11)).rename('ndsi')
    bsi = b11.add(b4).subtract(b8).subtract(b2).divide(b11.add(b4).add(b8).add(b2)).rename('bsi')

    return img.addBands([ndvi, ndmi, ndsi, bsi])


def s2_composite(year, start_month, end_month, region):
    start = f"{year}-{start_month:02d}-01"
    if end_month == 12:
        end = f"{year}-12-31"
    else:
        end = f"{year}-{end_month:02d}-30"
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(start, end)
            .map(mask_s2_sr)
            .map(add_s2_indices))
    comp = s2.median().select(['ndvi','ndmi','ndsi','bsi'])
    return comp


def s1_composite(year, region):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    s1 = s1.select(['VV','VH'])
    # S1 GRD is already in dB; take median directly
    med = s1.median()
    vv = med.select('VV').rename('s1_vv_db')
    vh = med.select('VH').rename('s1_vh_db')
    ratio = vv.subtract(vh).rename('s1_vv_minus_vh')
    return vv.addBands([vh, ratio])


def climate_composite(year, region):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate(start, end)
    pr = tc.select('pr').sum().rename('tc_pr')
    pet = tc.select('pet').sum().rename('tc_pet')
    aridity = pr.divide(pet.add(1e-6)).rename('tc_aridity')
    return pr.addBands([pet, aridity])


def topo_layers():
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope = ee.Terrain.slope(dem).rename('srtm_slope')
    return dem.rename('srtm_elev').addBands(slope)


def build_feature_image(year, region):
    # Annual S2 indices
    s2_annual = s2_composite(year, 1, 12, region)

    # Seasonal composites (southern hemisphere proxy)
    s2_wet = s2_composite(year, 1, 3, region).select(['ndvi','ndsi']).rename(['ndvi_wet','ndsi_wet'])
    s2_dry = s2_composite(year, 7, 9, region).select(['ndvi','ndsi']).rename(['ndvi_dry','ndsi_dry'])

    s1 = s1_composite(year, region)
    climate = climate_composite(year, region)
    topo = topo_layers()

    img = s2_annual.addBands([s2_wet, s2_dry, s1, climate, topo])
    return img


def to_fc(df, lat_col='lat', lon_col='lon', row_id_col='_row_id'):
    feats = []
    for lat, lon, rid in df[[lat_col, lon_col, row_id_col]].itertuples(index=False, name=None):
        if pd.isna(lat) or pd.isna(lon):
            continue
        feats.append(ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {'row_id': int(rid)}))
    return ee.FeatureCollection(feats)


def sample_year(df_year, year, region):
    img = build_feature_image(year, region)
    img = img.unmask(-9999)

    rows = []
    for start in range(0, len(df_year), CHUNK_SIZE):
        chunk = df_year.iloc[start:start+CHUNK_SIZE]
        fc = to_fc(chunk)
        samples = img.sampleRegions(collection=fc, properties=['row_id'], scale=30, tileScale=4, geometries=False)
        info = samples.getInfo()
        for feat in info.get('features', []):
            rows.append(feat.get('properties', {}))
        print(f"  Chunk {start//CHUNK_SIZE + 1}/{(len(df_year) - 1)//CHUNK_SIZE + 1}")

    if not rows:
        return pd.DataFrame(columns=['row_id'])

    df = pd.DataFrame(rows)
    return df


def main():
    ee.Initialize(project=PROJECT)

    df = pd.read_csv(IN_PATH)
    df = df.reset_index(drop=True)
    df['_row_id'] = df.index.astype(int)

    YEAR_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # region bounds
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # Sample per year and cache
    for year in sorted(df['year'].unique()):
        out_year = YEAR_OUT_DIR / f"features_{int(year)}.csv"
        if out_year.exists() and not FORCE_REDO:
            print(f"Skipping year {year} (cached)")
            continue
        df_y = df[df['year'] == year]
        print(f"Sampling year {year} ({len(df_y)} points)")
        samp = sample_year(df_y, int(year), region)
        samp['year'] = year
        samp = samp.rename(columns={'row_id': '_row_id'})
        samp.to_csv(out_year, index=False)
        print(f"  Wrote {out_year}")

    # Load cached samples
    parts = list(YEAR_OUT_DIR.glob("features_*.csv"))
    if not parts:
        raise SystemExit("No per-year feature files found.")
    samples = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    merged = df.merge(samples, on=['_row_id','year'], how='left')

    # Replace unmask sentinel with NaN
    merged = merged.replace(-9999, np.nan)

    merged.to_csv(OUT_PATH, index=False)
    print('Wrote', OUT_PATH, 'rows', len(merged), 'cols', len(merged.columns))


if __name__ == '__main__':
    main()
