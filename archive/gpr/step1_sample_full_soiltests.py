import ee
import numpy as np
import pandas as pd
import shapefile
from pathlib import Path

ee.Initialize()

base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
out_dir = base_dir / "exports" / "gpr_alphaearth_full"
out_dir.mkdir(parents=True, exist_ok=True)

sampling_dir = base_dir / "EW_WH_and_MG_Speirs"
shp_paths = sorted(sampling_dir.glob("*Soil_Sampling*.shp"))
if not shp_paths:
    raise FileNotFoundError(f"No soil sampling shapefiles found in {sampling_dir}")

exclude_fields = {"ID", "SampleID", "SmpDpth"}

def allowed_numeric_fields(reader: shapefile.Reader) -> set[str]:
    allowed = set()
    for field in reader.fields[1:]:
        name, ftype = field[0], field[1]
        if ftype not in ("N", "F"):
            continue
        if name.endswith("_U") or name in exclude_fields:
            continue
        allowed.add(name)
    return allowed


all_allowed = set()
for shp in shp_paths:
    reader = shapefile.Reader(str(shp))
    all_allowed |= allowed_numeric_fields(reader)

if not all_allowed:
    raise RuntimeError("No numeric soil test fields found.")

features = []
for shp in shp_paths:
    reader = shapefile.Reader(str(shp))
    fields = [f[0] for f in reader.fields[1:]]
    field_idx = {name: idx for idx, name in enumerate(fields)}

    for rec, geom in zip(reader.records(), reader.shapes()):
        if not geom.points:
            continue
        lon, lat = geom.points[0]
        props = {"lon": float(lon), "lat": float(lat)}
        for name in all_allowed:
            if name not in field_idx:
                continue
            val = rec[field_idx[name]]
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isnan(fval):
                continue
            props[name] = fval
        if len(props) <= 2:
            continue
        feat = ee.Feature(ee.Geometry.Point([lon, lat]), props)
        features.append(feat)

fc = ee.FeatureCollection(features)

alpha = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate("2020-01-01", "2024-12-31")
    .median()
)

sample = alpha.sampleRegions(
    collection=fc,
    properties=["lon", "lat"] + sorted(all_allowed),
    scale=10,
    geometries=False,
)

info = sample.getInfo()
rows = [f["properties"] for f in info.get("features", [])]
if not rows:
    raise RuntimeError("No AlphaEarth samples returned for soil test points.")

df = pd.DataFrame(rows)

band_cols = [c for c in df.columns if c.startswith("A")]
if len(band_cols) != 64:
    raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

# Ensure all target fields exist in the table
for name in sorted(all_allowed):
    if name not in df.columns:
        df[name] = np.nan

ordered_cols = band_cols + ["lon", "lat"] + sorted(all_allowed)
df = df[ordered_cols]

out_path = out_dir / "gpr_training_data_full.csv"
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"Rows: {len(df)}")
