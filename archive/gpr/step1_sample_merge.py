import ee
import pandas as pd
from pathlib import Path

ee.Initialize()

base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
out_dir = base_dir / "exports" / "gpr_alphaearth"
out_dir.mkdir(parents=True, exist_ok=True)

training_path = base_dir / "exports" / "alpha_5yr_training_data.csv"
points_path = base_dir / "exports" / "400_calibration_points.csv"

if not training_path.exists():
    raise FileNotFoundError(training_path)
if not points_path.exists():
    raise FileNotFoundError(points_path)

df_train = pd.read_csv(training_path)

band_cols = [c for c in df_train.columns if c.startswith("A")]
if len(band_cols) != 64:
    raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

# Load 400 points and map columns
_df_points = pd.read_csv(points_path)
col_map = {
    "Latitude": "lat",
    "Longitude": "lon",
    "pH (CaCl2)": "pH",
    "eCEC (cmol+/kg)": "CEC",
}
_df_points = _df_points.rename(columns=col_map)
required_cols = ["lat", "lon", "pH", "CEC"]
missing = [c for c in required_cols if c not in _df_points.columns]
if missing:
    raise ValueError(f"Missing columns in 400 points CSV: {missing}")

_df_points = _df_points.dropna(subset=required_cols)

# Build EE FeatureCollection
features = []
for row in _df_points.itertuples(index=False):
    geom = ee.Geometry.Point([float(row.lon), float(row.lat)])
    feat = ee.Feature(geom, {"pH": float(row.pH), "CEC": float(row.CEC)})
    features.append(feat)

fc = ee.FeatureCollection(features)

alpha = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate("2020-01-01", "2024-12-31")
    .median()
)

sample = alpha.sampleRegions(
    collection=fc,
    properties=["pH", "CEC"],
    scale=10,
    geometries=False,
)

info = sample.getInfo()
rows = [f["properties"] for f in info.get("features", [])]
if not rows:
    raise RuntimeError("No AlphaEarth samples returned for 400 points.")

df_embed = pd.DataFrame(rows)

# Keep only bands + targets
band_cols_embed = [c for c in df_embed.columns if c.startswith("A")]
# Align to training band order
missing_bands = [c for c in band_cols if c not in band_cols_embed]
if missing_bands:
    raise ValueError(f"Missing AlphaEarth bands in sampled data: {missing_bands}")

df_embed = df_embed[band_cols + ["pH", "CEC"]].dropna()

# Combine datasets
combined = pd.concat(
    [df_train[band_cols + ["pH", "CEC"]], df_embed],
    ignore_index=True,
)

combined_path = out_dir / "gpr_training_data_combined.csv"
embed_path = out_dir / "gpr_400_points_embeddings.csv"
combined.to_csv(combined_path, index=False)
df_embed.to_csv(embed_path, index=False)

print(f"Combined training rows: {len(combined)}")
print(f"Saved: {combined_path}")
print(f"Saved: {embed_path}")
