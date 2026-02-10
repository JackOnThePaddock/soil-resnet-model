import ee
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
import shapefile
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


ee.Initialize()

base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
out_dir = base_dir / "exports" / "gpr_alphaearth"
emb_dir = out_dir / "embeddings"
pred_dir = out_dir / "predictions"
emb_dir.mkdir(parents=True, exist_ok=True)
pred_dir.mkdir(parents=True, exist_ok=True)

combined_path = out_dir / "gpr_training_data_combined.csv"
if not combined_path.exists():
    raise FileNotFoundError(combined_path)

df = pd.read_csv(combined_path)
band_cols = [c for c in df.columns if c.startswith("A")]
if len(band_cols) != 64:
    raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

X = df[band_cols].values.astype(np.float32)
y_ph = df["pH"].values.astype(np.float32)
y_cec = df["CEC"].values.astype(np.float32)

# Model helpers

def build_gpr():
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)


def loocv_metrics(X_in, y_in):
    loo = LeaveOneOut()
    preds = np.zeros_like(y_in, dtype=np.float64)
    for train_idx, test_idx in loo.split(X_in):
        x_scaler = StandardScaler().fit(X_in[train_idx])
        y_scaler = StandardScaler().fit(y_in[train_idx].reshape(-1, 1))
        X_train = x_scaler.transform(X_in[train_idx])
        y_train = y_scaler.transform(y_in[train_idx].reshape(-1, 1)).ravel()
        model = build_gpr()
        model.fit(X_train, y_train)
        X_test = x_scaler.transform(X_in[test_idx])
        pred_scaled, _ = model.predict(X_test, return_std=True)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        preds[test_idx] = pred

    rmse = float(np.sqrt(mean_squared_error(y_in, preds)))
    mae = float(mean_absolute_error(y_in, preds))
    r2 = float(r2_score(y_in, preds))
    return rmse, mae, r2, preds


# LOOCV metrics
ph_rmse, ph_mae, ph_r2, ph_preds = loocv_metrics(X, y_ph)
cec_rmse, cec_mae, cec_r2, cec_preds = loocv_metrics(X, y_cec)

metrics_path = out_dir / "gpr_cv_metrics.csv"
metrics_df = pd.DataFrame(
    [
        {"target": "pH", "rmse": ph_rmse, "mae": ph_mae, "r2": ph_r2},
        {"target": "CEC", "rmse": cec_rmse, "mae": cec_mae, "r2": cec_r2},
    ]
)
metrics_df.to_csv(metrics_path, index=False)

preds_path = out_dir / "gpr_loocv_predictions.csv"
pd.DataFrame(
    {
        "pH_actual": y_ph,
        "pH_pred": ph_preds,
        "CEC_actual": y_cec,
        "CEC_pred": cec_preds,
    }
).to_csv(preds_path, index=False)

# Fit final models on full data
x_scaler = StandardScaler().fit(X)
X_scaled = x_scaler.transform(X)

y_scaler_ph = StandardScaler().fit(y_ph.reshape(-1, 1))
ph_model = build_gpr()
ph_model.fit(X_scaled, y_scaler_ph.transform(y_ph.reshape(-1, 1)).ravel())


y_scaler_cec = StandardScaler().fit(y_cec.reshape(-1, 1))
cec_model = build_gpr()
cec_model.fit(X_scaled, y_scaler_cec.transform(y_cec.reshape(-1, 1)).ravel())

kernel_path = out_dir / "gpr_model_kernels.txt"
with kernel_path.open("w", encoding="ascii") as f:
    f.write("pH kernel:\n")
    f.write(str(ph_model.kernel_))
    f.write("\n\nCEC kernel:\n")
    f.write(str(cec_model.kernel_))
    f.write("\n")

# AlphaEarth 5-year median for prediction
alpha = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate("2020-01-01", "2024-12-31")
    .median()
)

# Paddock boundaries
boundaries_path = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SpeirsBoundaries\boundaries\boundaries.shp")
if not boundaries_path.exists():
    raise FileNotFoundError(boundaries_path)

reader = shapefile.Reader(str(boundaries_path))
fields = [f[0] for f in reader.fields[1:]]


def safe_name(name):
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name[:64] if name else "paddock"


def shape_to_ee_polygon(shape):
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    rings = []
    for i in range(len(parts) - 1):
        ring = points[parts[i] : parts[i + 1]]
        rings.append(ring)
    return ee.Geometry.Polygon(rings)


def download_embeddings(name, geom, out_tif):
    if out_tif.exists():
        return out_tif
    img = alpha.clip(geom)
    url = img.getDownloadURL(
        {
            "scale": 10,
            "region": geom,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
    )
    tmp_path = out_tif.with_suffix(".download")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    with open(tmp_path, "rb") as f:
        signature = f.read(4)

    if signature.startswith(b"PK"):
        with zipfile.ZipFile(tmp_path, "r") as z:
            tif_names = [n for n in z.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise RuntimeError(f"No GeoTIFF found in {tmp_path}")
            z.extract(tif_names[0], path=out_tif.parent)
            extracted = out_tif.parent / tif_names[0]
            if extracted != out_tif:
                if out_tif.exists():
                    out_tif.unlink()
                extracted.rename(out_tif)
        tmp_path.unlink(missing_ok=True)
    else:
        if out_tif.exists():
            out_tif.unlink()
        tmp_path.rename(out_tif)
    return out_tif


def predict_raster(in_path, out_mean_path, out_std_path, model, x_scaler, y_scaler):
    nodata = -9999.0
    with rasterio.open(in_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="float32", nodata=nodata, compress="LZW")
        with rasterio.open(out_mean_path, "w", **profile) as dst_mean, rasterio.open(
            out_std_path, "w", **profile
        ) as dst_std:
            for _, window in src.block_windows(1):
                data = src.read(window=window, masked=True)
                # data shape: (bands, rows, cols)
                if np.isscalar(data.mask) or data.mask.shape == ():
                    rows, cols = data.shape[1], data.shape[2]
                    mask = np.zeros((rows, cols), dtype=bool)
                else:
                    mask = np.any(data.mask, axis=0)
                    rows, cols = mask.shape
                flat = data.filled(np.nan).reshape(data.shape[0], -1).T
                valid = ~mask.reshape(-1)
                preds = np.full(flat.shape[0], nodata, dtype=np.float32)
                stds = np.full(flat.shape[0], nodata, dtype=np.float32)
                if np.any(valid):
                    Xv = flat[valid]
                    Xv_scaled = x_scaler.transform(Xv)
                    mean_s, std_s = model.predict(Xv_scaled, return_std=True)
                    mean = y_scaler.inverse_transform(mean_s.reshape(-1, 1)).ravel()
                    std = std_s * y_scaler.scale_[0]
                    preds[valid] = mean.astype(np.float32)
                    stds[valid] = std.astype(np.float32)
                preds = preds.reshape(rows, cols)
                stds = stds.reshape(rows, cols)
                dst_mean.write(preds, 1, window=window)
                dst_std.write(stds, 1, window=window)


# Process each paddock
for rec, shape in zip(reader.records(), reader.shapes()):
    attrs = dict(zip(fields, rec))
    name = safe_name(str(attrs.get("FIELD_NAME", "paddock")))
    geom = shape_to_ee_polygon(shape)

    emb_path = emb_dir / f"{name}_alpha_5yr.tif"
    print(f"Downloading embeddings for {name}...")
    download_embeddings(name, geom, emb_path)

    ph_mean = pred_dir / f"{name}_pH_gpr_mean.tif"
    ph_std = pred_dir / f"{name}_pH_gpr_std.tif"
    cec_mean = pred_dir / f"{name}_CEC_gpr_mean.tif"
    cec_std = pred_dir / f"{name}_CEC_gpr_std.tif"

    print(f"Predicting pH for {name}...")
    predict_raster(emb_path, ph_mean, ph_std, ph_model, x_scaler, y_scaler_ph)

    print(f"Predicting CEC for {name}...")
    predict_raster(emb_path, cec_mean, cec_std, cec_model, x_scaler, y_scaler_cec)

print("Done.")
