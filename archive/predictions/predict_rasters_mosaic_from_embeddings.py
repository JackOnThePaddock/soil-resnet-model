import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


# Config
YEAR = 2024
BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
EMB_DIR = BASE_DIR / "Imagery" / "embeddings_annual"
OUT_DIR = BASE_DIR / "outputs" / "predictions" / f"rasters_{YEAR}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BASE = Path(r"C:\Users\jackc\Documents\National Soil Data Standardised")
TRAIN_MERGED = TRAIN_BASE / "by_year_cleaned_top10cm_metrics_alphaearth" / "merged"
METRICS_CSV = TRAIN_BASE / "outputs" / "alphaearth_merged_model_metrics.csv"


def band_cols(df):
    return [c for c in df.columns if re.fullmatch(r"A\d{2}", c)]


def load_best_params(target, model_name):
    df = pd.read_csv(METRICS_CSV)
    row = df[(df["target"] == target) & (df["model"] == model_name)].iloc[0]
    return ast.literal_eval(row["params"])


def train_models():
    models = {}

    # pH -> SVR_RBF
    ph_path = TRAIN_MERGED / "top10cm_ph_alphaearth_merged.csv"
    df_ph = pd.read_csv(ph_path)
    bands = sorted(band_cols(df_ph))
    df_ph = df_ph[bands + ["ph"]].dropna()
    X_ph = df_ph[bands].values
    y_ph = df_ph["ph"].values
    svr_params = load_best_params("ph", "SVR_RBF")
    svr = SVR(kernel="rbf", C=svr_params["svr__C"], gamma=svr_params["svr__gamma"], epsilon=svr_params["svr__epsilon"])
    ph_model = Pipeline([("scaler", StandardScaler()), ("svr", svr)])
    ph_model.fit(X_ph, y_ph)
    models["ph"] = ph_model

    # CEC -> CatBoost
    cec_path = TRAIN_MERGED / "top10cm_cec_cmolkg_alphaearth_merged.csv"
    df_cec = pd.read_csv(cec_path)
    bands = sorted(band_cols(df_cec))
    df_cec = df_cec[bands + ["cec_cmolkg"]].dropna()
    X_cec = df_cec[bands].values
    y_cec = df_cec["cec_cmolkg"].values
    cb_params = load_best_params("cec_cmolkg", "CatBoost")
    cec_model = CatBoostRegressor(
        iterations=500,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=4,
        **cb_params,
    )
    cec_model.fit(X_cec, y_cec)
    models["cec_cmolkg"] = cec_model

    # ESP -> RandomForest
    esp_path = TRAIN_MERGED / "top10cm_esp_pct_alphaearth_merged.csv"
    df_esp = pd.read_csv(esp_path)
    bands = sorted(band_cols(df_esp))
    df_esp = df_esp[bands + ["esp_pct"]].dropna()
    X_esp = df_esp[bands].values
    y_esp = df_esp["esp_pct"].values
    rf_params = load_best_params("esp_pct", "RandomForest")
    esp_model = RandomForestRegressor(
        n_estimators=rf_params.get("n_estimators", 300),
        max_features=rf_params.get("max_features", "sqrt"),
        random_state=42,
        n_jobs=-1,
    )
    esp_model.fit(X_esp, y_esp)
    models["esp_pct"] = esp_model

    return models


def predict_raster(in_path, out_path, model):
    with rasterio.open(in_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(window=window)  # (bands, rows, cols)
                bands, rows, cols = data.shape
                flat = np.moveaxis(data, 0, -1).reshape(-1, bands)
                valid = np.isfinite(flat).all(axis=1)
                if src.nodata is not None and np.isfinite(src.nodata):
                    valid &= ~(flat == src.nodata).any(axis=1)

                out = np.full((flat.shape[0],), -9999.0, dtype="float32")
                if valid.any():
                    preds = model.predict(flat[valid])
                    out[valid] = preds.astype("float32")

                out_img = out.reshape(rows, cols)
                dst.write(out_img, 1, window=window)


def main():
    models = train_models()

    # per-paddock predictions
    paddock_ph = []
    paddock_cec = []
    paddock_esp = []

    for tif in sorted(EMB_DIR.glob(f"*_{YEAR}.tif")):
        name = tif.stem
        out_ph = OUT_DIR / f"{name}_ph.tif"
        out_cec = OUT_DIR / f"{name}_cec.tif"
        out_esp = OUT_DIR / f"{name}_esp.tif"

        predict_raster(tif, out_ph, models["ph"])
        predict_raster(tif, out_cec, models["cec_cmolkg"])
        predict_raster(tif, out_esp, models["esp_pct"])

        paddock_ph.append(out_ph)
        paddock_cec.append(out_cec)
        paddock_esp.append(out_esp)

    # mosaics
    for label, files in [("ph", paddock_ph), ("cec", paddock_cec), ("esp", paddock_esp)]:
        if not files:
            continue
        srcs = [rasterio.open(p) for p in files]
        mosaic, out_transform = merge(srcs, nodata=-9999.0)
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "nodata": -9999.0,
                "count": 1,
                "dtype": "float32",
                "compress": "deflate",
            }
        )
        out_path = OUT_DIR / f"farm_mosaic_{label}_{YEAR}.tif"
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic[0].astype("float32"), 1)
        for s in srcs:
            s.close()

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
