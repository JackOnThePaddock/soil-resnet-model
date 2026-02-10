import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


# Paths
BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
EMB_DIR = BASE_DIR / "Imagery" / "embeddings_annual"
OUT_CSV = BASE_DIR / "outputs" / "predictions" / "paddock_medians_from_embeddings.csv"

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
    bands = band_cols(df_ph)
    df_ph = df_ph[bands + ["ph"]].dropna()
    X_ph = df_ph[bands]
    y_ph = df_ph["ph"].values
    svr_params = load_best_params("ph", "SVR_RBF")
    svr = SVR(kernel="rbf", C=svr_params["svr__C"], gamma=svr_params["svr__gamma"], epsilon=svr_params["svr__epsilon"])
    ph_model = Pipeline([("scaler", StandardScaler()), ("svr", svr)])
    ph_model.fit(X_ph, y_ph)
    models["ph"] = ph_model

    # CEC -> CatBoost
    cec_path = TRAIN_MERGED / "top10cm_cec_cmolkg_alphaearth_merged.csv"
    df_cec = pd.read_csv(cec_path)
    bands = band_cols(df_cec)
    df_cec = df_cec[bands + ["cec_cmolkg"]].dropna()
    X_cec = df_cec[bands]
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
    bands = band_cols(df_esp)
    df_esp = df_esp[bands + ["esp_pct"]].dropna()
    X_esp = df_esp[bands]
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


def parse_name_year(filename):
    stem = Path(filename).stem
    match = re.search(r"_(\d{4})$", stem)
    if match:
        year = int(match.group(1))
        paddock = stem[: match.start()]
    else:
        year = None
        paddock = stem
    return paddock, year


def read_embeddings(tif_path):
    with rasterio.open(tif_path) as ds:
        arr = ds.read()  # (bands, rows, cols)
        nodata = ds.nodata
    # reshape to (n_pixels, bands)
    bands, rows, cols = arr.shape
    flat = np.moveaxis(arr, 0, -1).reshape(-1, bands)
    valid = np.isfinite(flat).all(axis=1)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~(flat == nodata).any(axis=1)
    return flat, valid


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    models = train_models()

    rows = []
    for tif in sorted(EMB_DIR.glob("*.tif")):
        paddock, year = parse_name_year(tif.name)
        X, valid = read_embeddings(tif)
        if valid.sum() == 0:
            rows.append(
                {
                    "file": tif.name,
                    "paddock": paddock,
                    "year": year,
                    "n_pixels": len(X),
                    "n_valid": 0,
                    "ph_median": None,
                    "cec_median": None,
                    "esp_median": None,
                }
            )
            continue

        Xv = X[valid]
        ph_pred = models["ph"].predict(Xv)
        cec_pred = models["cec_cmolkg"].predict(Xv)
        esp_pred = models["esp_pct"].predict(Xv)

        rows.append(
            {
                "file": tif.name,
                "paddock": paddock,
                "year": year,
                "n_pixels": len(X),
                "n_valid": int(valid.sum()),
                "ph_median": float(np.nanmedian(ph_pred)),
                "cec_median": float(np.nanmedian(cec_pred)),
                "esp_median": float(np.nanmedian(esp_pred)),
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
