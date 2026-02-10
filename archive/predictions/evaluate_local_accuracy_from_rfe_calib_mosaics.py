from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
LOCAL_POINTS = BASE_DIR / "outputs" / "training" / "training_points_1x1.csv"
MOS_DIR = BASE_DIR / "outputs" / "predictions" / "rfe_calib_mosaics"
OUT_CSV = BASE_DIR / "outputs" / "predictions" / "local_accuracy_from_rfe_calib_mosaics.csv"
OUT_PRED = BASE_DIR / "outputs" / "predictions" / "local_point_predictions_from_rfe_calib_mosaics.csv"

NODATA = -9999.0


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def sample_raster(raster_path, coords_ll):
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        coords = [transformer.transform(lon, lat) for lon, lat in coords_ll]
        vals = []
        for val in src.sample(coords):
            v = float(val[0])
            if v == NODATA or np.isnan(v):
                vals.append(np.nan)
            else:
                vals.append(v)
        return np.array(vals, dtype="float32")


def main():
    df = pd.read_csv(LOCAL_POINTS)
    rename_targets = {"pH": "ph", "CEC": "cec_cmolkg", "ESP": "esp_pct"}
    df = df.rename(columns={k: v for k, v in rename_targets.items() if k in df.columns})

    required = ["lat", "lon", "ph", "cec_cmolkg", "esp_pct"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in local points")

    coords = list(zip(df["lon"].values, df["lat"].values))

    mosaics = {
        "2024": {
            "ph": MOS_DIR / "farm_mosaic_ph_2024.tif",
            "cec_cmolkg": MOS_DIR / "farm_mosaic_cec_2024.tif",
            "esp_pct": MOS_DIR / "farm_mosaic_esp_2024.tif",
        },
        "median_2017_2024": {
            "ph": MOS_DIR / "farm_mosaic_ph_median_2017_2024.tif",
            "cec_cmolkg": MOS_DIR / "farm_mosaic_cec_median_2017_2024.tif",
            "esp_pct": MOS_DIR / "farm_mosaic_esp_median_2017_2024.tif",
        },
        "mean_2017_2024": {
            "ph": MOS_DIR / "farm_mosaic_ph_mean_2017_2024.tif",
            "cec_cmolkg": MOS_DIR / "farm_mosaic_cec_mean_2017_2024.tif",
            "esp_pct": MOS_DIR / "farm_mosaic_esp_mean_2017_2024.tif",
        },
    }

    metrics_rows = []
    pred_rows = []

    for mosaic_name, paths in mosaics.items():
        if not all(p.exists() for p in paths.values()):
            continue

        preds = {}
        for target, path in paths.items():
            preds[target] = sample_raster(path, coords)

        for target in ["ph", "cec_cmolkg", "esp_pct"]:
            y_true = df[target].values
            y_pred = preds[target]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 3:
                continue
            rmse, mae, r2 = metrics(y_true[mask], y_pred[mask])
            metrics_rows.append(
                {
                    "mosaic": mosaic_name,
                    "target": target,
                    "n": int(mask.sum()),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                }
            )

        out = df[["paddock", "sample_id", "lat", "lon", "ph", "cec_cmolkg", "esp_pct"]].copy()
        out["mosaic"] = mosaic_name
        out["pred_ph"] = preds["ph"]
        out["pred_cec"] = preds["cec_cmolkg"]
        out["pred_esp"] = preds["esp_pct"]
        pred_rows.append(out)

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(OUT_CSV, index=False)
        print(f"Wrote {OUT_CSV}")
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(OUT_PRED, index=False)
        print(f"Wrote {OUT_PRED}")


if __name__ == "__main__":
    main()
