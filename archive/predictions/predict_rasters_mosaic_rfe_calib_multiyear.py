import json
from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.merge import merge
import joblib


BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
EMB_DIR = BASE_DIR / "Imagery" / "embeddings_annual"
OUT_ROOT = BASE_DIR / "outputs" / "predictions"
OUT_MOS = OUT_ROOT / "rfe_calib_mosaics"
OUT_MOS.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "outputs" / "models_rfe_calib"
SEL_BANDS = MODEL_DIR / "rfe_selected_bands.json"

NODATA = -9999.0


def year_from_name(path):
    m = re.search(r"_(\d{4})\.tif$", path.name)
    return int(m.group(1)) if m else None


def list_years():
    years = set()
    for tif in EMB_DIR.glob("*.tif"):
        yr = year_from_name(tif)
        if yr:
            years.add(yr)
    return sorted(years)


def band_name_to_index(band):
    return int(band.replace("A", ""))  # A00 -> 0


def predict_raster(in_path, out_path, base_model, cal_model, band_idx):
    with rasterio.open(in_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(window=window)  # (bands, rows, cols)
                bands, rows, cols = data.shape
                flat = np.moveaxis(data, 0, -1).reshape(-1, bands)
                valid = np.isfinite(flat).all(axis=1)
                if src.nodata is not None and np.isfinite(src.nodata):
                    valid &= ~(flat == src.nodata).any(axis=1)

                out = np.full((flat.shape[0],), NODATA, dtype="float32")
                if valid.any():
                    X = flat[valid][:, band_idx]
                    base_pred = base_model.predict(X)
                    X_cal = np.column_stack([base_pred, X])
                    cal_pred = cal_model.predict(X_cal)
                    out[valid] = cal_pred.astype("float32")

                out_img = out.reshape(rows, cols)
                dst.write(out_img, 1, window=window)


def mosaic_year(year, label, files):
    if not files:
        return None
    srcs = [rasterio.open(p) for p in files]
    mosaic, out_transform = merge(srcs, nodata=NODATA)
    out_meta = srcs[0].meta.copy()
    out_meta.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "nodata": NODATA,
            "count": 1,
            "dtype": "float32",
            "compress": "deflate",
        }
    )
    out_path = OUT_MOS / f"farm_mosaic_{label}_{year}.tif"
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic[0].astype("float32"), 1)
    for s in srcs:
        s.close()
    print(f"Wrote {out_path}")
    return out_path


def median_or_mean(label, mosaic_paths, out_path, mode="median"):
    srcs = [rasterio.open(p) for p in mosaic_paths]
    ref = srcs[0]
    for s in srcs[1:]:
        if (
            s.width != ref.width
            or s.height != ref.height
            or s.transform != ref.transform
            or s.crs != ref.crs
        ):
            raise RuntimeError("Mosaic grids do not match; reproject required.")

    profile = ref.profile.copy()
    profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")

    with rasterio.open(out_path, "w", **profile) as dst:
        for _, window in ref.block_windows(1):
            stack = []
            for s in srcs:
                arr = s.read(1, window=window).astype("float32")
                arr[arr == NODATA] = np.nan
                stack.append(arr)
            data_stack = np.stack(stack, axis=0)
            if mode == "mean":
                data = np.nanmean(data_stack, axis=0)
            else:
                data = np.nanmedian(data_stack, axis=0)
            data = np.where(np.isnan(data), NODATA, data).astype("float32")
            dst.write(data, 1, window=window)

    for s in srcs:
        s.close()
    print(f"Wrote {out_path}")


def main():
    sel = json.loads(SEL_BANDS.read_text(encoding="utf-8"))
    targets = ["ph", "cec_cmolkg", "esp_pct"]

    models = {}
    band_idx = {}
    for t in targets:
        base = joblib.load(MODEL_DIR / f"base_catboost_{t}.joblib")
        cal = joblib.load(MODEL_DIR / f"cal_catboost_{t}.joblib")
        models[t] = (base, cal)
        band_idx[t] = [band_name_to_index(b) for b in sel[t]]

    years = list_years()
    mosaics = {"ph": [], "cec": [], "esp": []}

    for year in years:
        out_dir_year = OUT_ROOT / f"rfe_calib_rasters_{year}"
        out_dir_year.mkdir(parents=True, exist_ok=True)

        paddock_ph = []
        paddock_cec = []
        paddock_esp = []

        for tif in sorted(EMB_DIR.glob(f"*_{year}.tif")):
            name = tif.stem
            out_ph = out_dir_year / f"{name}_ph.tif"
            out_cec = out_dir_year / f"{name}_cec.tif"
            out_esp = out_dir_year / f"{name}_esp.tif"

            if not out_ph.exists():
                predict_raster(tif, out_ph, models["ph"][0], models["ph"][1], band_idx["ph"])
            if not out_cec.exists():
                predict_raster(tif, out_cec, models["cec_cmolkg"][0], models["cec_cmolkg"][1], band_idx["cec_cmolkg"])
            if not out_esp.exists():
                predict_raster(tif, out_esp, models["esp_pct"][0], models["esp_pct"][1], band_idx["esp_pct"])

            paddock_ph.append(out_ph)
            paddock_cec.append(out_cec)
            paddock_esp.append(out_esp)

        mos_ph = mosaic_year(year, "ph", paddock_ph)
        mos_cec = mosaic_year(year, "cec", paddock_cec)
        mos_esp = mosaic_year(year, "esp", paddock_esp)

        if mos_ph:
            mosaics["ph"].append(mos_ph)
        if mos_cec:
            mosaics["cec"].append(mos_cec)
        if mos_esp:
            mosaics["esp"].append(mos_esp)

    if not years:
        return
    span = f"{years[0]}_{years[-1]}" if len(years) > 1 else f"{years[0]}"

    if mosaics["ph"]:
        median_or_mean("ph", mosaics["ph"], OUT_MOS / f"farm_mosaic_ph_median_{span}.tif", mode="median")
        median_or_mean("ph", mosaics["ph"], OUT_MOS / f"farm_mosaic_ph_mean_{span}.tif", mode="mean")
    if mosaics["cec"]:
        median_or_mean("cec", mosaics["cec"], OUT_MOS / f"farm_mosaic_cec_median_{span}.tif", mode="median")
        median_or_mean("cec", mosaics["cec"], OUT_MOS / f"farm_mosaic_cec_mean_{span}.tif", mode="mean")
    if mosaics["esp"]:
        median_or_mean("esp", mosaics["esp"], OUT_MOS / f"farm_mosaic_esp_median_{span}.tif", mode="median")
        median_or_mean("esp", mosaics["esp"], OUT_MOS / f"farm_mosaic_esp_mean_{span}.tif", mode="mean")


if __name__ == "__main__":
    main()
