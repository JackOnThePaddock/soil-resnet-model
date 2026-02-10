import ee
import numpy as np
import pandas as pd
import rasterio
import shapefile
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


def na_to_cmol(na_value, na_unit):
    try:
        val = float(na_value)
    except (TypeError, ValueError):
        return None
    unit = (na_unit or "").lower()
    if "cmol" in unit:
        return val
    # Assume mg/kg when unit is missing or not cmol(+)/kg
    return val / 230.0


def collect_soil_points(shp_paths, paddock_name):
    rows = []
    for shp_path in shp_paths:
        reader = shapefile.Reader(str(shp_path))
        fields = [f[0] for f in reader.fields[1:]]
        field_idx = {name: idx for idx, name in enumerate(fields)}
        for rec, geom in zip(reader.records(), reader.shapes()):
            if not geom.points:
                continue
            lon, lat = geom.points[0]
            na = rec[field_idx["Na"]] if "Na" in field_idx else None
            na_u = rec[field_idx["Na_U"]] if "Na_U" in field_idx else None
            na_cmol = na_to_cmol(na, na_u)
            if na_cmol is None or np.isnan(na_cmol):
                continue
            rows.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "paddock": paddock_name,
                    "na_cmol": float(na_cmol),
                }
            )
    return rows


def rf_loocv_metrics(X, y, n_estimators=200):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in loo.split(X):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


def predict_raster(in_path, out_path, model):
    nodata = -9999.0
    with rasterio.open(in_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="float32", nodata=nodata, compress="LZW")
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(window=window, masked=True)
                if np.isscalar(data.mask) or data.mask.shape == ():
                    rows, cols = data.shape[1], data.shape[2]
                    mask = np.zeros((rows, cols), dtype=bool)
                else:
                    mask = np.any(data.mask, axis=0)
                    rows, cols = mask.shape
                flat = data.filled(np.nan).reshape(data.shape[0], -1).T
                valid = ~mask.reshape(-1)
                preds = np.full(flat.shape[0], nodata, dtype=np.float32)
                if np.any(valid):
                    Xv = flat[valid]
                    preds[valid] = model.predict(Xv).astype(np.float32)
                preds = preds.reshape(rows, cols)
                dst.write(preds, 1, window=window)


def main():
    ee.Initialize()

    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
    out_dir = base_dir / "exports" / "rf_exchangeable_sodium"
    out_dir.mkdir(parents=True, exist_ok=True)

    hill_paths = [
        base_dir / "EW_WH_and_MG_Speirs" / "Hill_PDK_North_Soil_Sampling_2024-01-01T06-15-21Z.shp",
        base_dir / "EW_WH_and_MG_Speirs" / "Hill_Pdk_South_Soil_Sampling_2024-01-01T06-32-12Z.shp",
    ]
    lightning_paths = [
        base_dir / "EW_WH_and_MG_Speirs" / "Lightning_Tree_Soil_Sampling_2024-01-01T05-43-03Z.shp",
    ]

    points = []
    points += collect_soil_points(hill_paths, "HILLPDK")
    points += collect_soil_points(lightning_paths, "LIGHTNING_TREE")
    if not points:
        raise RuntimeError("No valid soil test points found.")

    features = [
        ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]), {"paddock": row["paddock"], "na_cmol": row["na_cmol"]})
        for row in points
    ]
    fc = ee.FeatureCollection(features)

    alpha = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate("2020-01-01", "2024-12-31")
        .median()
    )

    sample = alpha.sampleRegions(
        collection=fc,
        properties=["paddock", "na_cmol"],
        scale=10,
        geometries=False,
    )
    info = sample.getInfo()
    rows = [f["properties"] for f in info.get("features", [])]
    if not rows:
        raise RuntimeError("No AlphaEarth samples returned.")

    df = pd.DataFrame(rows)
    band_cols = [c for c in df.columns if c.startswith("A")]
    if len(band_cols) != 64:
        raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

    df.to_csv(out_dir / "na_training_data.csv", index=False)

    metrics = []
    for paddock in ["HILLPDK", "LIGHTNING_TREE"]:
        sub = df[df["paddock"] == paddock].dropna(subset=["na_cmol"])
        if sub.empty:
            continue
        X = sub[band_cols].values.astype(np.float32)
        y = sub["na_cmol"].values.astype(np.float32)

        rmse, mae, r2 = rf_loocv_metrics(X, y)
        metrics.append({"paddock": paddock, "samples": len(sub), "rmse": rmse, "mae": mae, "r2": r2})

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X, y)

        emb_path = base_dir / "exports" / "gpr_alphaearth" / "embeddings" / f"{paddock}_alpha_5yr.tif"
        if not emb_path.exists():
            raise FileNotFoundError(emb_path)
        out_path = out_dir / f"{paddock}_exchangeable_Na_cmol_rf.tif"
        predict_raster(emb_path, out_path, model)

    pd.DataFrame(metrics).to_csv(out_dir / "na_rf_loocv_metrics.csv", index=False)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
