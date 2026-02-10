import ee
import numpy as np
import pandas as pd
import rasterio
import shapefile
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
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
    return val / 230.0


def collect_shp_points(shp_paths, paddock_name):
    rows = []
    for shp_path in shp_paths:
        reader = shapefile.Reader(str(shp_path))
        fields = [f[0] for f in reader.fields[1:]]
        field_idx = {name: idx for idx, name in enumerate(fields)}
        for rec, geom in zip(reader.records(), reader.shapes()):
            if not geom.points:
                continue
            lon, lat = geom.points[0]
            pH = rec[field_idx["pH"]] if "pH" in field_idx else None
            cec = rec[field_idx["CEC"]] if "CEC" in field_idx else None
            na = rec[field_idx["Na"]] if "Na" in field_idx else None
            na_u = rec[field_idx["Na_U"]] if "Na_U" in field_idx else None
            na_cmol = na_to_cmol(na, na_u)
            esp = None
            if na_cmol is not None and cec not in (None, 0):
                try:
                    esp = float(na_cmol) / float(cec) * 100.0
                except (TypeError, ValueError, ZeroDivisionError):
                    esp = None
            rows.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "paddock": paddock_name,
                    "pH": pH,
                    "CEC": cec,
                    "ESP": esp,
                }
            )
    return rows


def collect_400_points(points_csv):
    df = pd.read_csv(points_csv)
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "lon": float(row.Longitude),
                "lat": float(row.Latitude),
                "paddock": "400",
                "pH": float(row.pH_CaCl2),
                "CEC": float(row.CEC_cmol_kg),
                "ESP": float(row.ESP_pct),
            }
        )
    return rows


def rf_loocv_metrics(X, y, n_estimators=300):
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


def select_best_features(X, y, band_cols, candidate_counts):
    estimator = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_features="sqrt",
    )
    rfe = RFE(estimator=estimator, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    order = np.argsort(ranking)
    ordered_features = [band_cols[i] for i in order]

    metrics = []
    best = None
    for n in candidate_counts:
        feats = ordered_features[:n]
        Xn = X[:, [band_cols.index(f) for f in feats]]
        rmse, mae, r2 = rf_loocv_metrics(Xn, y)
        metrics.append({"n_features": n, "rmse": rmse, "mae": mae, "r2": r2})
        if best is None or rmse < best["rmse"]:
            best = {"n_features": n, "rmse": rmse, "mae": mae, "r2": r2, "features": feats}
    return ordered_features, metrics, best


def predict_raster(in_path, out_path, model, band_cols, selected_features):
    nodata = -9999.0
    idx = [band_cols.index(f) for f in selected_features]
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
                    Xv = flat[valid][:, idx]
                    preds[valid] = model.predict(Xv).astype(np.float32)
                preds = preds.reshape(rows, cols)
                dst.write(preds, 1, window=window)


def main():
    ee.Initialize()

    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
    out_dir = base_dir / "exports" / "rf_combined_ph_cec_esp"
    out_dir.mkdir(parents=True, exist_ok=True)

    hill_paths = [
        base_dir / "EW_WH_and_MG_Speirs" / "Hill_PDK_North_Soil_Sampling_2024-01-01T06-15-21Z.shp",
        base_dir / "EW_WH_and_MG_Speirs" / "Hill_Pdk_South_Soil_Sampling_2024-01-01T06-32-12Z.shp",
    ]
    lightning_paths = [
        base_dir / "EW_WH_and_MG_Speirs" / "Lightning_Tree_Soil_Sampling_2024-01-01T05-43-03Z.shp",
    ]
    points_csv = base_dir / "exports" / "rf_exchangeable_sodium" / "400_na_points.csv"

    points = []
    points += collect_shp_points(hill_paths, "HILLPDK")
    points += collect_shp_points(lightning_paths, "LIGHTNING_TREE")
    if points_csv.exists():
        points += collect_400_points(points_csv)

    if not points:
        raise RuntimeError("No points found for training.")

    features = []
    for row in points:
        props = {"paddock": row["paddock"]}
        if row["pH"] is not None:
            props["pH"] = float(row["pH"])
        if row["CEC"] is not None:
            props["CEC"] = float(row["CEC"])
        if row["ESP"] is not None:
            props["ESP"] = float(row["ESP"])
        feat = ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]), props)
        features.append(feat)

    fc = ee.FeatureCollection(features)
    alpha = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate("2020-01-01", "2024-12-31")
        .median()
    )
    sample = alpha.sampleRegions(
        collection=fc,
        properties=["paddock", "pH", "CEC", "ESP"],
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

    df.to_csv(out_dir / "training_data_combined.csv", index=False)

    candidate_counts = [5, 8, 10, 12, 15, 20, 30, 40, 64]
    metrics_all = []
    best_rows = []

    for target in ["pH", "CEC", "ESP"]:
        sub = df.dropna(subset=[target])
        X = sub[band_cols].values.astype(np.float32)
        y = sub[target].values.astype(np.float32)
        ordered_features, metrics, best = select_best_features(X, y, band_cols, candidate_counts)
        for row in metrics:
            metrics_all.append({"target": target, **row})
        best_rows.append(
            {
                "target": target,
                "n_features": best["n_features"],
                "rmse": best["rmse"],
                "mae": best["mae"],
                "r2": best["r2"],
                "features": ",".join(best["features"]),
            }
        )

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        X_best = sub[best["features"]].values.astype(np.float32)
        model.fit(X_best, y)

        emb_dir = base_dir / "exports" / "gpr_alphaearth" / "embeddings"
        for paddock in ["HILLPDK", "LIGHTNING_TREE", "400"]:
            emb_path = emb_dir / f"{paddock}_alpha_5yr.tif"
            if not emb_path.exists():
                continue
            out_path = out_dir / f"{paddock}_{target}_rf_bestbands.tif"
            predict_raster(emb_path, out_path, model, band_cols, best["features"])

    pd.DataFrame(metrics_all).to_csv(out_dir / "feature_count_metrics.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / "best_features.csv", index=False)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
