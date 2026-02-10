import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold, LeaveOneGroupOut


def smooth_embeddings(in_path, out_path):
    if out_path.exists():
        return out_path
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="float32", compress="LZW")
        with rasterio.open(out_path, "w", **meta) as dst:
            for b in range(1, src.count + 1):
                data = src.read(b).astype(np.float32)
                smooth = uniform_filter(data, size=3, mode="nearest")
                dst.write(smooth, b)
    return out_path


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
    return metrics, best


def cv_metrics(X, y, groups, n_estimators=300):
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in kf.split(X):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rows.append(
        {
            "cv": "5fold",
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
    )

    logo = LeaveOneGroupOut()
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in logo.split(X, y, groups):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rows.append(
        {
            "cv": "leave_one_paddock_out",
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
    )
    return rows


def per_paddock_holdout(X, y, groups, n_estimators=300):
    rows = []
    for paddock in sorted(set(groups)):
        test_mask = groups == paddock
        train_mask = ~test_mask
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        y_true = y[test_mask]
        r2 = float("nan")
        if len(y_true) >= 2 and np.std(y_true) > 0:
            r2 = float(r2_score(y_true, preds))
        rows.append(
            {
                "paddock": paddock,
                "n_test": int(test_mask.sum()),
                "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
                "mae": float(mean_absolute_error(y_true, preds)),
                "r2": r2,
            }
        )
    return rows


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
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
    out_dir = base_dir / "exports" / "rf_combined_ph_cec_esp_3x3"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "training_data_combined_3x3.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    df = pd.read_csv(train_path)
    band_cols = [c for c in df.columns if c.startswith("A")]
    if len(band_cols) != 64:
        raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

    candidate_counts = [5, 8, 10, 12, 15, 20, 30, 40, 64]
    metrics_all = []
    best_rows = []
    cv_rows = []
    holdout_rows = []

    for target in ["pH", "CEC", "ESP"]:
        sub = df.dropna(subset=[target]).copy()
        X = sub[band_cols].values.astype(np.float32)
        y = sub[target].values.astype(np.float32)
        groups = sub["paddock"].astype(str).values

        metrics, best = select_best_features(X, y, band_cols, candidate_counts)
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

        # LOOCV all-64 vs best-bands
        rmse_all, mae_all, r2_all = rf_loocv_metrics(X, y)
        cv_rows.append(
            {
                "target": target,
                "feature_set": "all_64",
                "cv": "loocv",
                "rmse": rmse_all,
                "mae": mae_all,
                "r2": r2_all,
                "n_features": 64,
                "n_samples": len(y),
            }
        )

        X_best = sub[best["features"]].values.astype(np.float32)
        rmse_best, mae_best, r2_best = rf_loocv_metrics(X_best, y)
        cv_rows.append(
            {
                "target": target,
                "feature_set": "best_bands",
                "cv": "loocv",
                "rmse": rmse_best,
                "mae": mae_best,
                "r2": r2_best,
                "n_features": best["n_features"],
                "n_samples": len(y),
            }
        )

        # 5-fold + LOPO for best-bands and all-64
        for feats, label in [(band_cols, "all_64"), (best["features"], "best_bands")]:
            Xset = sub[feats].values.astype(np.float32)
            for row in cv_metrics(Xset, y, groups):
                cv_rows.append(
                    {
                        "target": target,
                        "feature_set": label,
                        "cv": row["cv"],
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "r2": row["r2"],
                        "n_features": len(feats),
                        "n_samples": len(y),
                        "n_groups": len(set(groups)),
                    }
                )

        # Per-paddock holdout for best-bands
        for row in per_paddock_holdout(X_best, y, groups):
            holdout_rows.append(
                {
                    "target": target,
                    "feature_set": "best_bands",
                    **row,
                }
            )

        # Train final model and predict paddocks
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X_best, y)

        emb_dir = base_dir / "exports" / "gpr_alphaearth" / "embeddings"
        smooth_dir = out_dir / "embeddings_3x3"
        smooth_dir.mkdir(parents=True, exist_ok=True)

        for paddock in ["HILLPDK", "LIGHTNING_TREE", "400"]:
            emb_path = emb_dir / f"{paddock}_alpha_5yr.tif"
            if not emb_path.exists():
                continue
            smooth_path = smooth_dir / f"{paddock}_alpha_5yr_3x3.tif"
            smooth_embeddings(emb_path, smooth_path)
            out_path = out_dir / f"{paddock}_{target}_rf_bestbands_3x3.tif"
            predict_raster(smooth_path, out_path, model, band_cols, best["features"])

    pd.DataFrame(metrics_all).to_csv(out_dir / "feature_count_metrics.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / "best_features.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out_dir / "cv_accuracy_summary.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(out_dir / "cv_paddock_holdout_summary.csv", index=False)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
