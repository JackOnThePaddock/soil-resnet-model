import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


def build_gpr() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)


def loocv_metrics(X_in: np.ndarray, y_in: np.ndarray):
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
    return rmse, mae, r2


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name)


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


def main() -> None:
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports")
    work_dir = base_dir / "gpr_alphaearth_full"
    data_path = work_dir / "gpr_training_data_full.csv"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    band_cols = [c for c in df.columns if c.startswith("A")]
    if len(band_cols) != 64:
        raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

    target_cols = [c for c in df.columns if c not in band_cols and c not in {"lon", "lat"}]
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]

    emb_dir = base_dir / "gpr_alphaearth" / "embeddings"
    emb_files = sorted([p for p in emb_dir.glob("*_alpha_5yr.tif")])
    if not emb_files:
        raise FileNotFoundError(f"No embeddings found in {emb_dir}")

    pred_dir = work_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    var_map = []
    metrics = []

    for target in sorted(target_cols):
        sub = df[band_cols + [target]].dropna()
        if len(sub) < 5:
            continue
        y_vals = sub[target].values.astype(np.float32)
        if np.nanstd(y_vals) == 0:
            continue

        X = sub[band_cols].values.astype(np.float32)
        rmse, mae, r2 = loocv_metrics(X, y_vals)
        metrics.append({"target": target, "samples": len(sub), "rmse": rmse, "mae": mae, "r2": r2})

        x_scaler = StandardScaler().fit(X)
        X_scaled = x_scaler.transform(X)
        y_scaler = StandardScaler().fit(y_vals.reshape(-1, 1))

        model = build_gpr()
        model.fit(X_scaled, y_scaler.transform(y_vals.reshape(-1, 1)).ravel())

        safe_var = safe_name(target)
        var_map.append({"target": target, "safe_name": safe_var})
        var_dir = pred_dir / safe_var
        var_dir.mkdir(parents=True, exist_ok=True)

        for emb_path in emb_files:
            paddock = emb_path.stem.replace("_alpha_5yr", "")
            out_mean = var_dir / f"{paddock}_gpr_mean.tif"
            out_std = var_dir / f"{paddock}_gpr_std.tif"
            if out_mean.exists() and out_std.exists():
                continue
            predict_raster(emb_path, out_mean, out_std, model, x_scaler, y_scaler)

    metrics_path = work_dir / "gpr_cv_metrics_full.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    var_map_path = work_dir / "variable_map.csv"
    pd.DataFrame(var_map).to_csv(var_map_path, index=False)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved variable map: {var_map_path}")


if __name__ == "__main__":
    main()
