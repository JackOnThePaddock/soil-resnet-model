"""Gaussian Process Regression baseline with Matern kernel."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


def build_gpr(random_state: int = 42) -> GaussianProcessRegressor:
    """Build GPR with Matern kernel."""
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=random_state)


def loocv_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    """Run LOOCV for GPR, return metrics."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in loo.split(X):
        x_scaler = StandardScaler().fit(X[train_idx])
        y_scaler = StandardScaler().fit(y[train_idx].reshape(-1, 1))
        model = build_gpr()
        model.fit(x_scaler.transform(X[train_idx]), y_scaler.transform(y[train_idx].reshape(-1, 1)).ravel())
        pred_scaled, _ = model.predict(x_scaler.transform(X[test_idx]), return_std=True)
        preds[test_idx] = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    return {
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
    }


def train_gpr_baseline(
    csv_path: str, targets: List[str], output_dir: str,
) -> pd.DataFrame:
    """Train GPR (LOOCV) for each target. Note: slow for large datasets."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    band_cols = sorted([c for c in df.columns if c.startswith("a")])

    results = []
    for target in [t.lower() for t in targets]:
        if target not in df.columns:
            continue
        dft = df[band_cols + [target]].dropna()
        if len(dft) < 5 or len(dft) > 500:
            print(f"  Skipping GPR for {target} (n={len(dft)}, max 500 for LOOCV)")
            continue
        X, y = dft[band_cols].values.astype(np.float32), dft[target].values.astype(np.float32)
        print(f"  GPR {target} (n={len(y)}, LOOCV)...")
        m = loocv_metrics(X, y)
        results.append({"target": target, "n": len(y), **m})
        print(f"    R2={m['r2']:.3f}, RMSE={m['rmse']:.3f}")

    out_df = pd.DataFrame(results)
    out_path = Path(output_dir) / "gpr_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df
