import numpy as np
import pandas as pd
from pathlib import Path

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
    return rmse, mae, r2, preds


def main() -> None:
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\gpr_alphaearth")
    data_path = base_dir / "gpr_training_data_combined.csv"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    band_cols = [c for c in df.columns if c.startswith("A")]
    if len(band_cols) != 64:
        raise ValueError(f"Expected 64 AlphaEarth bands, found {len(band_cols)}")

    X = df[band_cols].values.astype(np.float32)
    y_ph = df["pH"].values.astype(np.float32)
    y_cec = df["CEC"].values.astype(np.float32)

    ph_rmse, ph_mae, ph_r2, ph_preds = loocv_metrics(X, y_ph)
    cec_rmse, cec_mae, cec_r2, cec_preds = loocv_metrics(X, y_cec)

    metrics_df = pd.DataFrame(
        [
            {"target": "pH", "rmse": ph_rmse, "mae": ph_mae, "r2": ph_r2},
            {"target": "CEC", "rmse": cec_rmse, "mae": cec_mae, "r2": cec_r2},
        ]
    )
    metrics_path = base_dir / "gpr_cv_metrics_refresh.csv"
    metrics_df.to_csv(metrics_path, index=False)

    preds_path = base_dir / "gpr_loocv_predictions_refresh.csv"
    pd.DataFrame(
        {
            "pH_actual": y_ph,
            "pH_pred": ph_preds,
            "CEC_actual": y_cec,
            "CEC_pred": cec_preds,
        }
    ).to_csv(preds_path, index=False)

    print(metrics_df.to_string(index=False))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()
