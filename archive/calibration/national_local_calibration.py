import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")

NATIONAL_NA = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_top10cm_alphaearth_5yr.csv"
NATIONAL_ESP = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_pct_top10cm_alphaearth_5yr.csv"
NATIONAL_ALL = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
LOCAL_POINTS = BASE_DIR / "outputs" / "training" / "training_points_1x1.csv"
LOCAL_ALPHAEARTH = BASE_DIR / "outputs" / "training" / "training_points_alphaearth_5yr.csv"

OUT_DIR = BASE_DIR / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PRED_DIR = BASE_DIR / "outputs" / "predictions"
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = OUT_DIR / "national_local_calibration_metrics.csv"
OUT_PREDS = OUT_PRED_DIR / "local_calibrated_predictions.csv"


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def prepare_local():
    if LOCAL_ALPHAEARTH.exists():
        return pd.read_csv(LOCAL_ALPHAEARTH)
    df = pd.read_csv(LOCAL_POINTS)
    # normalize target column names
    rename_targets = {
        "Na_cmol": "na_cmolkg",
        "ESP": "esp_pct",
        "pH": "ph",
        "CEC": "cec_cmolkg",
    }
    df = df.rename(columns={k: v for k, v in rename_targets.items() if k in df.columns})
    # map b1..b64 to A00..A63
    band_map = {f"b{i}": f"A{i-1:02d}" for i in range(1, 65)}
    df = df.rename(columns=band_map)
    return df


def run_target(target, national_csv, model):
    band_cols = [f"A{i:02d}" for i in range(64)]

    nat = pd.read_csv(national_csv)
    nat = nat[band_cols + [target]].dropna()

    local = prepare_local()
    local = local[band_cols + [target, "paddock", "sample_id", "lat", "lon"]].dropna(subset=[target])
    # drop rows with missing bands
    local = local.dropna(subset=band_cols)

    X_nat = nat[band_cols].values
    y_nat = nat[target].values

    X_local = local[band_cols].values
    y_local = local[target].values

    # Fit base model on national data
    model.fit(X_nat, y_nat)

    # Holdout split for calibration test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_local, y_local, local.index.values, test_size=0.2, random_state=42
    )

    base_pred_train = model.predict(X_train)
    base_pred_test = model.predict(X_test)

    # Calibrator on local train
    calib = LinearRegression().fit(base_pred_train.reshape(-1, 1), y_train)
    calib_pred_test = calib.predict(base_pred_test.reshape(-1, 1))

    # Metrics
    base_rmse, base_mae, base_r2 = metrics(y_test, base_pred_test)
    cal_rmse, cal_mae, cal_r2 = metrics(y_test, calib_pred_test)

    # Calibrate using ALL local points (for paddock predictions)
    base_pred_all = model.predict(X_local)
    calib_all = LinearRegression().fit(base_pred_all.reshape(-1, 1), y_local)
    calib_pred_all = calib_all.predict(base_pred_all.reshape(-1, 1))

    # Prepare prediction output for all local points
    pred_df = local[["paddock", "sample_id", "lat", "lon", target]].copy()
    pred_df["target"] = target
    pred_df["pred_base"] = base_pred_all
    pred_df["pred_calibrated"] = calib_pred_all

    # add holdout flag
    pred_df["holdout_split"] = "train"
    pred_df.loc[idx_test, "holdout_split"] = "test"

    metrics_row = {
        "target": target,
        "n_national": len(y_nat),
        "n_local": len(y_local),
        "holdout_rmse_base": base_rmse,
        "holdout_mae_base": base_mae,
        "holdout_r2_base": base_r2,
        "holdout_rmse_cal": cal_rmse,
        "holdout_mae_cal": cal_mae,
        "holdout_r2_cal": cal_r2,
    }

    return metrics_row, pred_df


def main():
    # Best models from previous runs
    model_na = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, gamma=0.01, epsilon=0.01)),
    ])

    model_esp = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=10, p=1, weights="distance")),
    ])

    model_ph = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=1, gamma="scale", epsilon=0.2)),
    ])

    model_cec = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.2)),
    ])

    metrics_rows = []
    preds_list = []

    m_na, p_na = run_target("na_cmolkg", NATIONAL_NA, model_na)
    metrics_rows.append(m_na)
    preds_list.append(p_na)

    m_esp, p_esp = run_target("esp_pct", NATIONAL_ESP, model_esp)
    metrics_rows.append(m_esp)
    preds_list.append(p_esp)

    m_ph, p_ph = run_target("ph", NATIONAL_ALL, model_ph)
    metrics_rows.append(m_ph)
    preds_list.append(p_ph)

    m_cec, p_cec = run_target("cec_cmolkg", NATIONAL_ALL, model_cec)
    metrics_rows.append(m_cec)
    preds_list.append(p_cec)

    pd.DataFrame(metrics_rows).to_csv(OUT_METRICS, index=False)
    pd.concat(preds_list, ignore_index=True).to_csv(OUT_PREDS, index=False)

    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_PREDS}")


if __name__ == "__main__":
    main()
