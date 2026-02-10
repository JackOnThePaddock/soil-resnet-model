import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")

NATIONAL_NA = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_top10cm_alphaearth_5yr.csv"
NATIONAL_ESP = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_pct_top10cm_alphaearth_5yr.csv"
NATIONAL_ALL = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
LOCAL_ALPHAEARTH = BASE_DIR / "outputs" / "training" / "training_points_alphaearth_5yr.csv"

OUT_DIR = BASE_DIR / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PRED_DIR = BASE_DIR / "outputs" / "predictions"
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = OUT_DIR / "national_local_calibration_metrics_recomputed.csv"
OUT_PREDS = OUT_PRED_DIR / "local_calibrated_predictions_recomputed.csv"


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def prepare_local():
    df = pd.read_csv(LOCAL_ALPHAEARTH)
    # normalize target column names if needed
    rename_targets = {
        "Na_cmol": "na_cmolkg",
        "ESP": "esp_pct",
        "pH": "ph",
        "CEC": "cec_cmolkg",
    }
    df = df.rename(columns={k: v for k, v in rename_targets.items() if k in df.columns})
    return df


def build_candidates(include_knn=False):
    candidates = []

    # SVR (RBF)
    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    svr_grid = {
        "svr__C": [0.1, 1, 10, 100],
        "svr__gamma": ["scale", 0.1, 0.01],
        "svr__epsilon": [0.05, 0.1, 0.2],
    }
    candidates.append(("SVR_RBF", svr_pipe, svr_grid))

    # RandomForest
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    }
    candidates.append(("RandomForest", rf, rf_grid))

    # ExtraTrees
    et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    et_grid = {
        "n_estimators": [300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    }
    candidates.append(("ExtraTrees", et, et_grid))

    if include_knn:
        knn_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor()),
        ])
        knn_grid = {
            "knn__n_neighbors": [5, 10, 20],
            "knn__weights": ["distance"],
            "knn__p": [1, 2],
        }
        candidates.append(("KNN", knn_pipe, knn_grid))

    return candidates


def select_best_base_model(X, y, include_knn=False):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_name = None
    best_model = None
    best_params = None
    best_rmse = None

    for name, model, grid in build_candidates(include_knn=include_knn):
        search = GridSearchCV(model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X, y)
        rmse = -search.best_score_
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = search.best_estimator_
            best_params = search.best_params_

    return best_name, best_model, best_params, best_rmse


def run_target(target, national_csv, include_knn=False):
    band_cols = [f"A{i:02d}" for i in range(64)]

    nat = pd.read_csv(national_csv)
    nat = nat[band_cols + [target]].dropna()

    local = prepare_local()
    local = local[band_cols + [target, "paddock", "sample_id", "lat", "lon"]].dropna(subset=[target])
    local = local.dropna(subset=band_cols)

    X_nat = nat[band_cols].values
    y_nat = nat[target].values

    X_local = local[band_cols].values
    y_local = local[target].values

    # Select best base model on national
    best_name, best_model, best_params, best_rmse = select_best_base_model(X_nat, y_nat, include_knn=include_knn)

    # Fit base model on all national
    best_model.fit(X_nat, y_nat)

    # Holdout split for calibration test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_local, y_local, local.index.values, test_size=0.2, random_state=42
    )

    base_pred_train = best_model.predict(X_train)
    base_pred_test = best_model.predict(X_test)

    # Calibration models (on base predictions only)
    calibrators = {
        "Linear": LinearRegression(),
        "GBR": GradientBoostingRegressor(random_state=42),
        "RF": RandomForestRegressor(random_state=42, n_estimators=300, max_depth=5),
    }

    base_rmse, base_mae, base_r2 = metrics(y_test, base_pred_test)

    best_cal_name = None
    best_cal_model = None
    best_cal_rmse = None
    best_cal_metrics = None

    for cname, cmodel in calibrators.items():
        cmodel.fit(base_pred_train.reshape(-1, 1), y_train)
        cal_pred = cmodel.predict(base_pred_test.reshape(-1, 1))
        rmse, mae, r2 = metrics(y_test, cal_pred)
        if best_cal_rmse is None or rmse < best_cal_rmse:
            best_cal_rmse = rmse
            best_cal_name = cname
            best_cal_model = cmodel
            best_cal_metrics = (rmse, mae, r2)

    # Calibrate using ALL local points for predictions
    base_pred_all = best_model.predict(X_local)
    # linear calib for reference
    lin_cal = LinearRegression().fit(base_pred_all.reshape(-1, 1), y_local)
    lin_pred_all = lin_cal.predict(base_pred_all.reshape(-1, 1))

    # best calib on all data
    best_cal_model.fit(base_pred_all.reshape(-1, 1), y_local)
    best_pred_all = best_cal_model.predict(base_pred_all.reshape(-1, 1))

    pred_df = local[["paddock", "sample_id", "lat", "lon", target]].copy()
    pred_df["target"] = target
    pred_df["base_model"] = best_name
    pred_df["pred_base"] = base_pred_all
    pred_df["pred_cal_linear"] = lin_pred_all
    pred_df["pred_cal_best"] = best_pred_all
    pred_df["holdout_split"] = "train"
    pred_df.loc[idx_test, "holdout_split"] = "test"

    metrics_row = {
        "target": target,
        "n_national": len(y_nat),
        "n_local": len(y_local),
        "best_base_model": best_name,
        "best_base_params": best_params,
        "national_cv_rmse": best_rmse,
        "holdout_rmse_base": base_rmse,
        "holdout_mae_base": base_mae,
        "holdout_r2_base": base_r2,
        "best_calibrator": best_cal_name,
        "holdout_rmse_cal": best_cal_metrics[0],
        "holdout_mae_cal": best_cal_metrics[1],
        "holdout_r2_cal": best_cal_metrics[2],
    }

    return metrics_row, pred_df


def main():
    metrics_rows = []
    preds_list = []

    m_na, p_na = run_target("na_cmolkg", NATIONAL_NA, include_knn=False)
    metrics_rows.append(m_na)
    preds_list.append(p_na)

    m_esp, p_esp = run_target("esp_pct", NATIONAL_ESP, include_knn=True)
    metrics_rows.append(m_esp)
    preds_list.append(p_esp)

    m_ph, p_ph = run_target("ph", NATIONAL_ALL, include_knn=False)
    metrics_rows.append(m_ph)
    preds_list.append(p_ph)

    m_cec, p_cec = run_target("cec_cmolkg", NATIONAL_ALL, include_knn=False)
    metrics_rows.append(m_cec)
    preds_list.append(p_cec)

    pd.DataFrame(metrics_rows).to_csv(OUT_METRICS, index=False)
    pd.concat(preds_list, ignore_index=True).to_csv(OUT_PREDS, index=False)

    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_PREDS}")


if __name__ == "__main__":
    main()
