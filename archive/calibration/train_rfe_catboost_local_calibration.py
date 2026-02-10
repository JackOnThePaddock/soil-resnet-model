import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
import joblib


BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
TRAIN_BASE = Path(r"C:\Users\jackc\Documents\National Soil Data Standardised")
TRAIN_MERGED = TRAIN_BASE / "by_year_cleaned_top10cm_metrics_alphaearth" / "merged"
LOCAL_ALPHA = BASE_DIR / "outputs" / "training" / "training_points_alphaearth_2024.csv"

OUT_DIR = BASE_DIR / "outputs" / "models_rfe_calib"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BANDS = OUT_DIR / "rfe_selected_bands.json"
OUT_BAND_CV = OUT_DIR / "rfe_band_selection_cv.csv"
OUT_METRICS = OUT_DIR / "local_calibration_metrics_rfe_catboost.csv"


TARGETS = {
    "ph": TRAIN_MERGED / "top10cm_ph_alphaearth_merged.csv",
    "cec_cmolkg": TRAIN_MERGED / "top10cm_cec_cmolkg_alphaearth_merged.csv",
    "esp_pct": TRAIN_MERGED / "top10cm_esp_pct_alphaearth_merged.csv",
}


def band_cols(df):
    return [c for c in df.columns if re.fullmatch(r"A\d{2}", c)]


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def select_bands_catboost(df, target):
    bands = sorted(band_cols(df))
    df = df[bands + [target]].dropna()
    X = df[bands]
    y = df[target].values

    base_params = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "thread_count": 4,
    }
    model = CatBoostRegressor(**base_params)
    model.fit(X, y)
    importances = model.get_feature_importance()
    ranked = [bands[i] for i in np.argsort(importances)[::-1]]

    k_list = [8, 16, 24, 32, 40, 48, 56, 64]
    k_list = [k for k in k_list if k <= len(ranked)]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    best_k = None
    best_rmse = None

    for k in k_list:
        sel = ranked[:k]
        preds = np.zeros_like(y, dtype=float)
        for tr, te in cv.split(X):
            m = CatBoostRegressor(**base_params)
            m.fit(X.iloc[tr][sel], y[tr])
            preds[te] = m.predict(X.iloc[te][sel])
        rmse = np.sqrt(mean_squared_error(y, preds))
        cv_rows.append({"target": target, "k": k, "rmse": rmse})
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_k = k

    selected = ranked[:best_k] if best_k is not None else ranked
    return selected, cv_rows


def prepare_local():
    df = pd.read_csv(LOCAL_ALPHA)
    rename_targets = {"Na_cmol": "na_cmolkg", "ESP": "esp_pct", "pH": "ph", "CEC": "cec_cmolkg"}
    df = df.rename(columns={k: v for k, v in rename_targets.items() if k in df.columns})
    return df


def main():
    selected_bands = {}
    cv_rows_all = []
    metrics_rows = []

    local = prepare_local()

    for target, path in TARGETS.items():
        df = pd.read_csv(path)
        bands = sorted(band_cols(df))
        if target not in df.columns or not bands:
            continue

        sel_bands, cv_rows = select_bands_catboost(df, target)
        selected_bands[target] = sel_bands
        cv_rows_all.extend(cv_rows)

        # Train base model on national data
        df_nat = df[bands + [target]].dropna()
        X_nat = df_nat[sel_bands]
        y_nat = df_nat[target].values
        base_model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=4,
        )
        base_model.fit(X_nat, y_nat)

        # Local calibration
        local_t = local[sel_bands + [target, "paddock", "sample_id", "lat", "lon"]].dropna(subset=[target])
        local_t = local_t.dropna(subset=sel_bands)
        X_local = local_t[sel_bands]
        y_local = local_t[target].values

        idx_train, idx_test = train_test_split(local_t.index.values, test_size=0.2, random_state=42)
        X_train = X_local.loc[idx_train]
        X_test = X_local.loc[idx_test]
        y_train = y_local[local_t.index.get_indexer(idx_train)]
        y_test = y_local[local_t.index.get_indexer(idx_test)]

        base_pred_train = base_model.predict(X_train)
        base_pred_test = base_model.predict(X_test)

        X_cal_train = np.column_stack([base_pred_train, X_train.values])
        X_cal_test = np.column_stack([base_pred_test, X_test.values])

        cal_model = CatBoostRegressor(
            iterations=500,
            depth=4,
            learning_rate=0.1,
            l2_leaf_reg=3,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=4,
        )
        cal_model.fit(X_cal_train, y_train)
        cal_pred = cal_model.predict(X_cal_test)

        base_rmse, base_mae, base_r2 = metrics(y_test, base_pred_test)
        cal_rmse, cal_mae, cal_r2 = metrics(y_test, cal_pred)

        metrics_rows.append(
            {
                "target": target,
                "n_local": len(y_local),
                "n_national": len(y_nat),
                "n_bands": len(sel_bands),
                "base_rmse": base_rmse,
                "base_mae": base_mae,
                "base_r2": base_r2,
                "cal_rmse": cal_rmse,
                "cal_mae": cal_mae,
                "cal_r2": cal_r2,
            }
        )

        # Fit calibrator on all local points
        base_pred_all = base_model.predict(X_local)
        X_cal_all = np.column_stack([base_pred_all, X_local.values])
        cal_model.fit(X_cal_all, y_local)

        joblib.dump(base_model, OUT_DIR / f"base_catboost_{target}.joblib")
        joblib.dump(cal_model, OUT_DIR / f"cal_catboost_{target}.joblib")

    OUT_BANDS.write_text(json.dumps(selected_bands, indent=2), encoding="utf-8")
    pd.DataFrame(cv_rows_all).to_csv(OUT_BAND_CV, index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT_METRICS, index=False)

    print(f"Wrote {OUT_BANDS}")
    print(f"Wrote {OUT_BAND_CV}")
    print(f"Wrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
