import pandas as pd
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")

NATIONAL_ALL = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
LOCAL_ALPHAEARTH = BASE_DIR / "outputs" / "training" / "training_points_alphaearth_5yr.csv"

OUT_DIR = BASE_DIR / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PRED_DIR = BASE_DIR / "outputs" / "predictions"
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = OUT_DIR / "national_local_calibration_metrics_catboost_base_ph_cec.csv"
OUT_PREDS = OUT_PRED_DIR / "local_calibrated_predictions_catboost_base_ph_cec.csv"


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def prepare_local():
    df = pd.read_csv(LOCAL_ALPHAEARTH)
    rename_targets = {
        "Na_cmol": "na_cmolkg",
        "ESP": "esp_pct",
        "pH": "ph",
        "CEC": "cec_cmolkg",
    }
    df = df.rename(columns={k: v for k, v in rename_targets.items() if k in df.columns})
    return df


def run_target(target, cat_params):
    band_cols = [f"A{i:02d}" for i in range(64)]

    nat = pd.read_csv(NATIONAL_ALL)
    nat = nat[band_cols + [target]].dropna()

    local = prepare_local()
    local = local[band_cols + [target, "paddock", "sample_id", "lat", "lon"]].dropna(subset=[target])
    local = local.dropna(subset=band_cols)

    X_nat = nat[band_cols]
    y_nat = nat[target].values

    X_local = local[band_cols]
    y_local = local[target].values

    # Fit CatBoost base model on national
    base_model = CatBoostRegressor(
        iterations=800,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=4,
        **cat_params,
    )
    base_model.fit(X_nat, y_nat)

    # Holdout split
    idx_train, idx_test = train_test_split(local.index.values, test_size=0.2, random_state=42)
    X_train = X_local.loc[idx_train]
    X_test = X_local.loc[idx_test]
    y_train = y_local[local.index.get_indexer(idx_train)]
    y_test = y_local[local.index.get_indexer(idx_test)]

    base_pred_train = base_model.predict(X_train)
    base_pred_test = base_model.predict(X_test)

    base_rmse, base_mae, base_r2 = metrics(y_test, base_pred_test)

    # Calibrator: CatBoost on base_pred + bands
    X_cal_train = np.column_stack([base_pred_train, X_train.values])
    X_cal_test = np.column_stack([base_pred_test, X_test.values])

    cal_model = CatBoostRegressor(
        iterations=500,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=4,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
    )
    cal_model.fit(X_cal_train, y_train)
    cal_pred_test = cal_model.predict(X_cal_test)

    cal_rmse, cal_mae, cal_r2 = metrics(y_test, cal_pred_test)

    # Fit calibrator on ALL local
    base_pred_all = base_model.predict(X_local)
    X_cal_all = np.column_stack([base_pred_all, X_local.values])
    cal_model.fit(X_cal_all, y_local)
    cal_pred_all = cal_model.predict(X_cal_all)

    pred_df = local[["paddock", "sample_id", "lat", "lon", target]].copy()
    pred_df["target"] = target
    pred_df["base_model"] = "CatBoost"
    pred_df["pred_base"] = base_pred_all
    pred_df["pred_cal_catboost"] = cal_pred_all
    pred_df["holdout_split"] = "train"
    pred_df.loc[idx_test, "holdout_split"] = "test"

    metrics_row = {
        "target": target,
        "n_national": len(y_nat),
        "n_local": len(y_local),
        "base_model": "CatBoost",
        "base_params": cat_params,
        "holdout_rmse_base": base_rmse,
        "holdout_mae_base": base_mae,
        "holdout_r2_base": base_r2,
        "holdout_rmse_cal": cal_rmse,
        "holdout_mae_cal": cal_mae,
        "holdout_r2_cal": cal_r2,
    }

    return metrics_row, pred_df


def main():
    cat_params = {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 5}

    metrics_rows = []
    preds_list = []

    m_ph, p_ph = run_target("ph", cat_params)
    metrics_rows.append(m_ph)
    preds_list.append(p_ph)

    m_cec, p_cec = run_target("cec_cmolkg", cat_params)
    metrics_rows.append(m_cec)
    preds_list.append(p_cec)

    pd.DataFrame(metrics_rows).to_csv(OUT_METRICS, index=False)
    pd.concat(preds_list, ignore_index=True).to_csv(OUT_PREDS, index=False)

    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_PREDS}")


if __name__ == "__main__":
    main()
