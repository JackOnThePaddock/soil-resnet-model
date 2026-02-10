import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT = Path(r"C:/Users/jackc/Documents/EW WH & MG SPEIRS/SOIL Testing model Data")
DATA = PROJECT / "outputs" / "training" / "training_points_1x1_with_400.csv"
RFE_METRICS = PROJECT / "outputs" / "models" / "svr_rfe_with400_metrics.csv"
OUT = PROJECT / "outputs" / "models" / "esp_derived_vs_direct.csv"


def eval_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def load_best_row(df, target):
    sub = df[df["target"] == target].copy()
    sub = sub.sort_values("rmse_loocv").head(1)
    if sub.empty:
        raise RuntimeError(f"No rows for target {target}")
    return sub.iloc[0]


def make_model(best_row):
    params = json.loads(best_row["best_params"])
    model = make_pipeline(StandardScaler(), SVR(cache_size=500))
    model.set_params(**params)
    return model


def main():
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c.startswith("b")]

    rfe = pd.read_csv(RFE_METRICS)

    # Best models for CEC, Na_cmol, ESP
    row_cec = load_best_row(rfe, "CEC")
    row_na = load_best_row(rfe, "Na_cmol")
    row_esp = load_best_row(rfe, "ESP")

    # Features used by each best model
    feats_cec = row_cec["features"].split(";")
    feats_na = row_na["features"].split(";")
    feats_esp = row_esp["features"].split(";")

    loo = LeaveOneOut()

    # CEC predictions
    df_cec = df.dropna(subset=["CEC"] + feats_cec).reset_index(drop=True)
    X_cec = df_cec[feats_cec].values
    y_cec = df_cec["CEC"].values
    model_cec = make_model(row_cec)
    pred_cec = cross_val_predict(model_cec, X_cec, y_cec, cv=loo, n_jobs=1)

    # Na predictions
    df_na = df.dropna(subset=["Na_cmol"] + feats_na).reset_index(drop=True)
    X_na = df_na[feats_na].values
    y_na = df_na["Na_cmol"].values
    model_na = make_model(row_na)
    pred_na = cross_val_predict(model_na, X_na, y_na, cv=loo, n_jobs=1)

    # ESP direct predictions
    df_esp = df.dropna(subset=["ESP"] + feats_esp).reset_index(drop=True)
    X_esp = df_esp[feats_esp].values
    y_esp = df_esp["ESP"].values
    model_esp = make_model(row_esp)
    pred_esp_direct = cross_val_predict(model_esp, X_esp, y_esp, cv=loo, n_jobs=1)

    # Align derived ESP to rows with CEC+Na+ESP present
    common = df.dropna(subset=["CEC", "Na_cmol", "ESP"]).reset_index(drop=True)

    # Recompute predictions on common subset for fair comparison
    X_cec_c = common[feats_cec].values
    y_cec_c = common["CEC"].values
    pred_cec_c = cross_val_predict(model_cec, X_cec_c, y_cec_c, cv=loo, n_jobs=1)

    X_na_c = common[feats_na].values
    y_na_c = common["Na_cmol"].values
    pred_na_c = cross_val_predict(model_na, X_na_c, y_na_c, cv=loo, n_jobs=1)

    X_esp_c = common[feats_esp].values
    y_esp_c = common["ESP"].values
    pred_esp_direct_c = cross_val_predict(model_esp, X_esp_c, y_esp_c, cv=loo, n_jobs=1)

    # Derived ESP from predictions: ESP = (Na/CEC) * 100
    eps = 1e-6
    pred_esp_derived = (pred_na_c / (pred_cec_c + eps)) * 100.0

    # Metrics
    rmse_d, mae_d, r2_d = eval_metrics(y_esp_c, pred_esp_derived)
    rmse_m, mae_m, r2_m = eval_metrics(y_esp_c, pred_esp_direct_c)

    out = pd.DataFrame([
        {
            "method": "ESP_derived_from_Na_over_CEC",
            "n_samples": len(common),
            "rmse_loocv": rmse_d,
            "mae_loocv": mae_d,
            "r2_loocv": r2_d,
            "notes": "ESP = 100 * Na_cmol / CEC using LOOCV preds",
        },
        {
            "method": "ESP_direct_SVR",
            "n_samples": len(common),
            "rmse_loocv": rmse_m,
            "mae_loocv": mae_m,
            "r2_loocv": r2_m,
            "notes": "Best SVR+RFE model, LOOCV preds",
        },
    ])

    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
