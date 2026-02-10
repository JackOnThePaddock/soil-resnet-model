from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from cubist import Cubist
import cubist._make_names_string as mns
import cubist._make_data_string as mds
import re

# Patch Cubist's _escapes to handle float values (pandas 3.0 astype(str) keeps NaN as float)
def _escapes_safe(x):
    chars = [":", ";", "|"]
    x = [str(c) for c in x]
    for i in chars:
        x = [c.replace(i, f"\\{i}") for c in x]
    return [re.escape(c) for c in x]

mns._escapes = _escapes_safe
mds._escapes = _escapes_safe

PROJECT = Path(r"C:/Users/jackc/Documents/EW WH & MG SPEIRS/SOIL Testing model Data")
OUT_DIR = PROJECT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = PROJECT / "outputs" / "training" / "training_points_1x1_with_400.csv"
TARGETS = ["pH", "CEC", "Na_cmol", "ESP"]
RANDOM_STATE = 42


def eval_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def cv_predict_cubist(X, y, params, cv):
    preds = np.empty(len(y))
    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        model = Cubist(random_state=RANDOM_STATE, **params)
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)
    return preds


def main():
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c.startswith("b")]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    loo = LeaveOneOut()

    param_grid = {
        "n_committees": [1, 5, 10, 20],
        "neighbors": [None, 1, 5, 9],
        "unbiased": [False, True],
    }

    rows = []
    for target in TARGETS:
        df_t = df.dropna(subset=[target] + feature_cols).reset_index(drop=True)
        X = df_t[feature_cols]
        y = df_t[target]

        best = None
        best_rmse = np.inf

        for params in ParameterGrid(param_grid):
            pred_5 = cv_predict_cubist(X, y, params, kf)
            rmse_5, mae_5, r2_5 = eval_metrics(y, pred_5)
            if rmse_5 < best_rmse:
                best_rmse = rmse_5
                best = (params, rmse_5, mae_5, r2_5)

        best_params, rmse_5, mae_5, r2_5 = best

        pred_loo = cv_predict_cubist(X, y, best_params, loo)
        rmse_loo, mae_loo, r2_loo = eval_metrics(y, pred_loo)

        rows.append(
            {
                "target": target,
                "n_samples": len(df_t),
                "best_params": best_params,
                "rmse_5fold": rmse_5,
                "mae_5fold": mae_5,
                "r2_5fold": r2_5,
                "rmse_loocv": rmse_loo,
                "mae_loocv": mae_loo,
                "r2_loocv": r2_loo,
            }
        )

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "cubist_metrics_with400.csv"
    out.to_csv(out_path, index=False)
    print("Wrote", out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
