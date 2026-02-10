import json
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT = Path(r"C:/Users/jackc/Documents/EW WH & MG SPEIRS/SOIL Testing model Data")
OUT_DIR = PROJECT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = PROJECT / "outputs" / "training" / "training_points_1x1_with_400.csv"

TARGETS = ["pH", "CEC", "Na_cmol", "ESP"]
TOP_NS = [5, 8, 10, 12, 15, 20, 30, 40, 64]
RANDOM_STATE = 42


def eval_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c.startswith("b")]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    loo = LeaveOneOut()

    # Moderate SVR grid (rbf/linear/poly)
    param_grid = [
        {
            "svr__kernel": ["rbf"],
            "svr__C": [0.1, 1, 10],
            "svr__gamma": ["scale", 0.1, 0.01],
            "svr__epsilon": [0.01, 0.05, 0.1],
        },
        {
            "svr__kernel": ["linear"],
            "svr__C": [0.1, 1, 10],
            "svr__epsilon": [0.01, 0.05, 0.1],
        },
        {
            "svr__kernel": ["poly"],
            "svr__C": [0.1, 1, 10],
            "svr__gamma": ["scale", 0.1],
            "svr__epsilon": [0.01, 0.05],
            "svr__degree": [2, 3],
            "svr__coef0": [0, 1],
        },
    ]

    rows = []
    for target in TARGETS:
        df_t = df.dropna(subset=[target] + feature_cols)
        X = df_t[feature_cols]
        y = df_t[target].values

        # RFE ranking using linear SVR
        rfe_est = LinearSVR(C=1.0, epsilon=0.1, max_iter=10000, random_state=RANDOM_STATE)
        rfe = RFE(rfe_est, n_features_to_select=1, step=1)
        rfe.fit(X, y)
        rfe_order = (
            pd.Series(rfe.ranking_, index=feature_cols)
            .sort_values()
            .index.tolist()
        )

        for n in TOP_NS:
            n = min(n, len(rfe_order))
            subset = rfe_order[:n]
            Xs = X[subset].values

            pipeline = make_pipeline(StandardScaler(), SVR(cache_size=500))
            grid = GridSearchCV(
                pipeline,
                param_grid,
                scoring="neg_root_mean_squared_error",
                cv=kf,
                n_jobs=-1,
            )
            grid.fit(Xs, y)
            best_model = grid.best_estimator_

            pred_5fold = cross_val_predict(best_model, Xs, y, cv=kf, n_jobs=1)
            rmse_5, mae_5, r2_5 = eval_metrics(y, pred_5fold)

            pred_loo = cross_val_predict(best_model, Xs, y, cv=loo, n_jobs=1)
            rmse_loo, mae_loo, r2_loo = eval_metrics(y, pred_loo)

            rows.append(
                {
                    "target": target,
                    "feature_method": "rfe",
                    "top_n": n,
                    "n_samples": len(df_t),
                    "best_params": json.dumps(grid.best_params_),
                    "features": ";".join(subset),
                    "rmse_5fold": rmse_5,
                    "mae_5fold": mae_5,
                    "r2_5fold": r2_5,
                    "rmse_loocv": rmse_loo,
                    "mae_loocv": mae_loo,
                    "r2_loocv": r2_loo,
                }
            )

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "svr_rfe_with400_metrics.csv"
    out.to_csv(out_path, index=False)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
