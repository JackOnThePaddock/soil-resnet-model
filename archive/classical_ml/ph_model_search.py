import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT = Path(r"C:/Users/jackc/Documents/EW WH & MG SPEIRS/SOIL Testing model Data")
OUT_DIR = PROJECT / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "no400": PROJECT / "outputs" / "training" / "training_points_1x1.csv",
    "with400": PROJECT / "outputs" / "training" / "training_points_1x1_with_400.csv",
}

TOP_NS = [5, 8, 10, 12, 15, 20, 30, 40, 64]

MODEL_SPECS = [
    ("RF", RandomForestRegressor),
    ("ET", ExtraTreesRegressor),
]

N_ESTIMATORS_LIST = [300, 500]
MAX_FEATURES_LIST = ["sqrt"]

RANDOM_STATE = 42


def eval_cv(model, X, y, cv):
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    rmse = mean_squared_error(y, preds) ** 0.5
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return rmse, mae, r2


def get_feature_orders(X, y, features):
    # RF importance order
    rf_rank = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_rank.fit(X, y)
    rf_order = (
        pd.Series(rf_rank.feature_importances_, index=features)
        .sort_values(ascending=False)
        .index.tolist()
    )

    # RFE order (smaller estimator for speed)
    rfe_est = RandomForestRegressor(
        n_estimators=100,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rfe = RFE(rfe_est, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    rfe_order = (
        pd.Series(rfe.ranking_, index=features)
        .sort_values()
        .index.tolist()
    )

    return rf_order, rfe_order


def run_dataset(label, csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["pH"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith("b")]
    X = df[feature_cols]
    y = df["pH"]

    rf_order, rfe_order = get_feature_orders(X, y, feature_cols)

    # Save feature orders
    pd.DataFrame({"rf_importance_order": rf_order}).to_csv(
        OUT_DIR / f"ph_feature_order_rf_{label}.csv", index=False
    )
    pd.DataFrame({"rfe_order": rfe_order}).to_csv(
        OUT_DIR / f"ph_feature_order_rfe_{label}.csv", index=False
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    loo = LeaveOneOut()

    results = []
    for feat_method in ["all", "rf_importance", "rfe"]:
        for n in TOP_NS:
            if feat_method == "all":
                if n != 64:
                    continue
                subset = feature_cols
            else:
                order = rf_order if feat_method == "rf_importance" else rfe_order
                n = min(n, len(order))
                subset = order[:n]

            for model_name, model_cls in MODEL_SPECS:
                for n_estimators in N_ESTIMATORS_LIST:
                    for max_features in MAX_FEATURES_LIST:
                        model = model_cls(
                            n_estimators=n_estimators,
                            max_features=max_features,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        )
                        rmse, mae, r2 = eval_cv(model, X[subset], y, kf)
                        results.append(
                            {
                                "dataset": label,
                                "feature_method": feat_method,
                                "top_n": n,
                                "model": model_name,
                                "n_estimators": n_estimators,
                                "max_features": max_features,
                                "cv": "5fold",
                                "rmse": rmse,
                                "mae": mae,
                                "r2": r2,
                            }
                        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / f"ph_simulations_5fold_{label}.csv", index=False)

    # LOOCV for top 5 by RMSE
    top = results_df.sort_values("rmse").head(5).reset_index(drop=True)
    loocv_rows = []
    for _, row in top.iterrows():
        if row["feature_method"] == "all":
            subset = feature_cols
        else:
            order = rf_order if row["feature_method"] == "rf_importance" else rfe_order
            subset = order[: int(row["top_n"]) ]

        model_cls = RandomForestRegressor if row["model"] == "RF" else ExtraTreesRegressor
        model = model_cls(
            n_estimators=int(row["n_estimators"]),
            max_features=row["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rmse, mae, r2 = eval_cv(model, X[subset], y, loo)
        loocv_rows.append(
            {
                **row.to_dict(),
                "cv": "loocv",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    loocv_df = pd.DataFrame(loocv_rows)
    loocv_df.to_csv(OUT_DIR / f"ph_simulations_loocv_{label}.csv", index=False)


if __name__ == "__main__":
    for label, path in DATASETS.items():
        run_dataset(label, path)
    print("Done")
