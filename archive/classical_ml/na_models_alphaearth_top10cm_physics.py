import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_physics_5yr.csv"
OUT_DIR = BASE_DIR / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "alphaearth_top10cm_na_physics_model_metrics.csv"


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def run_cv(model, X, y, cv):
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return eval_metrics(y, preds)


def main():
    df = pd.read_csv(IN_CSV)
    band_cols = [c for c in df.columns if c.startswith("A")]
    extra_cols = ["twi", "slope_deg", "prescott"]
    feature_cols = band_cols + extra_cols

    target = "na_cmolkg"
    dft = df[feature_cols + [target]].copy()
    dft = dft.dropna(subset=[target])
    dft = dft.dropna(subset=feature_cols)

    X = dft[feature_cols].values
    y = dft[target].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # SVR (expanded grid)
    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])
    svr_grid = {
        "svr__C": [0.1, 1, 10, 100],
        "svr__gamma": ["scale", 0.1, 0.01, 0.001],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    }
    grid = GridSearchCV(svr_pipe, svr_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X, y)
    rmse, mae, r2 = run_cv(grid.best_estimator_, X, y, cv)
    results.append({"model": "SVR", "rmse": rmse, "mae": mae, "r2": r2, "params": grid.best_params_})

    # SVR with log1p target (if non-negative)
    if y.min() >= 0:
        svr_log = TransformedTargetRegressor(
            regressor=svr_pipe,
            func=np.log1p,
            inverse_func=np.expm1,
        )
        grid_log = GridSearchCV(svr_log, {
            "regressor__svr__C": [0.1, 1, 10, 100],
            "regressor__svr__gamma": ["scale", 0.1, 0.01],
            "regressor__svr__epsilon": [0.01, 0.05, 0.1],
        }, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid_log.fit(X, y)
        rmse, mae, r2 = run_cv(grid_log.best_estimator_, X, y, cv)
        results.append({"model": "SVR_log1p", "rmse": rmse, "mae": mae, "r2": r2, "params": grid_log.best_params_})

    # Random Forest
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [500],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"],
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    rf_search.fit(X, y)
    rmse, mae, r2 = run_cv(rf_search.best_estimator_, X, y, cv)
    results.append({"model": "RandomForest", "rmse": rmse, "mae": mae, "r2": r2, "params": rf_search.best_params_})

    # Extra Trees
    et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    et_grid = {
        "n_estimators": [500],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"],
    }
    et_search = GridSearchCV(et, et_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    et_search.fit(X, y)
    rmse, mae, r2 = run_cv(et_search.best_estimator_, X, y, cv)
    results.append({"model": "ExtraTrees", "rmse": rmse, "mae": mae, "r2": r2, "params": et_search.best_params_})

    # Gradient Boosting
    gbr = GradientBoostingRegressor(random_state=42)
    gbr_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0],
    }
    gbr_search = GridSearchCV(gbr, gbr_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gbr_search.fit(X, y)
    rmse, mae, r2 = run_cv(gbr_search.best_estimator_, X, y, cv)
    results.append({"model": "GradientBoosting", "rmse": rmse, "mae": mae, "r2": r2, "params": gbr_search.best_params_})

    # HistGradientBoosting
    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb_grid = {
        "max_depth": [None, 10, 20],
        "learning_rate": [0.05, 0.1],
        "max_iter": [300, 500],
    }
    hgb_search = GridSearchCV(hgb, hgb_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    hgb_search.fit(X, y)
    rmse, mae, r2 = run_cv(hgb_search.best_estimator_, X, y, cv)
    results.append({"model": "HistGradientBoosting", "rmse": rmse, "mae": mae, "r2": r2, "params": hgb_search.best_params_})

    # KNN
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])
    knn_grid = {
        "knn__n_neighbors": [5, 10, 20],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    knn_search = GridSearchCV(knn_pipe, knn_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    knn_search.fit(X, y)
    rmse, mae, r2 = run_cv(knn_search.best_estimator_, X, y, cv)
    results.append({"model": "KNN", "rmse": rmse, "mae": mae, "r2": r2, "params": knn_search.best_params_})

    # Ridge / Lasso
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])
    ridge_grid = {"ridge__alpha": [0.1, 1, 10, 100]}
    ridge_search = GridSearchCV(ridge_pipe, ridge_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    ridge_search.fit(X, y)
    rmse, mae, r2 = run_cv(ridge_search.best_estimator_, X, y, cv)
    results.append({"model": "Ridge", "rmse": rmse, "mae": mae, "r2": r2, "params": ridge_search.best_params_})

    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(max_iter=10000))
    ])
    lasso_grid = {"lasso__alpha": [0.001, 0.01, 0.1, 1]}
    lasso_search = GridSearchCV(lasso_pipe, lasso_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    lasso_search.fit(X, y)
    rmse, mae, r2 = run_cv(lasso_search.best_estimator_, X, y, cv)
    results.append({"model": "Lasso", "rmse": rmse, "mae": mae, "r2": r2, "params": lasso_search.best_params_})

    pd.DataFrame(results).sort_values("rmse").to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
