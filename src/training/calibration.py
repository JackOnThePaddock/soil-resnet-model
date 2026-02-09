"""National-to-local calibration using CatBoost."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def select_best_base_model(
    X: np.ndarray, y: np.ndarray, random_state: int = 42,
) -> Tuple[str, object, dict, float]:
    """Select best national-scale base model via CV."""
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    best_name, best_model, best_params, best_rmse = None, None, None, float("inf")

    svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    svr_grid = {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.1], "svr__epsilon": [0.1]}
    search = GridSearchCV(svr_pipe, svr_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X, y)
    rmse = -search.best_score_
    if rmse < best_rmse:
        best_name, best_model, best_params, best_rmse = "SVR_RBF", search.best_estimator_, search.best_params_, rmse

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_grid = {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 2], "max_features": ["sqrt"]}
    search = GridSearchCV(rf, rf_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X, y)
    rmse = -search.best_score_
    if rmse < best_rmse:
        best_name, best_model, best_params, best_rmse = "RandomForest", search.best_estimator_, search.best_params_, rmse

    et = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    search = GridSearchCV(et, rf_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X, y)
    rmse = -search.best_score_
    if rmse < best_rmse:
        best_name, best_model, best_params, best_rmse = "ExtraTrees", search.best_estimator_, search.best_params_, rmse

    return best_name, best_model, best_params, best_rmse


def calibrate_with_catboost(
    national_csv: str, local_csv: str, target: str,
    band_cols: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42,
) -> Dict:
    """Train national base model, then calibrate to local with CatBoost."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("catboost required: pip install catboost")

    if band_cols is None:
        band_cols = [f"A{i:02d}" for i in range(64)]

    nat = pd.read_csv(national_csv)
    nat.columns = nat.columns.str.lower()
    target_lower = target.lower()
    band_cols_lower = [c.lower() for c in band_cols]
    nat = nat[band_cols_lower + [target_lower]].dropna()

    local = pd.read_csv(local_csv)
    local.columns = local.columns.str.lower()
    local = local.dropna(subset=[target_lower] + band_cols_lower)

    X_nat, y_nat = nat[band_cols_lower].values, nat[target_lower].values
    X_local, y_local = local[band_cols_lower].values, local[target_lower].values

    best_name, best_model, best_params, _ = select_best_base_model(X_nat, y_nat, random_state)
    best_model.fit(X_nat, y_nat)

    idx_train, idx_test = train_test_split(np.arange(len(y_local)), test_size=test_size, random_state=random_state)
    base_pred_test = best_model.predict(X_local[idx_test])
    X_cal_train = np.column_stack([best_model.predict(X_local[idx_train]), X_local[idx_train]])
    X_cal_test = np.column_stack([base_pred_test, X_local[idx_test]])

    cal_model = CatBoostRegressor(iterations=500, loss_function="RMSE", random_seed=random_state,
                                   verbose=False, depth=6, learning_rate=0.1, l2_leaf_reg=3)
    cal_model.fit(X_cal_train, y_local[idx_train])
    cal_pred_test = cal_model.predict(X_cal_test)
    y_test = y_local[idx_test]

    return {
        "target": target, "base_model": best_name, "n_national": len(y_nat), "n_local": len(y_local),
        "base_r2": float(r2_score(y_test, base_pred_test)),
        "base_rmse": float(np.sqrt(mean_squared_error(y_test, base_pred_test))),
        "cal_r2": float(r2_score(y_test, cal_pred_test)),
        "cal_rmse": float(np.sqrt(mean_squared_error(y_test, cal_pred_test))),
    }
