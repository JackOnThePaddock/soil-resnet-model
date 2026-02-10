"""National-to-local calibration utilities using CatBoost."""

import re
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


def _feature_sort_key(column_name: str) -> Tuple[int, int, str]:
    """Sort feature columns by trailing numeric suffix when present."""
    match = re.search(r"(\d+)$", column_name)
    if match:
        return (0, int(match.group(1)), column_name)
    return (1, -1, column_name)


def _resolve_band_columns(
    national_df: pd.DataFrame,
    local_df: pd.DataFrame,
    band_cols: Optional[List[str]] = None,
) -> List[str]:
    """Resolve and validate shared feature columns between national and local tables."""
    nat_cols = national_df.columns.tolist()
    local_cols = local_df.columns.tolist()
    nat_set = set(nat_cols)
    local_set = set(local_cols)

    if band_cols:
        band_cols_lower = [c.lower() for c in band_cols]
        missing_nat = [c for c in band_cols_lower if c not in nat_set]
        missing_local = [c for c in band_cols_lower if c not in local_set]
        if missing_nat or missing_local:
            raise ValueError(
                f"Missing requested feature columns. national_missing={missing_nat}, local_missing={missing_local}"
            )
        return band_cols_lower

    shared = nat_set.intersection(local_set)
    for pattern in (r"^band_\d+$", r"^ae_\d+$", r"^a\d+$"):
        regex = re.compile(pattern)
        matched = [c for c in shared if regex.match(c)]
        if matched:
            return sorted(matched, key=_feature_sort_key)

    raise ValueError(
        "Could not infer shared feature columns. Pass `band_cols` explicitly."
    )


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

    nat = pd.read_csv(national_csv)
    nat.columns = nat.columns.str.lower()
    target_lower = target.lower()

    local = pd.read_csv(local_csv)
    local.columns = local.columns.str.lower()

    if target_lower not in nat.columns:
        raise ValueError(f"Target '{target}' not found in national data")
    if target_lower not in local.columns:
        raise ValueError(f"Target '{target}' not found in local data")

    band_cols_lower = _resolve_band_columns(nat, local, band_cols=band_cols)

    nat = nat[band_cols_lower + [target_lower]].dropna()
    local = local.dropna(subset=[target_lower] + band_cols_lower)

    X_nat, y_nat = nat[band_cols_lower].values, nat[target_lower].values
    X_local, y_local = local[band_cols_lower].values, local[target_lower].values

    if len(y_nat) < 20:
        raise ValueError(
            f"Not enough national samples for {target}: {len(y_nat)} (need >= 20)"
        )
    if len(y_local) < 5:
        raise ValueError(
            f"Not enough local samples for {target}: {len(y_local)} (need >= 5)"
        )

    best_name, best_model, best_params, _ = select_best_base_model(X_nat, y_nat, random_state)
    best_model.fit(X_nat, y_nat)

    n_local = len(y_local)
    requested_test_n = max(1, int(round(test_size * n_local)))
    test_n = min(requested_test_n, n_local - 1)
    idx_train, idx_test = train_test_split(
        np.arange(n_local),
        test_size=test_n,
        random_state=random_state,
    )
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
        "base_params": str(best_params),
    }


def calibrate_national_to_local(
    national_data: str,
    local_data: str,
    output_dir: str,
    targets: Optional[List[str]] = None,
    band_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calibrate national predictions to local data across one or more targets.

    Results are saved to `<output_dir>/calibration_metrics.csv`.
    """
    if targets is None:
        targets = ["ph", "cec", "esp", "soc", "ca", "mg", "na"]

    rows: List[Dict] = []
    for target in targets:
        try:
            row = calibrate_with_catboost(
                national_csv=national_data,
                local_csv=local_data,
                target=target,
                band_cols=band_cols,
                test_size=test_size,
                random_state=random_state,
            )
            rows.append(row)
            print(
                f"Calibrated {target}: base_r2={row['base_r2']:.3f}, cal_r2={row['cal_r2']:.3f}"
            )
        except ValueError as exc:
            print(f"Skipping {target}: {exc}")

    if not rows:
        raise ValueError("No calibration targets could be evaluated")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_file = output_path / "calibration_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved calibration metrics: {metrics_file}")
    return metrics_df
