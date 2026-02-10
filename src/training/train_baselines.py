"""Unified baseline model training (SVR, Random Forest, Extra Trees)."""

from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.evaluation.metrics import compute_metrics


def _feature_sort_key(column_name: str) -> Tuple[int, int, str]:
    """Sort feature columns by trailing numeric suffix when available."""
    match = re.search(r"(\d+)$", column_name)
    if match:
        return (0, int(match.group(1)), column_name)
    return (1, -1, column_name)


def _resolve_feature_columns(df: pd.DataFrame, feature_prefix: str = "auto") -> List[str]:
    """Resolve feature columns from explicit prefix or robust fallback patterns."""
    columns = df.columns.tolist()
    prefix = feature_prefix.lower().strip()

    if prefix and prefix != "auto":
        pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
        matched = [c for c in columns if pattern.match(c)]
        if matched:
            return sorted(matched, key=_feature_sort_key)
        matched = [c for c in columns if c.startswith(prefix)]
        if matched:
            return sorted(matched, key=_feature_sort_key)

    for fallback_pattern in (r"^band_\d+$", r"^ae_\d+$", r"^a\d+$"):
        pattern = re.compile(fallback_pattern)
        matched = [c for c in columns if pattern.match(c)]
        if matched:
            return sorted(matched, key=_feature_sort_key)

    return []


def train_svr(X: np.ndarray, y: np.ndarray, cv: KFold) -> Tuple[object, np.ndarray, dict]:
    """Train SVR with RBF kernel via grid search."""
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    param_grid = {
        "svr__C": [1, 10, 100],
        "svr__gamma": ["scale", 0.1, 0.01],
        "svr__epsilon": [0.05, 0.1, 0.2],
    }
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X, y)
    preds = cross_val_predict(grid.best_estimator_, X, y, cv=cv, n_jobs=-1)
    return grid.best_estimator_, preds, grid.best_params_


def train_random_forest(X: np.ndarray, y: np.ndarray, cv: KFold) -> Tuple[object, np.ndarray, dict]:
    """Train Random Forest via grid search."""
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    }
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X, y)
    preds = cross_val_predict(grid.best_estimator_, X, y, cv=cv, n_jobs=-1)
    return grid.best_estimator_, preds, grid.best_params_


def train_extra_trees(X: np.ndarray, y: np.ndarray, cv: KFold) -> Tuple[object, np.ndarray, dict]:
    """Train Extra Trees via grid search."""
    et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    }
    grid = GridSearchCV(et, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X, y)
    preds = cross_val_predict(grid.best_estimator_, X, y, cv=cv, n_jobs=-1)
    return grid.best_estimator_, preds, grid.best_params_


def train_all_baselines(
    csv_path: str,
    targets: List[str],
    output_dir: str,
    feature_prefix: str = "auto",
    n_features: int = 64,
) -> pd.DataFrame:
    """Train all baseline models on each target, save comparison metrics."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    band_cols = _resolve_feature_columns(df, feature_prefix=feature_prefix)
    if not band_cols:
        raise ValueError(
            f"No feature columns found. Tried prefix '{feature_prefix}' with band_/ae_/aNN fallbacks."
        )
    if len(band_cols) != int(n_features):
        raise ValueError(f"Expected {n_features} feature columns, found {len(band_cols)}")

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in [t.lower() for t in targets]:
        if target not in df.columns:
            print(f"  Skipping {target}: not in data")
            continue
        dft = df[band_cols + [target]].dropna()
        if len(dft) < 10:
            continue
        X = dft[band_cols].values
        y = dft[target].values
        print(f"\n  Target: {target} (n={len(y)})")

        for model_name, trainer in [("SVR", train_svr), ("RandomForest", train_random_forest), ("ExtraTrees", train_extra_trees)]:
            try:
                _, preds, params = trainer(X, y, cv)
                metrics = compute_metrics(y.reshape(-1, 1), preds.reshape(-1, 1), [target])
                m = metrics[target]
                results.append({"model": model_name, "target": target, "n": len(y),
                                "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"], "params": str(params)})
                print(f"    {model_name}: R2={m['r2']:.3f}, RMSE={m['rmse']:.3f}")
            except Exception as e:
                print(f"    {model_name}: FAILED - {e}")

    out_df = pd.DataFrame(results)
    out_path = Path(output_dir) / "baseline_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out_df
