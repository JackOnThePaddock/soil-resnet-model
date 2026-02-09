"""Cross-validation strategies for soil property prediction."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut

from src.evaluation.metrics import compute_metrics


def kfold_cv(
    X: np.ndarray, y: np.ndarray, target_name: str,
    n_splits: int = 5, random_state: int = 42,
) -> Dict[str, float]:
    """K-Fold cross-validation with Random Forest."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X):
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1, max_features="sqrt")
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return compute_metrics(y.reshape(-1, 1), preds.reshape(-1, 1), [target_name])[target_name]


def paddock_holdout_cv(
    df: pd.DataFrame,
    band_cols: List[str],
    target: str,
    group_col: str = "paddock",
) -> pd.DataFrame:
    """Leave-one-paddock-out cross-validation."""
    groups = df[group_col].astype(str).values
    X, y = df[band_cols].values, df[target].values
    rows = []

    for paddock in sorted(set(groups)):
        test_mask = groups == paddock
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_features="sqrt", n_jobs=-1)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        metrics = compute_metrics(y[test_mask].reshape(-1, 1), preds.reshape(-1, 1), [target])
        m = metrics.get(target, {})
        rows.append({"paddock": paddock, "n_test": int(test_mask.sum()), **m})

    return pd.DataFrame(rows)
