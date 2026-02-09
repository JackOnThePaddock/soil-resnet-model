"""Feature selection via RFE and Random Forest importance."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def rfe_select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int = 24,
) -> Tuple[List[str], np.ndarray]:
    """Select features using Recursive Feature Elimination with LinearSVR."""
    rfe = RFE(
        estimator=LinearSVR(C=1.0, epsilon=0.1, dual=False, loss="squared_epsilon_insensitive", max_iter=5000),
        n_features_to_select=n_features,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("rfe", rfe)])
    pipe.fit(X, y)
    support = pipe.named_steps["rfe"].support_
    selected = [f for f, keep in zip(feature_names, support) if keep]
    return selected, support


def rf_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 500,
) -> pd.DataFrame:
    """Compute Random Forest feature importance scores."""
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance = rf.feature_importances_
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
