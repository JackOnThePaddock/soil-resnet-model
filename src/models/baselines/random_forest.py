"""Random Forest and Extra Trees baselines."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from src.evaluation.metrics import compute_metrics


def train_rf_baseline(
    csv_path: str, targets: List[str], output_dir: str,
) -> pd.DataFrame:
    """Train Random Forest and Extra Trees for each target."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    band_cols = sorted([c for c in df.columns if c.startswith("a")])

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {"n_estimators": [300, 500], "max_depth": [None, 20],
                  "min_samples_leaf": [1, 2], "max_features": ["sqrt"]}

    for target in [t.lower() for t in targets]:
        if target not in df.columns:
            continue
        dft = df[band_cols + [target]].dropna()
        if len(dft) < 10:
            continue
        X, y = dft[band_cols].values, dft[target].values

        for name, cls in [("RandomForest", RandomForestRegressor), ("ExtraTrees", ExtraTreesRegressor)]:
            model = cls(random_state=42, n_jobs=-1)
            grid = GridSearchCV(model, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            grid.fit(X, y)
            preds = cross_val_predict(grid.best_estimator_, X, y, cv=cv, n_jobs=-1)
            metrics = compute_metrics(y.reshape(-1, 1), preds.reshape(-1, 1), [target])
            m = metrics[target]
            results.append({"model": name, "target": target, "n": len(y), "r2": m["r2"],
                            "rmse": m["rmse"], "mae": m["mae"], "best_params": str(grid.best_params_)})
            print(f"  {name} {target}: R2={m['r2']:.3f}")

    out_df = pd.DataFrame(results)
    out_path = Path(output_dir) / "rf_et_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df
