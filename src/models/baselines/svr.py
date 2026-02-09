"""SVR baseline with RBF kernel and grid search."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.evaluation.metrics import compute_metrics


def train_svr_baseline(
    csv_path: str, targets: List[str], output_dir: str,
) -> pd.DataFrame:
    """Train SVR models for each target, save metrics."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    band_cols = sorted([c for c in df.columns if c.startswith("a")])

    results = []
    for target in [t.lower() for t in targets]:
        if target not in df.columns:
            continue
        dft = df[band_cols + [target]].dropna()
        if len(dft) < 10:
            continue
        X, y = dft[band_cols].values, dft[target].values

        pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
        param_grid = {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.1, 0.01], "svr__epsilon": [0.05, 0.1, 0.2]}
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X, y)
        preds = cross_val_predict(grid.best_estimator_, X, y, cv=cv, n_jobs=-1)
        metrics = compute_metrics(y.reshape(-1, 1), preds.reshape(-1, 1), [target])
        m = metrics[target]
        results.append({"target": target, "n": len(y), "r2": m["r2"], "rmse": m["rmse"],
                        "mae": m["mae"], "best_params": str(grid.best_params_)})
        print(f"  SVR {target}: R2={m['r2']:.3f}")

    out_df = pd.DataFrame(results)
    out_path = Path(output_dir) / "svr_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df
