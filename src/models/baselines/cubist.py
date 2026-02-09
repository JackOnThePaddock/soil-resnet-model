"""Cubist baseline model with grid search."""

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid

from src.evaluation.metrics import compute_metrics


def _patch_cubist():
    """Patch Cubist's _escapes to handle float values."""
    try:
        from cubist import _make_names_string as mns, _make_data_string as mds

        def _safe_escapes(x):
            chars = [":", ";", "|"]
            x = [str(c) for c in x]
            for i in chars:
                x = [c.replace(i, f"\\{i}") for c in x]
            return [re.escape(c) for c in x]

        mns._escapes = _safe_escapes
        mds._escapes = _safe_escapes
    except Exception:
        pass


def train_cubist_baseline(
    csv_path: str, targets: List[str], output_dir: str,
) -> pd.DataFrame:
    """Train Cubist models for each target with grid search."""
    try:
        from cubist import Cubist
    except ImportError:
        print("cubist not installed: pip install cubist")
        return pd.DataFrame()

    _patch_cubist()
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    band_cols = sorted([c for c in df.columns if c.startswith("a") or c.startswith("b")])

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {"n_committees": [1, 5, 10, 20], "neighbors": [None, 1, 5], "unbiased": [False]}

    for target in [t.lower() for t in targets]:
        if target not in df.columns:
            continue
        dft = df[band_cols + [target]].dropna().reset_index(drop=True)
        if len(dft) < 10:
            continue
        X_df, y = dft[band_cols], dft[target].values

        best_rmse, best_params = np.inf, None
        for params in ParameterGrid(param_grid):
            preds = np.empty(len(y))
            for train_idx, test_idx in cv.split(X_df):
                model = Cubist(random_state=42, **params)
                model.fit(X_df.iloc[train_idx], y[train_idx])
                preds[test_idx] = model.predict(X_df.iloc[test_idx])
            rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_params = rmse, params

        # Final eval with best params
        preds = np.empty(len(y))
        for train_idx, test_idx in cv.split(X_df):
            model = Cubist(random_state=42, **best_params)
            model.fit(X_df.iloc[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X_df.iloc[test_idx])

        metrics = compute_metrics(y.reshape(-1, 1), preds.reshape(-1, 1), [target])
        m = metrics[target]
        results.append({"target": target, "n": len(y), "r2": m["r2"], "rmse": m["rmse"],
                        "mae": m["mae"], "best_params": str(best_params)})
        print(f"  Cubist {target}: R2={m['r2']:.3f}")

    out_df = pd.DataFrame(results)
    out_path = Path(output_dir) / "cubist_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df
