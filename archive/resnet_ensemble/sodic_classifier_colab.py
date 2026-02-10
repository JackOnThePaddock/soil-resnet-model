"""
Sodicity Classifier (Colab)
==========================
Binary classification: sodic_risk = 1 if ESP > threshold, else 0.

Usage (after uploading data files to Colab):
  !pip -q install xgboost scikit-learn pandas numpy
  !python sodic_classifier_colab.py --data-dir . --pattern "soil_data_*_alphaearth.csv" --threshold 6

Notes:
- Use RAW ESP values (not normalized). This script expects files that include `esp` and `band_0..band_63`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, recall_score, accuracy_score, average_precision_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Single CSV file with raw ESP and band_0..band_63")
    ap.add_argument("--data-dir", default=".", help="Directory to scan for files")
    ap.add_argument("--pattern", default="soil_data_*_alphaearth.csv", help="Glob pattern for input files")
    ap.add_argument("--threshold", type=float, default=6.0, help="ESP threshold for sodic class")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def find_band_cols(df: pd.DataFrame) -> List[str]:
    # Prefer band_0..band_63
    band_cols = [f"band_{i}" for i in range(64) if f"band_{i}" in df.columns]
    if len(band_cols) == 64:
        return band_cols
    # Fallback to A00..A63 if present
    a_cols = [f"a{i:02d}" for i in range(64) if f"a{i:02d}" in df.columns]
    if len(a_cols) == 64:
        return a_cols
    # Mixed/partial
    return band_cols if band_cols else a_cols


def load_data(args) -> pd.DataFrame:
    if args.data:
        p = Path(args.data)
        if not p.exists():
            raise FileNotFoundError(p)
        return pd.read_csv(p)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found with pattern {args.pattern} in {data_dir}")

    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def main():
    args = parse_args()

    df = load_data(args)
    df.columns = df.columns.str.lower()

    if "esp" not in df.columns:
        raise ValueError("ESP column not found. Use raw alphaearth files with ESP.")

    band_cols = find_band_cols(df)
    if len(band_cols) != 64:
        raise ValueError(f"Expected 64 band columns, found {len(band_cols)}")

    # Build target
    df = df.dropna(subset=["esp"]).copy()
    df["sodic_risk"] = (df["esp"] > args.threshold).astype(int)

    # Features
    X = df[band_cols].values
    y = df["sodic_risk"].values

    # Drop rows with any NaN bands
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    base_rate = pos / max(pos + neg, 1)

    print(f"Samples: {len(y)} | Sodic: {pos} | Non-sodic: {neg} | Base rate: {base_rate:.2%}")

    # Class weight
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    acc = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(accuracy_score)).mean()
    rec = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(recall_score)).mean()
    f1 = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(f1_score)).mean()
    pr_auc = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(average_precision_score)).mean()

    print(f"Accuracy: {acc:.2%}")
    print(f"Recall:   {rec:.2%}")
    print(f"F1:       {f1:.2%}")
    print(f"PR-AUC:   {pr_auc:.2%}")


if __name__ == "__main__":
    main()
