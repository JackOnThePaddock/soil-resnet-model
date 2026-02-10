#!/usr/bin/env python
"""Estimate gain from fused features (AlphaEarth vs AlphaEarth+BareEarth) via grouped RF CV."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_resnet import _find_feature_columns


def _build_groups(df: pd.DataFrame, mode: str = "latlon", round_dp: int = 4):
    mode = mode.lower()
    if mode == "site_id" and "site_id" in df.columns:
        return df["site_id"].astype(str).values
    if mode == "latlon" and {"lat", "lon"}.issubset(df.columns):
        return (
            df["lat"].round(round_dp).astype(str) + "_" + df["lon"].round(round_dp).astype(str)
        ).values
    return None


def _evaluate_dataset(
    df: pd.DataFrame,
    feature_prefix: str,
    targets: list[str],
    n_splits: int = 5,
    group_mode: str = "latlon",
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    feature_cols = _find_feature_columns(df.columns.tolist(), feature_prefix)
    if not feature_cols:
        raise ValueError(f"No features found for prefix '{feature_prefix}'")

    groups = _build_groups(df, mode=group_mode)
    rows = []

    for target in [t.lower() for t in targets]:
        if target not in df.columns:
            continue
        subset = df[feature_cols + [target]].dropna()
        if len(subset) < max(20, n_splits * 4):
            continue

        X = subset[feature_cols].values
        y = subset[target].values
        if groups is not None:
            g = groups[subset.index]
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups=g)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = splitter.split(X, y)

        preds = np.full(len(y), np.nan, dtype=np.float32)
        for tr, te in splits:
            model = RandomForestRegressor(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X[tr], y[tr])
            preds[te] = model.predict(X[te]).astype(np.float32)

        valid = np.isfinite(preds)
        rows.append(
            {
                "target": target,
                "n": int(valid.sum()),
                "r2": float(r2_score(y[valid], preds[valid])),
                "rmse": float(np.sqrt(mean_squared_error(y[valid], preds[valid]))),
                "mae": float(mean_absolute_error(y[valid], preds[valid])),
                "n_features": len(feature_cols),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AlphaEarth vs fused feature performance")
    parser.add_argument("--alpha-csv", type=str, required=True, help="AlphaEarth-only dataset CSV")
    parser.add_argument("--fused-csv", type=str, required=True, help="Fused dataset CSV")
    parser.add_argument("--alpha-prefix", type=str, default="band_", help="Feature prefix in alpha dataset")
    parser.add_argument("--fused-prefix", type=str, default="feat_", help="Feature prefix in fused dataset")
    parser.add_argument("--targets", type=str, default="ph,cec,esp,soc,ca,mg,na")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--group-mode", type=str, default="latlon", choices=["latlon", "site_id", "none"])
    parser.add_argument("--output-csv", type=str, default="results/metrics/bare_earth_ablation.csv")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    group_mode = args.group_mode if args.group_mode != "none" else "none"

    alpha_df = pd.read_csv(args.alpha_csv)
    fused_df = pd.read_csv(args.fused_csv)

    alpha_res = _evaluate_dataset(
        alpha_df, feature_prefix=args.alpha_prefix, targets=targets, n_splits=args.n_splits, group_mode=group_mode
    )
    alpha_res["dataset"] = "alphaearth"
    fused_res = _evaluate_dataset(
        fused_df, feature_prefix=args.fused_prefix, targets=targets, n_splits=args.n_splits, group_mode=group_mode
    )
    fused_res["dataset"] = "fused"

    combined = pd.concat([alpha_res, fused_res], ignore_index=True)
    pivot = combined.pivot_table(index="target", columns="dataset", values="r2")
    if {"alphaearth", "fused"}.issubset(set(pivot.columns)):
        gain = (pivot["fused"] - pivot["alphaearth"]).rename("r2_gain_fused_minus_alpha")
        gain_df = gain.reset_index()
        combined = combined.merge(gain_df, on="target", how="left")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved ablation metrics: {out_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
