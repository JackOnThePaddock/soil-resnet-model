import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

BASE = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
OUT_DIR = BASE / "exports" / "rf_combined_ph_cec_esp_3x3"
TRAIN_PATH = OUT_DIR / "training_data_combined_3x3.csv"
BASELINE_BEST = BASE / "exports" / "rf_combined_ph_cec_esp" / "best_features.csv"

train = pd.read_csv(TRAIN_PATH)
band_cols = [c for c in train.columns if c.startswith("A")]

baseline_best = pd.read_csv(BASELINE_BEST)


def loocv_preds(X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    return preds


def cv_preds(X, y, groups):
    rows = []
    # 5-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.full_like(y, np.nan, dtype=float)
    for tr, te in kf.split(X):
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rows.append(("5fold", preds))

    # leave-one-paddock-out
    logo = LeaveOneGroupOut()
    preds = np.full_like(y, np.nan, dtype=float)
    for tr, te in logo.split(X, y, groups):
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rows.append(("leave_one_paddock_out", preds))

    return rows


def metrics(y, preds):
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


def per_paddock_holdout(X, y, groups):
    rows = []
    for paddock in sorted(set(groups)):
        test_mask = groups == paddock
        train_mask = ~test_mask
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        y_true = y[test_mask]
        r2 = float("nan")
        if len(y_true) >= 2 and np.std(y_true) > 0:
            r2 = float(r2_score(y_true, preds))
        rows.append(
            {
                "paddock": paddock,
                "n_test": int(test_mask.sum()),
                "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
                "mae": float(mean_absolute_error(y_true, preds)),
                "r2": r2,
            }
        )
    return rows


cv_rows = []
holdout_rows = []

for target in ["pH", "CEC", "ESP"]:
    sub = train.dropna(subset=[target]).copy()
    y = sub[target].values.astype(float)
    groups = sub["paddock"].astype(str).values

    # all 64 bands
    X_all = sub[band_cols].values.astype(float)

    # LOOCV
    preds = loocv_preds(X_all, y)
    rmse, mae, r2 = metrics(y, preds)
    cv_rows.append({
        "target": target,
        "feature_set": "all_64",
        "cv": "loocv",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features": len(band_cols),
        "n_samples": len(y),
    })

    # 5-fold + LOPO
    for cv_name, preds in cv_preds(X_all, y, groups):
        rmse, mae, r2 = metrics(y, preds)
        cv_rows.append({
            "target": target,
            "feature_set": "all_64",
            "cv": cv_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_features": len(band_cols),
            "n_samples": len(y),
            "n_groups": len(set(groups)),
        })

    for row in per_paddock_holdout(X_all, y, groups):
        holdout_rows.append({
            "target": target,
            "feature_set": "all_64",
            **row,
        })

    # baseline best-bands (from 1x1 extraction)
    feat_row = baseline_best[baseline_best["target"] == target].iloc[0]
    feats = [f.strip() for f in str(feat_row["features"]).split(",") if f.strip()]
    X_best = sub[feats].values.astype(float)

    preds = loocv_preds(X_best, y)
    rmse, mae, r2 = metrics(y, preds)
    cv_rows.append({
        "target": target,
        "feature_set": "best_bands_1x1",
        "cv": "loocv",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features": len(feats),
        "n_samples": len(y),
    })

    for cv_name, preds in cv_preds(X_best, y, groups):
        rmse, mae, r2 = metrics(y, preds)
        cv_rows.append({
            "target": target,
            "feature_set": "best_bands_1x1",
            "cv": cv_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_features": len(feats),
            "n_samples": len(y),
            "n_groups": len(set(groups)),
        })

    for row in per_paddock_holdout(X_best, y, groups):
        holdout_rows.append({
            "target": target,
            "feature_set": "best_bands_1x1",
            **row,
        })


cv_df = pd.DataFrame(cv_rows)
holdout_df = pd.DataFrame(holdout_rows)

cv_df.to_csv(OUT_DIR / "cv_accuracy_summary.csv", index=False)
holdout_df.to_csv(OUT_DIR / "cv_paddock_holdout_summary.csv", index=False)

print(cv_df)
print(holdout_df)
print(f"saved {OUT_DIR / 'cv_accuracy_summary.csv'}")
print(f"saved {OUT_DIR / 'cv_paddock_holdout_summary.csv'}")
