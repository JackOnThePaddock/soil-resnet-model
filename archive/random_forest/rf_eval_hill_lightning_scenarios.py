import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_hill_lightning_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "1x1": Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\training_data_combined.csv"),
    "3x3": Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\training_data_combined_3x3.csv"),
}

PADD0CKS = {"HILLPDK", "LIGHTNING_TREE"}

# Keep this light to avoid long runtimes
CANDIDATE_COUNTS = [10, 20, 30, 64]
N_ESTIMATORS_RANK = 150
N_ESTIMATORS_CV = 150
N_ESTIMATORS_EVAL = 200
SEED = 42


def rf_model(n_estimators):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=SEED,
        max_features="sqrt",
    )


def loocv_metrics(X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in loo.split(X):
        model = rf_model(N_ESTIMATORS_EVAL)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


def kfold_metrics(X, y, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in kf.split(X):
        model = rf_model(N_ESTIMATORS_CV)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


def lopo_metrics(X, y, groups):
    logo = LeaveOneGroupOut()
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in logo.split(X, y, groups):
        model = rf_model(N_ESTIMATORS_EVAL)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


def per_paddock_holdout(X, y, groups):
    rows = []
    for paddock in sorted(set(groups)):
        test_mask = groups == paddock
        train_mask = ~test_mask
        model = rf_model(N_ESTIMATORS_EVAL)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        y_true = y[test_mask]
        r2 = float("nan")
        if len(y_true) >= 2 and np.std(y_true) > 0:
            r2 = float(r2_score(y_true, preds))
        rows.append({
            "paddock": paddock,
            "n_test": int(test_mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "mae": float(mean_absolute_error(y_true, preds)),
            "r2": r2,
        })
    return rows


def rank_features(X, y, band_cols):
    model = rf_model(N_ESTIMATORS_RANK)
    model.fit(X, y)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    return [band_cols[i] for i in order]


feature_metrics = []
best_rows = []
cv_rows = []
holdout_rows = []

for ds_name, path in DATASETS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["paddock"].isin(PADD0CKS)].copy()
    if df.empty:
        raise RuntimeError(f"No rows for paddocks in {ds_name}")

    band_cols = [c for c in df.columns if c.startswith("A")]
    if len(band_cols) != 64:
        raise ValueError(f"Expected 64 bands in {ds_name}, got {len(band_cols)}")

    for target in ["pH", "CEC", "ESP"]:
        sub = df.dropna(subset=[target]).copy()
        X = sub[band_cols].values.astype(np.float32)
        y = sub[target].values.astype(np.float32)
        groups = sub["paddock"].astype(str).values

        ordered_features = rank_features(X, y, band_cols)

        # Select best feature count by 3-fold RMSE
        best = None
        for n in CANDIDATE_COUNTS:
            feats = ordered_features[:n]
            Xn = sub[feats].values.astype(np.float32)
            rmse, mae, r2 = kfold_metrics(Xn, y)
            feature_metrics.append({
                "dataset": ds_name,
                "target": target,
                "n_features": n,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "selection_cv": "3fold",
            })
            if best is None or rmse < best["rmse"]:
                best = {"n_features": n, "rmse": rmse, "mae": mae, "r2": r2, "features": feats}

        best_rows.append({
            "dataset": ds_name,
            "target": target,
            "n_features": best["n_features"],
            "rmse": best["rmse"],
            "mae": best["mae"],
            "r2": best["r2"],
            "features": ",".join(best["features"]),
            "selection_cv": "3fold",
        })

        # Evaluate all-64 and best-bands
        for feats, label in [(band_cols, "all_64"), (best["features"], "best_bands")]:
            Xset = sub[feats].values.astype(np.float32)
            rmse, mae, r2 = loocv_metrics(Xset, y)
            cv_rows.append({
                "dataset": ds_name,
                "target": target,
                "feature_set": label,
                "cv": "loocv",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_features": len(feats),
                "n_samples": len(y),
            })

            rmse, mae, r2 = kfold_metrics(Xset, y)
            cv_rows.append({
                "dataset": ds_name,
                "target": target,
                "feature_set": label,
                "cv": "3fold",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_features": len(feats),
                "n_samples": len(y),
            })

            rmse, mae, r2 = lopo_metrics(Xset, y, groups)
            cv_rows.append({
                "dataset": ds_name,
                "target": target,
                "feature_set": label,
                "cv": "leave_one_paddock_out",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_features": len(feats),
                "n_samples": len(y),
                "n_groups": len(set(groups)),
            })

        # Per-paddock holdout for best-bands
        X_best = sub[best["features"]].values.astype(np.float32)
        for row in per_paddock_holdout(X_best, y, groups):
            holdout_rows.append({
                "dataset": ds_name,
                "target": target,
                "feature_set": "best_bands",
                **row,
            })


pd.DataFrame(feature_metrics).to_csv(OUT_DIR / "feature_count_metrics.csv", index=False)
pd.DataFrame(best_rows).to_csv(OUT_DIR / "best_features.csv", index=False)
pd.DataFrame(cv_rows).to_csv(OUT_DIR / "cv_accuracy_summary.csv", index=False)
pd.DataFrame(holdout_rows).to_csv(OUT_DIR / "cv_paddock_holdout_summary.csv", index=False)

print(f"Saved outputs to: {OUT_DIR}")
