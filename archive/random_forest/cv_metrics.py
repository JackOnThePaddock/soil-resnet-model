import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp"
train = pd.read_csv(BASE + "\\training_data_combined.csv")
best = pd.read_csv(BASE + "\\best_features.csv")

band_cols = [c for c in train.columns if c.startswith("A")]


def cv_predict(X, y, splits):
    preds = np.full_like(y, np.nan, dtype=float)
    for tr, te in splits:
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    return preds


def metrics(y, preds):
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


rows = []
for target in ["pH", "CEC", "ESP"]:
    sub = train.dropna(subset=[target]).copy()
    y = sub[target].values.astype(float)
    groups = sub["paddock"].astype(str).values

    feat_sets = {"all_64": band_cols}
    feat_row = best[best["target"] == target].iloc[0]
    feats = [f.strip() for f in str(feat_row["features"]).split(",") if f.strip()]
    feat_sets["best_bands"] = feats

    for set_name, feats in feat_sets.items():
        X = sub[feats].values.astype(float)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = cv_predict(X, y, kf.split(X))
        rmse, mae, r2 = metrics(y, preds)
        rows.append({
            "target": target,
            "feature_set": set_name,
            "cv": "5fold",
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_features": len(feats),
            "n_samples": len(y),
        })

        logo = LeaveOneGroupOut()
        preds = cv_predict(X, y, logo.split(X, y, groups))
        rmse, mae, r2 = metrics(y, preds)
        rows.append({
            "target": target,
            "feature_set": set_name,
            "cv": "leave_one_paddock_out",
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_features": len(feats),
            "n_samples": len(y),
            "n_groups": len(np.unique(groups)),
        })

out = pd.DataFrame(rows)
print(out)
out_path = BASE + "\\cv_accuracy_summary.csv"
out.to_csv(out_path, index=False)
print("saved", out_path)
