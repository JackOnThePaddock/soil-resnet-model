import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp"
train = pd.read_csv(BASE + "\\training_data_combined.csv")
best = pd.read_csv(BASE + "\\best_features.csv")

band_cols = [c for c in train.columns if c.startswith("A")]

rows = []
for target in ["pH", "CEC", "ESP"]:
    sub = train.dropna(subset=[target]).copy()
    y_all = sub[target].values.astype(float)
    groups = sub["paddock"].astype(str).values

    feat_sets = {"all_64": band_cols}
    feat_row = best[best["target"] == target].iloc[0]
    feats = [f.strip() for f in str(feat_row["features"]).split(",") if f.strip()]
    feat_sets["best_bands"] = feats

    for set_name, feats in feat_sets.items():
        X_all = sub[feats].values.astype(float)
        for paddock in sorted(set(groups)):
            test_mask = groups == paddock
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            model = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_features="sqrt",
            )
            model.fit(X_all[train_mask], y_all[train_mask])
            preds = model.predict(X_all[test_mask])
            y_true = y_all[test_mask]
            rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
            mae = float(mean_absolute_error(y_true, preds))
            r2 = float('nan')
            if len(y_true) >= 2 and np.std(y_true) > 0:
                r2 = float(r2_score(y_true, preds))
            rows.append({
                "target": target,
                "feature_set": set_name,
                "paddock": paddock,
                "n_test": int(test_mask.sum()),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            })

out = pd.DataFrame(rows)
print(out.sort_values(["target", "feature_set", "paddock"]).to_string(index=False))
out_path = BASE + "\\cv_paddock_holdout_summary.csv"
out.to_csv(out_path, index=False)
print("saved", out_path)
