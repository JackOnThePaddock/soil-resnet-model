import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp"
train = pd.read_csv(BASE + "\\training_data_combined.csv")
best = pd.read_csv(BASE + "\\best_features.csv")


def loocv_preds(X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in loo.split(X):
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
bias_rows = []

for target in ["pH", "CEC", "ESP"]:
    sub = train.dropna(subset=[target]).copy()
    y = sub[target].values.astype(np.float32)
    paddocks = sub["paddock"].astype(str).values

    feat_row = best[best["target"] == target].iloc[0]
    feats = [f.strip() for f in str(feat_row["features"]).split(",") if f.strip()]
    X = sub[feats].values.astype(np.float32)

    preds = loocv_preds(X, y)
    rmse, mae, r2 = metrics(y, preds)

    # Paddock mean bias from LOOCV predictions
    bias_map = {}
    for paddock in sorted(set(paddocks)):
        mask = paddocks == paddock
        bias = float(np.mean(preds[mask] - y[mask]))
        bias_map[paddock] = bias
        bias_rows.append({
            "target": target,
            "paddock": paddock,
            "mean_error": bias,
            "n_points": int(mask.sum()),
        })

    preds_corr = np.array([preds[i] - bias_map[paddocks[i]] for i in range(len(preds))])
    rmse_c, mae_c, r2_c = metrics(y, preds_corr)

    rows.append({
        "target": target,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_bias_corrected": rmse_c,
        "mae_bias_corrected": mae_c,
        "r2_bias_corrected": r2_c,
        "n_features": len(feats),
        "n_samples": len(y),
        "bias_method": "paddock_mean_from_loocv",
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))

out_path = BASE + "\\loocv_accuracy_bias_corrected.csv"
out.to_csv(out_path, index=False)

bias_df = pd.DataFrame(bias_rows)
bias_path = BASE + "\\paddock_bias_from_loocv.csv"
bias_df.to_csv(bias_path, index=False)

print("saved", out_path)
print("saved", bias_path)
