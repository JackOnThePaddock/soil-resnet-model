import os
import re
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3"
TRAIN_PATH = os.path.join(BASE, "training_data_combined_3x3.csv")
BEST_PATH = os.path.join(BASE, "best_features.csv")
IN_DIR = os.path.join(BASE, "clipped_ndvi35")
OUT_DIR = os.path.join(BASE, "clipped_ndvi35_bias_corrected")
NODATA = -9999.0

os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
best = pd.read_csv(BEST_PATH)

band_cols = [c for c in train.columns if c.startswith("A")]


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


bias_rows = []

# Compute bias per paddock using LOOCV predictions
bias_map = {}
for target in ["pH", "CEC", "ESP"]:
    sub = train.dropna(subset=[target]).copy()
    y = sub[target].values.astype(np.float32)
    groups = sub["paddock"].astype(str).values

    feat_row = best[best["target"] == target].iloc[0]
    feats = [f.strip() for f in str(feat_row["features"]).split(",") if f.strip()]
    X = sub[feats].values.astype(np.float32)

    preds = loocv_preds(X, y)
    for paddock in sorted(set(groups)):
        mask = groups == paddock
        mean_error = float(np.mean(preds[mask] - y[mask]))
        bias_map[(paddock.upper(), target.upper())] = mean_error
        bias_rows.append({
            "target": target,
            "paddock": paddock,
            "mean_error": mean_error,
            "n_points": int(mask.sum()),
        })

pd.DataFrame(bias_rows).to_csv(os.path.join(BASE, "paddock_bias_from_loocv.csv"), index=False)

pattern = re.compile(r"^(400|HILLPDK|LIGHTNING_TREE)_(pH|CEC|ESP)_rf_bestbands_3x3_clip_ndvi35\.tif$", re.IGNORECASE)

applied = []
for fname in os.listdir(IN_DIR):
    m = pattern.match(fname)
    if not m:
        continue
    paddock = m.group(1).upper()
    target = m.group(2).upper()
    key = (paddock, target)
    if key not in bias_map:
        print(f"No bias for {key}, skipping {fname}")
        continue
    bias = bias_map[key]
    in_path = os.path.join(IN_DIR, fname)
    out_name = fname.replace("_clip_ndvi35", "_clip_ndvi35_bias")
    out_path = os.path.join(OUT_DIR, out_name)

    with rasterio.open(in_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        mask = data == NODATA
        data[~mask] = data[~mask] - bias
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

    applied.append({
        "file": fname,
        "paddock": paddock,
        "target": target,
        "bias_applied": -bias,
    })
    print(f"Wrote {out_path}")

pd.DataFrame(applied).to_csv(os.path.join(OUT_DIR, "bias_applied_summary.csv"), index=False)
